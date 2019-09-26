import tensorflow as tf
import logging
from data_loader import DataLoader
import numpy as np
import tools
from scipy.sparse.linalg import svds
from models import *
#from alexnet_samples.alexnet import *
from class_identifier import ClassIdentifier
from performance_calculator import PerformanceCalculator


class CifarSolver(object):
    """cifar solver"""
    """
    class names: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    """

    def __init__(self, loadfile=None):
        self.logger = logging.getLogger('logger_master')
        self.dataLoader = DataLoader()
        self.dataset_name = "cifar10"

        if loadfile:
            self.loadfile_name = loadfile
        else:
            self.loadfile_name = "cifar10_pickle"

        self.dataLoader.loadData(self.dataset_name, self.loadfile_name)
        self.clustered = None
        self.projection_matrices = None
        self.class_size = 3
        self.performance_calculator = PerformanceCalculator(self.class_size)

    def loadTrainingData(self, size=None):
        self.training_images, self.training_labels = self.dataLoader.getTrainingData(size)
        # for i in range(len(self.training_images)):
        #     self.training_images[i] = self.training_images[i] /255.0
        self.training_images = self.training_images/255.0
        #self.training_images -= np.repeat(np.reshape(np.mean(self.training_images, axis=1), [50000,1]), 784, axis=1)
        self.logger.info("Using {} training images".format(len(self.training_images)))

    def loadTestData(self, size=None):
        self.test_images, self.test_labels = self.dataLoader.getTestData(size)
        # for i in range(len(self.test_images)):
        #     self.test_images[i] = self.test_images[i] / 255.0
        self.test_images = self.test_images/255.0
        #self.test_images -= np.repeat(np.reshape(np.mean(self.test_images, axis=1), [10000,1]), , axis=1)
        self.logger.info("Using {} test images".format(len(self.test_images)))

    def test_images_vectorize(self, hsv=False):
        if self.test_images is None:
            self.loadTestData()
        if hsv:
            temp = tools.convert_rgb_to_hsv_onlyh(self.test_images, sine=True)
            self.test_images_vector = np.reshape(temp, (len(self.test_images), 32 * 32))
        else:
            self.test_images_vector = np.reshape(self.test_images, (len(self.test_images), 32*32*3))

    def print_number_of_elements_per_class(self):
        unique, counts = np.unique(self.training_labels, return_counts=True)
        self.element_numbers = dict(zip(unique, counts))
        self.logger.info("Number of classes: {}".format(self.element_numbers))

    def printImageSizes(self):
        self.logger.info("Image size for cifar 10 {}".format(self.training_images[0].shape))

    def cluster_training_with_ground_truth(self, hsv=False):
        self.print_number_of_elements_per_class()
        self.logger.info("Clustering with ground truth")
        self.image_clustered_with_gt = {}

        for i in self.element_numbers.keys():
            self.logger.info("Clustering data {}".format(i))
            label_index = np.where(self.training_labels == int(i))
            temp = np.take(self.training_images, label_index[0], axis=0)
            if hsv:
                temp2 = tools.convert_rgb_to_hsv_onlyh(temp, sine=True)
                self.image_clustered_with_gt[i] = np.reshape(temp2, (5000,32,32,1))
            else:
                #self.image_clustered_with_gt[i] = np.reshape(temp, (5000, 32*32*3))
                self.image_clustered_with_gt[i] = temp

    def cluster_test_with_ground_truth(self, hsv=False):
        self.print_number_of_elements_per_class()
        self.logger.info("Clustering Testing Data")
        if self.test_images is None:
            self.load_test_data()

        self.clustered_test = {}
        for i in self.element_numbers.keys():
            self.logger.info("Clustring Data {}".format(i))
            label_index = np.where(self.test_labels == int(i))
            temp = np.take(self.test_images, label_index[0], axis=0)
            if hsv:
                temp2 = tools.convert_rgb_to_hsv_onlyh(temp, sine=True)
                self.clustered_test[i] = np.reshape(temp2, (1000, 32, 32,1))
            else:
                self.clustered_test[i] = np.reshape(temp, (1000, 32,32,3))

    def calculateProjectionMatrices(self):
        self.logger.info("Calculating Projection Matrices")
        self.projection_matrices = {}
        if self.clustered is None:
            self.clusterTrainingWithGoundTruth()

        for i, matrix in self.clustered.items():
            self.logger.info("Calculation projection matrix for {}".format(i))
            self.projection_matrices[i] = tools.calculateProjectionMatrix(matrix.astype(float), 100)

    def train(self, model_):
        self.model = model_

        n_training_classes = 3  # number of classes to be used while training. starts from index 0
        self.class_size = n_training_classes
        n_testing_classes = 3  # number of classes to be used while testing, starts from index 0
        class_identifiers = [ClassIdentifier(index, n_training_classes) for index in range(n_training_classes)]
        #self.image_clustered_with_gt[3] = self.image_clustered_with_gt[8]
        #self.image_clustered_with_gt[4] = self.image_clustered_with_gt[9]

        for i in range(n_training_classes):
            class_identifiers[i].create_train_inputs_model(self.dataset_name, self.model, 50)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for identifier in class_identifiers:
                self.logger.info("Initializing identifier {}".format(identifier.class_index))
                identifier.sess = sess
                identifier.create_training_set(self.image_clustered_with_gt)

            # original
            # labels = []
            # for i in range(n_training_classes):
            #     temp = np.zeros((n_training_classes, 1))
            #     temp[i] = 1
            #     labels.append(np.repeat(temp, len(self.image_clustered_with_gt[i]), axis=1))
            # label_set = np.transpose(np.concatenate(labels, axis=1))
            # input_set = np.concatenate(
            #     [self.image_clustered_with_gt[data] for data in range(n_training_classes)], axis=0)
            # dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
            # dataset = dataset.repeat(20000)
            # dataset = dataset.shuffle(buffer_size=10000)
            # batched_dataset = dataset.batch(50)
            # iterator = batched_dataset.make_initializable_iterator()
            # next_element = iterator.get_next()
            # sess.run(iterator.initializer)

            labels = []
            for i in range(n_training_classes):
                temp = np.zeros((n_training_classes, 1))
                temp[i] = 1
                labels.append(np.repeat(temp, len(self.image_clustered_with_gt[i]), axis=1))
            label_set = np.transpose(np.concatenate(labels, axis=1))
            input_set = np.concatenate(
                [self.image_clustered_with_gt[data] for data in range(n_training_classes)], axis=0)
            dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
            dataset = dataset.repeat(20000)
            dataset = dataset.shuffle(buffer_size=10000)
            batched_dataset = dataset.batch(50)
            iterator = batched_dataset.make_initializable_iterator()
            next_element = iterator.get_next()
            sess.run(iterator.initializer)

            self.logger.info("Training Started")
            for _ in range(50):
                for identifier in class_identifiers:
                    identifier.train_one_update(next_element, 10)
                    identifier.calculate_separation()
                    identifier.print_separations()
                    identifier.calculate_class_reference(4)
                # self.logger.info("Small Testing Started")
                # for p in range(10):
                #     self.logger.info("Testing for {}".format(p))
                #     for k in range(50):
                #         for j in range(10):
                #             self.performance_calculator.set_result(j, class_identifiers[j].check_class(self.clustered_test[p][k:k+1]))
                #         self.performance_calculator.check_correct(p)
                # self.performance_calculator.print_accuracy()
                # self.performance_calculator.reset()

                self.logger.info("Testing Started")
                for p in range(n_training_classes):
                    for i in range(n_training_classes):
                        self.performance_calculator.set_result(i, class_identifiers[i].check_class(
                                    self.clustered_test[p]))
                    self.performance_calculator.check_correct(p)
                self.performance_calculator.print_accuracy()
                self.performance_calculator.reset()


    def train_norm(self):
        def weight_variable(shape, name=None):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=name)

        def bias_variable(shape, name=None):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name)

        feature_size = 512
        rank = 50
        n_classes = 10

        # create dataset
        input_set = np.concatenate(
            [self.image_clustered_with_gt[data] for data in range(10)], axis=0)
        labels = [ np.ones(5000), np.zeros(5000*9)]
        label_set = np.concatenate(labels)

        dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
        dataset = dataset.repeat(2000)
        dataset = dataset.shuffle(buffer_size=10000)
        batched_dataset = dataset.batch(50)
        iterator = batched_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        # create model

        x = tf.placeholder(tf.float32, [None, 32, 32, 3], "x")
        y_ = tf.placeholder(tf.int32, [None])
        pro_input = tf.placeholder(tf.float32, [None, feature_size])

        [features, norm_weighted_mean, tf_training_model] = create_norm_vgg19(x, y_, pro_input)

        # train model
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)

            for m in range(10000):
                print("calculating projection matrix")
                temp = []
                for i in range(1000):
                    temp.append(np.transpose(sess.run(features,
                                                           feed_dict={
                                                               x: self.image_clustered_with_gt[0][i*50:50*(i+1)],
                                                               keep_prob: 1.0})))
                feature_matrix = np.concatenate(temp, axis=1)
                feature_matrix = feature_matrix / np.linalg.norm(feature_matrix)
                u, s, v = svds(feature_matrix, rank)
                pro = np.matmul(u, np.transpose(u))

                print("projection matrix calculated")

                for p in range(10):
                    print("traning {}".format(p))
                    for k in range(1000):
                        batch_xs, batch_ys = sess.run(next_element)
                        sess.run(tf_training_model,
                                 feed_dict={x: batch_xs, y_: batch_ys, pro_input: pro,
                                            keep_prob: 1.0})

                print("testing")
                temp = []
                for i in range(100):
                     temp.append(sess.run(norm_weighted_mean,
                                          feed_dict={x: self.image_clustered_with_gt[0][i*50:50*(i+1)],
                                                     pro_input: pro, keep_prob: 1.0}))

                norm_value = np.concatenate(temp)
                #if np.max(norm_value) != 0:
                max_norm_ref = np.max(norm_value)
                true_norm_mean = np.mean(norm_value)
                print("True norm max {}".format(max_norm_ref))
                print("True norm mean {}".format(true_norm_mean))

                wrong_norm_value = 0
                wrong_min = 1
                total_norm_diff = 0
                achieved_count = 0
                whole_wrong_mean = 100
                for j in range(0, 10):
                    if j != 0:
                        temp = []
                        for i in range(100):
                            temp.append(sess.run(norm_weighted_mean,
                                            feed_dict={x: self.image_clustered_with_gt[j][i * 50:50 * (i + 1)],
                                                       pro_input: pro, keep_prob: 1.0}))

                        norm_value = np.concatenate(temp)

                        wrong_norm_value = max(wrong_norm_value, np.max(norm_value))
                        wrong_min = min(wrong_min, np.min(norm_value))
                        wrong_mean = np.mean(norm_value)
                        print(i)
                        print("wrong mean {}".format(wrong_mean))
                        whole_wrong_mean = min(wrong_mean, whole_wrong_mean)
                        diff_value = np.min(norm_value) - max_norm_ref
                        if(diff_value>=0.0):
                            achieved_count += 1
                        total_norm_diff += diff_value
                        print("{} norm min {}".format(j, diff_value))
                        print("whole wrong mean{}".format(whole_wrong_mean))

                #wrong_norm_value *= 1.5
                #print("wrong norm ref: {}".format(wrong_norm_value))
                #norm_ref = (max_norm_ref + wrong_min)/2
                norm_ref = (whole_wrong_mean + true_norm_mean)/2
                print("norm ref: {}".format(norm_ref))
                print("total norm min: {}".format(total_norm_diff))
                print("achieved count : {}".format(achieved_count))
                #if achieved_count > 5:
                correct = 0
                incorrect = 0
                for q in range(10):
                    test_max = 0
                    for t in range(1000):
                        norm_value = sess.run(norm_weighted_mean,
                                              feed_dict={x: self.clustered_test[q][t:t + 1], pro_input: pro,
                                                         keep_prob: 1.0})
                        if q == 0:
                            test_max = max(test_max, norm_value)
                            if norm_value < norm_ref:
                                correct += 1
                            else:
                                incorrect += 1
                        else:
                            if norm_value > norm_ref:
                                correct += 1
                            else:
                                incorrect += 1

                    print("checking {}, correct {}, incorrect {}".format(q, correct, incorrect))
                print("accucacy: {}".format(correct / (correct + incorrect)))

    def train_norm2(self):
        class_size = 2
        test_size = 2
        class_identifiers = [ClassIdentifier(str(index)) for index in range(test_size)]

        for i in range(test_size):
            class_identifiers[i].create_train_inputs_model(8*8*64, class_size, 5000, i, "vgg19")

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(test_size):
                class_identifiers[i].sess = sess

            for _ in range(10):
                for i in range(test_size):
                    class_identifiers[i].create_training_set(self.image_clustered_with_gt[i], 10, True)
                    class_identifiers[i].create_label_set()
                    print("start training")
                    class_identifiers[i].train(self.image_clustered_with_gt, self.clustered_test, True)

                print("checking")
                for t in range(2):
                    correct = 0
                    incorrect = 0
                    both_true = 0
                    both_true_correct = 0
                    both_true_incorrect = 0
                    for k in range(1000):
                        [value1, check1] = class_identifiers[0].check_class(self.clustered_test[t][k:k + 1])
                        [value2, check2] = class_identifiers[1].check_class(self.clustered_test[t][k:k + 1])
                        if check1 and check2:
                            both_true += 1
                            if t==0:
                                if value1<=value2:
                                    both_true_correct +=1
                                    correct += 1
                                else:
                                    both_true_incorrect += 1
                                    incorrect += 1
                            else:
                                if value2 <= value1:
                                    both_true_correct += 1
                                    correct += 1
                                else:
                                    both_true_incorrect+=1
                                    incorrect += 1
                        elif check1:
                            if t==0:
                                correct += 1
                            else:
                                incorrect += 1
                        elif check2:
                            if t==1:
                                correct += 1
                            else:
                                incorrect +=1
                    print("checking {}".format(t))
                    print("correct {}, incorrect {}".format(correct, incorrect))
                    print("both true {}, both true correct {}, both true incorrect {}".format(both_true, both_true_correct,
                                                                                              both_true_incorrect))
                    print("accuracy {}".format(correct/(correct+incorrect)))
            # total_correct = 0
            # total_incorrect = 0
            # # for i in range(class_size):
            # print("checking for {}".format(i))
            #
            # for j in range(class_size):
            #     local_correct = 0
            #     local_incorrect = 0
            #     for k in range(1000):
            #         values = []
            #         for t in range(test_size):
            #             [value, check] = class_identifiers[t].check_class(self.clustered_test[j][k:k+1])
            #             if check:
            #                 values.append([value, t])
            #         if len(values) == 1:
            #             if values[0][1] == j:
            #                 local_correct += 1
            #             else:
            #                 local_incorrect += 1
            #         else:
            #             print("{} comparing with {} and values: {}".format(j, t, values))
            #     print("{} correct {}, incorrect {}".format(j, local_correct, local_incorrect))
            #     total_correct += local_correct
            #     total_incorrect += local_incorrect
            # print("total correct {} incorrect {}".format(total_correct, total_incorrect))
            # print("accuracy {}".format(total_correct / (total_incorrect + total_correct)))
