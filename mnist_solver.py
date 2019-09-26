import tensorflow as tf
import logging
from data_loader import DataLoader
import numpy as np
from class_identifier import ClassIdentifier
from performance_calculator import PerformanceCalculator
from models import *


class MnistSolver():
    def __init__(self, loadfile=None):
        self.logger = logging.getLogger('logger_master')
        self.dataLoader = DataLoader()
        self.dataset_name = "mnist"
        if loadfile:
            self.loadfile_name = loadfile
        else:
            self.loadfile_name = "mnist_pickle"
        self.dataLoader.loadData(self.dataset_name, self.loadfile_name)
        self.class_size = 10
        self.clustered = None
        self.projection_matrices = None
        self.model = None
        self.performance_calculator = PerformanceCalculator(self.class_size)
        self.batch = 50

    def loadTrainingData(self, size=None):
        self.training_images, self.training_labels = self.dataLoader.getTrainingData(size)
        self.logger.info("Using {} training images".format(len(self.training_images)))

    def loadTestData(self, size=None):
        self.test_images, self.test_labels = self.dataLoader.getTestData(size)
        self.logger.info("Using {} test images".format(len(self.test_images)))

    def print_number_of_elements_per_class(self):
        unique, counts = np.unique(self.training_labels, return_counts=True)
        self.element_numbers = dict(zip(unique, counts))
        self.logger.info("Number of classes: {}".format(self.element_numbers))

    def cluster_training_with_gound_truth(self):
        self.print_number_of_elements_per_class()
        self.image_clustered_with_gt = {}
        for i in self.element_numbers.keys():
            self.logger.debug("Clustring Training Data {}".format(i))
            label_index = np.where(self.training_labels==int(i))
            self.image_clustered_with_gt[i] = np.array(np.take(self.training_images,label_index[0], axis=0))

    def cluster_test_with_ground_truth(self):
        self.logger.info("Clustering Testing Data")
        if self.test_images is None:
            self.load_test_data()

        self.clustered_test = {}
        for i in self.element_numbers.keys():
            self.logger.debug("Clustring Test Data {}".format(i))
            label_index = np.where(self.test_labels == int(i))
            self.clustered_test[i] = np.take(self.test_images, label_index[0], axis=0)

    def train(self, model_):
        self.model = model_

        n_training_classes = 10 # number of classes to be used while training. starts from index 0
        n_testing_classes = 10 # number of classes to be used while testing, starts from index 0

        class_identifiers = [ClassIdentifier(index, n_training_classes) for index in range(n_training_classes)]

        for i in range(n_training_classes):
            class_identifiers[i].create_train_inputs_model(self.dataset_name, self.model, self.batch)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for identifier in class_identifiers:
                self.logger.info("Initializing identifier {}".format(identifier.class_index))
                identifier.sess = sess
                identifier.create_training_set(self.image_clustered_with_gt)

            #labels for binary classification
            # labels = []
            # for i in range(n_training_classes):
            #     temp = np.zeros((n_training_classes, 1))
            #     temp[i] = 1
            #     labels.append(np.repeat(temp,len(self.image_clustered_with_gt[i]), axis=1))
            # label_set = np.transpose(np.concatenate(labels, axis=1))


            self.logger.info("Training Started")

            for identifier in class_identifiers:
                identifier.train_until_separation_extra(10, True)
                identifier.calculate_class_reference(0)

            self.logger.info("Binary Testing Started For Test Set")
            for p in range(n_training_classes):
                for i in range(n_training_classes):
                    self.performance_calculator.set_result(i, class_identifiers[i].check_class(
                                self.clustered_test[p]))
                self.performance_calculator.check_correct(p)
            self.performance_calculator.print_accuracy()
            self.performance_calculator.reset()

            # self.logger.info("Probabilistic Testing Started For Test Set")
            # for p in range(n_training_classes):
            #     for i in range(n_training_classes):
            #         self.performance_calculator.set_probability(i, class_identifiers[i].get_probability(
            #             self.clustered_test[p]))
            #     self.performance_calculator.check_probability(p)
            # self.performance_calculator.print_accuracy()
            # self.performance_calculator.reset()

            # self.logger.info("Testing Started For Train Set")
            # for p in range(n_training_classes):
            #     for i in range(n_training_classes):
            #         self.performance_calculator.set_result(i, class_identifiers[i].check_class(
            #                     self.image_clustered_with_gt[p]))
            #     self.performance_calculator.check_correct(p)
            # self.performance_calculator.print_accuracy()
            # self.performance_calculator.reset()

    def train_fcn(self, model_):
        self.model = model_

        n_training_classes = 10 # number of classes to be used while training. starts from index 0
        n_testing_classes = 10 # number of classes to be used while testing, starts from index 0

        class_identifiers = [ClassIdentifier(index, n_training_classes) for index in range(n_training_classes)]

        input_placeholders = []
        pro_holders = []
        for i in range(n_training_classes):
            input_placeholders.append(class_identifiers[i].create_train_inputs_model(self.dataset_name, self.model, self.batch))
            pro_holders.append(class_identifiers[i].input_pro)

        output_set = []
        for identifier in class_identifiers:
            output_set.append(identifier.rejection_vector_model)

        output = tf.concat(output_set, axis=1)
        fcc_weights = {
            'w1': weight_variable([320, 10]),
            'b1': bias_variable([10]),
        }
        y_ = tf.placeholder(tf.float32, [None, 10])
        y = tf.nn.relu(tf.matmul(output, fcc_weights['w1']) + fcc_weights['b1'])
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))

        tf_training_model = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, var_list=[fcc_weights["w1"], fcc_weights["b1"]])

        tf_test_model = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        tf_accuracy_model = tf.reduce_mean(tf.cast(tf_test_model, tf.float32))

        input_set = np.concatenate([self.image_clustered_with_gt[data] for data in range(10)], axis=0)
        labels = [np.zeros((self.image_clustered_with_gt[i].shape[0], 10)) for i in range(10)]
        for i in range(10):
            labels[i][:, i][:] = 1
        label_set = np.concatenate(labels, axis=0)

        input_test = np.concatenate([self.clustered_test[data] for data in range(10)], axis=0)
        label_test_list = [np.zeros((len(self.clustered_test[i]), 10)) for i in range(10)]
        for i in range(10):
            label_test_list[i][:, i] = 1
        label_test = np.concatenate(label_test_list, axis=0)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for identifier in class_identifiers:
                self.logger.info("Initializing identifier {}".format(identifier.class_index))
                identifier.sess = sess
                identifier.create_training_set(self.image_clustered_with_gt)
                identifier.calculate_projection_matrix(identifier.train_set[identifier.class_index])

            self.logger.info("Training Started")

            for identifier in class_identifiers:
                identifier.create_first_dataset()
                identifier.train_until_separation_extra(10, True)
                identifier.calculate_class_reference(0)

            # not able to save due to insufficient RAM
            #saver = tf.train.Saver()
            #save_path = saver.save(sess, "/home/mustafa/tf_workspace/conv_separated.ckpt")
            #print("saved to:", save_path)

            dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
            dataset = dataset.repeat(50)
            dataset = dataset.shuffle(buffer_size=10000)
            batched_dataset = dataset.batch(self.batch)
            iterator = batched_dataset.make_initializable_iterator()
            next_element = iterator.get_next()

            test_dataset = tf.data.Dataset.from_tensor_slices((input_test, label_test))
            test_dataset = test_dataset.repeat(50)

            test_batched_dataset = test_dataset.batch(self.batch)
            test_iterator = test_batched_dataset.make_initializable_iterator()
            test_next_element = test_iterator.get_next()

            sess.run(test_iterator.initializer)
            sess.run(iterator.initializer)

            max_accu = 0
            for m in range(1, 50):
                for _ in range(1200):
                    batch_xs, batch_ys = sess.run(next_element)
                    input_dict = {input_name: batch_xs for input_name in input_placeholders}
                    input_dict.update(
                        {pro_holders[i]: class_identifiers[i].projection_matrix for i in range(n_training_classes)})
                    input_dict.update({y_: batch_ys})
                    sess.run(tf_training_model, feed_dict=input_dict)

                accuracy_value = 0

                for _ in range(200):
                    test_batch_xs, test_batch_ys = sess.run(test_next_element)
                    input_dict = {input_name: test_batch_xs for input_name in input_placeholders}
                    input_dict.update(
                    {pro_holders[i]: class_identifiers[i].projection_matrix for i in range(n_training_classes)})
                    input_dict.update({y_: test_batch_ys})
                    accuracy_value += sess.run(tf_accuracy_model, feed_dict=input_dict)
                accuracy_value /= 200
                max_accu = max(accuracy_value, max_accu)
                print("Epoch: {},  accuracy: {}, max accu {} ".format(m,accuracy_value, max_accu))
                print(m)


            self.logger.info("Testing Started For Test Set")
            for p in range(n_training_classes):
                for i in range(n_training_classes):
                    self.performance_calculator.set_result(i, class_identifiers[i].check_class(
                                self.clustered_test[p]))
                self.performance_calculator.check_correct(p)
            self.performance_calculator.print_accuracy()
            self.performance_calculator.reset()

            self.logger.info("Testing Started For Train Set")
            for p in range(n_training_classes):
                for i in range(n_training_classes):
                    self.performance_calculator.set_result(i, class_identifiers[i].check_class(
                                self.image_clustered_with_gt[p]))
                self.performance_calculator.check_correct(p)
            self.performance_calculator.print_accuracy()
            self.performance_calculator.reset()


