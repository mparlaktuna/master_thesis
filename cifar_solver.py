import tensorflow as tf
import logging
from data_loader import DataLoader
import numpy as np
import tools
#from alexnet_samples.alexnet2 import *
from alexnet_samples.alexnet import *

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
        self.class_size = 10

    def loadTrainingData(self, size=None):
        self.training_images, self.training_labels = self.dataLoader.getTrainingData(size)
        self.logger.info("Using {} training images".format(len(self.training_images)))

    def loadTestData(self, size=None):
        self.test_images, self.test_labels = self.dataLoader.getTestData(size)
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
                self.image_clustered_with_gt[i] = np.reshape(temp2, (5000,32*32))
            else:
                self.image_clustered_with_gt[i] = np.reshape(temp, (5000, 32*32*3))

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
                self.clustered_test[i] = np.reshape(temp2, (1000, 32 * 32))
            else:
                self.clustered_test[i] = np.reshape(temp, (1000, 32*32*3))

    def calculateProjectionMatrices(self):
        self.logger.info("Calculating Projection Matrices")
        self.projection_matrices = {}
        if self.clustered is None:
            self.clusterTrainingWithGoundTruth()

        for i, matrix in self.clustered.items():
            self.logger.info("Calculation projection matrix for {}".format(i))
            self.projection_matrices[i] = tools.calculateProjectionMatrix(matrix.astype(float), 100)

    def testWithNormOnly(self):
        """test the whole set using the norms to matrices"""
        if self.clustered is None:
            self.clusterTrainingWithGoundTruth()

        if self.projection_matrices is None:
            self.calculateProjectionMatrices()

        if self.test_images is None:
            self.load_test_data()
            self.test_images_vectorize()

        correct = 0
        incorrect = 0
        self.incorrects = np.zeros((self.class_size, self.class_size))

        for i, v in enumerate(self.test_images_vector):
            if i % 1000 == 0:
                self.logger.info("Image index {}".format(i))

            a = {str(index): tools.calculateNorm(np.transpose(v), P) for index, P in self.projection_matrices.items()}
            minimum_value = min(a, key=a.get)
            if int(minimum_value) == self.test_labels[i]:
                correct += 1
            else:
                incorrect += 1
                self.incorrects[self.test_labels[i], int(minimum_value)] += 1

        self.logger.info("Test with norms only:")
        self.logger.info("Correct values: {}".format(correct))
        self.logger.info("Incorrect values: {}".format(incorrect))
        self.logger.info("Performance: {}".format(correct / len(self.test_images)))

    def test_two_classes_with_tf_norm(self):
        x = tf.placeholder(tf.float32, [None, 1024], "x")
        p1 = tf.placeholder(tf.float32, [None, 4096], "p1")
        p2 = tf.placeholder(tf.float32, [None, 4096], "p2")
        y = tf.placeholder(tf.int32, [64], "y")
        c = tf.placeholder(tf.int32, [64], "c")        # Define loss and optimizer


        W_conv1 = tools.weight_variable([5, 5, 1, 32], "w1")
        b_conv1 = tools.bias_variable([32], "b1")
        #
        x_image = tf.reshape(x, [-1, 32, 32, 1])

        #
        h_conv1 = tf.nn.relu(tools.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = tools.max_pool_2x2(h_conv1)

        W_conv2 = tools.weight_variable([5, 5, 32, 64], "w2")
        b_conv2 = tools.bias_variable([64], "b2")

        h_conv2 = tf.nn.relu(tools.conv2d(h_pool1, W_conv2) + b_conv2)

        h_pool2 = tools.max_pool_2x2(h_conv2)
        print("pool", h_pool2.shape)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
        print("h shape", h_pool2_flat.shape)
        first_index = 0
        second_index = 1
        first_input = self.clustered[first_index]
        second_input = self.clustered[second_index]
        first_test = self.clustered_test[first_index]
        second_test = self.clustered_test[second_index]
        total_input = np.concatenate((first_input,second_input))
        total_truth = np.concatenate((np.zeros((5000,1)), np.ones((5000,1))), axis=0)

        test_input = []
        for i in range(10):
            test_input.append(self.clustered_test[i])

        errors1 = tf.transpose(h_pool2_flat) - tf.matmul(p1, tf.transpose(h_pool2_flat))
        errors2 = tf.transpose(h_pool2_flat) - tf.matmul(p2, tf.transpose(h_pool2_flat))

        norm1 = tf.sqrt(tf.reduce_sum(tf.square(errors1)))
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(errors2)))

        error = norm1 + 1 / norm2

        train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(error)

        # dataset = tf.data.Dataset.from_tensor_slices((total_input, total_truth))
        # dataset = dataset.repeat(150)
        # dataset = dataset.shuffle(buffer_size=10000)
        # batched_dataset = dataset.batch(64)
        # iterator = batched_dataset.make_initializable_iterator()
        # next_element = iterator.get_next()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # sess.run(iterator.initializer)

            for m in range(1, 1000):
                # out1 = sess.run(h_pool2_flat, feed_dict={x:first_input})
                # out2 = sess.run(h_pool2_flat, feed_dict={x:second_input})
                # pro1 = tools.calculateProjectionMatrix(out1)
                # pro2 = tools.calculateProjectionMatrix(out2)

                correct = 0
                incorrect = 0
                correct2 = 0
                incorrect2 = 0
                total_correct = 0
                total_incorrect = 0

                # for k in range(len(total_input)):
                #     i = np.random.randint(0,len(total_truth))
                #     if total_truth[i]==0:
                #         sess.run(train_step, feed_dict={x: total_input[i:i+1], p1: pro1, p2: pro2})
                #     else:
                #         sess.run(train_step, feed_dict={x: total_input[i:i+1], p1: pro2, p2: pro1})
                #     if k%100 == 0:
                #         #print("error: ", sess.run(error, feed_dict={x: first_input[i:i + 1], p1: pro1, p2: pro2}))
                #         a = sess.run(norm1, feed_dict={x:total_input[i:i+1], p1 :pro1, p2: pro2})
                #
                #         b = sess.run(norm2, feed_dict={x:total_input[i:i+1], p1: pro1, p2: pro2})
                #         print(k, a, b)
#                projs = []
#                for i in range(10):
                out1 = sess.run(h_pool2_flat, feed_dict={x: first_input})
                out2 = sess.run(h_pool2_flat, feed_dict={x: second_input})

                pro1 = tools.calculateProjectionMatrix(out1,20)
                pro2 = tools.calculateProjectionMatrix(out2,20)

                # print("training start")
                # for j in range(10):
                #     print("training for {}".format(j))
                #
                #     other_list = [other_index for other_index in range(10) if other_index != j]
                #     data_set = self.clustered[j]
                #     for i in range(len(data_set)):
                #         rand_comp = np.random.choice(other_list)
                #         sess.run(train_step, feed_dict={x: data_set[i:i + 1], p1: projs[j], p2: projs[rand_comp]})
                #         # if i%100 == 0:
                #         #     #print("error: ", sess.run(error, feed_dict={x: first_input[i:i + 1], p1: pro1, p2: pro2}))
                #         #     a = sess.run(norm1, feed_dict={x:first_input[i:i+1], p1 :pro1, p2: pro2})
                #         #
                #         #     b = sess.run(norm2, feed_dict={x: first_input[i:i + 1], p1: pro1, p2: pro2})
                #         #     print(i, a, b)
                # print("Epoch {}".format(m))

                for k in range(len(total_input)):
                    i = np.random.randint(0,len(total_truth))
                    if total_truth[i]==0:
                        sess.run(train_step, feed_dict={x: total_input[i:i+1], p1: pro1, p2: pro2})
                    else:
                        sess.run(train_step, feed_dict={x: total_input[i:i+1], p1: pro2, p2: pro1})
                    if k%100 == 0:
                        #print("error: ", sess.run(error, feed_dict={x: first_input[i:i + 1], p1: pro1, p2: pro2}))
                        a = sess.run(norm1, feed_dict={x:total_input[i:i+1], p1 :pro1, p2: pro2})

                        b = sess.run(norm2, feed_dict={x:total_input[i:i+1], p1: pro1, p2: pro2})
                        print(k, a, b)

                # for i in range(len(second_input)):
                #     sess.run(train_step, feed_dict={x: second_input[i:i + 1], p1: pro1, p2: pro2})
                #     if i % 100 == 0:
                #         # print("error: ", sess.run(error, feed_dict={x: first_input[i:i + 1], p1: pro1, p2: pro2}))
                #         a = sess.run(norm1, feed_dict={x: second_input[i:i + 1], p1: pro1, p2: pro2})
                #
                #         b = sess.run(norm2, feed_dict={x: second_input[i:i + 1], p1: pro1, p2: pro2})
                #         print("second", i, a, b)

                for i in range(len(first_test)):
                    a = sess.run(norm1, feed_dict={x: first_test[i:i + 1], p1: pro1})
                    b = sess.run(norm2, feed_dict={x: first_test[i:i + 1], p2: pro2})
                    if a<b:
                        correct +=1
                    else:
                        incorrect +=1
                print("First test Correct: {}, Incorrect: {}".format(correct, incorrect))

                for i in range(len(second_test)):
                    a = sess.run(norm1, feed_dict={x: second_test[i:i + 1], p1: pro1})
                    b = sess.run(norm2, feed_dict={x: second_test[i:i + 1], p2: pro2})
                    if b<a:
                        correct2 += 1
                    else:
                        incorrect2 += 1
                print("Second test Correct: {}, Incorrect: {}".format(correct2, incorrect2))
                print("Total Correct: {}, Incorrect: {}".format(correct+ correct2, incorrect + incorrect2))
                # if m % 4 == 0:
                #     projs = []
                #     for i in range(10):
                #         out = sess.run(h_pool2_flat, feed_dict={x: self.clustered[i]})
                #         projs.append(tools.calculateProjectionMatrix(out))
                #
                #     for i in range(10):
                #         test_data = self.clustered_test[i]
                #         correct = 0
                #         incorrect = 0
                #         for j in range(len(test_data)):
                #             norm_values = []
                #             for k in range(10):
                #                 norm_values.append(sess.run(norm1, feed_dict={x: test_data[j:j + 1], p1: projs[k]}))
                #             if i == norm_values.index(min(norm_values)):
                #                 correct += 1
                #             else:
                #                 incorrect += 1
                #         print("Test for {} Correct: {}, Incorrect: {}".format(i, correct, incorrect))
                #         total_correct += correct
                #         total_incorrect += incorrect
                #         # print("current index: ", i)
                #         # print(norm_values)
                #         # print(min(norm_values))
                #         # print("min index: ", norm_values.index(min(norm_values)))
                #
                #     print("Total Correct: {}, Incorrect: {}".format(total_correct, total_incorrect))
                #     print("Performance: {}".format(total_correct / (total_correct + total_incorrect) * 100))

    def test_two_classes_with_separate_train_and_norm(self):

        x = tf.placeholder(tf.float32, [None, 1024], "x")
        x_2 = tf.placeholder(tf.float32, [None, 1024], "x2")

        p1 = tf.placeholder(tf.float32, [None, 4096], "p1")
        p1_2 = tf.placeholder(tf.float32, [None, 4096], "p2")


        W_conv1 = tools.weight_variable([5, 5, 1, 32], "w1")
        b_conv1 = tools.bias_variable([32], "b1")

        W_conv1_2 = tools.weight_variable([5, 5, 1, 32], "w1_2")
        b_conv1_2 = tools.bias_variable([32], "b1_2")

        x_image = tf.reshape(x, [-1, 32, 32, 1])
        x_image_2 = tf.reshape(x_2, [-1, 32, 32, 1])
        #
        h_conv1 = tf.nn.relu(tools.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = tools.max_pool_2x2(h_conv1)

        h_conv1_2 = tf.nn.relu(tools.conv2d(x_image_2, W_conv1_2) + b_conv1_2)
        h_pool1_2 = tools.max_pool_2x2(h_conv1_2)

        W_conv2 = tools.weight_variable([5, 5, 32, 64], "w2")
        b_conv2 = tools.bias_variable([64], "b2")

        W_conv2_2 = tools.weight_variable([5, 5, 32, 64], "w2_2")
        b_conv2_2 = tools.bias_variable([64], "b2_2")

        h_conv2 = tf.nn.relu(tools.conv2d(h_pool1, W_conv2) + b_conv2)
        h_conv2_2 = tf.nn.relu(tools.conv2d(h_pool1_2, W_conv2_2) + b_conv2_2)

        h_pool2 = tools.max_pool_2x2(h_conv2)
        h_pool2_2 = tools.max_pool_2x2(h_conv2_2)

        h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
        h_pool2_flat_2 = tf.reshape(h_pool2_2, [-1, 8 * 8 * 64])

        dist = tf.transpose(h_pool2_flat) - tf.matmul(p1, tf.transpose(h_pool2_flat))
        dist2 = tf.transpose(h_pool2_flat) - tf.matmul(p1_2, tf.transpose(h_pool2_flat))

        dist_2 = tf.transpose(h_pool2_flat_2) - tf.matmul(p1_2, tf.transpose(h_pool2_flat_2))
        dist2_2 = tf.transpose(h_pool2_flat_2) - tf.matmul(p1, tf.transpose(h_pool2_flat_2))

        norm = tf.sqrt(tf.reduce_sum(tf.square(dist)))
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(dist2)))

        norm_2 = tf.sqrt(tf.reduce_sum(tf.square(dist_2)))
        norm2_2 = tf.sqrt(tf.reduce_sum(tf.square(dist2_2)))

        train_step = tf.train.GradientDescentOptimizer(1e-5).minimize(norm+1/norm2)
        train_step_2 = tf.train.GradientDescentOptimizer(1e-5).minimize(norm_2+1/norm2_2)

        # dataset = tf.data.Dataset.from_tensor_slices((total_input, total_truth))
        # dataset = dataset.repeat(150)
        # dataset = dataset.shuffle(buffer_size=10000)
        # batched_dataset = dataset.batch(64)
        # iterator = batched_dataset.make_initializable_iterator()
        # next_element = iterator.get_next()

        first_index = 0
        second_index = 1
        first_input = self.clustered[first_index]
        second_input = self.clustered[second_index]
        first_test = self.clustered_test[first_index]
        second_test = self.clustered_test[second_index]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for m in range(1, 1000):
                correct = 0
                global_incorrect = 0
                local_incorrect = 0
                correct2 = 0
                global_incorrect2 = 0
                local_incorrect2 = 0
                out1 = sess.run(h_pool2_flat, feed_dict={x: first_input})
                out2 = sess.run(h_pool2_flat_2, feed_dict={x_2: second_input})

                pro1 = tools.calculateProjectionMatrix(out1, 20)
                pro2 = tools.calculateProjectionMatrix(out2, 20)

                for i in range(len(first_input)):
                    sess.run(train_step, feed_dict={x: first_input[i:i + 1], p1: pro1, p1_2: pro2})
                    if i % 100 == 0:
                        # print("error: ", sess.run(error, feed_dict={x: first_input[i:i + 1], p1: pro1, p2: pro2}))
                        a = sess.run(norm, feed_dict={x: first_input[i:i + 1], p1: pro1})
                        print("first", i, a)

                for i in range(len(second_input)):
                    sess.run(train_step_2, feed_dict={x_2: second_input[i:i + 1], p1_2: pro2, p1:pro1})
                    if i % 100 == 0:
                        # print("error: ", sess.run(error, feed_dict={x: first_input[i:i + 1], p1: pro1, p2: pro2}))
                        a = sess.run(norm_2, feed_dict={x_2: second_input[i:i + 1], p1_2: pro2})
                        print("second", i, a)

                for i in range(len(first_test)):
                    a = sess.run(norm, feed_dict={x: first_test[i:i + 1], p1: pro1})
                    b = sess.run(norm, feed_dict={x: first_test[i:i + 1], p1: pro2})
                    c = sess.run(norm_2, feed_dict={x_2: first_test[i:i + 1], p1_2: pro1})
                    d = sess.run(norm_2, feed_dict={x_2: first_test[i:i + 1], p1_2: pro2})
                    if a > b:
                        local_incorrect += 1
                    else:
                        if a<c and a<d:
                            correct += 1
                        elif d<c:
                            global_incorrect += 1
                        else:
                            correct += 1
                print("First test Correct: {}, Incorrect: {} Global Incorrect: {}".format(correct, local_incorrect, global_incorrect))

                for i in range(len(second_test)):
                    a = sess.run(norm, feed_dict={x: first_test[i:i + 1], p1: pro1})
                    b = sess.run(norm, feed_dict={x: first_test[i:i + 1], p1: pro2})
                    c = sess.run(norm_2, feed_dict={x_2: second_test[i:i + 1], p1_2: pro1})
                    d = sess.run(norm_2, feed_dict={x_2: second_test[i:i + 1], p1_2: pro2})

                    if d > c:
                        local_incorrect2 += 1
                    else:
                        if d<a and d<b:
                            correct2 += 1
                        elif a<b:
                            global_incorrect2 += 1
                        else:
                            correct2 += 1
                print("Second test Correct: {}, Incorrect: {}, Global incorrect: {}".format(correct2, local_incorrect2, global_incorrect2))
                print("Total Correct: {}, Incorrect: {}".format(correct + correct2, local_incorrect + global_incorrect + local_incorrect2 + global_incorrect2))
                # if m % 4 == 0:
                #     projs = []
                #     for i in range(10):
                #         out = sess.run(h_pool2_flat, feed_dict={x: self.clustered[i]})
                #         projs.append(tools.calculateProjectionMatrix(out))
                #
                #     for i in range(10):
                #         test_data = self.clustered_test[i]
                #         correct = 0
                #         incorrect = 0
                #         for j in range(len(test_data)):
                #             norm_values = []
                #             for k in range(10):
                #                 norm_values.append(sess.run(norm1, feed_dict={x: test_data[j:j + 1], p1: projs[k]}))
                #             if i == norm_values.index(min(norm_values)):
                #                 correct += 1
                #             else:
                #                 incorrect += 1
                #         print("Test for {} Correct: {}, Incorrect: {}".format(i, correct, incorrect))
                #         total_correct += correct
                #         total_incorrect += incorrect
                #         # print("current index: ", i)
                #         # print(norm_values)
                #         # print(min(norm_values))
                #         # print("min index: ", norm_values.index(min(norm_values)))
                #
                #     print("Total Correct: {}, Incorrect: {}".format(total_correct, total_incorrect))
                #     print("Performance: {}".format(total_correct / (total_correct + total_incorrect) * 100))

    # def test_classes_with_alexnet_norm(self):
    #
    #     one_class_size = 5000
    #     input_set = np.concatenate([self.image_clustered_with_gt[data] for data in range(10)], axis=0)
    #     labels = [np.zeros((one_class_size, 10)) for i in range(10)]
    #     for i in range(10):
    #         labels[i][:, i] = 1
    #     label_set = np.concatenate(labels, axis=0)
    #
    #     input_test = np.concatenate([self.clustered_test[data] for data in range(10)], axis=0)
    #     label_test_list = [np.zeros((len(self.clustered_test[0]), 10)) for i in range(10)]
    #     for i in range(10):
    #         label_test_list[i][:, i] = 1
    #     label_test = np.concatenate(label_test_list, axis=0)
    #
    #     feature_size = 4096
    #     x = tf.placeholder(tf.float32, [None, 32*32*3], "x")
    #     p1 = tf.placeholder(tf.float32, [None, feature_size], "p1")
    #     p2 = tf.placeholder(tf.float32, [None, feature_size], "p2")
    #     p3 = tf.placeholder(tf.float32, [None, feature_size], "p3")
    #     p4 = tf.placeholder(tf.float32, [None, feature_size], "p4")
    #     p5 = tf.placeholder(tf.float32, [None, feature_size], "p5")
    #     p6 = tf.placeholder(tf.float32, [None, feature_size], "p6")
    #     p7 = tf.placeholder(tf.float32, [None, feature_size], "p7")
    #     p8 = tf.placeholder(tf.float32, [None, feature_size], "p8")
    #     p9 = tf.placeholder(tf.float32, [None, feature_size], "p9")
    #     p10 = tf.placeholder(tf.float32, [None, feature_size], "p10")
    #
    #     weights = {
    #         'wc1': tf.Variable(tf.random_normal([3, 3, 3, 64])),
    #         'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    #         'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    #         'wd1': tf.Variable(tf.random_normal([4 * 4 * 256, 1024])),
    #         'wd2': tf.Variable(tf.random_normal([1024, 1024])),
    #         'out': tf.Variable(tf.random_normal([1024, n_classes]))
    #     }
    #     biases = {
    #         'bc1': tf.Variable(tf.random_normal([64])),
    #         'bc2': tf.Variable(tf.random_normal([128])),
    #         'bc3': tf.Variable(tf.random_normal([256])),
    #         'bd1': tf.Variable(tf.random_normal([1024])),
    #         'bd2': tf.Variable(tf.random_normal([1024])),
    #         'out': tf.Variable(tf.random_normal([n_classes]))
    #     }
    #     keep_prob = tf.placeholder(tf.float32)
    #
    #     _X = tf.reshape(x, shape=[-1, 32, 32, 3])
    #
    #     # Convolution Layer
    #     conv1 = conv2d('conv1', _X, weights['wc1'], biases['bc1'])
    #     # Max Pooling (down-sampling)
    #     pool1 = max_pool('pool1', conv1, k=2)
    #     # Apply Normalization
    #     norm1 = norm('norm1', pool1, lsize=4)
    #     # Apply Dropout
    #     norm1 = tf.nn.dropout(norm1, keep_prob)
    #
    #     # Convolution Layer
    #     conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
    #     # Max Pooling (down-sampling)
    #     pool2 = max_pool('pool2', conv2, k=2)
    #     # Apply Normalization
    #     norm2 = norm('norm2', pool2, lsize=4)
    #     # Apply Dropout
    #     norm2 = tf.nn.dropout(norm2, keep_prob)
    #
    #     # Convolution Layer
    #     conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
    #     # Max Pooling (down-sampling)
    #     pool3 = max_pool('pool3', conv3, k=2)
    #     # Apply Normalization
    #     norm3 = norm('norm3', pool3, lsize=4)
    #     # Apply Dropout
    #     norm3 = tf.nn.dropout(norm3, keep_prob)
    #
    #     # Fully connected layer
    #     h_pool2_flat = tf.reshape(norm3, [-1, weights['wd1'].get_shape().as_list()[
    #         0]])  # Reshape conv3 output to fit dense layer input
    #
    #
    #     y_ = tf.placeholder(tf.float32, [None, 10])
    #     # Define loss and optimizer
    #
    #     errors1 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p1, tf.transpose(h_pool2_flat)))
    #     errors2 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p2, tf.transpose(h_pool2_flat)))
    #     errors3 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p3, tf.transpose(h_pool2_flat)))
    #     errors4 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p4, tf.transpose(h_pool2_flat)))
    #     errors5 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p5, tf.transpose(h_pool2_flat)))
    #     errors6 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p6, tf.transpose(h_pool2_flat)))
    #     errors7 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p7, tf.transpose(h_pool2_flat)))
    #     errors8 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p8, tf.transpose(h_pool2_flat)))
    #     errors9 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p9, tf.transpose(h_pool2_flat)))
    #     errors10 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p10, tf.transpose(h_pool2_flat)))
    #
    #     norm1 = tf.sqrt(tf.reduce_sum(tf.square(errors1), axis=1))
    #     norm2 = tf.sqrt(tf.reduce_sum(tf.square(errors2), axis=1))
    #     norm3 = tf.sqrt(tf.reduce_sum(tf.square(errors3), axis=1))
    #     norm4 = tf.sqrt(tf.reduce_sum(tf.square(errors4), axis=1))
    #     norm5 = tf.sqrt(tf.reduce_sum(tf.square(errors5), axis=1))
    #     norm6 = tf.sqrt(tf.reduce_sum(tf.square(errors6), axis=1))
    #     norm7 = tf.sqrt(tf.reduce_sum(tf.square(errors7), axis=1))
    #     norm8 = tf.sqrt(tf.reduce_sum(tf.square(errors8), axis=1))
    #     norm9 = tf.sqrt(tf.reduce_sum(tf.square(errors9), axis=1))
    #     norm10 = tf.sqrt(tf.reduce_sum(tf.square(errors10), axis=1))
    #
    #     out = tf.stack([norm1, norm2, norm3, norm4, norm5, norm6, norm7, norm8, norm9, norm10], axis=1)
    #
    #     normalized = tf.abs(1 - tf.nn.l2_normalize(out, axis=1))
    #     cross_entropy = tf.reduce_mean(tf.abs(y_ - normalized))
    #
    #     train_step = tf.train.GradientDescentOptimizer(10).minimize(cross_entropy)
    #     correct_prediction = tf.equal(tf.argmax(normalized, 1), tf.argmax(y_, 1))
    #
    #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #
    #     dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
    #     dataset = dataset.repeat(150)
    #     batched_dataset = dataset.batch(50)
    #     dataset = dataset.shuffle(buffer_size=10000)
    #     iterator = batched_dataset.make_initializable_iterator()
    #     next_element = iterator.get_next()
    #
    #     with tf.Session() as sess:
    #
    #         sess.run(tf.global_variables_initializer())
    #         sess.run(iterator.initializer)
    #         self.logger.info("Training start with alexnet")
    #
    #         for m in range(0, 10000):
    #             if m%100==0:
    #                 self.logger.info("Calculating projections")
    #                 out1 = sess.run(h_pool2_flat,
    #                                 feed_dict={x: self.image_clustered_with_gt[0], keep_prob: 1.0})
    #                 out2 = sess.run(h_pool2_flat,
    #                                 feed_dict={x: self.image_clustered_with_gt[1], keep_prob: 1.0})
    #                 out3 = sess.run(h_pool2_flat,
    #                                 feed_dict={x: self.image_clustered_with_gt[2], keep_prob: 1.0})
    #                 out4 = sess.run(h_pool2_flat,
    #                                 feed_dict={x: self.image_clustered_with_gt[3], keep_prob: 1.0})
    #                 out5 = sess.run(h_pool2_flat,
    #                                 feed_dict={x: self.image_clustered_with_gt[4], keep_prob: 1.0})
    #                 out6 = sess.run(h_pool2_flat,
    #                                 feed_dict={x: self.image_clustered_with_gt[5], keep_prob: 1.0})
    #                 out7 = sess.run(h_pool2_flat,
    #                                 feed_dict={x: self.image_clustered_with_gt[6], keep_prob: 1.0})
    #                 out8 = sess.run(h_pool2_flat,
    #                                 feed_dict={x: self.image_clustered_with_gt[7], keep_prob: 1.0})
    #                 out9 = sess.run(h_pool2_flat,
    #                                 feed_dict={x: self.image_clustered_with_gt[8], keep_prob: 1.0})
    #                 out10 = sess.run(h_pool2_flat,
    #                                  feed_dict={x: self.image_clustered_with_gt[9], keep_prob: 1.0})
    #
    #                 rank = 100
    #                 pro1 = tools.calculateProjectionMatrix(out1, rank)
    #                 pro2 = tools.calculateProjectionMatrix(out2, rank)
    #                 pro3 = tools.calculateProjectionMatrix(out3, rank)
    #                 pro4 = tools.calculateProjectionMatrix(out4, rank)
    #                 pro5 = tools.calculateProjectionMatrix(out5, rank)
    #                 pro6 = tools.calculateProjectionMatrix(out6, rank)
    #                 pro7 = tools.calculateProjectionMatrix(out7, rank)
    #                 pro8 = tools.calculateProjectionMatrix(out8, rank)
    #                 pro9 = tools.calculateProjectionMatrix(out9, rank)
    #                 pro10 = tools.calculateProjectionMatrix(out10, rank)
    #                 self.logger.info("Projections calculated")
    #
    #             for _ in range(100):
    #                 batch_xs, batch_ys = sess.run(next_element)
    #                 sess.run(train_step,
    #                          feed_dict={x: batch_xs, p1: pro1, p2: pro2, p3: pro3, p4: pro4, p5: pro5, p6: pro6,
    #                                     p7: pro7, p8: pro8,
    #                                     p9: pro9, p10: pro10, y_: batch_ys, keep_prob: 0.5})
    #
    #             accu = []
    #             for i in range(0,10000,1000):
    #                 accu.append(accuracy.eval(feed_dict={
    #                     x: input_test[i:i+1000], y_: label_test[i:i+1000], p1: pro1, p2: pro2, p3: pro3, p4: pro4, p5: pro5, p6: pro6, p7: pro7,
    #                     p8: pro8,
    #                     p9: pro9, p10: pro10, keep_prob: 1.0}))
    #
    #             print(accu)
    #             print('test accuracy {}'.format(sum(accu)/10))
    #
    # def test_two_classes_with_alexnet_norm(self):
    #
    #     one_class_size = 5000
    #     input_set = np.concatenate([self.image_clustered_with_gt[data] for data in range(2)], axis=0)
    #     labels = [np.zeros((one_class_size, 2)) for i in range(2)]
    #     for i in range(2):
    #         labels[i][:, i] = 1
    #     label_set = np.concatenate(labels, axis=0)
    #
    #     input_test = np.concatenate([self.clustered_test[data] for data in range(2)], axis=0)
    #     label_test_list = [np.zeros((len(self.clustered_test[0]), 2)) for i in range(2)]
    #     for i in range(2):
    #         label_test_list[i][:, i] = 1
    #     label_test = np.concatenate(label_test_list, axis=0)
    #
    #     feature_size = 4096
    #     x = tf.placeholder(tf.float32, [None, 32 * 32 * 3], "x")
    #     p1 = tf.placeholder(tf.float32, [None, feature_size], "p1")
    #     p2 = tf.placeholder(tf.float32, [None, feature_size], "p2")
    #
    #
    #     weights = {
    #         'wc1': tf.Variable(tf.random_normal([3, 3, 3, 64])),
    #         'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    #         'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    #         'wd1': tf.Variable(tf.random_normal([4 * 4 * 256, 1024])),
    #         'wd2': tf.Variable(tf.random_normal([1024, 1024])),
    #         'out': tf.Variable(tf.random_normal([1024, n_classes]))
    #     }
    #     biases = {
    #         'bc1': tf.Variable(tf.random_normal([64])),
    #         'bc2': tf.Variable(tf.random_normal([128])),
    #         'bc3': tf.Variable(tf.random_normal([256])),
    #         'bd1': tf.Variable(tf.random_normal([1024])),
    #         'bd2': tf.Variable(tf.random_normal([1024])),
    #         'out': tf.Variable(tf.random_normal([n_classes]))
    #     }
    #     keep_prob = tf.placeholder(tf.float32)
    #
    #     _X = tf.reshape(x, shape=[-1, 32, 32, 3])
    #
    #     # Convolution Layer
    #     conv1 = conv2d('conv1', _X, weights['wc1'], biases['bc1'])
    #     # Max Pooling (down-sampling)
    #     pool1 = max_pool('pool1', conv1, k=2)
    #     # Apply Normalization
    #     norm1 = norm('norm1', pool1, lsize=4)
    #     # Apply Dropout
    #     norm1 = tf.nn.dropout(norm1, keep_prob)
    #
    #     # Convolution Layer
    #     conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
    #     # Max Pooling (down-sampling)
    #     pool2 = max_pool('pool2', conv2, k=2)
    #     # Apply Normalization
    #     norm2 = norm('norm2', pool2, lsize=4)
    #     # Apply Dropout
    #     norm2 = tf.nn.dropout(norm2, keep_prob)
    #
    #     # Convolution Layer
    #     conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
    #     # Max Pooling (down-sampling)
    #     pool3 = max_pool('pool3', conv3, k=2)
    #     # Apply Normalization
    #     norm3 = norm('norm3', pool3, lsize=4)
    #     # Apply Dropout
    #     norm3 = tf.nn.dropout(norm3, keep_prob)
    #
    #     # Fully connected layer
    #     h_pool2_flat = tf.reshape(norm3, [-1, weights['wd1'].get_shape().as_list()[
    #         0]])  # Reshape conv3 output to fit dense layer input
    #
    #     y_ = tf.placeholder(tf.float32, [None, 2])
    #     # Define loss and optimizer
    #
    #     errors1 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p1, tf.transpose(h_pool2_flat)))
    #     errors2 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p2, tf.transpose(h_pool2_flat)))
    #
    #     norm1 = tf.sqrt(tf.reduce_sum(tf.square(errors1), axis=1))
    #     norm2 = tf.sqrt(tf.reduce_sum(tf.square(errors2), axis=1))
    #     # norm3 = tf.sqrt(tf.reduce_sum(tf.square(errors3), axis=1))
    #     # norm4 = tf.sqrt(tf.reduce_sum(tf.square(errors4), axis=1))
    #     # norm5 = tf.sqrt(tf.reduce_sum(tf.square(errors5), axis=1))
    #     # norm6 = tf.sqrt(tf.reduce_sum(tf.square(errors6), axis=1))
    #     # norm7 = tf.sqrt(tf.reduce_sum(tf.square(errors7), axis=1))
    #     # norm8 = tf.sqrt(tf.reduce_sum(tf.square(errors8), axis=1))
    #     # norm9 = tf.sqrt(tf.reduce_sum(tf.square(errors9), axis=1))
    #     # norm10 = tf.sqrt(tf.reduce_sum(tf.square(errors10), axis=1))
    #
    #     out = tf.stack([norm1, norm2], axis=1)
    #
    #     normalized = tf.abs(1 - tf.nn.l2_normalize(out, axis=1))
    #     #class_norm = tf.gather(tf.nn.l2_normalize(out, axis=1), tf.argmax(y_, 1), axis=1)
    #     cross_entropy = tf.reduce_mean(tf.abs(y_ - normalized))
    #
    #     train_step = tf.train.GradientDescentOptimizer(1).minimize(cross_entropy)
    #
    #     correct_prediction = tf.equal(tf.argmax(normalized, 1), tf.argmax(y_, 1))
    #
    #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #
    #     dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
    #     dataset = dataset.repeat(150)
    #     batched_dataset = dataset.batch(50)
    #     dataset = dataset.shuffle(buffer_size=10000)
    #
    #     iterator = batched_dataset.make_initializable_iterator()
    #     next_element = iterator.get_next()
    #
    #     with tf.Session() as sess:
    #
    #         sess.run(tf.global_variables_initializer())
    #         sess.run(iterator.initializer)
    #         self.logger.info("Training start with alexnet")
    #
    #         out1 = sess.run(h_pool2_flat,
    #                         feed_dict={x: self.image_clustered_with_gt[0], keep_prob: 1.0})
    #         out2 = sess.run(h_pool2_flat,
    #                         feed_dict={x: self.image_clustered_with_gt[1], keep_prob: 1.0})
    #
    #
    #         rank = 100
    #         pro1 = tools.calculateProjectionMatrix(out1, rank)
    #         pro2 = tools.calculateProjectionMatrix(out2, rank)
    #
    #         for m in range(1, 10000):
    #
    #             for _ in range(100):
    #                 batch_xs, batch_ys = sess.run(next_element)
    #                 sess.run(train_step,
    #                          feed_dict={x: batch_xs, p1: pro1, p2: pro2, y_: batch_ys, keep_prob: 0.5})
    #
    #             accu = []
    #             for i in range(0, 2000, 1000):
    #                 accu.append(accuracy.eval(feed_dict={
    #                     x: input_test[i:i + 1000], y_: label_test[i:i + 1000], p1: pro1, p2: pro2, keep_prob: 1.0}))
    #             print(accu)
    #             print('test accuracy {}'.format(sum(accu) / 2))
    #
    # def test_classes_with_alexnet_norm_trials(self):
    #
    #     one_class_size = 5000
    #     input_set = np.concatenate([self.image_clustered_with_gt[data] for data in range(10)], axis=0)
    #     labels = [np.zeros((one_class_size, 10)) for i in range(10)]
    #     for i in range(10):
    #         labels[i][:, i] = 1
    #     label_set = np.concatenate(labels, axis=0)
    #
    #     input_test = np.concatenate([self.clustered_test[data] for data in range(10)], axis=0)
    #     label_test_list = [np.zeros((len(self.clustered_test[0]), 10)) for i in range(10)]
    #     for i in range(10):
    #         label_test_list[i][:, i] = 1
    #     label_test = np.concatenate(label_test_list, axis=0)
    #
    #     feature_size = 4096
    #     x = tf.placeholder(tf.float32, [None, 32 * 32 * 3], "x")
    #     p1 = tf.placeholder(tf.float32, [None, feature_size], "p1")
    #     p2 = tf.placeholder(tf.float32, [None, feature_size], "p2")
    #     p3 = tf.placeholder(tf.float32, [None, feature_size], "p3")
    #     p4 = tf.placeholder(tf.float32, [None, feature_size], "p4")
    #     p5 = tf.placeholder(tf.float32, [None, feature_size], "p5")
    #     p6 = tf.placeholder(tf.float32, [None, feature_size], "p6")
    #     p7 = tf.placeholder(tf.float32, [None, feature_size], "p7")
    #     p8 = tf.placeholder(tf.float32, [None, feature_size], "p8")
    #     p9 = tf.placeholder(tf.float32, [None, feature_size], "p9")
    #     p10 = tf.placeholder(tf.float32, [None, feature_size], "p10")
    #
    #     weights = {
    #         'wc1': tf.Variable(tf.random_normal([3, 3, 3, 64])),
    #         'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    #         'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    #         'wd1': tf.Variable(tf.random_normal([4 * 4 * 256, 1024])),
    #         'wd2': tf.Variable(tf.random_normal([1024, 1024])),
    #         'out': tf.Variable(tf.random_normal([1024, n_classes]))
    #     }
    #     biases = {
    #         'bc1': tf.Variable(tf.random_normal([64])),
    #         'bc2': tf.Variable(tf.random_normal([128])),
    #         'bc3': tf.Variable(tf.random_normal([256])),
    #         'bd1': tf.Variable(tf.random_normal([1024])),
    #         'bd2': tf.Variable(tf.random_normal([1024])),
    #         'out': tf.Variable(tf.random_normal([n_classes]))
    #     }
    #     keep_prob = tf.placeholder(tf.float32)
    #
    #     _X = tf.reshape(x, shape=[-1, 32, 32, 3])
    #
    #     # Convolution Layer
    #     conv1 = conv2d('conv1', _X, weights['wc1'], biases['bc1'])
    #     # Max Pooling (down-sampling)
    #     pool1 = max_pool('pool1', conv1, k=2)
    #     # Apply Normalization
    #     norm1 = norm('norm1', pool1, lsize=4)
    #     # Apply Dropout
    #     norm1 = tf.nn.dropout(norm1, keep_prob)
    #
    #     # Convolution Layer
    #     conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
    #     # Max Pooling (down-sampling)
    #     pool2 = max_pool('pool2', conv2, k=2)
    #     # Apply Normalization
    #     norm2 = norm('norm2', pool2, lsize=4)
    #     # Apply Dropout
    #     norm2 = tf.nn.dropout(norm2, keep_prob)
    #
    #     # Convolution Layer
    #     conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
    #     # Max Pooling (down-sampling)
    #     pool3 = max_pool('pool3', conv3, k=2)
    #     # Apply Normalization
    #     norm3 = norm('norm3', pool3, lsize=4)
    #     # Apply Dropout
    #     norm3 = tf.nn.dropout(norm3, keep_prob)
    #
    #     # Fully connected layer
    #     h_pool2_flat = tf.reshape(norm3, [-1, weights['wd1'].get_shape().as_list()[
    #         0]])  # Reshape conv3 output to fit dense layer input
    #
    #     y_ = tf.placeholder(tf.float32, [None, 10])
    #     # Define loss and optimizer
    #
    #     errors1 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p1, tf.transpose(h_pool2_flat)))
    #     errors2 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p2, tf.transpose(h_pool2_flat)))
    #     errors3 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p3, tf.transpose(h_pool2_flat)))
    #     errors4 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p4, tf.transpose(h_pool2_flat)))
    #     errors5 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p5, tf.transpose(h_pool2_flat)))
    #     errors6 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p6, tf.transpose(h_pool2_flat)))
    #     errors7 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p7, tf.transpose(h_pool2_flat)))
    #     errors8 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p8, tf.transpose(h_pool2_flat)))
    #     errors9 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p9, tf.transpose(h_pool2_flat)))
    #     errors10 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p10, tf.transpose(h_pool2_flat)))
    #
    #     # norm1 = tf.sqrt(tf.reduce_sum(tf.square(errors1), axis=1))
    #     # norm2 = tf.sqrt(tf.reduce_sum(tf.square(errors2), axis=1))
    #     # norm3 = tf.sqrt(tf.reduce_sum(tf.square(errors3), axis=1))
    #     # norm4 = tf.sqrt(tf.reduce_sum(tf.square(errors4), axis=1))
    #     # norm5 = tf.sqrt(tf.reduce_sum(tf.square(errors5), axis=1))
    #     # norm6 = tf.sqrt(tf.reduce_sum(tf.square(errors6), axis=1))
    #     # norm7 = tf.sqrt(tf.reduce_sum(tf.square(errors7), axis=1))
    #     # norm8 = tf.sqrt(tf.reduce_sum(tf.square(errors8), axis=1))
    #     # norm9 = tf.sqrt(tf.reduce_sum(tf.square(errors9), axis=1))
    #     # norm10 = tf.sqrt(tf.reduce_sum(tf.square(errors10), axis=1))
    #
    #     norm1 = tf.reduce_mean(tf.square(errors1), axis=1)
    #     norm2 = tf.reduce_mean(tf.square(errors2), axis=1)
    #     norm3 = tf.reduce_mean(tf.square(errors3), axis=1)
    #     norm4 = tf.reduce_mean(tf.square(errors4), axis=1)
    #     norm5 = tf.reduce_mean(tf.square(errors5), axis=1)
    #     norm6 = tf.reduce_mean(tf.square(errors6), axis=1)
    #     norm7 = tf.reduce_mean(tf.square(errors7), axis=1)
    #     norm8 = tf.reduce_mean(tf.square(errors8), axis=1)
    #     norm9 = tf.reduce_mean(tf.square(errors9), axis=1)
    #     norm10 = tf.reduce_mean(tf.square(errors10), axis=1)
    #
    #     out = tf.stack([norm1, norm2, norm3, norm4, norm5, norm6, norm7, norm8, norm9, norm10], axis=1)
    #
    #     normalized = tf.abs(1 - tf.nn.l2_normalize(out, axis=1))
    #     class_norm = tf.gather(tf.nn.l2_normalize(out, axis=1), tf.argmax(y_,1), axis=1)
    #     #cross_entropy = tf.reduce_mean(tf.abs(y_ - normalized))
    #
    #     train_step = tf.train.GradientDescentOptimizer(1).minimize(class_norm)
    #
    #     correct_prediction = tf.equal(tf.argmax(normalized, 1), tf.argmax(y_, 1))
    #
    #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #
    #     dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
    #     dataset = dataset.repeat(150)
    #     batched_dataset = dataset.batch(50)
    #     dataset = dataset.shuffle(buffer_size=10000)
    #
    #     iterator = batched_dataset.make_initializable_iterator()
    #     next_element = iterator.get_next()
    #
    #     with tf.Session() as sess:
    #
    #         sess.run(tf.global_variables_initializer())
    #         sess.run(iterator.initializer)
    #         self.logger.info("Training start with alexnet")
    #
    #         out1 = sess.run(h_pool2_flat,
    #                         feed_dict={x: self.image_clustered_with_gt[0], keep_prob: 1.0})
    #         out2 = sess.run(h_pool2_flat,
    #                         feed_dict={x: self.image_clustered_with_gt[1], keep_prob: 1.0})
    #         out3 = sess.run(h_pool2_flat,
    #                         feed_dict={x: self.image_clustered_with_gt[2], keep_prob: 1.0})
    #         out4 = sess.run(h_pool2_flat,
    #                         feed_dict={x: self.image_clustered_with_gt[3], keep_prob: 1.0})
    #         out5 = sess.run(h_pool2_flat,
    #                         feed_dict={x: self.image_clustered_with_gt[4], keep_prob: 1.0})
    #         out6 = sess.run(h_pool2_flat,
    #                         feed_dict={x: self.image_clustered_with_gt[5], keep_prob: 1.0})
    #         out7 = sess.run(h_pool2_flat,
    #                         feed_dict={x: self.image_clustered_with_gt[6], keep_prob: 1.0})
    #         out8 = sess.run(h_pool2_flat,
    #                         feed_dict={x: self.image_clustered_with_gt[7], keep_prob: 1.0})
    #         out9 = sess.run(h_pool2_flat,
    #                         feed_dict={x: self.image_clustered_with_gt[8], keep_prob: 1.0})
    #         out10 = sess.run(h_pool2_flat,
    #                          feed_dict={x: self.image_clustered_with_gt[9], keep_prob: 1.0})
    #
    #         rank = 100
    #         pro1 = tools.calculateProjectionMatrix(out1, rank)
    #         pro2 = tools.calculateProjectionMatrix(out2, rank)
    #         pro3 = tools.calculateProjectionMatrix(out3, rank)
    #         pro4 = tools.calculateProjectionMatrix(out4, rank)
    #         pro5 = tools.calculateProjectionMatrix(out5, rank)
    #         pro6 = tools.calculateProjectionMatrix(out6, rank)
    #         pro7 = tools.calculateProjectionMatrix(out7, rank)
    #         pro8 = tools.calculateProjectionMatrix(out8, rank)
    #         pro9 = tools.calculateProjectionMatrix(out9, rank)
    #         pro10 = tools.calculateProjectionMatrix(out10, rank)
    #
    #         for m in range(1, 10000):
    #
    #
    #             for _ in range(100):
    #                 batch_xs, batch_ys = sess.run(next_element)
    #                 sess.run(train_step,
    #                          feed_dict={x: batch_xs, p1: pro1, p2: pro2, p3: pro3, p4: pro4, p5: pro5, p6: pro6,
    #                                     p7: pro7, p8: pro8,
    #                                     p9: pro9, p10: pro10, y_: batch_ys, keep_prob: 0.5})
    #
    #             accu = []
    #             for i in range(0, 10000, 1000):
    #                 accu.append(accuracy.eval(feed_dict={
    #                     x: input_test[i:i + 1000], y_: label_test[i:i + 1000], p1: pro1, p2: pro2, p3: pro3, p4: pro4,
    #                     p5: pro5, p6: pro6, p7: pro7,
    #                     p8: pro8,
    #                     p9: pro9, p10: pro10, keep_prob: 1.0}))
    #
    #             print(accu)
    #             print('test accuracy {}'.format(sum(accu) / 10))

    # def test_two_classes_with_tf_norm_separate(self):
    #     one_class_size = 5000
    #     class_size = 2
    #     input_set = np.concatenate([self.image_clustered_with_gt[data] for data in range(class_size)], axis=0)
    #     labels = [np.zeros((one_class_size, class_size)) for i in range(class_size)]
    #     for i in range(class_size):
    #         labels[i][:, i] = 1
    #     label_set = np.concatenate(labels, axis=0)
    #
    #     input_test = np.concatenate([self.clustered_test[data] for data in range(class_size)], axis=0)
    #     label_test_list = [np.zeros((len(self.clustered_test[0]), class_size)) for i in range(class_size)]
    #     for i in range(class_size):
    #         label_test_list[i][:, i] = 1
    #     label_test = np.concatenate(label_test_list, axis=0)
    #
    #     feature_size = 4096
    #     x = tf.placeholder(tf.float32, [None, 32 * 32 * 3], "x")
    #     p1 = tf.placeholder(tf.float32, [None, feature_size], "p1")
    #     p2 = tf.placeholder(tf.float32, [None, feature_size], "p2")
    #
    #     weights = {
    #         'wc1': tf.Variable(tf.random_normal([3, 3, 3, 64])),
    #         'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    #         'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    #         'wd1': tf.Variable(tf.random_normal([4 * 4 * 256, 1024])),
    #         'wd2': tf.Variable(tf.random_normal([1024, 1024])),
    #         'out': tf.Variable(tf.random_normal([1024, n_classes]))
    #     }
    #     biases = {
    #         'bc1': tf.Variable(tf.random_normal([64])),
    #         'bc2': tf.Variable(tf.random_normal([128])),
    #         'bc3': tf.Variable(tf.random_normal([256])),
    #         'bd1': tf.Variable(tf.random_normal([1024])),
    #         'bd2': tf.Variable(tf.random_normal([1024])),
    #         'out': tf.Variable(tf.random_normal([n_classes]))
    #     }
    #
    #     weights2 = {
    #         'wc1': tf.Variable(tf.random_normal([3, 3, 3, 64])),
    #         'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    #         'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    #         'wd1': tf.Variable(tf.random_normal([4 * 4 * 256, 1024])),
    #         'wd2': tf.Variable(tf.random_normal([1024, 1024])),
    #         'out': tf.Variable(tf.random_normal([1024, n_classes]))
    #     }
    #     biases2 = {
    #         'bc1': tf.Variable(tf.random_normal([64])),
    #         'bc2': tf.Variable(tf.random_normal([128])),
    #         'bc3': tf.Variable(tf.random_normal([256])),
    #         'bd1': tf.Variable(tf.random_normal([1024])),
    #         'bd2': tf.Variable(tf.random_normal([1024])),
    #         'out': tf.Variable(tf.random_normal([n_classes]))
    #     }
    #     keep_prob = tf.placeholder(tf.float32)
    #
    #     _X = tf.reshape(x, shape=[-1, 32, 32, 3])
    #
    #     # Convolution Layer
    #     conv1 = conv2d('conv1', _X, weights['wc1'], biases['bc1'])
    #     # Max Pooling (down-sampling)
    #     pool1 = max_pool('pool1', conv1, k=2)
    #     # Apply Normalization
    #     norm1 = norm('norm1', pool1, lsize=4)
    #     # Apply Dropout
    #     norm1 = tf.nn.dropout(norm1, keep_prob)
    #
    #     conv1_2 = conv2d('conv1_2', _X, weights2['wc1'], biases2['bc1'])
    #     # Max Pooling (down-sampling)
    #     pool1_2 = max_pool('pool1_2', conv1_2, k=2)
    #     # Apply Normalization
    #     norm1_2 = norm('norm1_2', pool1_2, lsize=4)
    #     # Apply Dropout
    #     norm1_2 = tf.nn.dropout(norm1_2, keep_prob)
    #
    #     # Convolution Layer
    #     conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
    #     # Max Pooling (down-sampling)
    #     pool2 = max_pool('pool2', conv2, k=2)
    #     # Apply Normalization
    #     norm2 = norm('norm2', pool2, lsize=4)
    #     # Apply Dropout
    #     norm2 = tf.nn.dropout(norm2, keep_prob)
    #
    #     conv2_2 = conv2d('conv2_2', norm1_2, weights2['wc2'], biases2['bc2'])
    #     # Max Pooling (down-sampling)
    #     pool2_2 = max_pool('pool2_2', conv2_2, k=2)
    #     # Apply Normalization
    #     norm2_2 = norm('norm2_2', pool2_2, lsize=4)
    #     # Apply Dropout
    #     norm2_2 = tf.nn.dropout(norm2_2, keep_prob)
    #
    #     # Convolution Layer
    #     conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
    #     # Max Pooling (down-sampling)
    #     pool3 = max_pool('pool3', conv3, k=2)
    #     # Apply Normalization
    #     norm3 = norm('norm3', pool3, lsize=4)
    #     # Apply Dropout
    #     norm3 = tf.nn.dropout(norm3, keep_prob)
    #
    #     conv3_2 = conv2d('conv3_2', norm2_2, weights2['wc3'], biases2['bc3'])
    #     # Max Pooling (down-sampling)
    #     pool3_2 = max_pool('pool3_2', conv3_2, k=2)
    #     # Apply Normalization
    #     norm3_2 = norm('norm3_2', pool3_2, lsize=4)
    #     # Apply Dropout
    #     norm3_2 = tf.nn.dropout(norm3_2, keep_prob)
    #
    #     # Fully connected layer
    #     h_pool2_flat = tf.reshape(norm3, [-1, weights['wd1'].get_shape().as_list()[0]])
    #
    #     h_pool2_flat_2 = tf.reshape(norm3_2, [-1, weights2['wd1'].get_shape().as_list()[0]])
    #
    #     y_ = tf.placeholder(tf.float32, [None, class_size])
    #     # Define loss and optimizer
    #
    #     errors1 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p1, tf.transpose(h_pool2_flat)))
    #     errors2 = tf.transpose(tf.transpose(h_pool2_flat_2) - tf.matmul(p2, tf.transpose(h_pool2_flat_2)))
    #
    #     norm1 = tf.reduce_mean(tf.square(errors1), axis=1)
    #     norm2 = tf.reduce_mean(tf.square(errors2), axis=1)
    #
    #     out = tf.stack([norm1, norm2], axis=1)
    #
    #     normalized = tf.abs(1 - tf.nn.l2_normalize(out, axis=1))
    #     cross_entropy = tf.reduce_mean(tf.abs(y_ - normalized))
    #
    #     train_step = tf.train.GradientDescentOptimizer(1e-1).minimize(cross_entropy)
    #
    #     correct_prediction = tf.equal(tf.argmax(normalized, 1), tf.argmax(y_, 1))
    #
    #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #
    #     dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
    #     dataset = dataset.repeat(150)
    #     batched_dataset = dataset.batch(50)
    #     dataset = dataset.shuffle(buffer_size=10000)
    #     iterator = batched_dataset.make_initializable_iterator()
    #     next_element = iterator.get_next()
    #
    #     with tf.Session() as sess:
    #
    #         sess.run(tf.global_variables_initializer())
    #         sess.run(iterator.initializer)
    #         self.logger.info("Training start with alexnet")
    #
    #         for m in range(0, 10000):
    #             if m%1000==0:
    #                 self.logger.info("Calculating projections")
    #                 out1 = sess.run(h_pool2_flat,
    #                                 feed_dict={x: self.image_clustered_with_gt[0], keep_prob: 1.0})
    #                 out2 = sess.run(h_pool2_flat_2,
    #                                 feed_dict={x: self.image_clustered_with_gt[1], keep_prob: 1.0})
    #                 rank = 200
    #                 pro1 = tools.calculateProjectionMatrix(out1, rank)
    #                 pro2 = tools.calculateProjectionMatrix(out2, rank)
    #
    #                 self.logger.info("Projections calculated")
    #
    #             for _ in range(200):
    #                 batch_xs, batch_ys = sess.run(next_element)
    #                 sess.run(train_step,
    #                          feed_dict={x: batch_xs, p1: pro1, p2: pro2, y_: batch_ys, keep_prob: 0.5})
    #
    #             accu = []
    #             for i in range(0, 2000, 1000):
    #                 accu.append(accuracy.eval(feed_dict={
    #                     x: input_test[i:i + 1000], y_: label_test[i:i + 1000], p1: pro1, p2: pro2, keep_prob: 1.0}))
    #
    #             print(accu)
    #             print('test accuracy {}'.format(sum(accu) / class_size))
    #             print('test accuracy {}'.format(sum(accu) / 10))


    def test_classes_with_alexnet_norm2(self):

        one_class_size = 5000
        input_set = np.concatenate([self.image_clustered_with_gt[data] for data in range(10)], axis=0)
        labels = [np.zeros((one_class_size, 10)) for i in range(10)]
        for i in range(10):
            labels[i][:, i] = 1
        label_set = np.concatenate(labels, axis=0)

        input_test = np.concatenate([self.clustered_test[data] for data in range(10)], axis=0)
        label_test_list = [np.zeros((len(self.clustered_test[0]), 10)) for i in range(10)]
        for i in range(10):
            label_test_list[i][:, i] = 1
        label_test = np.concatenate(label_test_list, axis=0)

        feature_size = 1024
        x = tf.placeholder(tf.float32, [None, 32 * 32 * 3], "x")
        p1 = tf.placeholder(tf.float32, [None, feature_size], "p1")
        p2 = tf.placeholder(tf.float32, [None, feature_size], "p2")
        p3 = tf.placeholder(tf.float32, [None, feature_size], "p3")
        p4 = tf.placeholder(tf.float32, [None, feature_size], "p4")
        p5 = tf.placeholder(tf.float32, [None, feature_size], "p5")
        p6 = tf.placeholder(tf.float32, [None, feature_size], "p6")
        p7 = tf.placeholder(tf.float32, [None, feature_size], "p7")
        p8 = tf.placeholder(tf.float32, [None, feature_size], "p8")
        p9 = tf.placeholder(tf.float32, [None, feature_size], "p9")
        p10 = tf.placeholder(tf.float32, [None, feature_size], "p10")

        keep_prob = tf.placeholder(tf.float32)

        _X = tf.reshape(x, shape=[-1, 32, 32, 3])

        alexnet = AlexNet(_X, keep_prob, 10, "alexnet_samples/bvlc_alexnet.npy")
        h_pool2_flat = alexnet.flattened
        y_ = tf.placeholder(tf.float32, [None, 10])
        # Define loss and optimizer

        errors1 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p1, tf.transpose(h_pool2_flat)))
        errors2 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p2, tf.transpose(h_pool2_flat)))
        errors3 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p3, tf.transpose(h_pool2_flat)))
        errors4 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p4, tf.transpose(h_pool2_flat)))
        errors5 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p5, tf.transpose(h_pool2_flat)))
        errors6 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p6, tf.transpose(h_pool2_flat)))
        errors7 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p7, tf.transpose(h_pool2_flat)))
        errors8 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p8, tf.transpose(h_pool2_flat)))
        errors9 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p9, tf.transpose(h_pool2_flat)))
        errors10 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p10, tf.transpose(h_pool2_flat)))

        norm1 = tf.sqrt(tf.reduce_sum(tf.square(errors1), axis=1))
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(errors2), axis=1))
        norm3 = tf.sqrt(tf.reduce_sum(tf.square(errors3), axis=1))
        norm4 = tf.sqrt(tf.reduce_sum(tf.square(errors4), axis=1))
        norm5 = tf.sqrt(tf.reduce_sum(tf.square(errors5), axis=1))
        norm6 = tf.sqrt(tf.reduce_sum(tf.square(errors6), axis=1))
        norm7 = tf.sqrt(tf.reduce_sum(tf.square(errors7), axis=1))
        norm8 = tf.sqrt(tf.reduce_sum(tf.square(errors8), axis=1))
        norm9 = tf.sqrt(tf.reduce_sum(tf.square(errors9), axis=1))
        norm10 = tf.sqrt(tf.reduce_sum(tf.square(errors10), axis=1))

        norm_stack = tf.stack([norm1, norm2, norm3, norm4, norm5, norm6, norm7, norm8, norm9, norm10], axis=1)

        normalized = tf.abs(1 - tf.nn.l2_normalize(norm_stack, axis=1))
        cross_entropy = tf.reduce_mean(tf.abs(y_ - normalized))

        train_step = tf.train.GradientDescentOptimizer(10).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(normalized, 1), tf.argmax(y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
        dataset = dataset.repeat(150)
        dataset = dataset.shuffle(buffer_size=10000)
        batched_dataset = dataset.batch(50)
        iterator = batched_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            self.logger.info("Training start with alexnet")

            for m in range(0, 10000):
                if m % 100 == 0:
                    self.logger.info("Calculating projections")
                    out1 = sess.run(h_pool2_flat,
                                    feed_dict={x: self.image_clustered_with_gt[0], keep_prob: 1.0})
                    out2 = sess.run(h_pool2_flat,
                                    feed_dict={x: self.image_clustered_with_gt[1], keep_prob: 1.0})
                    out3 = sess.run(h_pool2_flat,
                                    feed_dict={x: self.image_clustered_with_gt[2], keep_prob: 1.0})
                    out4 = sess.run(h_pool2_flat,
                                    feed_dict={x: self.image_clustered_with_gt[3], keep_prob: 1.0})
                    out5 = sess.run(h_pool2_flat,
                                    feed_dict={x: self.image_clustered_with_gt[4], keep_prob: 1.0})
                    out6 = sess.run(h_pool2_flat,
                                    feed_dict={x: self.image_clustered_with_gt[5], keep_prob: 1.0})
                    out7 = sess.run(h_pool2_flat,
                                    feed_dict={x: self.image_clustered_with_gt[6], keep_prob: 1.0})
                    out8 = sess.run(h_pool2_flat,
                                    feed_dict={x: self.image_clustered_with_gt[7], keep_prob: 1.0})
                    out9 = sess.run(h_pool2_flat,
                                    feed_dict={x: self.image_clustered_with_gt[8], keep_prob: 1.0})
                    out10 = sess.run(h_pool2_flat,
                                     feed_dict={x: self.image_clustered_with_gt[9], keep_prob: 1.0})

                    rank = 100
                    pro1 = tools.calculateProjectionMatrix(out1, rank)
                    pro2 = tools.calculateProjectionMatrix(out2, rank)
                    pro3 = tools.calculateProjectionMatrix(out3, rank)
                    pro4 = tools.calculateProjectionMatrix(out4, rank)
                    pro5 = tools.calculateProjectionMatrix(out5, rank)
                    pro6 = tools.calculateProjectionMatrix(out6, rank)
                    pro7 = tools.calculateProjectionMatrix(out7, rank)
                    pro8 = tools.calculateProjectionMatrix(out8, rank)
                    pro9 = tools.calculateProjectionMatrix(out9, rank)
                    pro10 = tools.calculateProjectionMatrix(out10, rank)
                    self.logger.info("Projections calculated")

                for _ in range(100):
                    batch_xs, batch_ys = sess.run(next_element)
                    # print(
                    # sess.run(normalized,
                    #          feed_dict={x: batch_xs, p1: pro1, p2: pro2, p3: pro3, p4: pro4, p5: pro5, p6: pro6,
                    #                     p7: pro7, p8: pro8,
                    #                     p9: pro9, p10: pro10, y_: batch_ys, keep_prob: 0.5})
                    # )
                    sess.run(train_step,
                             feed_dict={x: batch_xs, p1: pro1, p2: pro2, p3: pro3, p4: pro4, p5: pro5, p6: pro6,
                                        p7: pro7, p8: pro8,
                                        p9: pro9, p10: pro10, y_: batch_ys, keep_prob: 0.5})
                accu = []
                for i in range(0, 10000, 1000):
                    accu.append(accuracy.eval(feed_dict={
                        x: input_test[i:i + 1000], y_: label_test[i:i + 1000], p1: pro1, p2: pro2, p3: pro3, p4: pro4,
                        p5: pro5, p6: pro6, p7: pro7,
                        p8: pro8,
                        p9: pro9, p10: pro10, keep_prob: 1.0}))

                print(accu)
                print('test accuracy {}'.format(sum(accu) / 10))