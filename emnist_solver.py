import tensorflow as tf
import logging
from data_loader import DataLoader
import numpy as np
import tools as tools
import pandas as pd
import os, sys
from six.moves import xrange
from tensorflow.python import debug as tf_debug
import pdb
from scipy.sparse.linalg import svds
import pickle
from class_identifier import ClassIdentifier

from random import randint
from scipy import linalg


number_to_class = "0 1 2 3 4 5 6 7 8 9 " \
                  "a b c d e f g h i j k l m n o p q r s t u v w x y z " \
                  "A B D E F G H N Q R T".split()


class EmnistSolver(object):
    """
    Emnist dataset solver functions
    """
    def __init__(self, loadfile=None):
        self.logger = logging.getLogger("logger_master")
        self.dataLoader = DataLoader()
        self.dataset_name = "emnist"
        self.setName = "emnist"
        self.class_size = 47

        if loadfile:
            self.loadfile_name = loadfile
        else:
            self.loadfile_name = "emnist_byclass"
        self.dataLoader.loadData(self.dataset_name, self.loadfile_name)

        self.clustered = None
        self.clustered_text = None
        self.tf_training_model = None
        self.improved_training_model = None
        self.tf_test_model = None
        self.tf_accuracy_model = None
        self.improved_test_model = None
        self.train_data_set = None
        self.test_data_set = None
#        self.tf_saver = tf.train.Saver()
        self.training_images = None
        self.test_images = None
        self.training_labels = None
        self.test_labels = None
        self.tf_training_features = None
        self.tf_training_feature_vector = None

    def load_training_data(self, size=None):
        self.training_images, self.training_labels = self.dataLoader.getTrainingData(size)
        self.logger.info("Using {} training images".format(len(self.training_images)))
        #self.training_images = self.training_images.astype(float) / 255.0
        #self.training_images -= np.repeat(np.reshape(np.mean(self.training_images, axis=1), [112800,1]), 784, axis=1)

    def load_test_data(self, size=None):
        self.test_images, self.test_labels = self.dataLoader.getTestData(size)
        #self.test_images = self.test_images.astype(float) / 255.0
        #self.test_images -= np.repeat(np.reshape(np.mean(self.test_images, axis=1), [18800,1]), 784, axis=1)
        #self.logger.info("Using {} test images".format(len(self.test_images)))

    def print_number_of_elements_per_class(self):
        unique, counts = np.unique(self.training_labels, return_counts=True)
        self.element_numbers = dict(zip(unique, counts))
        self.logger.info("Number of classes: {}".format(self.element_numbers))

    def cluster_training_with_gound_truth(self):
        self.print_number_of_elements_per_class()
        self.logger.info("Clustering Training Data")
        self.image_clustered_with_gt = {}
        for i in self.element_numbers.keys():
            self.logger.info("Clustering Data {}".format(i))
            label_index = np.where(self.training_labels == int(i))
            #print(label_index[0])
            self.image_clustered_with_gt[number_to_class[i]] = np.take(self.training_images, label_index[0], axis=0)

    def create_total_cluster_set(self):
        self.logger.info("Create total cluster set")
        self.whole_set = {}
        # for i in self.element_numbers.keys():
        #     self.whole_set[i] = np.concatenate(self.image_clustered_with_gt[number_to_class[i]], self.clustered_test[str(i)])

    def cluster_test_with_ground_truth(self):
        self.print_number_of_elements_per_class()
        self.logger.info("Clustering Testing Data")
        if self.test_images is None:
            self.load_test_data()

        self.clustered_test = {}
        for i in self.element_numbers.keys():
            self.logger.info("Clustring Data {}".format(i))
            label_index = np.where(self.test_labels == int(i))
            self.clustered_test[i] = np.take(self.test_images, label_index[0], axis=0)

    def create_tf_training_model(self):
        self.logger.info("Creating tf training model")
        self.x = tf.placeholder(tf.float32, [None, 784], "x")
        x_image = tf.reshape(self.x, [-1, 28, 28, 1])

        W_conv1 = tools.weight_variable([5, 5, 1, 32])
        b_conv1 = tools.bias_variable([32])
        h_conv1 = tf.nn.relu(tools.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = tools.max_pool_2x2(h_conv1)

        W_conv2 = tools.weight_variable([5, 5, 32, 64])
        b_conv2 = tools.bias_variable([64])
        h_conv2 = tf.nn.relu(tools.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = tools.max_pool_2x2(h_conv2)
        self.features = tf.reshape(h_pool2, [-1,49,64])

        self.h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

        W_fc1 = tools.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = tools.bias_variable([1024])
        h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, W_fc1) + b_fc1)
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        W_fc2 = tools.weight_variable([1024, 47])
        b_fc2 = tools.bias_variable([47])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        y_ = tf.placeholder(tf.float32, [None, 47])

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        self.tf_training_model = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        self.tf_test_model = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        self.tf_accuracy_model = tf.reduce_mean(tf.cast(self.tf_test_model, tf.float32))


    def train_one_norm(self):
        class_size = 8
        test_size = 8
        class_identifiers = [ClassIdentifier(str(index)) for index in range(test_size)]

        for i in range(test_size):
            class_identifiers[i].create_train_inputs_model(3136, class_size, 2800, i, "tf")

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(test_size):
                class_identifiers[i].sess = sess
                class_identifiers[i].create_training_set(self.image_clustered_with_gt[number_to_class[i]], 10, True)

                class_identifiers[i].create_label_set()
                for _ in range(10):
                    class_identifiers[i].train(self.image_clustered_with_gt, self.clustered_test)

                    total_correct = 0
                    total_incorrect = 0
                    #for i in range(class_size):
                    print("checking for {}".format(i))
                    local_correct = 0
                    local_incorrect = 0
                    for j in range(10):
                        result = class_identifiers[i].check_class(self.clustered_test[j])
                        temp = np.sum(result)
                        if j == i:
                            local_correct += temp
                            local_incorrect += 400-temp
                        else:
                            local_incorrect += temp
                            local_correct += 400-temp
                        print("{} correct {}, incorrect {}".format(j, local_correct, local_incorrect))
                    total_correct += local_correct
                    total_incorrect += local_incorrect
                    print("total correct {} incorrect {}".format(total_correct, total_incorrect))
                    print("accuracy {}".format(total_correct/(total_incorrect+total_correct)))

    def train_one_lda_norm(self):

        def weight_variable(shape, name=None):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=name)

        def bias_variable(shape, name=None):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name)

        feature_size = 7*7*64
        batch_size = 60
        rank = 100
        class_size = 10

        saver = tf.train.Saver()

        self.logger.info("Creating tf training model with angle separation")

        x = tf.placeholder(tf.float32, [None, 784], "x")
        wrong_norm_ref = tf.placeholder(tf.float32)
        keep_prob = tf.placeholder(tf.float32)
        y_ = tf.placeholder(tf.int32, [None])
        pro_input = tf.placeholder(tf.float32, [None, feature_size])

        x_image = tf.reshape(x, [-1, 28, 28, 1])

        W_conv1 = tools.weight_variable([3, 3, 1, 32])
        b_conv1 = tools.bias_variable([32])

        h_conv1 = tf.nn.relu(tools.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = tools.max_pool(h_conv1)

        W_conv2 = tools.weight_variable([3, 3, 32, 64])
        b_conv2 = tools.bias_variable([64])

        h_conv2 = tf.nn.relu(tools.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = tools.max_pool(h_conv2)

        features = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

        #features = deeplda_mnist.create_norm_network(x, keep_prob)

        features_norm_tra = tf.transpose(tf.nn.l2_normalize(features, axis=1))

        norm = tf.transpose(features_norm_tra - tf.matmul(pro_input, features_norm_tra))

        fcc_weights = {
        'W_fc1' : weight_variable([feature_size, 256]),
        'b_fc1' : bias_variable([256]),
        'W_fc2' : weight_variable([256, 1]),
        'b_fc2' : bias_variable([1])
        }
        norm_weighted = tf.nn.relu(tf.matmul(norm, fcc_weights['W_fc1']) + fcc_weights['b_fc1'])
        norm_weighted_mean = tf.reduce_sum(norm_weighted, axis=1)
        #norm_summed = tf.nn.relu(tf.matmul(norm_weighted, fcc_weights['W_fc2']) + fcc_weights['b_fc2'])
        #
        # #norm_minimized = norm
        #

        #wrong_norm = wrong_norm_ref-norm_weighted_mean
        wrong_norm = 1 / norm_weighted_mean
        #
        norm_minimize = tf.where(tf.greater(y_, 0), norm_weighted_mean, wrong_norm)

        tf_training_model = tf.train.AdamOptimizer(1e-4).minimize(norm_minimize)

        input_set = np.concatenate(
            [np.concatenate((self.image_clustered_with_gt[number_to_class[data]], self.clustered_test[data])) for data in range(class_size)], axis=0)
        labels = [ np.ones(2800), np.zeros(2800*9)]
        label_set = np.concatenate(labels)

        # input_test = np.concatenate([self.clustered_test[data] for data in range(class_size)], axis=0)
        # label_test_list = [np.zeros((len(self.clustered_test[0]), class_size)) for i in range(class_size)]
        # for i in range(class_size):
        #     label_test_list[i][:, i] = 1
        # label_test = np.concatenate(label_test_list, axis=0)

        dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
        dataset = dataset.repeat(20000)
        dataset = dataset.shuffle(buffer_size=10000)
        batched_dataset = dataset.batch(batch_size)
        iterator = batched_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            wrong_norm_value = 1

            for m in range(10000):


                #print("first norm max: {}".format(norm_value[0]))

                # for _ in range(10):
                #     norm_value = sess.run(norm,
                #                    feed_dict={x: self.image_clustered_with_gt[number_to_class[0]][0:2400], pro_input: pro, keep_prob: 1.0})
                #     print("norm value shape {}".format(norm_value.shape))
                #     #print("first norm max: {}".format(norm_value[0]))
                #
                #     norm_value_index = sess.run(norm,
                #                           feed_dict={x: self.image_clustered_with_gt[number_to_class[0]][0:1],
                #                                      pro_input: pro,
                #                                      keep_prob: 1.0})
                #
                #     print("second norm shape: {}".format(norm_value_index.shape))
                #     diff = norm_value[0] - norm_value_index[0]
                #     print("diff: {}".format(diff))
                #     print("diff sum: {}".format(np.sum(diff)))

                max_norm_ref = 1
                # while max_norm_ref > 0.05:

                feature_matrix = np.transpose(sess.run(features,
                                                       feed_dict={
                                                           x: self.image_clustered_with_gt[number_to_class[0]],
                                                           keep_prob: 1.0}))

                feature_matrix = feature_matrix / np.linalg.norm(feature_matrix)
                u, s, v = svds(feature_matrix, rank)
                pro = np.matmul(u, np.transpose(u))

                # for k in range(40):
                #     sess.run(tf_training_model,
                #              feed_dict={x: self.image_clustered_with_gt[number_to_class[0]][k * 60:60 * (k + 1)],
                #                         y_: np.ones(60), pro_input: pro,
                #                         keep_prob: 1.0, wrong_norm_ref: wrong_norm_value})

                for p in range(10):
                    print("traning {}".format(p))
                    for k in range(400):
                        batch_xs, batch_ys = sess.run(next_element)
                        sess.run(tf_training_model,
                                 feed_dict={x: batch_xs, y_: batch_ys, pro_input: pro,
                                            keep_prob: 1.0, wrong_norm_ref: wrong_norm_value})
                norm_value = sess.run(norm_weighted_mean,
                                      feed_dict={x: self.image_clustered_with_gt[number_to_class[0]][0:2800],
                                                 pro_input: pro, keep_prob: 1.0})
                max_norm_ref = np.max(norm_value)
                print("max {}".format(max_norm_ref))

                print("testing")
                norm_value = sess.run(norm_weighted_mean,
                                      feed_dict={x: self.image_clustered_with_gt[number_to_class[0]][0:2800],
                                                 pro_input: pro, keep_prob: 1.0})
                if np.max(norm_value) != 0:
                    max_norm_ref = np.max(norm_value)
                print("True norm max {}".format(max_norm_ref))
                wrong_norm_value = 0
                wrong_min = 1
                total_norm_diff = 0
                achieved_count = 0
                for j in range(0, 10):
                    if j != 0:
                        norm_value = sess.run(norm_weighted_mean,
                                              feed_dict={x: self.image_clustered_with_gt[number_to_class[j]],
                                                         pro_input: pro,
                                                         keep_prob: 1.0})

                        wrong_norm_value = max(wrong_norm_value, np.max(norm_value))
                        wrong_min = min(wrong_min, np.min(norm_value))
                        diff_value = np.min(norm_value) - max_norm_ref
                        if(diff_value>=0.0):
                            achieved_count += 1
                        total_norm_diff += diff_value
                        print("{} norm min {}".format(j, diff_value))

                #wrong_norm_value *= 1.5
                #print("wrong norm ref: {}".format(wrong_norm_value))
                norm_ref = (max_norm_ref + wrong_min)/2
                print("norm ref: {}".format(norm_ref))
                print("total norm min: {}".format(total_norm_diff))
                print("achieved count : {}".format(achieved_count))
                if achieved_count > 7:
                    correct = 0
                    incorrect = 0
                    for q in range(10):
                        test_max = 0
                        for t in range(400):
                            norm_value = sess.run(norm_weighted_mean,
                                            feed_dict={x: self.clustered_test[q][t:t+1], pro_input: pro,
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
                        print(test_max)
                    print("accucacy: {}".format(correct/(correct + incorrect)))

