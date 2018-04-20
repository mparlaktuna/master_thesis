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
import deeplda_mnist
from scipy.sparse.linalg import svds
import pickle

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



    def create_improved_test_model(self):
        self.f = tf.placeholder(tf.float32, [None, 49, 64], "f")
        #self.P = tf.matmul(tf.reshape(self.u2, (2400,1,49)), tf.reshape(self.u1, (49,1)), transpose_a=True)
        # self.p_diag = tf.diag_part(self.P)
        # angles = tf.acos(self.p_diag)
        # self.feature_angles = tf.reduce_sum(tf.square(tf.sin(angles)))

    def create_tf_training_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.training_images, self.training_label_vectors))
        dataset = dataset.repeat(150)
        batched_dataset = dataset.batch(64)
        iterator = batched_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

    def run_tf_training_features(self):
        self.logger.info("Running tf training feature matrices")
        self.tf_training_features = {}
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            self.tf_saver.restore(sess, os.getcwd() + "/emnisttfmodel1.ckpt")
            for class_name, images in self.image_clustered_with_gt.items():
                self.tf_training_features[class_name] = sess.run(self.features, feed_dict={self.x: images, self.keep_prob: 1.0})
        self.logger.info("Finished tf training feature matrices")

    def run_tf_training_features_vector(self):
        self.logger.info("Running tf training feature vectors")
        self.tf_training_feature_vector = {}
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            self.tf_saver.restore(sess, os.getcwd() + "/emnisttfmodel1.ckpt")
            for class_name, images in self.image_clustered_with_gt.items():
                self.tf_training_feature_vector[class_name] = sess.run(self.h_pool2_flat,
                                                                 feed_dict={self.x: images, self.keep_prob: 1.0})
        self.logger.info("Finished tf training feature vector")

    def calculate_features_svd(self, save_file=False):
        self.tf_training_feature_svds = {}
        for name, matrices in self.tf_training_features.items():
            self.logger.info("Calculating feature svd for {}".format(name))
            self.tf_training_feature_svds[name] = [svds(matrices[i: i + 1,:,:].reshape(49,64), 20) for i in range(len(matrices))]

            #self.tf_training_feature_svds

        if save_file:
            f = open("feature_svd.pck", "wb")
            pickle.dump(self.tf_training_feature_svds, f)
            f.close()

    def check_angles(self):
        self.sets = {}
        self.logger.info("Starting angle check")
        for name, matrices in self.tf_training_features.items():
            self.sets[name] = set()
            for i in range(5):
                for j in range(i+1,len(matrices)):
                    angles = tools.calculateIndividualAngles(matrices[i], matrices[j], 10)
                    print(angles)
                    a = input()
                    #indexes = np.where(angles<0.4)[0].tolist()
                    #self.sets[name].update(indexes)
            self.logger.info("For {} close angles are {}".format(name, self.sets[name]))

    def calculate_angle_between_feature_vectors(self):
        for name in self.tf_training_feature_vector.keys():
            for name2 in self.tf_training_feature_vector.keys():
                if not name == name2:
                    angles = tools.calculateIndividualAngles(self.tf_training_feature_vector[name], self.tf_training_feature_vector[name2], 20)
                    print("Between {} and {} is {}".format(name, name2, angles))
                    a = input()

    def calculate_angle_between_feature_vector_svd(self):
        for name in self.tf_training_feature_vector.keys():
            angles = tools.calculateAngleBetweenSvds(self.tf_training_feature_vector[name], 20)
            print("For {} angle is {}".format(name, angles))
            a = input()

    def calculate_sigmas_of_feature_vectors(self):
        for name in self.tf_training_feature_vector.keys():
            u,sigmas,v = svds(self.tf_training_feature_vector[name], 20)
            print(sigmas)

    def load_features_svd(self):
        f = open("feature_svd.pck", "rb")
        self.tf_training_feature_svds = pickle.load(f)
        f.close()

    def run_improved_tf_test(self):
        self.logger.info("Running improved tf test")
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            self.tf_saver.restore(sess, os.getcwd() + "/emnisttfmodel1.ckpt")
            for i in range(len(self.test_images)):
                feature = sess.run(self.features, feed_dict={self.x: self.test_images[i:i+1], self.keep_prob: 1.0})
                angles = {}
                for name, svd_values in self.tf_training_feature_svds.items():

                    angles[name] = min([tools.calculateAnglesAfterSvd(svds(feature.reshape(49,64),20), np.array(svd_value)) for svd_value in svd_values])
                    print(angles[name])
            #print(angle.shape)
            #print(angle)

    def test_two_classes_with_tf_norm(self):
        x = tf.placeholder(tf.float32, [None, 784], "x")
        p1 = tf.placeholder(tf.float32, [None, 3136], "p1")
        p2 = tf.placeholder(tf.float32, [None, 3136], "p2")
        y = tf.placeholder(tf.int32, [64], "y")
        c = tf.placeholder(tf.int32, [64], "c")
        y_ = tf.placeholder(tf.float32, [None, 2])
        # Define loss and optimizer


        W_conv1 = tools.weight_variable([5, 5, 1, 32], "w1")
        b_conv1 = tools.bias_variable([32], "b1")
        #
        x_image = tf.reshape(x, [-1, 28, 28, 1])

        #
        h_conv1 = tf.nn.relu(tools.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = tools.max_pool_2x2(h_conv1)

        W_conv2 = tools.weight_variable([5, 5, 32, 64], "w2")
        b_conv2 = tools.bias_variable([64], "b2")

        h_conv2 = tf.nn.relu(tools.conv2d(h_pool1, W_conv2) + b_conv2)

        h_pool2 = tools.max_pool_2x2(h_conv2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

        first_index = number_to_class[0]
        second_index = number_to_class[1]
        first_input = self.image_clustered_with_gt[first_index]
        second_input = self.image_clustered_with_gt[second_index]
        first_test = self.clustered_test[0]
        second_test = self.clustered_test[1]
        input_set = np.concatenate((first_input, second_input), axis=0)
        first_label = np.stack((np.ones(len(first_input)),np.zeros(len(first_input))), axis=1)
        second_label = np.stack((np.zeros(len(second_input)),np.ones(len(second_input))), axis=1)
        label_set = np.concatenate((first_label, second_label), axis=0)

        input_test = np.concatenate((first_test, second_test))
        first_test = np.stack((np.ones(len(first_test)), np.zeros(len(first_test))), axis=1)
        second_test = np.stack((np.zeros(len(second_test)), np.ones(len(second_test))), axis=1)
        label_test = np.concatenate((first_test, second_test), axis=0)

        errors1 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p1, tf.transpose(h_pool2_flat)))
        errors2 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p2,tf.transpose(h_pool2_flat)))

        norm1 = tf.sqrt(tf.reduce_sum(tf.square(errors1), axis=1))
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(errors2), axis=1))

        out = tf.stack([norm1, norm2], axis=1)
        #cross_entropy = tf.reduce_mean(
        #    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=out))

        normalized = tf.abs(1-tf.nn.l2_normalize(out, axis=1))
        cross_entropy = tf.reduce_mean(tf.abs(y_-normalized))

        train_step = tf.train.GradientDescentOptimizer(1e-5).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(normalized, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
        dataset = dataset.repeat(150)
        batched_dataset = dataset.batch(64)
        dataset = dataset.shuffle(buffer_size=10000)

        iterator = batched_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            self.logger.info("Training start")
            for m in range(1, 200000):
                out1 = sess.run(h_pool2_flat, feed_dict={x:first_input})
                out2 = sess.run(h_pool2_flat, feed_dict={x:second_input})
                pro1 = tools.calculateProjectionMatrix(out1)
                pro2 = tools.calculateProjectionMatrix(out2)
                for _ in range(40):
                    batch_xs, batch_ys = sess.run(next_element)

                    sess.run(train_step, feed_dict={x: batch_xs, p1: pro1, p2: pro2, y_:batch_ys, keep_prob:0.5})

                print('test accuracy %g' % accuracy.eval(feed_dict={
                x: input_test, y_: label_test, p1:pro1, p2: pro2, keep_prob: 1.0}))

    def test_classes_with_tf_norm(self):
        x = tf.placeholder(tf.float32, [None, 784], "x")
        p1 = tf.placeholder(tf.float32, [None, 3136], "p1")
        p2 = tf.placeholder(tf.float32, [None, 3136], "p2")
        p3 = tf.placeholder(tf.float32, [None, 3136], "p3")
        p4 = tf.placeholder(tf.float32, [None, 3136], "p4")
        p5 = tf.placeholder(tf.float32, [None, 3136], "p5")
        p6 = tf.placeholder(tf.float32, [None, 3136], "p6")
        p7 = tf.placeholder(tf.float32, [None, 3136], "p7")
        p8 = tf.placeholder(tf.float32, [None, 3136], "p8")
        p9 = tf.placeholder(tf.float32, [None, 3136], "p9")
        p10 = tf.placeholder(tf.float32, [None, 3136], "p10")

        y = tf.placeholder(tf.int32, [64], "y")
        c = tf.placeholder(tf.int32, [64], "c")
        y_ = tf.placeholder(tf.float32, [None, 10])
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            self.tf_saver.restore(sess, os.getcwd() + "/emnisttfmodel1.ckpt")
            for class_name, images in self.image_clustered_with_gt.items():
                self.tf_training_features[class_name] = sess.run(self.features, feed_dict={self.x: images, self.keep_prob: 1.0})
        self.logger.info("Finished tf training feature matrices")

    def run_tf_training_features_vector(self):
        self.logger.info("Running tf training feature vectors")
        self.tf_training_feature_vector = {}
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            self.tf_saver.restore(sess, os.getcwd() + "/emnisttfmodel1.ckpt")
            for class_name, images in self.image_clustered_with_gt.items():
                self.tf_training_feature_vector[class_name] = sess.run(self.h_pool2_flat,
                                                                 feed_dict={self.x: images, self.keep_prob: 1.0})
        self.logger.info("Finished tf training feature vector")

    def calculate_features_svd(self, save_file=False):
        self.tf_training_feature_svds = {}
        for name, matrices in self.tf_training_features.items():
            self.logger.info("Calculating feature svd for {}".format(name))
            self.tf_training_feature_svds[name] = [svds(matrices[i: i + 1,:,:].reshape(49,64), 20) for i in range(len(matrices))]

            #self.tf_training_feature_svds

        if save_file:
            f = open("feature_svd.pck", "wb")
            pickle.dump(self.tf_training_feature_svds, f)
            f.close()

    def check_angles(self):
        self.sets = {}
        self.logger.info("Starting angle check")
        for name, matrices in self.tf_training_features.items():
            self.sets[name] = set()
            for i in range(5):
                for j in range(i+1,len(matrices)):
                    angles = tools.calculateIndividualAngles(matrices[i], matrices[j], 10)
                    print(angles)
                    a = input()
                    #indexes = np.where(angles<0.4)[0].tolist()
                    #self.sets[name].update(indexes)
            self.logger.info("For {} close angles are {}".format(name, self.sets[name]))

    def calculate_angle_between_feature_vectors(self):
        for name in self.tf_training_feature_vector.keys():
            for name2 in self.tf_training_feature_vector.keys():
                if not name == name2:
                    angles = tools.calculateIndividualAngles(self.tf_training_feature_vector[name], self.tf_training_feature_vector[name2], 20)
                    print("Between {} and {} is {}".format(name, name2, angles))
                    a = input()

    def calculate_angle_between_feature_vector_svd(self):
        for name in self.tf_training_feature_vector.keys():
            angles = tools.calculateAngleBetweenSvds(self.tf_training_feature_vector[name], 20)
            print("For {} angle is {}".format(name, angles))
            a = input()

    def calculate_sigmas_of_feature_vectors(self):
        for name in self.tf_training_feature_vector.keys():
            u,sigmas,v = svds(self.tf_training_feature_vector[name], 20)
            print(sigmas)

    def load_features_svd(self):
        f = open("feature_svd.pck", "rb")
        self.tf_training_feature_svds = pickle.load(f)
        f.close()

    def run_improved_tf_test(self):
        self.logger.info("Running improved tf test")
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            self.tf_saver.restore(sess, os.getcwd() + "/emnisttfmodel1.ckpt")
            for i in range(len(self.test_images)):
                feature = sess.run(self.features, feed_dict={self.x: self.test_images[i:i+1], self.keep_prob: 1.0})
                angles = {}
                for name, svd_values in self.tf_training_feature_svds.items():

                    angles[name] = min([tools.calculateAnglesAfterSvd(svds(feature.reshape(49,64),20), np.array(svd_value)) for svd_value in svd_values])
                    print(angles[name])
            #print(angle.shape)
            #print(angle)

    def test_two_classes_with_tf_norm(self):
        x = tf.placeholder(tf.float32, [None, 784], "x")
        p1 = tf.placeholder(tf.float32, [None, 3136], "p1")
        p2 = tf.placeholder(tf.float32, [None, 3136], "p2")
        y = tf.placeholder(tf.int32, [64], "y")
        c = tf.placeholder(tf.int32, [64], "c")
        y_ = tf.placeholder(tf.float32, [None, 2])
        # Define loss and optimizer


        W_conv1 = tools.weight_variable([5, 5, 1, 32], "w1")
        b_conv1 = tools.bias_variable([32], "b1")
        #
        x_image = tf.reshape(x, [-1, 28, 28, 1])

        #
        h_conv1 = tf.nn.relu(tools.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = tools.max_pool_2x2(h_conv1)

        W_conv2 = tools.weight_variable([5, 5, 32, 64], "w2")
        b_conv2 = tools.bias_variable([64], "b2")

        h_conv2 = tf.nn.relu(tools.conv2d(h_pool1, W_conv2) + b_conv2)

        h_pool2 = tools.max_pool_2x2(h_conv2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

        input_set = np.concatenate([self.image_clustered_with_gt[number_to_class[data]] for data in range(10)], axis=0)
        labels = [np.zeros((2400,10)) for i in range(10)]
        for i in range(10):
            labels[i][:,i] = 1
        label_set = np.concatenate(labels, axis=0)

        input_test = np.concatenate([self.clustered_test[data] for data in range(10)], axis=0)
        label_test_list = [np.zeros((len(self.clustered_test[0]),10)) for i in range(10)]
        for i in range(10):
            label_test_list[i][:,i] = 1
        label_test = np.concatenate(label_test_list, axis=0)

        errors1 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p1, tf.transpose(h_pool2_flat)))
        errors2 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p2,tf.transpose(h_pool2_flat)))
        errors3 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p3, tf.transpose(h_pool2_flat)))
        errors4 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p4,tf.transpose(h_pool2_flat)))
        errors5 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p5, tf.transpose(h_pool2_flat)))
        errors6 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p6,tf.transpose(h_pool2_flat)))
        errors7 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p7, tf.transpose(h_pool2_flat)))
        errors8 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p8,tf.transpose(h_pool2_flat)))
        errors9 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p9, tf.transpose(h_pool2_flat)))
        errors10 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p10,tf.transpose(h_pool2_flat)))

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

        out = tf.stack([norm1, norm2, norm3, norm4, norm5, norm6, norm7, norm8, norm9, norm10], axis=1)

        normalized = tf.abs(1 - tf.nn.l2_normalize(out, axis=1))
        cross_entropy = tf.reduce_mean(tf.abs(y_ - normalized))

        train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(normalized, 1), tf.argmax(y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
        dataset = dataset.repeat(150)
        batched_dataset = dataset.batch(64)
        dataset = dataset.shuffle(buffer_size=10000)

        first_index = number_to_class[0]
        second_index = number_to_class[1]
        first_input = self.image_clustered_with_gt[first_index]
        second_input = self.image_clustered_with_gt[second_index]
        first_test = self.clustered_test[0]
        second_test = self.clustered_test[1]
        input_set = np.concatenate((first_input, second_input), axis=0)
        first_label = np.stack((np.ones(len(first_input)),np.zeros(len(first_input))), axis=1)
        second_label = np.stack((np.zeros(len(second_input)),np.ones(len(second_input))), axis=1)
        label_set = np.concatenate((first_label, second_label), axis=0)

        input_test = np.concatenate((first_test, second_test))
        first_test = np.stack((np.ones(len(first_test)), np.zeros(len(first_test))), axis=1)
        second_test = np.stack((np.zeros(len(second_test)), np.ones(len(second_test))), axis=1)
        label_test = np.concatenate((first_test, second_test), axis=0)

        errors1 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p1, tf.transpose(h_pool2_flat)))
        errors2 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p2,tf.transpose(h_pool2_flat)))

        norm1 = tf.sqrt(tf.reduce_sum(tf.square(errors1), axis=1))
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(errors2), axis=1))

        out = tf.stack([norm1, norm2], axis=1)
        #cross_entropy = tf.reduce_mean(
        #    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=out))

        normalized = tf.abs(1-tf.nn.l2_normalize(out, axis=1))
        cross_entropy = tf.reduce_mean(tf.abs(y_-normalized))

        train_step = tf.train.GradientDescentOptimizer(1e-5).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(normalized, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
        dataset = dataset.repeat(150)
        batched_dataset = dataset.batch(64)
        iterator = batched_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            self.logger.info("Training start")

            for m in range(1, 200000):
                ofset = 0
                out1 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[0+ofset]]})
                out2 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[1+ofset]]})
                out3 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[2+ofset]]})
                out4 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[3+ofset]]})
                out5 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[4+ofset]]})
                out6 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[5+ofset]]})
                out7 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[6+ofset]]})
                out8 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[7+ofset]]})
                out9 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[8+ofset]]})
                out10 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[9+ofset]]})

                rank = 120
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

                for _ in range(100):
                    batch_xs, batch_ys = sess.run(next_element)

                    sess.run(train_step, feed_dict={x: batch_xs, p1: pro1, p2: pro2, p3: pro3, p4: pro4,p5: pro5, p6: pro6,p7: pro7, p8: pro8,
                                                    p9: pro9, p10: pro10, y_:batch_ys, keep_prob:0.5})

                print('test accuracy %g' % accuracy.eval(feed_dict={
                x: input_test, y_: label_test,  p1: pro1, p2: pro2, p3: pro3, p4: pro4,p5: pro5, p6: pro6,p7: pro7, p8: pro8,
                                                    p9: pro9, p10: pro10, keep_prob: 1.0}))

    def test_classes_with_alexnet_norm(self):

        input_set = np.concatenate([self.image_clustered_with_gt[number_to_class[data]] for data in range(10)], axis=0)
        labels = [np.zeros((2400, 10)) for i in range(10)]
        for i in range(10):
            labels[i][:, i] = 1
        label_set = np.concatenate(labels, axis=0)

        input_test = np.concatenate([self.clustered_test[data] for data in range(10)], axis=0)
        label_test_list = [np.zeros((len(self.clustered_test[0]), 10)) for i in range(10)]
        for i in range(10):
            label_test_list[i][:, i] = 1
        label_test = np.concatenate(label_test_list, axis=0)


        feature_size  = 4096
        x = tf.placeholder(tf.float32, [None, 784], "x")
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

        weights = {
            'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
            'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
            'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
            'wd1': tf.Variable(tf.random_normal([4 * 4 * 256, 1024])),
            'wd2': tf.Variable(tf.random_normal([1024, 1024])),
            'out': tf.Variable(tf.random_normal([1024, n_classes]))
        }
        biases = {
            'bc1': tf.Variable(tf.random_normal([64])),
            'bc2': tf.Variable(tf.random_normal([128])),
            'bc3': tf.Variable(tf.random_normal([256])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'bd2': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }
        keep_prob = tf.placeholder(tf.float32)

        _X = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer
        conv1 = conv2d('conv1', _X, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        pool1 = max_pool('pool1', conv1, k=2)
        # Apply Normalization
        norm1 = norm('norm1', pool1, lsize=4)
        # Apply Dropout
        norm1 = tf.nn.dropout(norm1, keep_prob)

        # Convolution Layer
        conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        pool2 = max_pool('pool2', conv2, k=2)
        # Apply Normalization
        norm2 = norm('norm2', pool2, lsize=4)
        # Apply Dropout
        norm2 = tf.nn.dropout(norm2, keep_prob)

        # Convolution Layer
        conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
        # Max Pooling (down-sampling)
        pool3 = max_pool('pool3', conv3, k=2)
        # Apply Normalization
        norm3 = norm('norm3', pool3, lsize=4)
        # Apply Dropout
        norm3 = tf.nn.dropout(norm3, keep_prob)

        # Fully connected layer
        h_pool2_flat = tf.reshape(norm3, [-1, weights['wd1'].get_shape().as_list()[
            0]])  # Reshape conv3 output to fit dense layer input

        print(h_pool2_flat.shape)

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

        out = tf.stack([norm1, norm2, norm3, norm4, norm5, norm6, norm7, norm8, norm9, norm10], axis=1)

        normalized = tf.abs(1 - tf.nn.l2_normalize(out, axis=1))
        cross_entropy = tf.reduce_mean(tf.abs(y_ - normalized))

        train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(normalized, 1), tf.argmax(y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
        dataset = dataset.repeat(150)
        dataset = dataset.shuffle(buffer_size=10000)

        batched_dataset = dataset.batch(64)

        iterator = batched_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            self.logger.info("Training start with alexnet")
            for m in range(1, 200000):
                out1 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[0]], keep_prob:1.0})
                out2 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[1]], keep_prob:1.0})
                out3 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[2]], keep_prob:1.0})
                out4 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[3]], keep_prob:1.0})
                out5 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[4]], keep_prob:1.0})
                out6 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[5]], keep_prob:1.0})
                out7 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[6]], keep_prob:1.0})
                out8 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[7]], keep_prob:1.0})
                out9 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[8]], keep_prob:1.0})
                out10 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[9]], keep_prob:1.0})

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

                for _ in range(40):
                    batch_xs, batch_ys = sess.run(next_element)

                    sess.run(train_step,
                             feed_dict={x: batch_xs, p1: pro1, p2: pro2, p3: pro3, p4: pro4, p5: pro5, p6: pro6,
                                        p7: pro7, p8: pro8,
                                        p9: pro9, p10: pro10, y_: batch_ys, keep_prob: 0.5})

                print('test accuracy %g' % accuracy.eval(feed_dict={
                    x: input_test, y_: label_test, p1: pro1, p2: pro2, p3: pro3, p4: pro4, p5: pro5, p6: pro6, p7: pro7,
                    p8: pro8,
                    p9: pro9, p10: pro10, keep_prob: 1.0}))

            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            self.logger.info("Training start")
            for m in range(1, 200000):
                out1 = sess.run(h_pool2_flat, feed_dict={x:first_input})
                out2 = sess.run(h_pool2_flat, feed_dict={x:second_input})
                pro1 = tools.calculateProjectionMatrix(out1)
                pro2 = tools.calculateProjectionMatrix(out2)
                for _ in range(40):
                    batch_xs, batch_ys = sess.run(next_element)

                    sess.run(train_step, feed_dict={x: batch_xs, p1: pro1, p2: pro2, y_:batch_ys, keep_prob:0.5})

                print('test accuracy %g' % accuracy.eval(feed_dict={
                x: input_test, y_: label_test, p1:pro1, p2: pro2, keep_prob: 1.0}))

    def test_classes_with_tf_norm(self):
        x = tf.placeholder(tf.float32, [None, 784], "x")
        p1 = tf.placeholder(tf.float32, [None, 3136], "p1")
        p2 = tf.placeholder(tf.float32, [None, 3136], "p2")
        p3 = tf.placeholder(tf.float32, [None, 3136], "p3")
        p4 = tf.placeholder(tf.float32, [None, 3136], "p4")
        p5 = tf.placeholder(tf.float32, [None, 3136], "p5")
        p6 = tf.placeholder(tf.float32, [None, 3136], "p6")
        p7 = tf.placeholder(tf.float32, [None, 3136], "p7")
        p8 = tf.placeholder(tf.float32, [None, 3136], "p8")
        p9 = tf.placeholder(tf.float32, [None, 3136], "p9")
        p10 = tf.placeholder(tf.float32, [None, 3136], "p10")

        y = tf.placeholder(tf.int32, [64], "y")
        c = tf.placeholder(tf.int32, [64], "c")
        y_ = tf.placeholder(tf.float32, [None, 10])
        # Define loss and optimizer

        W_conv1 = tools.weight_variable([5, 5, 1, 32], "w1")
        b_conv1 = tools.bias_variable([32], "b1")
        #
        x_image = tf.reshape(x, [-1, 28, 28, 1])

        #
        h_conv1 = tf.nn.relu(tools.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = tools.max_pool_2x2(h_conv1)

        W_conv2 = tools.weight_variable([5, 5, 32, 64], "w2")
        b_conv2 = tools.bias_variable([64], "b2")

        h_conv2 = tf.nn.relu(tools.conv2d(h_pool1, W_conv2) + b_conv2)

        h_pool2 = tools.max_pool_2x2(h_conv2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

        input_set = np.concatenate([self.image_clustered_with_gt[number_to_class[data]] for data in range(10)], axis=0)
        labels = [np.zeros((2400,10)) for i in range(10)]
        for i in range(10):
            labels[i][:,i] = 1
        label_set = np.concatenate(labels, axis=0)

        input_test = np.concatenate([self.clustered_test[data] for data in range(10)], axis=0)
        label_test_list = [np.zeros((len(self.clustered_test[0]),10)) for i in range(10)]
        for i in range(10):
            label_test_list[i][:,i] = 1
        label_test = np.concatenate(label_test_list, axis=0)

        errors1 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p1, tf.transpose(h_pool2_flat)))
        errors2 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p2,tf.transpose(h_pool2_flat)))
        errors3 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p3, tf.transpose(h_pool2_flat)))
        errors4 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p4,tf.transpose(h_pool2_flat)))
        errors5 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p5, tf.transpose(h_pool2_flat)))
        errors6 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p6,tf.transpose(h_pool2_flat)))
        errors7 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p7, tf.transpose(h_pool2_flat)))
        errors8 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p8,tf.transpose(h_pool2_flat)))
        errors9 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p9, tf.transpose(h_pool2_flat)))
        errors10 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p10,tf.transpose(h_pool2_flat)))

        # normal norm calculation
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

        errors = tf.stack([errors1, errors2, errors3, errors4, errors5, errors6, errors7, errors8, errors9, errors10], axis=2)
        print("errors {}".format(errors.shape))

        out = tf.stack([norm1, norm2, norm3, norm4, norm5, norm6, norm7, norm8, norm9, norm10], axis=1)

        normalized = tf.abs(1 - tf.nn.l2_normalize(out, axis=1))
        #class_norm = tf.gather(tf.nn.l2_normalize(out, axis=1), tf.argmax(y_,1), axis=1)
        cross_entropy = tf.reduce_mean(tf.abs(y_ - normalized))

        train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(normalized, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
        dataset = dataset.repeat(150)
        dataset = dataset.shuffle(buffer_size=10000)
        batched_dataset = dataset.batch(64)
        iterator = batched_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            self.logger.info("Training start")

            ofset = 0
            out1 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[0 + ofset]]})
            out2 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[1 + ofset]]})
            out3 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[2 + ofset]]})
            out4 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[3 + ofset]]})
            out5 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[4 + ofset]]})
            out6 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[5 + ofset]]})
            out7 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[6 + ofset]]})
            out8 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[7 + ofset]]})
            out9 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[8 + ofset]]})
            out10 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[9 + ofset]]})

            rank = 10
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

            for m in range(1, 200000):

                for _ in range(100):
                    batch_xs, batch_ys = sess.run(next_element)
                    sess.run(train_step, feed_dict={x: batch_xs, p1: pro1, p2: pro2, p3: pro3, p4: pro4,p5: pro5, p6: pro6,p7: pro7, p8: pro8,
                                                     p9: pro9, p10: pro10, y_:batch_ys, keep_prob:0.5})

                print('test accuracy %g' % accuracy.eval(feed_dict={
                x: input_test, y_: label_test,  p1: pro1, p2: pro2, p3: pro3, p4: pro4,p5: pro5, p6: pro6,p7: pro7, p8: pro8,
                                                    p9: pro9, p10: pro10, keep_prob: 1.0}))
        out = tf.stack([norm1, norm2, norm3, norm4, norm5, norm6, norm7, norm8, norm9, norm10], axis=1)

        normalized = tf.abs(1 - tf.nn.l2_normalize(out, axis=1))
        cross_entropy = tf.reduce_mean(tf.abs(y_ - normalized))

        train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(normalized, 1), tf.argmax(y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
        dataset = dataset.repeat(150)
        batched_dataset = dataset.batch(64)
        iterator = batched_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

    def test_two_classes_with_separate_norm(self):
        x = tf.placeholder(tf.float32, [None, 784], "x")
        p1 = tf.placeholder(tf.float32, [None, 3136], "p1")
        p2 = tf.placeholder(tf.float32, [None, 3136], "p2")
        y_ = tf.placeholder(tf.float32, [None, 2])
        # Define loss and optimizer

        W_conv1 = tools.weight_variable([5, 5, 1, 32], "w1")
        b_conv1 = tools.bias_variable([32], "b1")

        W_conv1_2 = tools.weight_variable([5, 5, 1, 32], "w1_2")
        b_conv1_2 = tools.bias_variable([32], "b1_2")
        #
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        #
        h_conv1 = tf.nn.relu(tools.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = tools.max_pool_2x2(h_conv1)

        h_conv1_2 = tf.nn.relu(tools.conv2d(x_image, W_conv1_2) + b_conv1_2)
        h_pool1_2 = tools.max_pool_2x2(h_conv1_2)

        W_conv2 = tools.weight_variable([5, 5, 32, 64], "w2")
        b_conv2 = tools.bias_variable([64], "b2")

        W_conv2_2= tools.weight_variable([5, 5, 32, 64], "w2_2")
        b_conv2_2 = tools.bias_variable([64], "b2_2")

        h_conv2 = tf.nn.relu(tools.conv2d(h_pool1, W_conv2) + b_conv2)

        h_conv2_2 = tf.nn.relu(tools.conv2d(h_pool1_2, W_conv2_2) + b_conv2_2)

        h_pool2 = tools.max_pool_2x2(h_conv2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

        h_pool2_2 = tools.max_pool_2x2(h_conv2_2)
        h_pool2_flat_2 = tf.reshape(h_pool2_2, [-1, 7 * 7 * 64])

        first_index = number_to_class[0]
        second_index = number_to_class[1]
        first_input = self.image_clustered_with_gt[first_index]
        second_input = self.image_clustered_with_gt[second_index]
        first_test = self.clustered_test[0]
        second_test = self.clustered_test[1]
        input_set = np.concatenate((first_input, second_input), axis=0)
        first_label = np.stack((np.ones(len(first_input)), np.zeros(len(first_input))), axis=1)
        second_label = np.stack((np.zeros(len(second_input)), np.ones(len(second_input))), axis=1)
        label_set = np.concatenate((first_label, second_label), axis=0)

        input_test = np.concatenate((first_test, second_test))
        first_test = np.stack((np.ones(len(first_test)), np.zeros(len(first_test))), axis=1)
        second_test = np.stack((np.zeros(len(second_test)), np.ones(len(second_test))), axis=1)
        label_test = np.concatenate((first_test, second_test), axis=0)

        errors1 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p1, tf.transpose(h_pool2_flat)))
        errors2 = tf.transpose(tf.transpose(h_pool2_flat_2) - tf.matmul(p2, tf.transpose(h_pool2_flat_2)))

        norm1 = tf.sqrt(tf.reduce_sum(tf.square(errors1), axis=1))
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(errors2), axis=1))

        out = tf.stack([norm1, norm2], axis=1)
        # cross_entropy = tf.reduce_mean(
        #    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=out))

        normalized = tf.abs(1 - tf.nn.l2_normalize(out, axis=1))
        #cross_entropy = tf.reduce_mean(tf.abs(y_ - normalized))
        class_error = 1-tf.gather(normalized, tf.argmax(y_,1), axis=1)

        train_step = tf.train.GradientDescentOptimizer(1e-5).minimize(class_error)
        correct_prediction = tf.equal(tf.argmax(normalized, 1), tf.argmax(y_, 1))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            self.logger.info("Training start")

            for m in range(1, 200000):
                ofset = 0
                out1 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[0+ofset]]})
                out2 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[1+ofset]]})
                out3 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[2+ofset]]})
                out4 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[3+ofset]]})
                out5 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[4+ofset]]})
                out6 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[5+ofset]]})
                out7 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[6+ofset]]})
                out8 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[7+ofset]]})
                out9 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[8+ofset]]})
                out10 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[9+ofset]]})

                rank = 120
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

                for _ in range(100):
                    batch_xs, batch_ys = sess.run(next_element)

                    sess.run(train_step, feed_dict={x: batch_xs, p1: pro1, p2: pro2, p3: pro3, p4: pro4,p5: pro5, p6: pro6,p7: pro7, p8: pro8,
                                                    p9: pro9, p10: pro10, y_:batch_ys, keep_prob:0.5})

                print('test accuracy %g' % accuracy.eval(feed_dict={
                x: input_test, y_: label_test,  p1: pro1, p2: pro2, p3: pro3, p4: pro4,p5: pro5, p6: pro6,p7: pro7, p8: pro8,
                                                    p9: pro9, p10: pro10, keep_prob: 1.0}))

    def test_classes_with_alexnet_norm(self):
        input_set = np.concatenate([self.image_clustered_with_gt[number_to_class[data]] for data in range(10)], axis=0)
        labels = [np.zeros((2400, 10)) for i in range(10)]
        for i in range(10):
            labels[i][:, i] = 1
        label_set = np.concatenate(labels, axis=0)

        input_test = np.concatenate([self.clustered_test[data] for data in range(10)], axis=0)
        label_test_list = [np.zeros((len(self.clustered_test[0]), 10)) for i in range(10)]
        for i in range(10):
            label_test_list[i][:, i] = 1
        label_test = np.concatenate(label_test_list, axis=0)


        feature_size  = 4096
        x = tf.placeholder(tf.float32, [None, 784], "x")
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

        weights = {
            'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
            'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
            'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
            'wd1': tf.Variable(tf.random_normal([4 * 4 * 256, 1024])),
            'wd2': tf.Variable(tf.random_normal([1024, 1024])),
            'out': tf.Variable(tf.random_normal([1024, n_classes]))
        }
        biases = {
            'bc1': tf.Variable(tf.random_normal([64])),
            'bc2': tf.Variable(tf.random_normal([128])),
            'bc3': tf.Variable(tf.random_normal([256])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'bd2': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }
        keep_prob = tf.placeholder(tf.float32)

        _X = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer
        conv1 = conv2d('conv1', _X, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        pool1 = max_pool('pool1', conv1, k=2)
        # Apply Normalization
        norm1 = norm('norm1', pool1, lsize=4)
        # Apply Dropout
        norm1 = tf.nn.dropout(norm1, keep_prob)

        # Convolution Layer
        conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        pool2 = max_pool('pool2', conv2, k=2)
        # Apply Normalization
        norm2 = norm('norm2', pool2, lsize=4)
        # Apply Dropout
        norm2 = tf.nn.dropout(norm2, keep_prob)

        # Convolution Layer
        conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
        # Max Pooling (down-sampling)
        pool3 = max_pool('pool3', conv3, k=2)
        # Apply Normalization
        norm3 = norm('norm3', pool3, lsize=4)
        # Apply Dropout
        norm3 = tf.nn.dropout(norm3, keep_prob)

        # Fully connected layer
        h_pool2_flat = tf.reshape(norm3, [-1, weights['wd1'].get_shape().as_list()[
            0]])  # Reshape conv3 output to fit dense layer input

        print(h_pool2_flat.shape)

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

        out = tf.stack([norm1, norm2, norm3, norm4, norm5, norm6, norm7, norm8, norm9, norm10], axis=1)

        normalized = tf.abs(1 - tf.nn.l2_normalize(out, axis=1))
        cross_entropy = tf.reduce_mean(tf.abs(y_ - normalized))

        train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(normalized, 1), tf.argmax(y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
        dataset = dataset.repeat(150)
        dataset = dataset.shuffle(buffer_size=10000)
        batched_dataset = dataset.batch(64)
        iterator = batched_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            self.logger.info("Training start")

            out1 = sess.run(h_pool2_flat, feed_dict={x: first_input})
            out2 = sess.run(h_pool2_flat_2, feed_dict={x: second_input})
            pro1 = tools.calculateProjectionMatrix(out1, 100)
            pro2 = tools.calculateProjectionMatrix(out2, 100)

            for m in range(1, 200000):
                for _ in range(40):
                    batch_xs, batch_ys = sess.run(next_element)
                    #print(batch_ys)
                    #print(sess.run(normalized, feed_dict={x: batch_xs, p1: pro1, p2: pro2, y_: batch_ys, keep_prob: 0.5}))
                    sess.run(train_step, feed_dict={x: batch_xs, p1: pro1, p2: pro2, y_: batch_ys, keep_prob: 0.5})

                print('test accuracy %g' % accuracy.eval(feed_dict={
                    x: input_test, y_: label_test, p1: pro1, p2: pro2, keep_prob: 1.0}))

    def test_two_classes_with_separate_norm(self):
        x = tf.placeholder(tf.float32, [None, 784], "x")
        p1 = tf.placeholder(tf.float32, [None, 3136], "p1")
        p2 = tf.placeholder(tf.float32, [None, 3136], "p2")
        y_ = tf.placeholder(tf.float32, [None, 2])
        # Define loss and optimizer

        W_conv1 = tools.weight_variable([5, 5, 1, 32], "w1")
        b_conv1 = tools.bias_variable([32], "b1")

        W_conv1_2 = tools.weight_variable([5, 5, 1, 32], "w1_2")
        b_conv1_2 = tools.bias_variable([32], "b1_2")
        #
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        #
        h_conv1 = tf.nn.relu(tools.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = tools.max_pool_2x2(h_conv1)

        h_conv1_2 = tf.nn.relu(tools.conv2d(x_image, W_conv1_2) + b_conv1_2)
        h_pool1_2 = tools.max_pool_2x2(h_conv1_2)

        W_conv2 = tools.weight_variable([5, 5, 32, 64], "w2")
        b_conv2 = tools.bias_variable([64], "b2")

        W_conv2_2 = tools.weight_variable([5, 5, 32, 64], "w2_2")
        b_conv2_2 = tools.bias_variable([64], "b2_2")

        h_conv2 = tf.nn.relu(tools.conv2d(h_pool1, W_conv2) + b_conv2)

        h_conv2_2 = tf.nn.relu(tools.conv2d(h_pool1_2, W_conv2_2) + b_conv2_2)

        h_pool2 = tools.max_pool_2x2(h_conv2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

        h_pool2_2 = tools.max_pool_2x2(h_conv2_2)
        h_pool2_flat_2 = tf.reshape(h_pool2_2, [-1, 7 * 7 * 64])

        first_index = number_to_class[0]
        second_index = number_to_class[1]
        first_input = self.image_clustered_with_gt[first_index]
        second_input = self.image_clustered_with_gt[second_index]
        first_test = self.clustered_test[0]
        second_test = self.clustered_test[1]
        input_set = np.concatenate((first_input, second_input), axis=0)
        first_label = np.stack((np.ones(len(first_input)), np.zeros(len(first_input))), axis=1)
        second_label = np.stack((np.zeros(len(second_input)), np.ones(len(second_input))), axis=1)
        label_set = np.concatenate((first_label, second_label), axis=0)

        input_test = np.concatenate((first_test, second_test))
        first_test = np.stack((np.ones(len(first_test)), np.zeros(len(first_test))), axis=1)
        second_test = np.stack((np.zeros(len(second_test)), np.ones(len(second_test))), axis=1)
        label_test = np.concatenate((first_test, second_test), axis=0)

        errors1 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p1, tf.transpose(h_pool2_flat)))
        errors2 = tf.transpose(tf.transpose(h_pool2_flat_2) - tf.matmul(p2, tf.transpose(h_pool2_flat_2)))

        norm1 = tf.sqrt(tf.reduce_sum(tf.square(errors1), axis=1))
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(errors2), axis=1))

        out = tf.stack([norm1, norm2], axis=1)
        # cross_entropy = tf.reduce_mean(
        #    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=out))

        normalized = tf.abs(1 - tf.nn.l2_normalize(out, axis=1))
        # cross_entropy = tf.reduce_mean(tf.abs(y_ - normalized))
        class_error = 1 - tf.gather(normalized, tf.argmax(y_, 1), axis=1)

        train_step = tf.train.GradientDescentOptimizer(1e-5).minimize(class_error)
        correct_prediction = tf.equal(tf.argmax(normalized, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
        dataset = dataset.repeat(150)
        dataset = dataset.shuffle(buffer_size=10000)
        batched_dataset = dataset.batch(64)
        iterator = batched_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            self.logger.info("Training start")

            out1 = sess.run(h_pool2_flat, feed_dict={x: first_input})
            out2 = sess.run(h_pool2_flat_2, feed_dict={x: second_input})
            pro1 = tools.calculateProjectionMatrix(out1, 100)
            pro2 = tools.calculateProjectionMatrix(out2, 100)

            for m in range(1, 200000):
                for _ in range(40):
                    batch_xs, batch_ys = sess.run(next_element)
                    # print(batch_ys)
                    # print(sess.run(normalized, feed_dict={x: batch_xs, p1: pro1, p2: pro2, y_: batch_ys, keep_prob: 0.5}))
                    sess.run(train_step, feed_dict={x: batch_xs, p1: pro1, p2: pro2, y_: batch_ys, keep_prob: 0.5})

                print('test accuracy %g' % accuracy.eval(feed_dict={
                    x: input_test, y_: label_test, p1: pro1, p2: pro2, keep_prob: 1.0}))

    def test_classes_with_tf_norm_separate(self):
        input_set = np.concatenate([self.image_clustered_with_gt[number_to_class[data]] for data in range(10)], axis=0)
        labels = [np.zeros((2400, 10)) for i in range(10)]
        for i in range(10):
            labels[i][:, i] = 1
        label_set = np.concatenate(labels, axis=0)

        input_test = np.concatenate([self.clustered_test[data] for data in range(10)], axis=0)
        label_test_list = [np.zeros((len(self.clustered_test[0]), 10)) for i in range(10)]
        for i in range(10):
            label_test_list[i][:, i] = 1
        label_test = np.concatenate(label_test_list, axis=0)

        x = tf.placeholder(tf.float32, [None, 784], "x")
        p1 = tf.placeholder(tf.float32, [None, 3136], "p1")
        p2 = tf.placeholder(tf.float32, [None, 3136], "p2")
        p3 = tf.placeholder(tf.float32, [None, 3136], "p3")
        p4 = tf.placeholder(tf.float32, [None, 3136], "p4")
        p5 = tf.placeholder(tf.float32, [None, 3136], "p5")
        p6 = tf.placeholder(tf.float32, [None, 3136], "p6")
        p7 = tf.placeholder(tf.float32, [None, 3136], "p7")
        p8 = tf.placeholder(tf.float32, [None, 3136], "p8")
        p9 = tf.placeholder(tf.float32, [None, 3136], "p9")
        p10 = tf.placeholder(tf.float32, [None, 3136], "p10")

        p = [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10]
        y_ = tf.placeholder(tf.float32, [None, 10])
        # Define loss and optimizer

        h_pool2_flat = []
        norms = []
        for i in range(10):
            W_conv1 = tools.weight_variable([5, 5, 1, 32])
            b_conv1 = tools.bias_variable([32])
            W_conv2 = tools.weight_variable([5, 5, 32, 64])
            b_conv2 = tools.bias_variable([64])
            h_pool2_flat.append(tools.create_new_h_pool2_flat(x,W_conv1, W_conv2, b_conv1, b_conv2))
            error = tf.transpose(tf.transpose(h_pool2_flat[i]) - tf.matmul(p[i], tf.transpose(h_pool2_flat[i])))
            norms.append(tf.sqrt(tf.reduce_sum(tf.square(error), axis=1)))

        # normal norm calculation
        out = tf.stack(norms, axis=1)

        normalized = tf.abs(1 - tf.nn.l2_normalize(out, axis=1))
        #class_error = 1 - tf.gather(normalized, tf.argmax(y_, 1), axis=1)
        cross_entropy = tf.reduce_mean(tf.abs(y_ - normalized))

        train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(normalized, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
        dataset = dataset.repeat(2000)
        dataset = dataset.shuffle(buffer_size=10000)
        batched_dataset = dataset.batch(64)
        iterator = batched_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            self.logger.info("Training start")

            for m in range(0, 200000):
                if m%100 == 0:
                    self.logger.info("Calculating projecetions")
                    ofset = 0;
                    out1 = sess.run(h_pool2_flat[0],
                                    feed_dict={x: self.image_clustered_with_gt[number_to_class[0 + ofset]]})
                    out2 = sess.run(h_pool2_flat[1],
                                    feed_dict={x: self.image_clustered_with_gt[number_to_class[1 + ofset]]})
                    out3 = sess.run(h_pool2_flat[2],
                                    feed_dict={x: self.image_clustered_with_gt[number_to_class[2 + ofset]]})
                    out4 = sess.run(h_pool2_flat[3],
                                    feed_dict={x: self.image_clustered_with_gt[number_to_class[3 + ofset]]})
                    out5 = sess.run(h_pool2_flat[4],
                                    feed_dict={x: self.image_clustered_with_gt[number_to_class[4 + ofset]]})
                    out6 = sess.run(h_pool2_flat[5],
                                    feed_dict={x: self.image_clustered_with_gt[number_to_class[5 + ofset]]})
                    out7 = sess.run(h_pool2_flat[6],
                                    feed_dict={x: self.image_clustered_with_gt[number_to_class[6 + ofset]]})
                    out8 = sess.run(h_pool2_flat[7],
                                    feed_dict={x: self.image_clustered_with_gt[number_to_class[7 + ofset]]})
                    out9 = sess.run(h_pool2_flat[8],
                                    feed_dict={x: self.image_clustered_with_gt[number_to_class[8 + ofset]]})
                    out10 = sess.run(h_pool2_flat[9],
                                     feed_dict={x: self.image_clustered_with_gt[number_to_class[9 + ofset]]})

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
                    sess.run(train_step,
                             feed_dict={x: batch_xs, p1: pro1, p2: pro2, p3: pro3, p4: pro4, p5: pro5, p6: pro6,
                                        p7: pro7, p8: pro8,
                                        p9: pro9, p10: pro10, y_: batch_ys, keep_prob: 0.5})

                self.logger.info("Testing")
                accu = []
                for i in range(0,4000,400):
                    accu.append(accuracy.eval(feed_dict={
                    x: input_test[i:i+400], y_: label_test[i:i+400], p1: pro1, p2: pro2, p3: pro3, p4: pro4, p5: pro5, p6: pro6, p7: pro7,
                    p8: pro8,
                    p9: pro9, p10: pro10, keep_prob: 1.0}))

                print(accu)
                print("Accuracy: {}".format(sum(accu)/10))
            self.logger.info("Training start with alexnet")
            for m in range(1, 200000):
                out1 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[0]], keep_prob:1.0})
                out2 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[1]], keep_prob:1.0})
                out3 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[2]], keep_prob:1.0})
                out4 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[3]], keep_prob:1.0})
                out5 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[4]], keep_prob:1.0})
                out6 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[5]], keep_prob:1.0})
                out7 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[6]], keep_prob:1.0})
                out8 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[7]], keep_prob:1.0})
                out9 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[8]], keep_prob:1.0})
                out10 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[9]], keep_prob:1.0})

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

                for _ in range(40):
                    batch_xs, batch_ys = sess.run(next_element)

                    sess.run(train_step,
                             feed_dict={x: batch_xs, p1: pro1, p2: pro2, p3: pro3, p4: pro4, p5: pro5, p6: pro6,
                                        p7: pro7, p8: pro8,
                                        p9: pro9, p10: pro10, y_: batch_ys, keep_prob: 0.5})

                print('test accuracy %g' % accuracy.eval(feed_dict={
                    x: input_test, y_: label_test, p1: pro1, p2: pro2, p3: pro3, p4: pro4, p5: pro5, p6: pro6, p7: pro7,
                    p8: pro8,
                    p9: pro9, p10: pro10, keep_prob: 1.0}))

    def test_classes_with_tf_norm_trials(self):
        x = tf.placeholder(tf.float32, [None, 784], "x")
        p1 = tf.placeholder(tf.float32, [None, 3136], "p1")
        p2 = tf.placeholder(tf.float32, [None, 3136], "p2")
        p3 = tf.placeholder(tf.float32, [None, 3136], "p3")
        p4 = tf.placeholder(tf.float32, [None, 3136], "p4")
        p5 = tf.placeholder(tf.float32, [None, 3136], "p5")
        p6 = tf.placeholder(tf.float32, [None, 3136], "p6")
        p7 = tf.placeholder(tf.float32, [None, 3136], "p7")
        p8 = tf.placeholder(tf.float32, [None, 3136], "p8")
        p9 = tf.placeholder(tf.float32, [None, 3136], "p9")
        p10 = tf.placeholder(tf.float32, [None, 3136], "p10")

        y = tf.placeholder(tf.int32, [64], "y")
        c = tf.placeholder(tf.int32, [64], "c")
        y_ = tf.placeholder(tf.float32, [None, 10])
        # Define loss and optimizer

        W_conv1 = tools.weight_variable([5, 5, 1, 32], "w1")
        b_conv1 = tools.bias_variable([32], "b1")
        #
        x_image = tf.reshape(x, [-1, 28, 28, 1])

        #
        h_conv1 = tf.nn.relu(tools.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = tools.max_pool_2x2(h_conv1)

        W_conv2 = tools.weight_variable([5, 5, 32, 64], "w2")
        b_conv2 = tools.bias_variable([64], "b2")

        h_conv2 = tf.nn.relu(tools.conv2d(h_pool1, W_conv2) + b_conv2)

        h_pool2 = tools.max_pool_2x2(h_conv2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

        input_set = np.concatenate([self.image_clustered_with_gt[number_to_class[data]] for data in range(10)], axis=0)
        labels = [np.zeros((2400,10)) for i in range(10)]
        for i in range(10):
            labels[i][:,i] = 1
        label_set = np.concatenate(labels, axis=0)

        input_test = np.concatenate([self.clustered_test[data] for data in range(10)], axis=0)
        label_test_list = [np.zeros((len(self.clustered_test[0]),10)) for i in range(10)]
        for i in range(10):
            label_test_list[i][:,i] = 1
        label_test = np.concatenate(label_test_list, axis=0)

        errors1 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p1, tf.transpose(h_pool2_flat)))
        errors2 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p2,tf.transpose(h_pool2_flat)))
        errors3 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p3, tf.transpose(h_pool2_flat)))
        errors4 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p4,tf.transpose(h_pool2_flat)))
        errors5 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p5, tf.transpose(h_pool2_flat)))
        errors6 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p6,tf.transpose(h_pool2_flat)))
        errors7 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p7, tf.transpose(h_pool2_flat)))
        errors8 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p8,tf.transpose(h_pool2_flat)))
        errors9 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p9, tf.transpose(h_pool2_flat)))
        errors10 = tf.transpose(tf.transpose(h_pool2_flat) - tf.matmul(p10,tf.transpose(h_pool2_flat)))

        # normal norm calculation
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

        # mean calculation
        # norm1 = tf.reduce_mean(tf.square(errors1), axis=1)
        # norm2 = tf.reduce_mean(tf.square(errors2), axis=1)
        # norm3 = tf.reduce_mean(tf.square(errors3), axis=1)
        # norm4 = tf.reduce_mean(tf.square(errors4), axis=1)
        # norm5 = tf.reduce_mean(tf.square(errors5), axis=1)
        # norm6 = tf.reduce_mean(tf.square(errors6), axis=1)
        # norm7 = tf.reduce_mean(tf.square(errors7), axis=1)
        # norm8 = tf.reduce_mean(tf.square(errors8), axis=1)
        # norm9 = tf.reduce_mean(tf.square(errors9), axis=1)
        # norm10 = tf.reduce_mean(tf.square(errors10), axis=1)

        errors = tf.stack([errors1, errors2, errors3, errors4, errors5, errors6, errors7, errors8, errors9, errors10], axis=2)
        print("errors {}".format(errors.shape))

        out = tf.stack([norm1, norm2, norm3, norm4, norm5, norm6, norm7, norm8, norm9, norm10], axis=1)

        normalized = tf.abs(1 - tf.nn.l2_normalize(out, axis=1))
        #class_norm = tf.gather(tf.nn.l2_normalize(out, axis=1), tf.argmax(y_,1), axis=1)
        cross_entropy = tf.reduce_mean(tf.abs(y_ - normalized))

        train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(normalized, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
        dataset = dataset.repeat(150)
        batched_dataset = dataset.batch(64)
        iterator = batched_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            self.logger.info("Training start")

            ofset = 0
            out1 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[0 + ofset]]})
            out2 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[1 + ofset]]})
            out3 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[2 + ofset]]})
            out4 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[3 + ofset]]})
            out5 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[4 + ofset]]})
            out6 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[5 + ofset]]})
            out7 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[6 + ofset]]})
            out8 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[7 + ofset]]})
            out9 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[8 + ofset]]})
            out10 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[number_to_class[9 + ofset]]})

            rank = 10
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

            for m in range(1, 200000):

                for _ in range(100):
                    batch_xs, batch_ys = sess.run(next_element)
                    sess.run(train_step, feed_dict={x: batch_xs, p1: pro1, p2: pro2, p3: pro3, p4: pro4,p5: pro5, p6: pro6,p7: pro7, p8: pro8,
                                                     p9: pro9, p10: pro10, y_:batch_ys, keep_prob:0.5})

                print('test accuracy %g' % accuracy.eval(feed_dict={
                x: input_test, y_: label_test,  p1: pro1, p2: pro2, p3: pro3, p4: pro4,p5: pro5, p6: pro6,p7: pro7, p8: pro8,
                                                    p9: pro9, p10: pro10, keep_prob: 1.0}))


    def test_classes_with_tf_norm_separate2(self):
        input_set = np.concatenate([self.image_clustered_with_gt[number_to_class[data]] for data in range(10)], axis=0)
        labels = [np.zeros((2400, 10)) for i in range(10)]
        for i in range(10):
            labels[i][:, i][:] = 1
        label_set = np.concatenate(labels, axis=0)

        input_test = np.concatenate([self.clustered_test[data] for data in range(10)], axis=0)
        label_test_list = [np.zeros((len(self.clustered_test[0]), 10)) for i in range(10)]
        for i in range(10):
            label_test_list[i][:, i] = 1
        label_test = np.concatenate(label_test_list, axis=0)

        x = tf.placeholder(tf.float32, [None, 784], "x")
        p1 = tf.placeholder(tf.float32, [None, 3136], "p1")
        p2 = tf.placeholder(tf.float32, [None, 3136], "p2")
        p3 = tf.placeholder(tf.float32, [None, 3136], "p3")
        p4 = tf.placeholder(tf.float32, [None, 3136], "p4")
        p5 = tf.placeholder(tf.float32, [None, 3136], "p5")
        p6 = tf.placeholder(tf.float32, [None, 3136], "p6")
        p7 = tf.placeholder(tf.float32, [None, 3136], "p7")
        p8 = tf.placeholder(tf.float32, [None, 3136], "p8")
        p9 = tf.placeholder(tf.float32, [None, 3136], "p9")
        p10 = tf.placeholder(tf.float32, [None, 3136], "p10")

        p = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]
        y_ = tf.placeholder(tf.float32, [None, 10])
        # Define loss and optimizer

        h_pool2_flat = []
        norms = []
        for i in range(10):
            W_conv1 = tools.weight_variable([5, 5, 1, 32])
            b_conv1 = tools.bias_variable([32])
            W_conv2 = tools.weight_variable([5, 5, 32, 64])
            b_conv2 = tools.bias_variable([64])
            h_pool2_flat.append(tools.create_new_h_pool2_flat(x, W_conv1, W_conv2, b_conv1, b_conv2))
            local_norms = []
            for j in range(10):
                local_error = tf.transpose(tf.transpose(h_pool2_flat[i]) - tf.matmul(p[j], tf.transpose(h_pool2_flat[i])))
                local_norms.append(tf.sqrt(tf.reduce_sum(tf.square(local_error), axis=1)))
            temp = tf.stack(local_norms, axis=1)
            norms.append(temp)

        # normal norm calculation
        norm_stack = tf.stack(norms, axis=2)
        normalized = tf.abs(1-tf.nn.l2_normalize(norm_stack, axis=1))
        norm_mean = tf.reduce_min(normalized, axis=2)
        #out2 = tf.gather(out, tf.argmax(y_,1),axis=2)
        # normalized = tf.abs(1 - tf.nn.l2_normalize(out, axis=1))
        # # class_error = 1 - tf.gather(normalized, tf.argmax(y_, 1), axis=1)
        cross_entropy = tf.reduce_mean(tf.abs(y_ - norm_mean))
        #
        train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(cross_entropy)
        #
        correct_prediction = tf.equal(tf.argmax(norm_mean, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
        dataset = dataset.repeat(2000)
        dataset = dataset.shuffle(buffer_size=10000)
        batched_dataset = dataset.batch(64)
        iterator = batched_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            self.logger.info("Training start")

            for m in range(0, 200000):
                if m % 100 == 0:
                    self.logger.info("Calculating projecetions")
                    ofset = 0;
                    out1 = sess.run(h_pool2_flat[0],
                                    feed_dict={x: self.image_clustered_with_gt[number_to_class[0 + ofset]]})
                    out2 = sess.run(h_pool2_flat[1],
                                    feed_dict={x: self.image_clustered_with_gt[number_to_class[1 + ofset]]})
                    out3 = sess.run(h_pool2_flat[2],
                                    feed_dict={x: self.image_clustered_with_gt[number_to_class[2 + ofset]]})
                    out4 = sess.run(h_pool2_flat[3],
                                    feed_dict={x: self.image_clustered_with_gt[number_to_class[3 + ofset]]})
                    out5 = sess.run(h_pool2_flat[4],
                                    feed_dict={x: self.image_clustered_with_gt[number_to_class[4 + ofset]]})
                    out6 = sess.run(h_pool2_flat[5],
                                    feed_dict={x: self.image_clustered_with_gt[number_to_class[5 + ofset]]})
                    out7 = sess.run(h_pool2_flat[6],
                                    feed_dict={x: self.image_clustered_with_gt[number_to_class[6 + ofset]]})
                    out8 = sess.run(h_pool2_flat[7],
                                    feed_dict={x: self.image_clustered_with_gt[number_to_class[7 + ofset]]})
                    out9 = sess.run(h_pool2_flat[8],
                                    feed_dict={x: self.image_clustered_with_gt[number_to_class[8 + ofset]]})
                    out10 = sess.run(h_pool2_flat[9],
                                     feed_dict={x: self.image_clustered_with_gt[number_to_class[9 + ofset]]})

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
                    sess.run(train_step,
                             feed_dict={x: batch_xs, p1: pro1, p2: pro2, p3: pro3, p4: pro4, p5: pro5, p6: pro6,
                                        p7: pro7, p8: pro8,
                                        p9: pro9, p10: pro10, y_: batch_ys, keep_prob: 0.5})


                self.logger.info("Testing")
                accu = []
                for i in range(0, 4000, 400):
                    accu.append(accuracy.eval(feed_dict={
                        x: input_test[i:i + 400], y_: label_test[i:i + 400], p1: pro1, p2: pro2, p3: pro3, p4: pro4,
                        p5: pro5, p6: pro6, p7: pro7,
                        p8: pro8,
                        p9: pro9, p10: pro10, keep_prob: 1.0}))

                print(accu)
                print("Accuracy: {}".format(sum(accu) / 10))

    def train_tf_with_mid_angle_separation(self):
        self.logger.info("Creating tf training model with angle separation")
        x = tf.placeholder(tf.float32, [None, 784], "x")
        x2 = tf.placeholder(tf.float32, [None, 784], "x2")

        x_image = tf.reshape(x, [-1, 28, 28, 1])
        x_image2 = tf.reshape(x2, [-1, 28, 28, 1])

        W_conv1 = tools.weight_variable([5, 5, 1, 32])
        b_conv1 = tools.bias_variable([32])

        h_conv1 = tf.nn.relu(tools.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = tools.max_pool_2x2(h_conv1)

        h_conv1_2 = tf.nn.relu(tools.conv2d(x_image2, W_conv1) + b_conv1)
        h_pool1_2 = tools.max_pool_2x2(h_conv1_2)

        W_conv2 = tools.weight_variable([5, 5, 32, 64])
        b_conv2 = tools.bias_variable([64])
        h_conv2 = tf.nn.relu(tools.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = tools.max_pool_2x2(h_conv2)

        h_conv2_2 = tf.nn.relu(tools.conv2d(h_pool1_2, W_conv2) + b_conv2)
        h_pool2_2 = tools.max_pool_2x2(h_conv2_2)

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_pool2_flat_2 = tf.reshape(h_pool2_2, [-1, 7 * 7 * 64])

        s1, u1, v1 = tf.svd(tf.transpose(tf.reshape(h_pool2_flat, [2400,3136])), full_matrices=True, compute_uv=True, name="svd")
        s2, u2, v2 = tf.svd(tf.transpose(tf.reshape(h_pool2_flat_2, [2400,3136])), full_matrices=True, compute_uv=True, name="svd2")

        #tf.matmul(tf.transpose(u1), u2)

        #s, u, v = tf.svd(tf.matmul(tf.transpose(u1), u2), full_matrices=True, compute_uv=True, name="svd")
        #s, u, v = tf.svd(tf.matmul(tf.transpose(h_pool2_flat), h_pool2_flat_2), full_matrices=True, compute_uv=True, name="svd")

        p_diag = tf.diag_part(tf.matmul(tf.transpose(u1), u2))
        angles = tf.reduce_sum(tf.square(tf.sin(tf.acos(p_diag))))
        train_angles = tf.train.AdamOptimizer(1e-4).minimize(3136-angles)

        W_fc1 = tools.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = tools.bias_variable([1024])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        W_fc2 = tools.weight_variable([1024, 2])
        b_fc2 = tools.bias_variable([2])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        y_ = tf.placeholder(tf.float32, [None, 2])

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
        tf_training_model = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        tf_test_model = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        tf_accuracy_model = tf.reduce_mean(tf.cast(tf_test_model, tf.float32))

        index1 = 2
        index2 = 8
        first_index = number_to_class[index1]
        second_index = number_to_class[index2]
        first_input = self.image_clustered_with_gt[first_index]
        second_input = self.image_clustered_with_gt[second_index]
        first_test = self.clustered_test[index1]
        second_test = self.clustered_test[index2]
        input_set = np.concatenate((first_input, second_input), axis=0)
        first_label = np.stack((np.ones(len(first_input)), np.zeros(len(first_input))), axis=1)
        second_label = np.stack((np.zeros(len(second_input)), np.ones(len(second_input))), axis=1)
        label_set = np.concatenate((first_label, second_label), axis=0)

        input_test = np.concatenate((first_test, second_test))
        first_test = np.stack((np.ones(len(first_test)), np.zeros(len(first_test))), axis=1)
        second_test = np.stack((np.zeros(len(second_test)), np.ones(len(second_test))), axis=1)
        label_test = np.concatenate((first_test, second_test), axis=0)

        dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
        dataset = dataset.repeat(150)
        dataset = dataset.shuffle(buffer_size=10000)
        batched_dataset = dataset.batch(64)
        iterator = batched_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            self.logger.info("Training start")

            for m in range(1, 200000):
                for _ in range(40):
                    batch_xs, batch_ys = sess.run(next_element)

                    # u1_values = sess.run(u1, feed_dict={x: self.image_clustered_with_gt[number_to_class[index1]],
                    #                                            x2: self.image_clustered_with_gt[
                    #                                                number_to_class[index1]]})
                    #
                    # u2_values = sess.run(u2, feed_dict={x: self.image_clustered_with_gt[number_to_class[index1]],
                    #                                     x2: self.image_clustered_with_gt[
                    #                                         number_to_class[index1]]})
                    #
                    # print(u1_values.shape)
                    # print(u2_values.shape)
                    #for i in range(0, 10):
                    sess.run(train_angles,
                             feed_dict={x: self.image_clustered_with_gt[number_to_class[0]],
                                        x2: self.image_clustered_with_gt[number_to_class[1]]})
                    angle_values = sess.run(angles,
                                            feed_dict={x: self.image_clustered_with_gt[number_to_class[0]],
                                                       x2: self.image_clustered_with_gt[number_to_class[1]]})
                    print(angle_values)


                    # print(batch_ys)
                    # print(sess.run(normalized, feed_dict={x: batch_xs, p1: pro1, p2: pro2, y_: batch_ys, keep_prob: 0.5}))
                    #sess.run(tf_training_model, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

                # print('test accuracy %g' % tf_accuracy_model.eval(feed_dict={
                #     x: input_test, y_: label_test, keep_prob: 1.0}))

    def train_lda(self):
        def weight_variable(shape, name=None):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=name)

        def bias_variable(shape, name=None):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name)

        weights = {
            "w1": tf.Variable(tf.random_normal([3, 3, 1, 64])),
            "w2": tf.Variable(tf.random_normal([3, 3, 64, 64])),
            "w3": tf.Variable(tf.random_normal([3, 3, 64, 96])),
            "w4": tf.Variable(tf.random_normal([3, 3, 96, 96])),
            "w5": tf.Variable(tf.random_normal([3, 3, 96, 256])),
            "w6": tf.Variable(tf.random_normal([3, 3, 256, 256])),
            "w7": tf.Variable(tf.random_normal([1, 1, 256, 10])),
        }
        fcc_weights = {
        'W_fc1' : weight_variable([7 * 7 * 10, 256]),
        'b_fc1' : bias_variable([256]),
        'W_fc2' : weight_variable([256, 10]),
        'b_fc2' : bias_variable([10])
        }

        feature_size = 490
        self.logger.info("Creating tf training model with angle separation")
        x = tf.placeholder(tf.float32, [None, 784], "x")

        #
        # x input to the network 
        # y is the truth 
        # o_u is the Q from outside or previous set
        #  

        keep_prob = tf.placeholder(tf.float32)
        y_ = tf.placeholder(tf.float32, [None, 10])
        o_u = tf.placeholder(tf.float32, [feature_size, feature_size])

        [y_conv, features] = deeplda_mnist.lda_create_network(x, weights, fcc_weights, keep_prob)

        #
        # Creates the lda network 
        # Features is the fewatures layer 
        # y_conv is the entire network with the fcc
        #  

        # #regular training
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))

        #regular training
        #tf_training_model = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        #only fcc training
        tf_training_model = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, var_list=[fcc_weights["W_fc1"], fcc_weights["b_fc1"], fcc_weights["W_fc2"], fcc_weights["b_fc2"]])

        tf_test_model = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        tf_accuracy_model = tf.reduce_mean(tf.cast(tf_test_model, tf.float32))

        # flat_features = tf.reshape(features, [-1, feature_size])
        m0, m1 = tf.split(features, 2, axis=1)
        #
        # s, u, v = tf.svd(tf.transpose(tf.reshape(flat_features, [feature_size, feature_size])), full_matrices=True, compute_uv=True,
        #                     name="svd")
        s1, u1, v1 = tf.svd(tf.reshape(m0, [feature_size, feature_size]), full_matrices=True,
                         compute_uv=True,
                         name="svd")

        s2, u2, v2 = tf.svd(tf.reshape(m1, [feature_size, feature_size]), full_matrices=True,
                         compute_uv=True,
                         name="svd2")

        #b_s1 = tf.nn.top_k(s1, 20)
        #kth_s1 = tf.reduce_min(b_s1.values)
        #top2_s1 = tf.greater_equal(s1, kth_s1)
        #s1_rec = tf.where(top2_s1, s1, tf.zeros(feature_size))

        #b_s2 = tf.nn.top_k(s2, 20)
        #kth_s2 = tf.reduce_min(b_s2.values)
        #top2_s2 = tf.greater_equal(s2, kth_s2)
        #s2_rec = tf.where(top2_s2, s2, tf.zeros(feature_size))

        #rank approximation code for each one get r1 and r2
        r1 = 100
        r2 = 100

        q1 = u1[:,0:r1]
        q2 = u2[:,0:r2]

        A = tf.matmul(q1, q2, transpose_a=True)


        #m0_rec = tf.matmul(tf.matmul(u1, tf.diag(s1_rec)), tf.transpose(v1))
        #m1_rec = tf.matmul(tf.matmul(u2, tf.diag(s2_rec)), tf.transpose(v2))
        #m0_norm = tf.nn.l2_normalize(m0_rec)
        #m1_norm = tf.nn.l2_normalize(m1_rec)

        # p_diag = tf.diag_part(tf.matmul(tf.transpose(u1), u2))
        # angles = feature_size - tf.reduce_sum(tf.sqrt(tf.square(tf.sin(tf.acos(p_diag)))))
        # a = tf.matmul(tf.transpose(m0_norm), m1_norm)
        #a_norm = tf.nn.l2_normalize(a)

        s = tf.svd(tf.reshape(A,[feature_size, feature_size]), full_matrices=True, compute_uv=False, name="svd3")
        #angles = tf.reduce_sum(tf.sqrt(tf.square(tf.sin(tf.acos(s)))))
        #acos = tf.acos(tf.minimum(1.0,s))
        #acos_nonan = tf.where(tf.is_nan(acos_value), tf.zeros(acos_value.shape), acos_value)

        angles = tf.sqrt(tf.reduce_sum(tf.square(s)))

        train_angles = tf.train.GradientDescentOptimizer(1).minimize(angles)

        # image_clustered_with_gt class imputs are classified
        input_set = np.concatenate([self.image_clustered_with_gt[number_to_class[data]] for data in range(10)], axis=0)
        labels = [np.zeros((2400, 10)) for i in range(10)]
        for i in range(10):
            labels[i][:, i][:] = 1
        label_set = np.concatenate(labels, axis=0)

        input_test = np.concatenate([self.clustered_test[data] for data in range(10)], axis=0)
        label_test_list = [np.zeros((len(self.clustered_test[0]), 10)) for i in range(10)]
        for i in range(10):
            label_test_list[i][:, i] = 1
        label_test = np.concatenate(label_test_list, axis=0)

        # first_label = np.stack((np.ones(len(first_input)), np.zeros(len(first_input))), axis=1)
        # second_label = np.stack((np.zeros(len(second_input)), np.ones(len(second_input))), axis=1)
        # label_set = np.concatenate((first_label, second_label), axis=0)
        #
        # input_test = np.concatenate((first_test, second_test))
        # first_test = np.stack((np.ones(len(first_test)), np.zeros(len(first_test))), axis=1)
        # second_test = np.stack((np.zeros(len(second_test)), np.ones(len(second_test))), axis=1)
        # label_test = np.concatenate((first_test, second_test), axis=0)
        #
        dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
        dataset = dataset.repeat(200)
        dataset = dataset.shuffle(buffer_size=10000)
        batched_dataset = dataset.batch(60)
        iterator = batched_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            self.logger.info("Training start")


            #train conv network
            # for _ in range(2):
            #     for first_index in range(10):
            #         for second_index in range(first_index+1, 10):
            #             mean_angle = 1.0
            #             m = 0
            #             while mean_angle>0.5:
            #                 angle_value = 0
            #
            #                 for k in range(4):
            #                     angle_value += sess.run([angles,train_angles], feed_dict={x: np.concatenate((self.image_clustered_with_gt[number_to_class[first_index]][index_list[k]:index_list[k+1]],
            #                                                                              self.image_clustered_with_gt[
            #                                                                                  number_to_class[second_index]][index_list[k]:index_list[k+1]]))})[0]
            #
            #                     # angle_value += sess.run(, feed_dict={x: np.concatenate((self.image_clustered_with_gt[number_to_class[first_index]][index_list[k]:index_list[k+1]],
            #                     #                                                          self.image_clustered_with_gt[
            #                     #                                                              number_to_class[second_index]][index_list[k]:index_list[k+1]]))})
            #
            #                 mean_angle = angle_value/5.0;
            #                 m += 1
            #                 print("epoch {}, first index: {}, second index: {}, mean angle: {}".format(m, first_index, second_index, mean_angle))

            # for m in range(10):
            #     for k in range(1,10):
            #         angle_value = 1.0
            #         while angle_value>0.2:
            #             first_space = sess.run(flat_features,feed_dict={x: self.image_clustered_with_gt[number_to_class[0]][0:490]})
            #             for j in range(1,k):
            #                 first_space += sess.run(flat_features,feed_dict={x: self.image_clustered_with_gt[number_to_class[j]][0:490]})
            #
            #             [__, angle_value] = sess.run([train_angles, angles], feed_dict={x: self.image_clustered_with_gt[number_to_class[k]][0:490], o_u: first_space})
            #             print("j: {} k: {}, angle: {}".format(m, k, angle_value))
            #
            # [m1_val, m2_val] = sess.run([m0,m1], feed_dict={x: np.concatenate((self.image_clustered_with_gt[
            #                                                                         number_to_class[0]][0:490],
            #                                                                     self.image_clustered_with_gt[
            #                                                                         number_to_class[0]][0:490])) })
            #print(u1_val-u2_val)
            #print(angle_values)
            #print(m1_val - m2_val)
            # n_u, n_s, n_v = linalg.svd(np.matmul(np.transpose(np.reshape(u1_val,(490,490))), np.reshape(u2_val,(490,490))))
            #
            # sess.run(train_angles, feed_dict={x: np.concatenate((self.image_clustered_with_gt[
            #                                                      number_to_class[0]][0:490],
            #                                                  self.image_clustered_with_gt[
            #                                                      number_to_class[1]][0:490]))})
            #print("sigmas", n_s)
            # for k in range(0,10):
            #     angle_value = 1.0
            #     # min_value = 0.0
            #     # min_angle_values = []
            #     # #j = 0
            #     if k == 0:
            #         continue
            #     #while angle_value>min_value:
            #     while angle_value>0.1:
            #
            #         sess.run(train_angles, feed_dict={x: np.concatenate((self.image_clustered_with_gt[number_to_class[0]][0:490],
            #                                                   self.image_clustered_with_gt[number_to_class[k]][0:490]))
            #                                 })
            #         angle_value = sess.run(angles, feed_dict={x: np.concatenate((self.image_clustered_with_gt[
            #                                                                  number_to_class[0]][0:490],
            #                                                              self.image_clustered_with_gt[
            #                                                                  number_to_class[k]][0:490]))})
            #         print("k {}, angle_value {}".format(k, angle_value))
            #
            pro = []
            index_list = [(0,490), (490,980), (980,1470), (1470,1960), (1910, 2400)]
            #
            # min1_value = sess.run(test, feed_dict={x: np.concatenate((self.image_clustered_with_gt[
            #                                                                 number_to_class[0]][0:490],
            #                                                              self.image_clustered_with_gt[
            #                                                                  number_to_class[1]][0:490]))})
            # print("min1 value: ", min1_value)
            # print(min1_value.shape)

            for i in range(10):
                feature_matrix = np.zeros((490,490))
                for index in index_list:
                    feature_matrix += sess.run(features, feed_dict={x: self.image_clustered_with_gt[number_to_class[i]][index[0]:index[1]]})
                feature_matrix /= 5.0
                pro_temp = tools.calculateProjectionMatrix(feature_matrix, 100)
                pro_temp = pro_temp/np.linalg.norm(pro_temp)
                pro.append(pro_temp)

            correct = 0
            incorrect = 0
            for j in range(10):
                for image in self.clustered_test[j]:
                    norm = []
                    for i in range(10):
                        norm.append(tools.calculateNorm(np.transpose(sess.run(features, feed_dict={x: np.reshape(image, [1,784])})), pro[i]))
                    if j == norm.index(min(norm)):
                        correct += 1
                    else:
                        incorrect += 1
                        #print(norm)
                print("j {}, correct:{}, incorrect: {}, accuracy {}".format(j, correct, incorrect, correct/(correct+incorrect)))


            max_accu = 0
            for m in range(1, 200):
                for _ in range(400):
                    batch_xs, batch_ys = sess.run(next_element)
                    sess.run(tf_training_model, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})

                accuracy_value = sess.run(tf_accuracy_model, feed_dict={x: input_test, y_: label_test, keep_prob: 1.0})
                max_accu = max(accuracy_value, max_accu)

                print("Epoch: {},  accuracy: {}, max accu {} ".format(m,accuracy_value, max_accu))

                # for k in range(0,10):
                #     angle_value = 2.0
                #     # min_value = 0.0
                #     # min_angle_values = []
                #     # #j = 0
                #     if k == 0:
                #         continue
                #     #while angle_value>min_value:
                #     while angle_value>1.0:
                #         angle_value = sess.run(angles, feed_dict={x: np.concatenate((self.image_clustered_with_gt[
                #                                                                           number_to_class[0]][0:490],
                #                                                                       self.image_clustered_with_gt[
                #                                                                           number_to_class[k]][0:490]))})
                #
                #         sess.run(train_angles, feed_dict={x: np.concatenate((self.image_clustered_with_gt[number_to_class[0]][0:490],
                #                                                   self.image_clustered_with_gt[number_to_class[k]][0:490]))
                #                                })
                #         print("k {}, angle_value {}".format(k, angle_value))

                # accuracy_value = sess.run(tf_accuracy_model,
                #                           feed_dict={x: input_test, y_: label_test, keep_prob: 1.0})
                # max_accu = max(accuracy_value, max_accu)
                #
                # print("Epoch: {},  2 nd accuracy: {}, max accu {} ".format(m, accuracy_value, max_accu))

                    # u1_values = sess.run(u1, feed_dict={x: self.image_clustered_with_gt[number_to_class[index1]],
                    #                                            x2: self.image_clustered_with_gt[
                    #                                                number_to_class[index1]]})
                    #
                    # u2_values = sess.run(u2, feed_dict={x: self.image_clustered_with_gt[number_to_class[index1]],
                    #                                     x2: self.image_clustered_with_gt[
                    #                                         number_to_class[index1]]})
                    #
                    # print(u1_values.shape)
                    # print(u2_values.shape)
                    # for i in range(0, 10):
                    # sess.run(train_angles,
                    #          feed_dict={x: self.image_clustered_with_gt[number_to_class[0]],
                    #                     x2: self.image_clustered_with_gt[number_to_class[1]]})
                    # angle_values = sess.run(angles,
                    #                         feed_dict={x: self.image_clustered_with_gt[number_to_class[0]],
                    #                                    x2: self.image_clustered_with_gt[number_to_class[1]]})
                    # print(angle_values)
                    #
                    # print(batch_ys)
                    # print(sess.run(normalized, feed_dict={x: batch_xs, p1: pro1, p2: pro2, y_: batch_ys, keep_prob: 0.5}))
                    # sess.run(tf_training_model, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

    def train_save_lda(self):
        self.logger.info("Creating tf training model of lda, train and save")

        def weight_variable(shape, name=None):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=name)

        def bias_variable(shape, name=None):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name)

        weights = {
            "w1": tf.Variable(tf.random_normal([3, 3, 1, 64])),
            "w2": tf.Variable(tf.random_normal([3, 3, 64, 64])),
            "w3": tf.Variable(tf.random_normal([3, 3, 64, 96])),
            "w4": tf.Variable(tf.random_normal([3, 3, 96, 96])),
            "w5": tf.Variable(tf.random_normal([3, 3, 96, 256])),
            "w6": tf.Variable(tf.random_normal([3, 3, 256, 256])),
            "w7": tf.Variable(tf.random_normal([1, 1, 256, 10])),
        }
        fcc_weights = {
            'W_fc1': weight_variable([7 * 7 * 10, 256]),
            'b_fc1': bias_variable([256]),
            'W_fc2': weight_variable([256, 10]),
            'b_fc2': bias_variable([10])
        }

        feature_size = 490
        x = tf.placeholder(tf.float32, [None, 784], "x")
        keep_prob = tf.placeholder(tf.float32)
        y_ = tf.placeholder(tf.float32, [None, 10])
        saver = tf.train.Saver()

        [y_conv, features] = deeplda_mnist.lda_create_network(x, weights, fcc_weights, keep_prob)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
        tf_training_model = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        tf_test_model = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        tf_accuracy_model = tf.reduce_mean(tf.cast(tf_test_model, tf.float32))

        input_set = np.concatenate([self.image_clustered_with_gt[number_to_class[data]] for data in range(10)], axis=0)
        labels = [np.zeros((2400, 10)) for i in range(10)]
        for i in range(10):
            labels[i][:, i][:] = 1
        label_set = np.concatenate(labels, axis=0)

        input_test = np.concatenate([self.clustered_test[data] for data in range(10)], axis=0)
        label_test_list = [np.zeros((len(self.clustered_test[0]), 10)) for i in range(10)]
        for i in range(10):
            label_test_list[i][:, i] = 1
        label_test = np.concatenate(label_test_list, axis=0)

        dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
        dataset = dataset.repeat(200)
        dataset = dataset.shuffle(buffer_size=10000)
        batched_dataset = dataset.batch(60)
        iterator = batched_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            self.logger.info("Training start")

            max_accu = 0
            for m in range(1, 200):
                for _ in range(400):
                    batch_xs, batch_ys = sess.run(next_element)
                    sess.run(tf_training_model, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})

                accuracy_value = sess.run(tf_accuracy_model, feed_dict={x: input_test, y_: label_test, keep_prob: 1.0})
                max_accu = max(accuracy_value, max_accu)

                print("Epoch: {},  accuracy: {}, max accu {} ".format(m,accuracy_value, max_accu))

            save_path = saver.save(sess, "/home/mustafa/tf_workspace/master_thesis/tf_model.ckpt")
            print("Model saved in path: %s" % save_path)

    def train_lda_norm(self):
        def weight_variable(shape, name=None):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=name)

        def bias_variable(shape, name=None):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name)

        weights = {
            "w1": tf.Variable(tf.random_normal([3, 3, 1, 64])),
            "w2": tf.Variable(tf.random_normal([3, 3, 64, 64])),
            "w3": tf.Variable(tf.random_normal([3, 3, 64, 96])),
            "w4": tf.Variable(tf.random_normal([3, 3, 96, 96])),
            "w5": tf.Variable(tf.random_normal([3, 3, 96, 256])),
            "w6": tf.Variable(tf.random_normal([3, 3, 256, 256])),
            "w7": tf.Variable(tf.random_normal([1, 1, 256, 10])),
        }
        fcc_weights = {
            'W_fc1': weight_variable([7 * 7 * 10, 256]),
            'b_fc1': bias_variable([256]),
            'W_fc2': weight_variable([256, 10]),
            'b_fc2': bias_variable([10])
        }

        feature_size = 490
        batch_size = 60

        self.logger.info("Creating tf training model with angle separation")
        x = tf.placeholder(tf.float32, [None, 784], "x")
        keep_prob = tf.placeholder(tf.float32)
        y_ = tf.placeholder(tf.float32, [None, 10])
        pro_input = tf.placeholder(tf.float32, [None, feature_size])

        [y_conv, features] = deeplda_mnist.lda_create_network(x, weights, fcc_weights, keep_prob)
        features = tf.nn.l2_normalize(features)

        s, u, v = tf.svd(tf.transpose(tf.reshape(features, [feature_size, feature_size])), full_matrices=True,
                            compute_uv=True,
                            name="svd")

        pro = tf.matmul(u, tf.transpose(u))

        projections = tf.split(pro_input, 10)
        norms = []
        norm1 = tf.sqrt(tf.reduce_sum(tf.square(tf.transpose(features) - tf.matmul(projections[0], tf.transpose(features))), axis=0))
        norm2 = tf.sqrt(
            tf.reduce_sum(tf.square(tf.transpose(features) - tf.matmul(projections[1], tf.transpose(features))),
                          axis=0))
        for p_m in projections:
           norms.append(tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(tf.transpose(features) - tf.matmul(p_m, tf.transpose(features))), axis=0)),  [-1,1]))

        a = tf.concat(norms, axis=1)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=a))

        #regular training
        tf_training_model = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        tf_test_model = tf.equal(tf.argmax(a, 1), tf.argmax(y_, 1))
        tf_accuracy_model = tf.reduce_mean(tf.cast(tf_test_model, tf.float32))

        input_set = np.concatenate([self.image_clustered_with_gt[number_to_class[data]] for data in range(10)], axis=0)
        labels = [np.zeros((2400, 10)) for i in range(10)]
        for i in range(10):
            labels[i][:, i][:] = 1
        label_set = np.concatenate(labels, axis=0)

        input_test = np.concatenate([self.clustered_test[data] for data in range(10)], axis=0)
        label_test_list = [np.zeros((len(self.clustered_test[0]), 10)) for i in range(10)]
        for i in range(10):
            label_test_list[i][:, i] = 1
        label_test = np.concatenate(label_test_list, axis=0)

        dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
        dataset = dataset.repeat(200)
        dataset = dataset.shuffle(buffer_size=10000)
        batched_dataset = dataset.batch(batch_size)
        iterator = batched_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            self.logger.info("Training start")

            for e in range(200):
                #print("calculating projections")
                proj_values = []

                #print("diff" , np.sum(self.image_clustered_with_gt[number_to_class[0]][0:490] - self.image_clustered_with_gt[number_to_class[1]][0:490]))
                for i in range(10):
                    pro_temp = sess.run(pro, feed_dict={x: self.image_clustered_with_gt[number_to_class[i]][0:490]})
                    pro_temp = pro_temp / np.linalg.norm(pro_temp)

                    proj_values.append(pro_temp)
                    # print("projections calculated")
                # print(np.sqrt(np.sum(np.square(proj_values[0] - proj_values[2]))))
                # print(np.sqrt(np.sum(np.square(proj_values[0] - proj_values[3]))))
                pro_array = np.concatenate(proj_values, axis=0)

                for i in range(40):

                    batch_xs, batch_ys = sess.run(next_element)
                    #sess.run(tf_training_model, feed_dict={x: batch_xs, y_: batch_ys, pro_input: pro_array})
                    pro1, pro2 = sess.run([norm1, norm2], feed_dict={x: self.image_clustered_with_gt[number_to_class[0]][0:1], pro_input: pro_array})
                    print(pro1)
                    print(pro2)

                accuracy = 0
                for i in range(65):
                    accuracy += sess.run(tf_accuracy_model, feed_dict={x:input_test[i*60:i*60+60], y_: label_test[i*60:i*60+60], pro_input: pro_array})
                print(accuracy/65)
