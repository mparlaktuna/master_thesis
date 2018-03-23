import tensorflow as tf
import logging
from data_loader import DataLoader
import numpy as np
import tools


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
            label_index = np.where(self.training_labels==int(i))
            self.image_clustered_with_gt[i] = np.array(np.take(self.training_images,label_index[0], axis=0))

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

    def calculate_projection_matrices(self):
        self.projection_matrices = {}
        for i, matrix in self.image_clustered_with_gt.items():
            self.logger.info("Calculation projection matrix for {}".format(i))
            self.projection_matrices[i] = tools.calculateProjectionMatrix(matrix.astype(float), 40)

    def test_with_norm_only(self):
        """test the whole set using the norms to matrices"""
        if self.clustered is None:
            self.cluster_training_with_gound_truth()

        if self.projection_matrices is None:
            self.calculate_projection_matrices()

        correct = 0
        incorrect = 0
        for i, v in enumerate(self.test_images):
            if i%100:
                self.logger.info("Image index {}".format(i))
            a = {str(index): tools.calculateNorm(np.transpose(v), P) for index, P in self.projection_matrices.items()}
            minimum_value = min(a, key=a.get)
            if int(minimum_value) == self.test_labels[i]:
                correct += 1
            else:
                incorrect += 1

        self.logger.info("Test with norms only:")
        self.logger.info("Correct values: {}".format(correct))
        self.logger.info("Incorrect values: {}".format(incorrect))
        self.logger.info("Performance: {}".format(correct/len(self.test_images)))

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

        input_set = np.concatenate([self.image_clustered_with_gt[data] for data in range(10)], axis=0)
        labels = [np.zeros((len(self.image_clustered_with_gt[i]),10)) for i in range(10)]
        for i in range(10):
            labels[i][:,i] = 1
        label_set = np.concatenate(labels, axis=0)

        input_test = np.concatenate([self.clustered_test[data] for data in range(10)], axis=0)
        label_test_list = [np.zeros((len(self.clustered_test[i]),10)) for i in range(10)]
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

        out = tf.stack([norm1, norm2, norm3, norm4, norm5, norm6, norm7, norm8, norm9, norm10], axis=1)

        normalized = tf.abs(1 - tf.nn.l2_normalize(out, axis=1))
        cross_entropy = tf.reduce_mean(tf.abs(y_ - normalized))

        train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)

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

            self.logger.info("Calculating projections")
            out1 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[0]})
            out2 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[1]})
            out3 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[2]})
            out4 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[3]})
            out5 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[4]})
            out6 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[5]})
            out7 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[6]})
            out8 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[7]})
            out9 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[8]})
            out10 = sess.run(h_pool2_flat, feed_dict={x: self.image_clustered_with_gt[9]})

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
            print('no training test accuracy %g' % accuracy.eval(feed_dict={
                x: input_test, y_: label_test, p1: pro1, p2: pro2, p3: pro3, p4: pro4, p5: pro5, p6: pro6, p7: pro7,
                p8: pro8,
                p9: pro9, p10: pro10}))

            for m in range(1, 200000):

                for _ in range(100):
                    batch_xs, batch_ys = sess.run(next_element)
                    sess.run(train_step, feed_dict={x: batch_xs, p1: pro1, p2: pro2, p3: pro3, p4: pro4,p5: pro5, p6: pro6,p7: pro7, p8: pro8,
                                                     p9: pro9, p10: pro10, y_:batch_ys})

                print('test accuracy %g' % accuracy.eval(feed_dict={
                x: input_test, y_: label_test,  p1: pro1, p2: pro2, p3: pro3, p4: pro4,p5: pro5, p6: pro6,p7: pro7, p8: pro8,
                                                    p9: pro9, p10: pro10}))

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

        s1, u1, v1 = tf.svd(tf.transpose(tf.reshape(h_pool2_flat, [3136, 3136])), full_matrices=True, compute_uv=True, name="svd")
        s2, u2, v2 = tf.svd(tf.transpose(tf.reshape(h_pool2_flat_2, [3136, 3136])), full_matrices=True, compute_uv=True, name="svd2")

        # tf.matmul(tf.transpose(u1), u2)

        # s, u, v = tf.svd(tf.matmul(tf.transpose(u1), u2), full_matrices=True, compute_uv=True, name="svd")
        # s, u, v = tf.svd(tf.matmul(tf.transpose(h_pool2_flat), h_pool2_flat_2), full_matrices=True, compute_uv=True, name="svd")

        #s, u, v = tf.svd(tf.matmul(u1, u2), full_matrices=True, compute_uv=True, name="svd3")
        p_diag = tf.diag_part(tf.matmul(tf.transpose(u1), u2))
        angles = 3136-tf.reduce_sum(tf.square(tf.sin(tf.acos(p_diag))))
        train_angles = tf.train.AdamOptimizer(1e-4).minimize(angles)

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
        first_index = 2
        second_index = 8
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

            for m in range(1, 200000):
                for _ in range(40):
                    batch_xs, batch_ys = sess.run(next_element)

                    # s_values = sess.run(angles, feed_dict={x: self.image_clustered_with_gt[0][0:3136],
                    #                                            x2: self.image_clustered_with_gt[1][0:3136]})
                    #
                    # u2_values = sess.run(u2, feed_dict={x: self.image_clustered_with_gt[number_to_class[index1]],
                    #                                     x2: self.image_clustered_with_gt[
                    #                                         number_to_class[index1]]})
                    #
                    # print(s_values)
                    # print(u2_values.shape)
                    # for i in range(0, 10):
                    print("traing start")
                    # sess.run(train_angles,
                    #          feed_dict={x: self.image_clustered_with_gt[0][0:3136],
                    #                     x2: self.image_clustered_with_gt[1][0:3136]})
                    angle_values = sess.run([train_angles,angles],
                                            feed_dict={x: self.image_clustered_with_gt[0][0:3136],
                                                       x2: self.image_clustered_with_gt[1][0:3136]})
                    print(angle_values)

                    # print(batch_ys)
                    # print(sess.run(normalized, feed_dict={x: batch_xs, p1: pro1, p2: pro2, y_: batch_ys, keep_prob: 0.5}))
                    # sess.run(tf_training_model, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

                # print('test accuracy %g' % tf_accuracy_model.eval(feed_dict={
                #     x: input_test, y_: label_test, keep_prob: 1.0}))

    def train_lda_with_mid_angle_separation(self):
        def conv2d(name, l_input, w):
            return tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool(name, l_input, k):
            return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

        def norm(name, l_input, lsize=4):
            return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

        def batch_norm(input):
            return tf.contrib.layers.batch_norm(input)

        self.logger.info("Creating tf training model with angle separation")
        x = tf.placeholder(tf.float32, [None, 784], "x")

        x_image = tf.reshape(x, [-1, 28, 28, 1])

        weights = {
            "w1": tf.Variable(tf.random_normal([3, 3, 1, 64])),
            "w2": tf.Variable(tf.random_normal([3, 3, 64, 64])),
            "w3": tf.Variable(tf.random_normal([3, 3, 64, 96])),
            "w4": tf.Variable(tf.random_normal([3, 3, 96, 96])),
            "w5": tf.Variable(tf.random_normal([3, 3, 96, 256])),
            "w6": tf.Variable(tf.random_normal([3, 3, 256, 256])),
            "w7": tf.Variable(tf.random_normal([1, 1, 256, 10])),
        }

        conv1 = conv2d("conv1", x_image, weights["w1"])
        norm1 = batch_norm(conv1)
        conv2 = conv2d("conv2", norm1, weights["w2"])
        norm2 = batch_norm(conv2)
        pool1 = max_pool("pool1", norm2, 2)

        conv3 = conv2d("conv3", pool1, weights["w3"])
        norm3 = batch_norm(conv3)
        conv4 = conv2d("conv4", norm3, weights["w4"])
        norm4 = batch_norm(conv4)
        pool2 = max_pool("pool2", norm4, 2)

        conv5 = conv2d("conv5", pool2, weights["w5"])
        norm5 = batch_norm(conv5)
        conv6 = conv2d("conv6", norm5, weights["w6"])
        norm6 = batch_norm(conv6)
        conv7 = conv2d("conv7", norm6, weights["w7"])
        norm7 = batch_norm(conv7)
        out1 = tf.reshape(norm7, [-1, 7 * 7 * 10])

        m1, m2 = tf.split(out1, 2)
        # m1 = tf.placeholder(tf.float32, [490, 490], "m1")
        # m2 = tf.placeholder(tf.float32, [490, 490], "m2")

        # h_pool2_flat = tf.reshape(out1, [-1, 7 * 7 * 64])
        # h_pool2_flat_2 = tf.reshape(out1, [-1, 7 * 7 * 64])
        #
        s1, u1, v1 = tf.svd(tf.transpose(tf.reshape(m1, [490,490])), full_matrices=True, compute_uv=True,
                            name="svd")
        s2, u2, v2 = tf.svd(tf.transpose(tf.reshape(m2, [490,490])), full_matrices=True, compute_uv=True,
                            name="svd2")

        # tf.matmul(tf.transpose(u1), u2)

        # s, u, v = tf.svd(tf.matmul(tf.transpose(u1), u2), full_matrices=True, compute_uv=True, name="svd")
        # s, u, v = tf.svd(tf.matmul(tf.transpose(h_pool2_flat), h_pool2_flat_2), full_matrices=True, compute_uv=True, name="svd")

        # s, u, v = tf.svd(tf.matmul(u1, u2), full_matrices=True, compute_uv=True, name="svd3")
        p_diag = tf.diag_part(tf.matmul(tf.transpose(u1), u2))
        angles = 490 - tf.reduce_sum(tf.square(tf.sin(tf.acos(p_diag))))
        train_angles = tf.train.AdamOptimizer(1e-4).minimize(angles)

        W_fc1 = tools.weight_variable([7 * 7 * 10, 1024])
        b_fc1 = tools.bias_variable([1024])
        h_fc1 = tf.nn.relu(tf.matmul(out1, W_fc1) + b_fc1)
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
        first_index = 2
        second_index = 8
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

            for m in range(1, 200000):
                for _ in range(40):
                    batch_xs, batch_ys = sess.run(next_element)

                    out1_val = sess.run([train_angles, angles], feed_dict={x: np.concatenate([self.image_clustered_with_gt[0][0:490], self.image_clustered_with_gt[1][0:490]])})
                    #out2_val = sess.run(out1, feed_dict={x: })
                    print(out1_val)
                    # angles = sess.run(u2, feed_dict={m1:out1_val, m2: out2_val})
                    # #
                    # print(angles.shape)
                    # print(u2_values.shape)
                    # for i in range(0, 10):
                    print("traing start")
                    # sess.run(train_angles,
                    #          feed_dict={x: self.image_clustered_with_gt[0][0:3136],
                    #                     x2: self.image_clustered_with_gt[1][0:3136]})
                    # angle_values = sess.run([train_angles, angles],
                    #                         feed_dict={x: self.image_clustered_with_gt[0][0:3136],
                    #                                    x2: self.image_clustered_with_gt[1][0:3136]})
                    # print(angle_values)

                    # print(batch_ys)
                    # print(sess.run(normalized, feed_dict={x: batch_xs, p1: pro1, p2: pro2, y_: batch_ys, keep_prob: 0.5}))
                    # sess.run(tf_training_model, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

                # print('test accuracy %g' % tf_accuracy_model.eval(feed_dict={
                #     x: input_test, y_: label_test, keep_prob: 1.0}))