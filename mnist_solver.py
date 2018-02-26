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
