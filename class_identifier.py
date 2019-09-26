import tensorflow as tf
import numpy as np
from models import *
from scipy.sparse.linalg import svds
import scipy.misc
import logging
import random
import math
import matplotlib.pyplot as plt


class SeparationValue:
    def __init__(self):
        self.mean = 0
        self.max = 0
        self.min = 0


class ClassIdentifier:
    def __init__(self, index, class_size_):
        """
        :param index: given index number for the class
        """
        self.class_index = index
        self.logger = logging.getLogger('logger_master')
        self.save_name = "/home/mustafa/tf_workspace/master_thesis/train_saves/train_{}".format(self.class_index)
        self.element_per_class = 0
        self.class_size = class_size_
        self.train_set = None
        self.test_set = None
        self.label_set = None
        self.projection_matrix = None
        self.class_reference = 0
        self.class_separation = [SeparationValue() for _ in range(self.class_size)]
        self.nonclass_min_ = 0
        self.feature_model = None
        self.training_model = None
        self.norm_model = None
        self.rank =100
        self.input_x = None
        self.input_y = None
        self.keep_prob = None
        self.keep_prob_value = 0.8
        self.input_pro = None
        self.feature_size = 0
        self.sess = None
        self.set_name = None
        self.model_name = None
        self.batch_size = 0
        self.next_element = None
        self.currenct_epoch = 0
        self.train_set_size = 0
        self.projection_values = []
        self.separation_finished = False
        self.next_train_focus = []

    def create_train_inputs_model(self, set_name_, model_name_, step_size_):
        """
        Creates the initial values for the training
        :param set_name_: Name of the dataset used e.g mnist, cifar10
        :param model_name_: Model used for training, e.g base
        :param step_size_: Size of the batch going through calculations
        :return:
        """
        with tf.variable_scope("{}_".format(self.class_index)):

            self.model_name = model_name_
            self.set_name = set_name_
            self.batch_size = step_size_
            self.feature_size = model_details[self.set_name][self.model_name]["feature_size"]
            self.logger.debug("Feature size {}".format(self.feature_size))
            if self.set_name == "mnist":
                self.input_x = tf.placeholder(tf.float32, [None, 784], "x")
            elif self.set_name == "cifar10":
                self.input_x = tf.placeholder(tf.float32, [None, 32, 32, 3], "x")
            self.input_y = tf.placeholder(tf.int32, [None])
            self.keep_prob = tf.placeholder(tf.float32)
            self.input_pro = tf.placeholder(tf.float32, [None, self.feature_size], name="projection_matrix")
            self.currenct_epoch = 0
            self.create_models()
            return self.input_x

    def modelReady(self):
        """
        Check of feature training and norm models have been initialized
        :return:
        """
        return self.feature_model and self.training_model and self.norm_model

    def create_models(self):
        """creates the models with given set name and model name"""
        [self.feature_model, self.norm_model, self.training_model, self.rejection_vector_model] = \
            model_details[self.set_name][self.model_name]["model"](self.input_x, self.input_y, self.input_pro, self.keep_prob)

    def create_training_set(self, train_set_, whole_set=True):
        """creates the training set from performance"""
        if whole_set:
            self.train_set = train_set_
#            assert self.class_size == len(self.train_set)
        self.train_set_size = 0
        for i in range(self.class_size):
            print("i", i)
            self.train_set_size += self.train_set[i].shape[0]

        return self.train_set

    def create_first_dataset(self):
        labels = []
        for i in range(self.class_size):

            temp = np.zeros((self.class_size, 1))
            temp[i] = 1
            labels.append(np.repeat(temp, len(self.train_set[i]), axis=1))
        label_set = np.transpose(np.concatenate(labels, axis=1))
        input_set = np.concatenate(
            [self.train_set[data] for data in range(len(self.train_set))], axis=0)

        dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
        dataset = dataset.repeat(20)
        dataset = dataset.shuffle(buffer_size=10000)
        batched_dataset = dataset.batch(self.batch_size)
        self.iterator = batched_dataset.make_initializable_iterator()
        self.next_element = self.iterator.get_next()
        self.train_set_size = label_set.shape[0]

    def calculate_projection_matrix(self, train_set):
        """
        Calculate the projection matrix using the training set and feature model
        Updates the projection_matrix value
        :param self:
        :return:
        """
        self.logger.info("calculating projection matrix")
        train_size = train_set.shape[0]
        iter_size = int(train_size / self.batch_size)
        temp = []

        for i in range(iter_size):
            temp.append(np.transpose(self.sess.run(self.feature_model, feed_dict={self.keep_prob:1.0, self.input_x: train_set[i*self.batch_size:(i + 1) * self.batch_size]})))
        temp.append(np.transpose(self.sess.run(self.feature_model, feed_dict={self.keep_prob:1.0,
            self.input_x: train_set[iter_size * self.batch_size:train_set.shape[0]]})))

        feature_matrix = np.concatenate(temp, axis=1)

        feature_matrix = feature_matrix / np.linalg.norm(feature_matrix)
        u, s, v = svds(feature_matrix)
        self.projection_matrix = np.matmul(u, np.transpose(u))
        #print(self.projection_matrix)
        self.logger.info("projection matrix calculated")

    def train_one_update(self, next_element, projection_update_rate=10):
        self.calculate_projection_matrix(self.train_set[self.class_index])
        for j in range(projection_update_rate):
            print("Training class {} epoch {}".format(self.class_index, self.currenct_epoch))
            for k in range(int(self.train_set_size/self.batch_size)):
                batch_xs, batch_ys = self.sess.run(next_element)
                batch_ys = batch_ys[:,self.class_index]
                self.sess.run(self.training_model,
                         feed_dict={self.input_x: batch_xs, self.input_y: batch_ys, self.keep_prob:self.keep_prob_value,
                                    self.input_pro: self.projection_matrix})

            self.currenct_epoch += 1

    def train_until_separation(self, next_element, projection_update_rate=10):
        while not self.separation_finished:
            self.calculate_projection_matrix(self.train_set[self.class_index])
            for j in range(projection_update_rate):
                print("Training class {} epoch {}".format(self.class_index, self.currenct_epoch))
                for k in range(int(self.train_set_size / self.batch_size)):
                    batch_xs, batch_ys = self.sess.run(next_element)
                    batch_ys = batch_ys[:,self.class_index]
                    self.sess.run(self.training_model,
                             feed_dict={self.input_x: batch_xs, self.input_y: batch_ys,
                                        self.input_pro: self.projection_matrix})
                self.currenct_epoch += 1

            self.calculate_separation()
            self.print_separations()

    def train_until_separation_extra(self, projection_update_rate=10, fast_training=False):
        self.create_first_dataset()
        self.sess.run(self.iterator.initializer)
        self.separation_finished = False
        while not self.separation_finished:
            self.calculate_projection_matrix(self.train_set[self.class_index])
            for j in range(projection_update_rate):
                print("Training class {} epoch {}".format(self.class_index, self.currenct_epoch))
                for k in range(math.ceil(self.train_set_size / self.batch_size)):
                    batch_xs, batch_ys = self.sess.run(self.next_element)
                    batch_ys = batch_ys[:,self.class_index]
                    self.sess.run(self.training_model,
                             feed_dict={self.input_x: batch_xs, self.input_y: batch_ys,
                                        self.input_pro: self.projection_matrix})
                self.currenct_epoch += 1

            self.calculate_separation()
            self.print_separations()
            #self.separation_finished = True
            if fast_training:
                #self.create_next_dataset_uniform()
                self.create_next_dataset_less()
        if self.separation_finished:
            saver = tf.train.Saver()


    def calculate_separation(self):
        """
        Calculate separation values that indicates identifier performance
        :return:
        """
        self.logger.info("Checking separation values for {}".format(self.class_index))
        self.projection_values = []
        for i in range(len(self.train_set)):
            iter_size = int(self.train_set[i].shape[0] / self.batch_size)
            temp = []
            for k in range(iter_size):
                temp.append(self.sess.run(self.norm_model, feed_dict={self.keep_prob:1.0,
                    self.input_x: self.train_set[i][k * self.batch_size:self.batch_size * (k + 1)],
                    self.input_pro: self.projection_matrix}))

            temp.append(self.sess.run(self.norm_model, feed_dict={self.keep_prob:1.0,
                self.input_x: self.train_set[i][iter_size * self.batch_size:self.train_set[i].shape[0]],
                self.input_pro: self.projection_matrix}))

            self.projection_values.append(np.concatenate(temp))

        #plot
        # max_arg = np.argmax(self.projection_values[self.class_index])
        # max_image = self.train_set[self.class_index][max_arg]
        # image = np.reshape(max_image, (28,28))
        # plt.imshow(image, cmap='gray')
        # plt.show()

        min_nonclass = 100
        self.class_total = np.sum(self.projection_values)
        self.projection_values = self.projection_values / self.class_total
        print("class total value: ", self.class_total)
        # calculate class mean max min
        for i in range(self.class_size):
            self.class_separation[i].min = np.min(self.projection_values[i])
            self.class_separation[i].mean = np.mean(self.projection_values[i])
            self.class_separation[i].max = np.max(self.projection_values[i])
            if i != self.class_index:
                min_nonclass = min(min_nonclass, self.class_separation[i].min)

        self.next_train_focus = []
        for i in range(self.class_size):
            if i == self.class_index:
                self.next_train_focus.append(np.where(self.projection_values[i]>min_nonclass))
            else:
                self.next_train_focus.append(np.where(self.projection_values[i]<self.class_separation[self.class_index].max))

        self.separation_finished = True
        for i in range(self.class_size):
            if i != self.class_index:
                self.separation_finished = self.separation_finished and (self.class_separation[self.class_index].max < self.class_separation[i].min)

        if self.separation_finished:
            self.logger.info("Separation is finished for {}".format(self.class_index))

    def create_next_dataset_less(self):
        remaining_size = 0
        for data in range(len(self.train_set)):
            remaining_size += self.next_train_focus[data][0].shape[0]
            print(data, np.take(self.train_set[data], self.next_train_focus[data][0], axis=0).shape)
        if remaining_size == 0:
            return
        input_set = np.concatenate(
            [np.take(self.train_set[data], self.next_train_focus[data][0], axis=0) for data in range(len(self.train_set)) if self.next_train_focus[data][0].shape[0]>0], axis=0)

        labels = []
        for i in range(self.class_size):
            if self.next_train_focus[i][0].shape[0]>0:
                temp = np.zeros((self.class_size, 1))
                temp[i] = 1
                labels.append(np.repeat(temp, self.next_train_focus[i][0].shape[0], axis=1))
        label_set = np.transpose(np.concatenate(labels, axis=1))
        self.train_set_size = label_set.shape[0]
        print("next input set size: {}".format(input_set.shape))
        print("next focus train set size: {}".format(self.train_set_size))
        dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
        if remaining_size < self.batch_size:
            dataset = dataset.repeat(20*math.ceil(self.batch_size/remaining_size))
        else:
            dataset = dataset.repeat(20)
        dataset = dataset.shuffle(buffer_size=10000)
        batched_dataset = dataset.batch(self.batch_size)
        self.iterator = batched_dataset.make_initializable_iterator()
        self.next_element = self.iterator.get_next()
        self.sess.run(self.iterator.initializer)

    def create_next_dataset_uniform(self):
        dataset = []
        labelset = []
        n_regions = 10
        for data in range(len(self.train_set)):
            step = (self.class_separation[data].max - self.class_separation[data].min)/n_regions
            first_value = self.class_separation[data].min
            second_value = self.class_separation[data].min + step
            selected_data = []
            for i in range(n_regions):
                temp = np.logical_and(first_value < self.projection_values[data],self.projection_values[data] < second_value)
                arguments = np.argwhere(temp)
                if arguments.shape[0]>0:
                    r_arg = np.random.choice(arguments[0], 100)
                    selected_data.append(self.train_set[data][r_arg])

                second_value += step
                first_value += step
            t = np.concatenate(selected_data, axis=0)
            temp = np.zeros((self.class_size, 1))
            temp[data] = 1
            labelset.append(np.repeat(temp, t.shape[0], axis=1))
            dataset.append(t)
        input_set = np.concatenate(dataset)
        label_set = np.transpose(np.concatenate(labelset, axis=1))
        self.train_set_size = label_set.shape[0]
        print("next input set size: {}".format(input_set.shape))
        print("next focus train set size: {}".format(self.train_set_size))
        dataset = tf.data.Dataset.from_tensor_slices((input_set, label_set))
        dataset = dataset.repeat(20)
        dataset = dataset.shuffle(buffer_size=10000)
        batched_dataset = dataset.batch(self.batch_size)
        self.iterator = batched_dataset.make_initializable_iterator()
        self.next_element = self.iterator.get_next()
        self.sess.run(self.iterator.initializer)

    def print_separations(self):
        self.logger.info("For class {}".format(self.class_index))
        for i in range(self.class_size):
            if i == self.class_index:
                self.logger.info("Self: min {}, mean: {}, max: {}".format(self.class_separation[i].min,
                                                                          self.class_separation[i].mean,
                                                                          self.class_separation[i].max))
            else:
                self.logger.info("Class {}: min {}, mean: {}, max: {}, separation {}".format(i, self.class_separation[i].min,
                                                                          self.class_separation[i].mean,
                                                                          self.class_separation[i].max,
                                                                         self.class_separation[i].min-self.class_separation[self.class_index].max))

    def calculate_class_reference(self, type_id):
        """
        :param type_id: calculation type for class reference value, types are given below
            0: using the maximum projection value for this class
            1: using minimum of a nonclass projection value
            2: mean of the minimum of nonclass and max of self class
            3: mean of the mean of the self class and min of means of nonclasses
            4: mean of self class
        :return:
        """
        if type_id == 0:
            self.class_reference = self.class_separation[self.class_index].max
        elif type_id == 1:
            values = [self.class_separation[i].min for i in range(self.class_size) if i != self.class_index]
            self.class_reference = min(values)
        elif type_id == 2:
            values = [self.class_separation[i].min for i in range(self.class_size) if i != self.class_index]
            self.class_reference = (min(values) + self.class_separation[self.class_index].max)/2
        elif type_id == 3:
            values = [self.class_separation[i].mean for i in range(self.class_size) if i != self.class_index]
            self.class_reference = (min(values) + self.class_separation[self.class_index].mean)/2
        elif type_id == 4:
            self.class_reference = self.class_separation[self.class_index].mean

        self.logger.info("Class reference value: {}".format(self.class_reference))

    def get_projections(self, input_):
        projection_values = self.sess.run(self.norm_model,
                          feed_dict={self.keep_prob:1.0, self.input_x: input_, self.input_pro: self.projection_matrix})
        return projection_values

    def check_class(self, input_):
        projections = self.get_projections(input_)
        projections = projections/self.class_total
        return projections, np.less_equal(projections, np.full(projections.shape, self.class_reference))

    def get_probability(self, input_):
        values = [self.class_separation[i].mean for i in range(self.class_size) if i != self.class_index]
        projections = self.get_projections(input_)
        diff2 = np.repeat(self.class_separation[self.class_index].mean-min(values),projections.shape[0])
        diff1 = projections - np.repeat(self.class_separation[self.class_index].mean, projections.shape[0])

        prob = np.repeat(1.0, projections.shape[0]) - np.abs(np.divide(diff1,diff2))
        return np.where(projections<=self.class_separation[self.class_index].mean, np.repeat(1.0,projections.shape[0]),
                        prob)

    def calculate_optimum_training_set(self):
        pass
        # else:
        #     self.calculate_projection_matrix(train_set_)
        #     norms = self.get_norms(train_set_)
        #     max_value = np.max(norms)
        #     min_value = np.min(norms)
        #     print("shape: {}, max: {}, min {}".format(norms.shape, max_value , min_value))
        #     step = (max_value - min_value) / n_sample_sets
        #     n_samples_per_set = self.element_per_class / n_sample_sets
        #     chosen_args = []
        #
        #     for i in range(n_sample_sets):
        #         args = np.argwhere(min_value<=norms)
        #         args2 = np.argwhere(norms[args]<=(min_value+step))
        #         if args2.shape[0]>0:
        #             if args2.shape[0]<=n_samples_per_set:
        #                 chosen_args.append(args2[:,0])
        #             else:
        #                 chosen_args.append(np.random.choice(args2[:,0], int(n_samples_per_set)))
        #         min_value += step
        #
        #     chosen_args = np.concatenate(chosen_args)
        #     chosen_args = np.unique(chosen_args)
        #     while chosen_args.shape[0]<self.element_per_class:
        #         chosen_args = np.append(chosen_args, np.random.choice(range(0,train_set_.shape[0]), self.element_per_class-chosen_args.shape[0]))
        #         chosen_args = np.unique(chosen_args)
        #
        #     self.class_set = train_set_[chosen_args]
        #     print(self.class_set.shape)
        #
        #     self.create_label_set()
        #
        #     self.calculate_projection_matrix(self.class_set)  # recalculate projection matrix with new created training set
