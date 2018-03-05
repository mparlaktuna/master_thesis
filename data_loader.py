import os, sys
import logging
from mnist import MNIST
import numpy as np
import pickle
import scipy.io


class DataLoader(object):
    "loads datasets"

    def __init__(self):
        self._dataSetLocations = {"cifar10":"~/tf_workspace/data/cifar10_data/",
                                  "mnist": "~/tf.workspace/data/mnist_data/",
                                  "emnist": "~/tf_workspace/data/emnist_data/"}
        self.logger = logging.getLogger('logger_master')
        self.dataset_directory = ""
        self.data = {}

    def createMatlab(self, dataset_name):
        """
        creates a matlab for the given dataset_name
        :return:
        """
        self.logger.debug("create matlab for:" + dataset_name)

    def loadData(self, dataset_name, file_name):
        """loads the data set and returns dataset"""
        if dataset_name in self._dataSetLocations:
            self.logger.debug("loading all data: " + dataset_name)
            self.dataset_directory = os.path.join(os.path.dirname(__file__), self._dataSetLocations[dataset_name])
            self.logger.debug("loading from directory: " + self.dataset_directory)

            if file_name == "mnist_raw":
                mndata = MNIST(self.dataset_directory)
                images, labels = mndata.load_training()
                test_images, test_labels = mndata.load_testing()
                images = np.array(images, dtype=float)
                labels = np.array(labels)
                test_images = np.array(test_images, dtype=float)
                test_labels = np.array(test_labels)

                self.data = {"training_images": images, "training_labels": labels, "test_images": test_images, "test_labels": test_labels}
                self.printDataSize()

            elif file_name == "mnist_pickle":
                with open(self.dataset_directory + file_name, 'rb') as fo:
                    self.data = pickle.load(fo)
                self.printDataSize()

            elif file_name == "cifar10":
                batch_names = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
                images = []
                labels = []
                for batch in batch_names:
                    a = self.unpickle(self.dataset_directory + batch)
                    images.extend(self._convert_images(a[b'data']))
                    labels.extend(a[b'labels'])

                test = self.unpickle(self.dataset_directory + "test_batch")
                test_images = self._convert_images(np.array(test[b'data']))
                test_labels = np.array(test[b'labels'], dtype=np.int)

                images = np.array(images, dtype=np.float)
                labels = np.array(labels, dtype=np.int)

                self.data = {"training_images": images, "training_labels": labels, "test_images": test_images, "test_labels": test_labels}
                self.printDataSize()

            elif file_name == "cifar10_pickle":
                self.logger.debug("loading cifar10_pickle")
                with open(self.dataset_directory + file_name, 'rb') as fo:
                    self.data = pickle.load(fo)
                self.printDataSize()

            elif file_name in ["emnist-byclass.mat", "emnist-balanced.mat"]:
                self.logger.debug("loading {}".format(file_name))
                mat = scipy.io.loadmat(self.dataset_directory + file_name)
                images = np.array(mat['dataset'][0,0][0][0,0][0])
                labels = mat['dataset'][0,0][0][0,0][1]
                test_images = np.array(mat['dataset'][0,0][1][0,0][0])
                test_labels = mat['dataset'][0,0][1][0,0][1]

                self.data = {"training_images": images, "training_labels": labels, "test_images": test_images,
                              "test_labels": test_labels}
                self.printDataSize()

            elif file_name in ["emnist_byclass", "emnist_balanced"]:
                self.logger.debug("loading {}".format(file_name))
                with open(self.dataset_directory + file_name, 'rb') as fo:
                    self.data = pickle.load(fo)
                self.printDataSize()

            else:
                self.logger.debug("data is not loaded")
                raise KeyError("given dataset name is not in list: " + dataset_name)

        else:
            self.logger.debug("given dataset name is not in list: " + dataset_name)
            raise KeyError("given dataset name is not in list: " + dataset_name)

    def printDataSize(self):
        """prints the size of the data"""
        self.logger.debug("loaded training image size: " + str(len(self.data["training_images"])))
        self.logger.debug("loaded training label size: " + str(len(self.data["training_labels"])))
        self.logger.debug("loaded test image size: " + str(len(self.data["test_images"])))
        self.logger.debug("loaded test label size: " + str(len(self.data["test_labels"])))

    def unpickle(self, file):
        """unpickle function to load cifar10_raw"""
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def _convert_images(self,raw):
        """
        Convert images from the CIFAR-10 format and
        return a 4-dim array with shape: [image_number, height, width, channel]
        where the pixels are floats between 0.0 and 1.0.
        """
        # Convert the raw images from the data-files to floating-points.
        raw_float = np.array(raw, dtype=float) / 255.0
        # Reshape the array to 4-dimensions.
        images = raw_float.reshape([-1, 3, 32, 32])
        # Reorder the indices of the array.
        images = images.transpose([0, 2, 3, 1])
        return images

    def saveDataSet(self, save_name, outside_data=None, dataset_name=None, size=None, matlab=False):
        """saves the given sie of the data set loaded"""
        if dataset_name:
            self.dataset_directory = os.path.join(os.path.dirname(__file__), self._dataSetLocations[dataset_name])
        if self.data or outside_data:
            self.logger.debug("start saving data set with name " + save_name + " to " + self.dataset_directory)
            if size is None:
                self.logger.debug("saving all data")
                output = open(self.dataset_directory+save_name, 'wb')
                if outside_data:
                    pickle.dump(outside_data, output)
                else:
                    pickle.dump(self.data, output)
            else:
                self.logger.debug("saving size: " + str(size))
        else:
            self.logger.debug("there is no loaded data")

    def loadDataSet(self, file_name):
        self.logger.debug("loading {}".format(file_name))
        with open(self.dataset_directory + file_name, 'rb') as fo:
            loaded_data = pickle.load(fo)
        return loaded_data

    def getTrainingData(self, size=None):
        """returns the training [0-size] elements in the data set"""
        if size is None:
            return self.data["training_images"], self.data["training_labels"]
        else:
            if size<len(self.data["training_images"]):
                return self.data["training_images"][0:size], self.data["training_labels"][0:size]
            else:
                return self.data["training_images"], self.data["training_labels"]

    def getTestData(self, size=None):
        """returns the test [0-size] elements in the data set"""
        if size is None:
            return self.data["test_images"], self.data["test_labels"]
        else:
            if size<len(self.data["test_images"]):
                return self.data["test_images"][0:size], self.data["test_labels"][0:size]
            else:
                return self.data["test_images"], self.data["test_labels"]

