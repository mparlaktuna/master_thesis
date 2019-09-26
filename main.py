import logging.config
from tools import *
from datetime import datetime
from data_loader import DataLoader
from mnist_solver import MnistSolver
from cifar_solver import CifarSolver
from emnist_solver import EmnistSolver
import tensorflow as tf



def main():
    """
    Mustafa Parlaktuna, mparlaktuna@gmail.com
    :return:
    """
    subproject_name = "cifar10"
    run_description = "running tensorflow angle tests"
    log_file_name = datetime.now().strftime('%H:%M:%S-%d-%m-%Y') + ".log"
    createLoggers(log_file_name)
    if is_logging:
        with open(log_history_file, 'a') as f:
            f.write(subproject_name + "\t" + run_description + "\t" + log_file_name + "\n")


    try:
        logger = logging.getLogger('logger_master')
        logger.info("Starting Subproject " + subproject_name)
        if subproject_name == "mnist":
            mnist = MnistSolver()
            mnist.loadTrainingData()
            mnist.loadTestData()
            mnist.cluster_training_with_gound_truth()
            mnist.cluster_test_with_ground_truth()
            mnist.train("layer_norm")
            #mnist.train_fcn("conv2")

        elif subproject_name == "emnist":
            emnist = EmnistSolver("emnist_balanced")
            emnist.load_training_data()
            emnist.load_test_data()
            emnist.cluster_training_with_gound_truth()
            emnist.cluster_test_with_ground_truth()
            emnist.train_one_norm()

        elif subproject_name == "cifar10":
            cifar = CifarSolver()
            cifar.loadTrainingData()
            cifar.loadTestData()
            cifar.test_images_vectorize()
            cifar.cluster_training_with_ground_truth()
            cifar.cluster_test_with_ground_truth()
            cifar.train("conv")

    except KeyError:
        logger.debug("Unable to load dataset")
    #
    # #sample run cifar
    # #cifar10 = cifar10_data()
    #
    # #sample run minst
    # mnist = Mnist()
    # mnist.loadDataSet('asd')


if __name__ == '__main__':
    main()

