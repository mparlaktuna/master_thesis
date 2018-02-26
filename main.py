import logging.config
from tools import *
from datetime import datetime
from data_loader import DataLoader
from mnist_solver import MnistSolver
from cifar_solver import CifarSolver
from emnist_solver import EmnistSolver


def main():
    """
    Mustafa Parlaktuna, mparlaktuna@gmail.com
    :return:
    """
    subproject_name = "emnist"
    run_description = "running tensorflow angle tests"
    log_file_name = datetime.now().strftime('%H:%M:%S-%d-%m-%Y') + ".log"
    createLoggers(log_file_name)
    if is_logging:
        with open(log_history_file, 'a') as f:
            f.write(subproject_name + "\t" + run_description + "\t" + log_file_name + "\n")

    logger = logging.getLogger('logger_master')
    logger.info("Starting Subproject " + subproject_name)


    try:
        #mnist test
        # mnist = MnistSolver()
        # mnist.loadTrainingData()
        # mnist.loadTestData()
        # mnist.cluster_training_with_gound_truth()
        # mnist.cluster_test_with_ground_truth()
        # #mnist.test_with_norm_only()
        # mnist.test_classes_with_tf_norm()

        cifar = CifarSolver()
        cifar.loadTrainingData()
        cifar.loadTestData()
        cifar.test_images_vectorize()
        cifar.cluster_training_with_ground_truth()
        cifar.cluster_test_with_ground_truth()
        # cifar.test_two_classes_with_tf_norm_separate()
        #cifar.test_two_classes_with_alexnet_norm()
        #cifar.test_classes_with_alexnet_norm()
        cifar.test_classes_with_alexnet_norm2()

        #emnist run
        # emnist = EmnistSolver("emnist_balanced")
        # emnist.load_training_data()
        # emnist.load_test_data()
        # emnist.cluster_training_with_gound_truth()
        # emnist.cluster_test_with_ground_truth()
        # emnist.test_classes_with_tf_norm_separate2()
        #emnist.test_two_classes_with_separate_norm()
        # emnist.test_classes_with_tf_norm_trials()

        # emnist.test_two_classes_with_tf_norm()
        #emnist.test_classes_with_tf_norm()
        # emnist.test_classes_with_alexnet_norm()
        # emnist.calculate_angle_between_feature_vector_svd()







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


"""
some trial code below for future usage
# cifar10 = CifarSolver()
# cifar10.load_training_data()
# cifar10.print_number_of_elements_per_class()
# cifar10.printImageSizes()

#mnist samples
# mnist = MnistSolver("mnist_pickle")
# mnist.load_training_data()
#
# mnist.load_test_data()
# mnist.testWithNormOnly()

#save outside dataset
#dataLoader.saveDataSet("mnist_clustered", mnist.clustered, "mnist")

#plot all data
#plotSvd(mnist.clustered[1])

#get angles
#angle = calculateSubspaceAngles(mnist.training_images, mnist.test_images)
#logger.debug("Angles between: {}".format(angle))

#dataLoader.loadData("mnist_modified")
#dataLoader.saveDataSet("mnist_modified")
#images, labels = dataLoader.getTrainingData(20000)  

dataLoader.loadData("emnist", "emnist-byclass.mat")
dataLoader.saveDataSet("emnist_byclass")
  
"""