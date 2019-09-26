import logging
import numpy as np


class PerformanceCalculator:
    """
    Calculates the performance with given check and value from an identifier
    A check if an identifier says it identifies that input as its class
    Value is the projection distance calculated
    The guess is found using following rules and performance is updated accordingly
    Cases:
    1)If no checks the smallest value is the guess, if correct it is a value_correct
    2)If only one check but the smallest value is at another index, the guess is the check index, if correct check_correct
    3)If only one check and the smallest value is at that index the guess is that index, if correct sure_correct
    4)If multiple checks, the smallest value is the guess, if correct notsure_correct
    If incorrect increase incorrect
    """
    def __init__(self, n_classes_):
        self.logger = logging.getLogger('logger_master')
        self.performances = [Performance() for n in range(n_classes_)]
        self.check_values = [CheckValue() for n in range(n_classes_)]
        self.accuracies = []

    def reset(self):
        for performance in self.performances:
            performance.reset()

    def add_sure_correct(self, guess):
        self.performances[guess].sure_correct += 1
        self.performances[guess].correct += 1

    def add_value_correct(self, guess):
        self.performances[guess].check_correct += 1
        self.performances[guess].correct += 1

    def add_check_correct(self, guess):
        self.performances[guess].value_correct += 1
        self.performances[guess].correct += 1

    def add_notsure_correct(self, guess):
        self.performances[guess].not_sure_correct += 1
        self.performances[guess].correct += 1

    def add_incorrect(self, guess):
        self.performances[guess].incorrect += 1

    def get_total_accuracy(self):
        pass

    def print_accuracy(self):
        total_correct = 0
        total_incorrect = 0
        for i, performance in enumerate(self.performances):
            total = performance.correct + performance.incorrect
            total_correct += performance.correct
            total_incorrect += performance.incorrect
            if total>0:
                accuracy = performance.correct / total
                self.logger.info(
                    "Guess {}, sure correct: {}, check correct {}, value correct {}, not sure correct {}, incorrect {}, accuracy {}".format(
                        i, performance.sure_correct, performance.check_correct, performance.value_correct,
                        performance.not_sure_correct, performance.incorrect, accuracy))
        accuracy = total_correct/(total_incorrect+total_correct)
        self.accuracies.append(accuracy)
        self.logger.info("Total Correct {}, Total Incorrect {}, Total Accuracy {}".format(total_correct, total_incorrect, accuracy))
        self.logger.info("Best accuracy {}".format(max(self.accuracies)))

    def get_accuracy(self, index):
        return self.performances[index].correct/(self.performances[index].correct + self.performances[index].incorrect)

    def set_result(self, identifier_index, result):
        self.check_values[identifier_index].set_value_check(result)

    def set_probability(self, identifier_index, probability):
        self.check_values[identifier_index].set_probability(probability)

    def get_min_value_index(self):
        temp = [x.current_value for x in self.check_values]
        values = np.stack(temp)
        min_index = np.argmin(values, axis=0)
        return min_index

    def get_check_indexes(self):
        checks = np.stack([x.current_check for x in self.check_values])
        temp = np.split(checks, checks.shape[1], axis=1)
        check_list = [x.tolist() for x in temp]
        check_flat = []
        for l in check_list:
            check_flat.append([item for sublist in l for item in sublist])
        return check_flat

    def print_check_values(self):
        self.logger.info("Checks")
        self.logger.info([i.current_check for i in self.check_values])

        self.logger.info("Values")
        self.logger.info([i.current_value for i in self.check_values])

    def check_correct(self, truth):
        #self.print_check_values()
        min_index_list = self.get_min_value_index()
        check_index_list = self.get_check_indexes()

        for i in range(len(min_index_list)):
            check_indexes = [index for index, check in enumerate(check_index_list[i]) if check]
            min_index = min_index_list[i]

            if len(check_indexes) == 0:  # case 1
                guess = min_index
                if guess == truth:
                    self.add_value_correct(truth)
                else:
                    self.add_incorrect(truth)

            if len(check_indexes) == 1:  # cases 2 and 3
                guess = check_indexes[0]
                if guess == truth:
                    if min_index == truth:  # case 3
                        self.add_sure_correct(truth)
                    else:  # case 2
                        self.add_check_correct(truth)
                else:
                    self.add_incorrect(truth)

            if len(check_indexes) > 1: # case 4
                guess = min_index
                if guess == truth:
                    self.add_notsure_correct(truth)
                else:
                    self.add_incorrect(truth)

    def check_probability(self, truth):
        temp = [x.probability for x in self.check_values]
        prob = np.stack(temp)
        est_out = np.argmax(prob, axis=0)
        self.performances[truth].correct = np.count_nonzero(np.where(np.equal(est_out, np.repeat(truth, est_out.shape)), 1, 0));
        self.performances[truth].sure_correct = self.performances[truth].correct
        self.performances[truth].incorrect = est_out.shape[0] - self.performances[truth].correct

class Performance:
    def __init__(self):
        self.correct = 0
        self.incorrect = 0
        self.value_correct = 0
        self.check_correct = 0
        self.sure_correct = 0
        self.not_sure_correct = 0

    def reset(self):
        self.correct = 0
        self.incorrect = 0
        self.value_correct = 0
        self.check_correct = 0
        self.sure_correct = 0
        self.not_sure_correct = 0


class CheckValue:
    def __init__(self):
        self.current_value = 0
        self.current_check = False
        self.probability = 0

    def set_value_check(self, result):
        self.current_value = result[0]
        self.current_check = result[1]

    def set_probability(self, prob):
        self.probability = prob
