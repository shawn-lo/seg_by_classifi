import numpy as np
from io import StringIO
import operator
from sklearn.neighbors import KDTree

class nearestneighbors:
    def __init__(self, k_value):
        self.k_value = k_value
        self.accuracy = 0
        self.predicted_label = []

    # knn() : classifier
    # @param {ndarray} train_data: training data
    # @param {ndarray} test_data: testing data
    # @return {array} : the index is the number of oversegment, the value is the number of segment label
    def knn(self, feature_train, feature_test, train_label, test_label, reduced_size=5):
        if type(feature_train) is np.ndarray and type(feature_test) is np.ndarray:
            train_number = feature_train.shape[0]
            test_number = feature_test.shape[0]
        elif type(feature_train) is list and type(feature_test) is list:
            train_number = len(feature_train)
            test_number = len(feature_test)
        else:
            print('Error: In knn, the input should be \'numpy.ndarray\' or \'list\'.')

        correct = 0
        wrong = 0

        for i in range(0, test_number):
            labels = {}
            tree = KDTree(feature_train)
            dist, ind = tree.query(feature_test[i], k = self.k_value)
            count0 = 0
            count1 = 0
            for j in ind[0]:
                if train_label[j] in labels:
                    count = labels.get(train_label[j])
                    count += 1
                    labels[train_label[j]] = count
                else:
                    labels[train_label[j]] = 1
            #print(labels)
            predicted_label = max(labels.keys(), key=(lambda key: labels[key]))
            self.predicted_label.append(predicted_label)

            if predicted_label == test_label[i]:
                correct += 1
            else:
                wrong += 1

        print('There are ',correct,' correct and ', wrong, ' wrong.')
        self.accuracy = correct/(correct+wrong)
        #return self

    def predict(self):
        return self.predicted_label

    def get_accuracy(self):
        return self.accuracy


