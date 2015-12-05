import numpy as np
from io import StringIO
import scipy
from scipy import ndimage
from pylab import *
from plyfile import PlyData, PlyElement

import NearestNeighbors as nn
import parsing
import analysis

from sklearn.svm import SVC
from sklearn import tree

if __name__ == '__main__':
    # load raw data
    parser = parsing.parsing()
    raw_data = parser.parse_ply_data('./Data/Cornell/features/office_data_nodefeats.txt')

    # create analysis utils
    analyser = analysis.analysis()

#=========================================================
# K-NN
#=========================================================
    # classify
    kSet = [3]
    experiment_number = 10
    for k in kSet:
        print('This is for k = ', k, '\n')
        k_classifier = nn.nearestneighbors(k)

        for i in range(0, experiment_number):
            np.random.shuffle(raw_data)
            # 50-50
            raw_train_data, raw_test_data = np.split(raw_data, 2)
            # Get train data and test data
            train_data = parser.parse_feature_data(raw_train_data)
            test_data = parser.parse_feature_data(raw_test_data)
            # Get train label and test label
            train_label = parser.parse_label(raw_train_data)
            test_label = parser.parse_label(raw_test_data)

            # reduce the classes number, reduced_classes {(class label, number)}
            reduced_size = 2
            reduced_classes = analyser.reduce_classes(parser.parse_label(raw_data), reduced_size)
            #print(reduced_classes)
            reduced_label = np.zeros(reduced_size)
            for i in range(reduced_size):
                reduced_label[i] = reduced_classes[i][0]

            # Deal with data, make it fit for new class label.
            reduced_train_data, reduced_train_label = parser.parse_reduced_data(train_data, train_label, reduced_label)
            reduced_test_data, reduced_test_label = parser.parse_reduced_data(test_data, test_label, reduced_label)

            #print(reduced_train_label)

            acc = k_classifier.knn(reduced_train_data, reduced_test_data, reduced_train_label, reduced_test_label)
            #print(acc)

#========================================================
# SVM
#========================================================
for i in range(0, experiment_number):
            np.random.shuffle(raw_data)
            # 50-50
            raw_train_data, raw_test_data = np.split(raw_data, 2)
            # Get train data and test data
            train_data = parser.parse_feature_data(raw_train_data)
            test_data = parser.parse_feature_data(raw_test_data)
            # Get train label and test label
            train_label = parser.parse_label(raw_train_data)
            test_label = parser.parse_label(raw_test_data)

            # reduce the classes number, reduced_classes {(class label, number)}
            reduced_size = 5
            reduced_classes = analyser.reduce_classes(train_label, reduced_size)
            reduced_label = np.zeros(reduced_size)
            for i in range(reduced_size):
                reduced_label[i] = reduced_classes[i][0]

            # Deal with data, make it fit for new class label.
            reduced_train_data, reduced_train_label = parser.parse_reduced_data(train_data, train_label, reduced_label)
            reduced_test_data, reduced_test_label = parser.parse_reduced_data(test_data, test_label, reduced_label)
            print(len(reduced_train_data))
            print(len(reduced_train_label))
            #print(type(reduced_train_data), type(reduced_train_label))
            clf = SVC()
            #clf = tree.DecisionTreeClassifier()
            clf.fit(np.asarray(reduced_train_data), np.asarray(reduced_train_label))
            predict = clf.predict(np.asarray(reduced_test_data))
            correct = 0
            wrong = 0
            for i in range(0, len(predict)):
                if predict[i] == reduced_test_label[i]:
                    correct += 1
                else:
                    wrong += 1
            print(correct,' ', wrong)
            print(correct/(correct+wrong))

