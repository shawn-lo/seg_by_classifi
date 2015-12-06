import numpy as np
from io import StringIO
import scipy
from scipy import ndimage
from pylab import *
from plyfile import PlyData, PlyElement

import NearestNeighbors as nn
import parsing
import analysis

from sklearn import svm
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

if __name__ == '__main__':
    # load raw txt data
    parser = parsing.parsing()
    extracted_features_path = './Data/Cornell/features/office_data_nodefeats.txt'
    raw_data = parser.parse_txt_data(extracted_features_path)
    # load ply data, here just test for office scene 1
    pic_path_set = ['./Data/Cornell/office_data_ply/scene20.ply']
    for pic_path in pic_path_set:
        office_data = parser.parse_ply_data(pic_path)
        print(office_data)


    # create analysis utils
    analyser = analysis.analysis()

#=========================================================
# K-NN
#=========================================================
    # classify
    kSet = [3]
    experiment_number = 1

    for k in kSet:
        print('This is for k = ', k, '\n')
        #k_classifier = nn.nearestneighbors(k)

        for i in range(0, experiment_number):
            k_classifier = nn.nearestneighbors(k)
            np.random.shuffle(raw_data)
            # 50-50
            #raw_train_data, raw_test_data = analyser.split(raw_data, 0.8)
            raw_train_data, raw_test_data = np.split(raw_data, 2)
            # Get train data and test data
            train_data = parser.parse_feature_data(raw_train_data)
            test_data = parser.parse_feature_data(raw_test_data)
            # Get train label and test label
            train_label = parser.parse_label(raw_train_data)
            test_label = parser.parse_label(raw_test_data)

            # reduce the classes number, reduced_classes {(class label, number)}
            reduced_size = 5
            reduced_classes = analyser.reduce_classes(parser.parse_label(raw_data), reduced_size)
            #print(reduced_classes)
            reduced_label = np.zeros(reduced_size)
            for i in range(reduced_size):
                reduced_label[i] = reduced_classes[i][0]

            # Deal with data, make it fit for new class label.
            reduced_train_data, reduced_train_label = parser.parse_reduced_data(train_data, train_label, reduced_label)
            reduced_train_overseg = parser.parse_reduced_overseg(raw_train_data, train_label, reduced_label)

            reduced_test_data, reduced_test_label = parser.parse_reduced_data(test_data, test_label, reduced_label)
            reduced_test_overseg = parser.parse_reduced_overseg(raw_test_data, test_label, reduced_label)

            # over-segmentation to segmentation
            overseg_dict = analyser.over2seg(raw_data, reduced_label)

            k_classifier.knn(reduced_train_data, reduced_test_data, reduced_train_label, reduced_test_label)
            acc = k_classifier.get_accuracy()
            print(acc)
            predicted = k_classifier.predict()

            test_overseg_dict = analyser.construct_dict(reduced_test_overseg, predicted)

#======
# plot
#======
            plotdata = parser.parse2plot(office_data, test_overseg_dict)
            ply_plot = np.array(plotdata, dtype=[('x', 'f4'), ('y', 'f4'), ('z','f4'), ('red','u1'), ('green', 'u1'), ('blue','u1')])
            #print(ply_plot)
            el = PlyElement.describe(ply_plot, 'vertex')
            PlyData([el]).write('test.ply')

            with open('./result.txt', 'w') as f:
                string = 'The accuracy of Knn for k = ' + str(k) + ' is '+ str(acc) + '\n'
                f.write(string)



#========================================================
# SVM
#========================================================
    for i in range(0, experiment_number):
        np.random.shuffle(raw_data)
        # 50-50
        raw_train_data, raw_test_data = analyser.split(raw_data, 0.8)
        #raw_train_data, raw_test_data = np.split(raw_data, 2)
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
        reduced_train_overseg = parser.parse_reduced_overseg(raw_train_data, train_label, reduced_label)
        reduced_test_data, reduced_test_label = parser.parse_reduced_data(test_data, test_label, reduced_label)
        reduced_test_overseg = parser.parse_reduced_overseg(raw_test_data, test_label, reduced_label)

        overseg_dict = analyser.over2seg(raw_data, reduced_label)


        #print(type(reduced_train_data), type(reduced_train_label))
        #clf = SVC(decision_function_shape='ovr')
        #clf = RandomForestClassifier(n_estimators=20)
        #clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2),n_estimators=600,learning_rate=1.5,algorithm="SAMME")
        clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2),n_estimators=1000,learning_rate=1.5,algorithm="SAMME")

        #clf = tree.DecisionTreeClassifier()
        clf.fit(np.asarray(reduced_train_data), np.asarray(reduced_train_label))
        predicted = clf.predict(np.asarray(reduced_test_data))
        test_overseg_dict = analyser.construct_dict(reduced_test_overseg, predicted)

        correct = 0
        wrong = 0
        for i in range(0, len(predicted)):
            if predicted[i] == reduced_test_label[i]:
                correct += 1
            else:
                wrong += 1
        #print(correct,' ', wrong)
        #print('The accuracy of SVM is: ',correct/(correct+wrong))
        adv_acc = correct/(correct+wrong)
        print(adv_acc)
        with open('./result.txt', 'a') as f:
            string = 'The accuracy of SVM is '+ str(adv_acc) + '\n'
            f.write(string)

#======
# plot
#======
            plotdata = parser.parse2plot(office_data, test_overseg_dict)
            ply_plot = np.array(plotdata, dtype=[('x', 'f4'), ('y', 'f4'), ('z','f4'), ('red','u1'), ('green', 'u1'), ('blue','u1')])
            #print(ply_plot)
            el = PlyElement.describe(ply_plot, 'vertex')
            PlyData([el]).write('test2.ply')




