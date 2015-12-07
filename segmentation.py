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
    pic_path_set = ['./Data/Cornell/office_data_ply/scene5.ply']
    for pic_path in pic_path_set:
        office_data = parser.parse_ply_data(pic_path)
        print(office_data)


    # create analysis utils
    analyser = analysis.analysis()

#=========================================================
# K-NN
#=========================================================
    # classify
#    kSet = [3]
#    experiment_number = 1
#
#    for k in kSet:
#        print('This is for k = ', k, '\n')
#        #k_classifier = nn.nearestneighbors(k)
#
#        for i in range(0, experiment_number):
#            k_classifier = nn.nearestneighbors(k)
#            np.random.shuffle(raw_data)
#            # 50-50
#            raw_train_data, raw_test_data = analyser.split(raw_data, 0.8)
#            #raw_train_data, raw_test_data = np.split(raw_data, 2)
#            # Get train data and test data
#            train_data = parser.parse_feature_data(raw_train_data)
#            test_data = parser.parse_feature_data(raw_test_data)
#            # Get train label and test label
#            train_label = parser.parse_label(raw_train_data)
#            test_label = parser.parse_label(raw_test_data)
#
#            # reduce the classes number, reduced_classes {(class label, number)}
#            reduced_size = 5
#            reduced_classes = analyser.reduce_classes(parser.parse_label(raw_data), reduced_size)
#            #print(reduced_classes)
#            reduced_label = np.zeros(reduced_size)
#            for i in range(reduced_size):
#                reduced_label[i] = reduced_classes[i][0]
#
#            # Deal with data, make it fit for new class label.
#            reduced_train_data, reduced_train_label = parser.parse_reduced_data(train_data, train_label, reduced_label)
#            reduced_train_overseg = parser.parse_reduced_overseg(raw_train_data, train_label, reduced_label)
#
#            reduced_test_data, reduced_test_label = parser.parse_reduced_data(test_data, test_label, reduced_label)
#            reduced_test_overseg = parser.parse_reduced_overseg(raw_test_data, test_label, reduced_label)
#
#            # over-segmentation to segmentation
#            overseg_dict = analyser.over2seg(raw_data, reduced_label)
#
#            k_classifier.knn(reduced_train_data, reduced_test_data, reduced_train_label, reduced_test_label)
#            acc = k_classifier.get_accuracy()
#            print(acc)
#            predicted = k_classifier.predict()
#
#            test_overseg_dict = analyser.construct_dict(reduced_test_overseg, predicted)
#
##======
## plot
##======
#            plotdata = parser.parse2plot(office_data, test_overseg_dict)
#            ply_plot = np.array(plotdata, dtype=[('x', 'f4'), ('y', 'f4'), ('z','f4'), ('red','u1'), ('green', 'u1'), ('blue','u1')])
#            #print(ply_plot)
#            el = PlyElement.describe(ply_plot, 'vertex')
#            PlyData([el]).write('test.ply')
#
#            with open('./result.txt', 'w') as f:
#                string = 'The accuracy of Knn for k = ' + str(k) + ' is '+ str(acc) + '\n'
#                f.write(string)


#========================================================
# SVM
#========================================================
    experiment_number = 5
    accuracy_list = [0,0,0,0,0,0,0,0,0,0,0]
    np.random.shuffle(raw_data)

    for i in range(0, experiment_number):
        # 50-50
        raw_train_data, raw_test_data = analyser.split(raw_data, 0.8)
        #raw_train_data, raw_test_data = analyser.cross_validation(raw_data, i)
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
        # 1, knn
        k_classifier = nn.nearestneighbors(3)
        k_classifier.knn(reduced_train_data, reduced_test_data, reduced_train_label, reduced_test_label)
        acc = k_classifier.get_accuracy()
        accuracy_list[0] += acc
        print(acc)
        predicted = k_classifier.predict()
        with open('./result.txt', 'w') as f:
            string = 'The accuracy of Knn for k = ' + str(3) + ' is '+ str(acc) + '\n'
            f.write(string)



        #print(type(reduced_train_data), type(reduced_train_label))
        #clf = SVC(decision_function_shape='ovo', kernel='linear')
        #clf = RandomForestClassifier(n_estimators=20)
        # 2, decision tree vs adaboost decision tree
        clf_t = tree.DecisionTreeClassifier(max_depth=2)
        clf_a = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2),n_estimators=600,learning_rate=1.5,algorithm="SAMME")
        #clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2),n_estimators=1000,learning_rate=1.5,algorithm="SAMME")

        clf_t.fit(np.asarray(reduced_train_data), np.asarray(reduced_train_label))
        clf_a.fit(np.asarray(reduced_train_data), np.asarray(reduced_train_label))

        predicted_t = clf_t.predict(np.asarray(reduced_test_data))
        predicted_a = clf_a.predict(np.asarray(reduced_test_data))
        test_overseg_dict_t = analyser.construct_dict(reduced_test_overseg, predicted_t)
        test_overseg_dict_a = analyser.construct_dict(reduced_test_overseg, predicted_a)

        correct = 0
        wrong = 0
        for i in range(0, len(predicted_t)):
            if predicted_t[i] == reduced_test_label[i]:
                correct += 1
            else:
                wrong += 1
        #print(correct,' ', wrong)
        #print('The accuracy of SVM is: ',correct/(correct+wrong))
        adv_acc = correct/(correct+wrong)
        print(adv_acc)
        accuracy_list[1] += adv_acc
        with open('./result.txt', 'a') as f:
            string = 'The accuracy of Decision Tree is '+ str(adv_acc) + '\n'
            f.write(string)

        correct = 0
        wrong = 0
        for i in range(0, len(predicted_a)):
            if predicted_a[i] == reduced_test_label[i]:
                correct += 1
            else:
                wrong += 1
        #print(correct,' ', wrong)
        #print('The accuracy of SVM is: ',correct/(correct+wrong))
        adv_acc = correct/(correct+wrong)
        print(adv_acc)
        accuracy_list[2] += adv_acc
        with open('./result.txt', 'a') as f:
            string = 'The accuracy of Decision Tree(Adaboost) is '+ str(adv_acc) + '\n'
            f.write(string)


#====================================
# test
#====================================

#== Test SVM(Linear), OVR vs OVO ==#
        clf_ovo = SVC(decision_function_shape='ovo', kernel='linear')
        clf_ovr = SVC(decision_function_shape='ovr', kernel='linear')
        clf_ovo.fit(np.asarray(reduced_train_data), np.asarray(reduced_train_label))
        clf_ovr.fit(np.asarray(reduced_train_data), np.asarray(reduced_train_label))
        predicted_ovo = clf_ovo.predict(np.asarray(reduced_test_data))
        predicted_ovr = clf_ovr.predict(np.asarray(reduced_test_data))
        test_overseg_dict_ovo = analyser.construct_dict(reduced_test_overseg, predicted_ovo)
        test_overseg_dict_ovr = analyser.construct_dict(reduced_test_overseg, predicted_ovr)

        correct = 0
        wrong = 0
        for i in range(0, len(predicted_ovo)):
            if predicted_ovo[i] == reduced_test_label[i]:
                correct += 1
            else:
                wrong += 1
        #print(correct,' ', wrong)
        #print('The accuracy of SVM is: ',correct/(correct+wrong))
        adv_acc = correct/(correct+wrong)
        print(adv_acc)
        accuracy_list[3] += adv_acc
        with open('./result.txt', 'a') as f:
            string = 'The accuracy of SVM(OVO) is '+ str(adv_acc) + '\n'
            f.write(string)

        correct = 0
        wrong = 0
        for i in range(0, len(predicted_ovr)):
            if predicted_ovr[i] == reduced_test_label[i]:
                correct += 1
            else:
                wrong += 1
        #print(correct,' ', wrong)
        #print('The accuracy of SVM is: ',correct/(correct+wrong))
        adv_acc = correct/(correct+wrong)
        print(adv_acc)
        accuracy_list[4] += adv_acc
        with open('./result.txt', 'a') as f:
            string = 'The accuracy of SVM(OVR) is '+ str(adv_acc) + '\n'
            f.write(string)

#== Test SVM(Linear) vs LinearSVM , ovr==#
        clf_1 = SVC(decision_function_shape='ovr', kernel='linear')
        clf_2 = svm.LinearSVC(multi_class='ovr')
        clf_1.fit(np.asarray(reduced_train_data), np.asarray(reduced_train_label))
        clf_2.fit(np.asarray(reduced_train_data), np.asarray(reduced_train_label))
        predicted_1 = clf_1.predict(np.asarray(reduced_test_data))
        predicted_2 = clf_2.predict(np.asarray(reduced_test_data))
        test_overseg_dict_1 = analyser.construct_dict(reduced_test_overseg, predicted_1)
        test_overseg_dict_2 = analyser.construct_dict(reduced_test_overseg, predicted_2)

        correct = 0
        wrong = 0
        for i in range(0, len(predicted_1)):
            if predicted_1[i] == reduced_test_label[i]:
                correct += 1
            else:
                wrong += 1
        #print(correct,' ', wrong)
        #print('The accuracy of SVM is: ',correct/(correct+wrong))
        adv_acc = correct/(correct+wrong)
        print(adv_acc)
        accuracy_list[5] += adv_acc
        with open('./result.txt', 'a') as f:
            string = 'The accuracy of SVM is '+ str(adv_acc) + '\n'
            f.write(string)

        correct = 0
        wrong = 0
        for i in range(0, len(predicted_2)):
            if predicted_2[i] == reduced_test_label[i]:
                correct += 1
            else:
                wrong += 1
        #print(correct,' ', wrong)
        #print('The accuracy of SVM is: ',correct/(correct+wrong))
        adv_acc = correct/(correct+wrong)
        print(adv_acc)
        accuracy_list[6] += adv_acc
        with open('./result.txt', 'a') as f:
            string = 'The accuracy of LinearSVM is '+ str(adv_acc) + '\n'
            f.write(string)

#== Test C,  SVM(RBF) vs SVM(RBF) , ovr==#
        clf_c1 = SVC(decision_function_shape='ovr', kernel='rbf',)
        clf_c2 = SVC(decision_function_shape='ovr', kernel='rbf', C=1000)
        clf_c1.fit(np.asarray(reduced_train_data), np.asarray(reduced_train_label))
        clf_c2.fit(np.asarray(reduced_train_data), np.asarray(reduced_train_label))
        predicted_c1 = clf_c1.predict(np.asarray(reduced_test_data))
        predicted_c2 = clf_c2.predict(np.asarray(reduced_test_data))
        test_overseg_dict_c1 = analyser.construct_dict(reduced_test_overseg, predicted_c1)
        test_overseg_dict_c2 = analyser.construct_dict(reduced_test_overseg, predicted_c2)

        correct = 0
        wrong = 0
        for i in range(0, len(predicted_c1)):
            if predicted_c1[i] == reduced_test_label[i]:
                correct += 1
            else:
                wrong += 1
        #print(correct,' ', wrong)
        #print('The accuracy of SVM is: ',correct/(correct+wrong))
        adv_acc = correct/(correct+wrong)
        print(adv_acc)
        accuracy_list[7] += adv_acc
        with open('./result.txt', 'a') as f:
            string = 'The accuracy of SVM with C=1 is '+ str(adv_acc) + '\n'
            f.write(string)

        correct = 0
        wrong = 0
        for i in range(0, len(predicted_c2)):
            if predicted_c2[i] == reduced_test_label[i]:
                correct += 1
            else:
                wrong += 1
        #print(correct,' ', wrong)
        #print('The accuracy of SVM is: ',correct/(correct+wrong))
        adv_acc = correct/(correct+wrong)
        print(adv_acc)
        accuracy_list[8] += adv_acc
        with open('./result.txt', 'a') as f:
            string = 'The accuracy of SVM with C=100 is '+ str(adv_acc) + '\n'
            f.write(string)

    for i in range(0,11):
        accuracy_list[i] = accuracy_list[i]/experiment_number
    print(accuracy_list)
    with open('./result.txt', 'a') as f:
        f.write(str(accuracy_list))

'''
#== Test ,  SVM(Poly2) vs SVM(Poly3) , ovr==#
        clf_p1 = SVC(decision_function_shape='ovr', kernel='poly', degree=2)
        clf_p2 = SVC(decision_function_shape='ovr', kernel='poly', degree=3)
        clf_p1.fit(np.asarray(reduced_train_data), np.asarray(reduced_train_label))
        clf_p2.fit(np.asarray(reduced_train_data), np.asarray(reduced_train_label))
        predicted_p1 = clf_p1.predict(np.asarray(reduced_test_data))
        predicted_p2 = clf_p2.predict(np.asarray(reduced_test_data))
        test_overseg_dict_p1 = analyser.construct_dict(reduced_test_overseg, predicted_p1)
        test_overseg_dict_p2 = analyser.construct_dict(reduced_test_overseg, predicted_p2)

        correct = 0
        wrong = 0
        for i in range(0, len(predicted_p1)):
            if predicted_p1[i] == reduced_test_label[i]:
                correct += 1
            else:
                wrong += 1
        #print(correct,' ', wrong)
        #print('The accuracy of SVM is: ',correct/(correct+wrong))
        adv_acc = correct/(correct+wrong)
        print(adv_acc)
        accuracy_list[9] += adv_acc
        with open('./result.txt', 'a') as f:
            string = 'The accuracy of SVM(poly) with degree=2 is '+ str(adv_acc) + '\n'
            f.write(string)

        correct = 0
        wrong = 0
        for i in range(0, len(predicted_p2)):
            if predicted_p2[i] == reduced_test_label[i]:
                correct += 1
            else:
                wrong += 1
        #print(correct,' ', wrong)
        #print('The accuracy of SVM is: ',correct/(correct+wrong))
        adv_acc = correct/(correct+wrong)
        print(adv_acc)
        accuracy_list[10] += adv_acc
        with open('./result.txt', 'a') as f:
            string = 'The accuracy of SVM(poly) with degree=3 is '+ str(adv_acc) + '\n'
            f.write(string)


#======
# plot
#======
            plotdata = parser.parse2plot(office_data, test_overseg_dict)
            ply_plot = np.array(plotdata, dtype=[('x', 'f4'), ('y', 'f4'), ('z','f4'), ('red','u1'), ('green', 'u1'), ('blue','u1')])
            #print(ply_plot)
            el = PlyElement.describe(ply_plot, 'vertex')
            PlyData([el]).write('test2.ply')
'''



