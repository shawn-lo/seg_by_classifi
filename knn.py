import numpy as np
from io import StringIO
from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
#============================================================
#
#============================================================

# knn() : classifier
# @param {int} k_value : the number of nearest neighbors
# @param {ndarray} train_data: training data
# @param {ndarray} test_data: testing data
# @return {array} : the index is the number of oversegment, the value is the number of segment label
def knn(k_value, train_data, test_data):
    # Training
    train_number = train_data.shape[0]
    # features from 4 to 55
    feature_train = np.zeros((train_number, 52))
    for i in range(0, train_number):
        feature_train[i][0:52] = train_data[i][3:55]
    #print(feature_train)

    # Testing
    correct = 0
    wrong = 0

    test_number = test_data.shape[0]
    feature_test = np.zeros((test_number, 52))

    for i in range(0, test_number):
        feature_test[i][0:52] = test_data[i][3:55]
    #print(feature_test)

    correct = 0
    wrong = 0
    # test data, label start from 1
    for i in range(0, test_number):
        labels = np.zeros(129)
        tree = KDTree(feature_train)
        dist, ind = tree.query(feature_test[i], k = k_value)
        count0 = 0
        count1 = 0
        for j in ind[0]:
            index = train_data[j][2]
            labels[index-1] += 1
        #print(labels)
        indice = np.argmax(labels)
        #print(indice)
        label = indice+1
        #label = train_data[indice][2]
        #print(label)
        if label == test_data[i][2]:
            correct += 1
        else:
            wrong += 1
    print(correct,' ',wrong)
    return correct/(correct+wrong)

# parse_features_data(): parsing raw data to ndarray
# @param {string} path: path of raw data about extracted features
def parse_features_data(path):
    i = 0
    data_type = []
    features_data = np.zeros((1108,55),dtype='float32')
    with open(path) as input_data:
        for line in input_data:
            # get features name and construct data_type
            if i == 0:
                print('Feature data parsing begins.')
            elif i < 56:
                feature_name = line[1:-1]
                if i < 4:
                    data_type.append((feature_name, 'int16'))
                else:
                    data_type.append((feature_name, 'float32'))
            # get features data
            else:
                raw_string = line[:-1]
                raw_features_data = np.fromstring(raw_string, sep='\t')
                raw_featuers_data = np.array(raw_features_data, dtype='float32')
                features_data[i-56] = raw_features_data
            i += 1
    return features_data

if __name__ == '__main__':
    # load raw data
    raw_data = parse_features_data('./Data/Cornell/features/office_data_nodefeats.txt')
    #print(raw_data)

    # classify
    kSet = [5]
    experiment_number = 1
    for k in kSet:
        print('This is for k = ', k, '\n')
        for i in range(0, experiment_number):
            #np.random.shuffle(raw_data)
            # 50-50
            train_data, test_data = np.split(raw_data, 2)
            print(knn(k,train_data, test_data))
#============================================================
            # Training
            train_number = train_data.shape[0]
            # features from 4 to 55
            feature_train = np.zeros((train_number, 52))
            for i in range(0, train_number):
                feature_train[i][0:52] = train_data[i][3:55]
            #print(feature_train)
            # Testing
            test_number = test_data.shape[0]
            feature_test = np.zeros((test_number, 52))

            for i in range(0, test_number):
                feature_test[i][0:52] = test_data[i][3:55]
            #print(feature_test)

            # training_label construct
            training_label = np.zeros(train_number)
            for i in range(0, train_number):
                training_label[i] = train_data[i][2]
            #print(training_label)

            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(feature_train, training_label)
            total = 0
            correct = 0
            for i in range(0, test_number):
                l = neigh.predict(feature_test[i])
                if l == test_data[i][2]:
                    correct+=1
                total += 1
            print(float(correct)/float(total))
#===================================================
            # DT
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(feature_train, training_label)
            total = 0
            correct = 0
            for i in range(0, test_number):
                l = clf.predict(feature_test[i])
                if l == test_data[i][2]:
                    correct += 1
                total += 1
            print(float(correct)/float(total))

