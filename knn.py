import numpy as np
from sklearn.neighbors import KDTree

def knn(k_value, train_data, test_data, labels):
    # Training
    number = train_data.shape[0]
    # use X,Y,Z, RGB as feature
    feature_train = np.zeros((number,6))
    for i in range(0, number):
        feature_train[i][0:6] = train_data[['x','y','z','red','green','blue','cameraIndex', 'distance','segment','label']][i]
        #feature_train[i][0:6] = train_data[i][0:6]

    # Testing
    predict_data = test_data
    correct = 0
    wrong = 0
    amount = test_data.shape[0]
    feature_test = np.zeros((amount, 6))
    for i in range(0, amount):
        feature_test[i][0:6] = test_data[['x','y','z','red','green','blue','cameraIndex', 'distance','segment','label']][i]

    for i in range(0, amount):
        tree = KDTree(feature_train)
        dist, ind = tree.query(feature_test[i], k=k_value)
        for j in ind[0]:
            index = train_data[j][9]
            labels[index] += 1
        print('Yes')
        # Find the max label
        predictLabel = np.argmax(labels)
        predict_data[i][9] = predictLabel
        # Check predict label
        if predictLabel == test_data[i][9]:
            correct += 1
        else:
            wrong += 1

    print('The correct rate is: ', correct/(correct+wrong))



