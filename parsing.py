import numpy as np

class parsing:
    def __init__(self):
        pass
    # parse_ply_data(): parsing raw data to ndarray
    # @param {string} path: path of raw data about extracted features
    # @return {numpy array}: features data
    def parse_ply_data(self, path):
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

    def parse_feature_data(self, raw_data):
        size = raw_data.shape[0]
        feature = np.zeros((size, 52))
        for i in range(0, size):
            feature[i][0:52] = raw_data[i][3:55]
        return feature

    #@return {list}
    def parse_label(self, raw_data):
        size = raw_data.shape[0]
        label = list(map(lambda x: x, raw_data[:,2]))
        return label

    def parse_reduced_data(self, raw_data, raw_label, class_set):
        size = raw_data.shape[0]
        data = []
        label = []
        for i in range(0, size):
            if raw_label[i] in class_set:
                data.append(raw_data[i])
                label.append(raw_label[i])
        return [data, label]
