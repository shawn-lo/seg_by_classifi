import numpy as np
import heapq

class analysis:
    def __init__(self):
        pass

    def split(self, dataset, percent):
        np.random.shuffle(dataset)
        size = dataset.shape[0]
        pivot = int(size*percent)
        train = dataset[:pivot,:]
        test = dataset[pivot:,:]
        print('There are ', len(train), ' training data.')
        print(type(train))
        print('There are ', len(test), 'testing data.')
        return [train, test]


    # @param {list} l: list of classes
    # @return {list} largest: list with top n
    def reduce_classes(self, l, n=5):
        # construct dict
        d = {}
        for i in l:
            if i in d:
                count = d.get(i)
                count += 1
                d[i] = count
            else:
                d[i] = 1
        # find top n
        heap = [(-value, key) for key,value in d.items()]
        largest = heapq.nsmallest(n, heap)
        largest = list((key, -value) for value, key in largest)
        return largest

    # @param {ndarray} data: raw data with feature seg_number, seg_label
    # @param {ndarray} labels: reduced label
    # @return {dict} d (key,value): key->overseg, value->seg
    def over2seg(self, data, labels):
        d = {}
        for item in data:
            over = item[0]
            seg = item[1]
            if seg in labels:
                d[over] = seg
        return d

    def construct_dict(self,keyset, valueset):
        d = {}
        size = len(keyset)
        for i in range(0, size):
            d[keyset[i]] = valueset[i]
        return d


