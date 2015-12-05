import numpy as np
import heapq

class analysis:
    def __init__(self):
        pass

    #@param {list} l: list of classes
    #@return {list} largest: list with top n
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

