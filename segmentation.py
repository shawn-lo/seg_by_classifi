import numpy as np
from io import StringIO
import scipy
from scipy import ndimage
from pylab import *
from plyfile import PlyData, PlyElement
import knn as ks

# read ply data
plyData = PlyData.read('./Data/Cornell/office_data_ply/scene1.ply')
dataset = plyData['vertex'][:]
print(plyData)
# get labels
labels_name = np.loadtxt('./Data/Cornell/features/labels.txt',converters={0: lambda x: unicode(x, 'utf-8')}, dtype='U2')
labels_num = labels_name.shape[0]
labels = np.zeros(labels_num)
# split into train data and test data
n = 1
for number in range(0, n):
    np.random.shuffle(dataset)
    train_data, test_data = np.split(dataset, 2)
    ks.knn(5,train_data, test_data, labels)

#    vertex = np.array(plydata['vertex'][:], dtype=[('x', 'f4'), ('y','f4'), ('z', 'f4'),
#    ('red','u1'), ('green','u1'),('blue','u1'),
#    ('cameraIndex','u1'),('distance','f4'),
#    ('segment','u1'),('label','u1')])
#el = PlyElement.describe(vertex, 'vertex')
#PlyData([el]).write('test.ply')
