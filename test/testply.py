import numpy as np
from io import StringIO
import re
import scipy
from scipy import ndimage
from PIL import Image
from pylab import *
from plyfile import PlyData, PlyElement


plydata = PlyData.read('office_data_ply/scene1.ply')
#data = np.asarray(plydata['vertex'][:])
#print(data)
for d in plydata['vertex'][:]:
    d[3] = 0
    d[4] = 0
    d[5] = 0
#print(plydata['vertex'][:])
vertex = np.array(plydata['vertex'][:], dtype=[('x', 'f4'), ('y','f4'), ('z', 'f4'),
    ('red','u1'), ('green','u1'),('blue','u1'),
    ('cameraIndex','u1'),('distance','f4'),
    ('segment','u1'),('label','u1')])
el = PlyElement.describe(vertex, 'vertex')
PlyData([el]).write('test.ply')

result = PlyData.read('test.ply')
print(result['vertex'][:])
#print(plydata.elements[0].data)

