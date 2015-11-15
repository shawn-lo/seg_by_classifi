import numpy as np
from io import StringIO
import re
import scipy
from scipy import ndimage
from PIL import Image
from pylab import *
from plyfile import PlyData, PlyElement
#arr = np.zeros((480, 640))
#
#a = np.loadtxt('scene25_5.txt')
#h,w = a.shape[:2]
#print(a.shape)
#rgb = np.zeros((h,w,3), dtype=np.uint8)
#for y in range(0, h):
#    for x in range(0, w):
#        rgb[y][x][0] = a[y][x]*10
#        rgb[y][x][1] = a[y][x]
#        rgb[y][x][2] = a[y][x]*10
#print(rgb)
#scipy.misc.imsave('test.png', rgb)
#
#r = np.zeros((255, 255, 3), dtype=np.uint8)
#r[..., 0] = np.arange(255)
#r[..., 1] = 55
#r[..., 2] = 1 - np.arange(255)
##print(r)
#scipy.misc.imsave('test2.png', r)

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

