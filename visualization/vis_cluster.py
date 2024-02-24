import open3d as o3d
import open3d.ml as _ml3d
import os
import json
import open3d.ml.torch as ml3d  # just switch to open3d.ml.tf for tf usage
import scipy.io
import numpy as np
#pointsNumpy = np.load("grid_1.npy", allow_pickle=True)
#print(pointsNumpy)

colors = {
    0:[230/255, 25/255, 75/255],
    1:[60/255, 180/255, 75/255],
    2:[255/255, 225/255, 25/255],
    3: [0/255, 130/255, 200/255],
    4:[245/255, 130/255, 48/255],
    5:[70/255, 240/255, 240/255],
    6:[240/255, 50/255, 230/255],
    7:[250/255, 190/255, 212/255],
    8:[0/255, 128/255, 128/255],
    9:[220/255, 190/255, 255/255],
    10:[170/255, 110/255, 40/255],
    11:[255/255, 250/255, 200/255],
    12:[128/255, 0/255, 0/255],
    13:[170/255, 255/255, 195/255],
    14:[0/255, 0/255, 128/255],
    15:[128/255, 128/255, 128/255],
    -1 :[0/255, 0/255, 0/255]
}
v = ml3d.vis.Visualizer()
lut = ml3d.vis.LabelLUT()
points = np.load('clusters/clusters_grid_0_mobile.npy')
for val in np.unique(points[:,4]):
    #print(colors[int(val)])
    lut.add_label("class" + str(val), val)
v.set_lut("labels", lut)
v.set_lut("cluster", lut)

data = {"points": points[:,:3].astype(np.float32), "labels": points[:,3].astype(np.int32), "cluster": points[:,4].astype(np.int32), "feat": None,
         "name": "cluster" }
v.visualize([data])