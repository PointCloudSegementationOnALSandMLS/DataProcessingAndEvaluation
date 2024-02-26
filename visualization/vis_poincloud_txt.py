import open3d as o3d
import open3d.ml as _ml3d
import os
import json
import open3d.ml.torch as ml3d  # just switch to open3d.ml.tf for tf usage
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def loadData(numpyCloud, txtfile):
    
    data = np.load(numpyCloud)

    points = np.array(data[:, :3], dtype=np.float32) 
    kdtree = cKDTree(points)

    num_points = len(points)
    isolated_points = []

    for i in range(num_points):
        # Query the KD-tree to find the index of the nearest neighbor
        _, neighbor_index = kdtree.query(points[i], k=2)
        
        # Skip the first neighbor (itself), take the second neighbor
        point1 = points[i]
        point2 = points[neighbor_index[1]]
        
        distance = np.linalg.norm(point2 - point1)
        if(distance>2.5):
            print(point1)
            isolated_points.append(i)

            # Set a threshold distance to define isolation (you can adjust this)

    data = np.delete(data, isolated_points, axis=0)

    with open(txtfile, 'r') as file:
# Read the lines of the file and store them in an array
            pred = file.readlines()
            pred = np.array(pred).astype(np.int32)

    return {"data": data, pred: "pred"}

semantic_labels = {
    0: {"class": 'Unclassified', "color": [1.0, 1.0, 1.0]},
    1: {"class": 'Natural', "color": [0.0, 1.0, 0.0]},
    2: {"class": 'Building', "color": [1.0, 0.0, 0.0]},
    3: {"class": 'Pole', "color": [1.0, 0.5, 0.0]},
    4: {"class": 'Ground', "color": [0.5, 0.5, 0.5]},
    5: {"class": 'Car', "color": [0.0, 0.0, 1.0]},
    6: {"class": 'Wall', "color": [0.4, 0.0, 0.4]},
    7: {"class": 'Fence', "color": [1.0, 0.2, 1.0]},
    8: {"class": 'Hedge', "color": [0.0, 0.4, 0.0]}
}
v = ml3d.vis.Visualizer()
lut = ml3d.vis.LabelLUT()

for val in sorted(semantic_labels.keys()):
    lut.add_label(semantic_labels[val]["class"], val, semantic_labels[val]["color"])
v.set_lut("labels", lut)
v.set_lut("pred", lut)

data = np.load('mls/street-seperate/test/test_grid.npy', "mls/kpconv/prediction/test_grid.txt")

data = {"points": points[:,:3].astype(np.float32), "labels": points[:,3].astype(np.int32), "feat": points[:,4].astype(np.float32),
         "name": "test" }
v.visualize([data])