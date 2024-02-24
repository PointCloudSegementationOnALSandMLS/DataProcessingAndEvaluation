import open3d as o3d
import open3d.ml as _ml3d
import os
import json
import open3d.ml.torch as ml3d  # just switch to open3d.ml.tf for tf usage
import scipy.io
import numpy as np
#pointsNumpy = np.load("grid_1.npy", allow_pickle=True)
#print(pointsNumpy)
semantic_labels = {
    0: {"class": 'Unclassified', "color": [1.0, 1.0, 1.0]},
    1: {"class": 'Natural', "color": [0.0, 1.0, 0.0]},
    2: {"class": 'Building', "color": [1.0, 0.0, 0.0]},
    3: {"class": 'Pole', "color": [1.0, 0.5, 0.0]},
    4: {"class": 'Street/ Ground', "color": [0.5, 0.5, 0.5]},
    5: {"class": 'Car', "color": [0.0, 0.0, 1.0]},
    6: {"class": 'Wall/ Vertical', "color": [0.4, 0.0, 0.4]},
    7: {"class": 'Fence', "color": [1.0, 0.2, 1.0]},
    8: {"class": 'Hedge', "color": [0.0, 0.4, 0.0]}
}


v = ml3d.vis.Visualizer()
lut = ml3d.vis.LabelLUT()

for val in sorted(semantic_labels.keys()):
    lut.add_label(semantic_labels[val]["class"], val, semantic_labels[val]["color"])
v.set_lut("labels", lut)
v.set_lut("pred", lut)


predictions = json.load( open("" ) )  # Load pedictions 

# In the most predictions, we not saved the point to save disk memory, then the points have to included like this:
points = np.load('/test/test_grid.npy')[:,:3]

data = {"points": np.array(points.astype(np.float32)), "labels": predictions["label"], "feat": None, "pred": predictions["pred"],
         "name": "essen" }
v.visualize([data])