import open3d as o3d
import open3d.ml as _ml3d
import os
import json
import open3d.ml.torch as ml3d  # just switch to open3d.ml.tf for tf usage
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

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
#data = json.load( open("predections/grid_2_mobile_pred.json" ) )  
points = np.load('Daten_Essen/Allee/grids/mobile/LSplits/hedgeOriginal/train/grid_1.npy')

data = {"points": points[:,:3].astype(np.float32), "labels": points[:,3].astype(np.int32), "feat": points[:,4].astype(np.float32),
         "name": "test" }
v.visualize([data])