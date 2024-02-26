import numpy as np
import open3d as o3d
import open3d.ml as _ml3d
import os
import json
import open3d.ml.torch as ml3d  # just switch to open3d.ml.tf for tf usage
import scipy.io
import numpy as np
import laspy

#print(pointsNumpy)
semantic_labels = {
        0: 'Unclassified',
        1: 'Natural',
        2: 'Building',
        3: 'Pole',
        4: 'Ground',
        5: 'Car',
        6: 'Wall',
        7: 'Fence',
    }
v = ml3d.vis.Visualizer()
lut = ml3d.vis.LabelLUT()

for val in sorted(semantic_labels.keys()):
    lut.add_label(semantic_labels[val], val)
v.set_lut("labels", lut)
v.set_lut("pred", lut)
#data = json.load( open( "predicted_essen_mobile_200.json" ) )  
#points = np.load('Daten_Essen/Allee/grids/mobile/Lsplits/normalized/train/grid_normalized_5.npy')
points = np.load("Daten_Essen/Allee/grids/areial/normalized/train/grid_normalized_1.npy", allow_pickle=True)
las = laspy.read('Daten_Essen/Allee/grids/areial/grids/grid_1/grid_1.las')
laspoints= np.stack([las.x, las.y, las.z, las.classification], axis=0).transpose((1, 0))
originClasses =  points[:,3].astype(np.int32)
print(originClasses)
newClasses = np.copy(originClasses)
lasClasses= laspoints[:,3].astype(np.int32)
print(lasClasses)
mask = (lasClasses == 2) | (lasClasses == 26) | (lasClasses == 21)
print(mask)
newClasses[mask] = 4
print(newClasses)
data = {"points": points[:,:3].astype(np.float32), "labels": originClasses, "feat": points[:,4].astype(np.float32), "pred": newClasses, "name": "test" }
v.visualize([data])
