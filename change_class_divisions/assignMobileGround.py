import numpy as np
import open3d as o3d
import open3d.ml as _ml3d
import os
import json
import open3d.ml.torch as ml3d  # just switch to open3d.ml.tf for tf usage
import scipy.io
import numpy as np
import laspy

for i in range(0,11):
#data = json.load( open( "predicted_essen_mobile_200.json" ) )  
#points = np.load('Daten_Essen/Allee/grids/mobile/Lsplits/normalized/train/grid_normalized_5.npy')
    try:
        points = np.load("Daten_Essen/Allee/grids/mobile/Lsplits/normalized/train/grid_normalized_{}.npy".format(i), allow_pickle=True)
    except:
        points = np.load("Daten_Essen/Allee/grids/mobile/Lsplits/normalized/validation/grid_normalized_{}.npy".format(i), allow_pickle=True)
    try:
        groundPoints= scipy.io.loadmat('./Daten_Essen/Allee/grids/mobile/Lsplits/ground/grid_{}/VoxelLabelData/grid_{}_Label_1.mat'.format(i,i))["L"]
    except:
        print("error")
        groundPoints= points

    rawpoints = points[:,:3]
    print(points[:,:3], groundPoints[:,:3])
    groundClasses= groundPoints[:, 3]
    originClasses =  points[:,3].astype(np.int32)
    unique_values, counts = np.unique(groundClasses, return_counts=True)
    print(unique_values, counts)
    print(originClasses)
    newClasses = np.copy(originClasses)
    print(groundClasses)
    mask = (groundClasses == 4) 
    print(mask)
    newClasses[mask] = 4
    print(newClasses)
    complete = np.c_[ rawpoints, newClasses, points[:,4]]
    np.save("Daten_Essen/Allee/grids/mobile/Lsplits/ground/grid_{}".format(i), complete)

