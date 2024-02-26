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
        points = np.load("Daten_Essen/Allee/grids/aerial/normalized/train/grid_normalized_{}.npy".format(i), allow_pickle=True)
    except:
        points = np.load("Daten_Essen/Allee/grids/aerial/normalized/validation/grid_normalized_{}.npy".format(i), allow_pickle=True)
    las = laspy.read('Daten_Essen/Allee/grids/aerial/grids/grid_{}/grid_{}.las'.format(i,i))
    laspoints= np.stack([las.x, las.y, las.z, las.classification], axis=0).transpose((1, 0))
    rawpoints = points[:,:3]
    originClasses =  points[:,3].astype(np.int32)
    print(originClasses)
    newClasses = np.copy(originClasses)
    lasClasses= laspoints[:,3].astype(np.int32)
    print(lasClasses)
    mask = (lasClasses == 2) | (lasClasses == 26) | (lasClasses == 21)
    print(mask)
    newClasses[mask] = 4
    print(newClasses)
    complete = np.c_[ rawpoints, newClasses, points[:,4]]
    np.save("Daten_Essen/Allee/grids/aerial/ground/grid_{}".format(i), complete)
