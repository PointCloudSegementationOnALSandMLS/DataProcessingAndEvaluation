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
        points = np.load("./Daten_Essen/Allee/grids/mobile/LSplits/normalized/train/grid_normalized_{}.npy".format(i), allow_pickle=True)
    except: 
        points = np.load("./Daten_Essen/Allee/grids/mobile/LSplits/normalized/validation/grid_normalized_{}.npy".format(i), allow_pickle=True)

    rawpoints = points[:,:3]
    newClasses = np.copy(points[:,3])
    #print(newClasses)
    mask = (newClasses == 7) 
    newClasses[mask] = 6
    #print(newClasses)
    complete = np.c_[ rawpoints, newClasses, points[:,4]]
    np.save("Daten_Essen/Allee/grids/mobile/LSplits/vertical/grid_{}".format(i), complete)

points = np.load("./Daten_Essen/Allee/grids/mobile/LSplits/normalized/test/test_grid.npy")
rawpoints = points[:,:3]
newClasses = np.copy(points[:,3])
print(newClasses)
mask = (newClasses == 7) 
newClasses[mask] = 6
print(newClasses)
complete = np.c_[ rawpoints, newClasses, points[:,4]]
np.save("Daten_Essen/Allee/grids/mobile/LSplits/vertical/test_grid.npy", complete)