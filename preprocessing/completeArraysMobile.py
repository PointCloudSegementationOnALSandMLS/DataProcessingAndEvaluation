import laspy
import numpy as np
import scipy.io
import json
offsets = []
for i in range(0,11):
    labeled_points= np.load("Daten_Essen/Allee/grids/mobile/Lsplits/grid_complete_{}.npy".format(i), allow_pickle=True)
    las = laspy.read('Daten_Essen/Allee/grids/mobile/Lsplits/grid_{}.las'.format(i))
    print(list(las.point_format.dimension_names))
    #print(list(las.X))
    point_data = np.stack([las.x, las.y, las.z, las.intensity], axis=0).transpose((1, 0))
    print(labeled_points)
    print(point_data)
    offset= np.min(labeled_points, axis= 0)
    offsets.append(offset.tolist())
    
    normPoints= labeled_points - offset
    complete = np.c_[ normPoints, point_data[:,3 ]]
    print(complete)
    
    np.save("Daten_Essen/Allee/grids/mobile/Lsplits/grid_normalized_{}".format(i), complete)
print(offsets)
json.dump( offsets, open( "mobile_offset.json", 'w' ) )