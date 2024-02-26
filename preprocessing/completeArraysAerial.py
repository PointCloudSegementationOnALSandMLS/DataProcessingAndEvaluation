import laspy
import numpy as np
import scipy.io
import json
offsets = []
for i in [0,6,10]:
    test= scipy.io.loadmat('Daten_Essen/Allee/grids/aerial/numpy/grid_{}_Label_1.mat'.format(i))
    print(test["L"])
    print(np.unique(test["L"][:,3]))
    las = laspy.read('Daten_Essen/Allee/grids/aerial/grids/grid_{}/grid_{}.las'.format(i,i))
    las
    print(list(las.point_format.dimension_names))
    print(las.header.scale)
    print(las.header.offset)
    #print(list(las.X))
    point_data = np.stack([las.X, las.Y, las.Z, las.intensity], axis=0).transpose((1, 0))
    print(point_data)
    points = test["L"]
    offset= np.min(points, axis= 0)
    offsets.append(offset.tolist())
    print(offsets)
    normPoints= points - offset
    complete = np.c_[ normPoints, point_data[:,3 ]]
    print(complete)
    
    np.save("Daten_Essen/Allee/grids/aerial/normalized/grid_{}".format(i), complete)
#json.dump( offsets, open( "aerial_offset.json", 'w' ) )