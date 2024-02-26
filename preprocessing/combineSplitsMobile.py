import laspy
import numpy as np
import scipy.io
import json
points = scipy.io.loadmat('Daten_Essen/Allee/grids/mobile/Lsplits/grid_0/grid_0_Label_1.mat')
print(points["L"])
np.save("Daten_Essen/Allee/grids/mobile/Lsplits/grid_complete_0", points["L"])
points = scipy.io.loadmat('Daten_Essen/Allee/grids/mobile/Lsplits/grid_10/grid_10_Label_1.mat')
print(points["L"])
np.save("Daten_Essen/Allee/grids/mobile/Lsplits/grid_complete_10", points["L"])
print(point)
for i in range(5,10):
    complete = np.empty((0, 4))
    print(np.shape(complete)) 
    for j in range(0,4):
        points = scipy.io.loadmat('Daten_Essen/Allee/grids/mobile/Lsplits/grid_{}/split{}/split{}_Label_1.mat'.format(i,j,j))
        print("points", points["L"])
        print(np.shape(points))
        print(np.unique(points["L"][:,3]))
        complete = np.concatenate((complete, points["L"]))
    
    np.save("Daten_Essen/Allee/grids/mobile/Lsplits/grid_complete_{}".format(i), complete)
#json.dump( offsets, open( "aerial_offset.json", 'w' ) )