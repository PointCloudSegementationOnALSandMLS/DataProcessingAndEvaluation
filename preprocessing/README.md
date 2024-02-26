## Preprocessing

The first step in processing the data was to calculate the gird int which the point cloud is diveded to. This was done using the calculateGrids.py script. The result was an geojson like the all.geojson. This was then used to split the original comple las file into these grids using the split_las.py file. 

For labeling the grids, we had to split each mobile grid into 4 smaller grids. After the labeling these were combined again using the combienSplitsMobile.py script. Further, the labeled point cloud exported from the LiDAR labeler did not included the intensity. Therefore we added this again to the point clouds with the scrips combineArraysMobile.py and combineArraysAerial.py 

For training the networs it was necessary to know the amount of points in each class. With the script countclasses.py we calculated these values.