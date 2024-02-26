import numpy as np
from sklearn.cluster import DBSCAN
import json
from scipy.spatial.distance import cdist


array= np.load("/validation/grid_0.npy")
data = json.load( open("/grid_0_pred.json" ) )  
print(array[:,:3])
point_cloud = np.c_[ array[:,:3], np.array(data["pred"])]

# Filter points with class 6
class_6_points = point_cloud[point_cloud[:, 3] == 6]

# Filter points with class 4
class_4_points = point_cloud[point_cloud[:, 3] == 4]




# Set the parameters for DBSCAN
eps = 2  # Maximum distance between two points to be considered neighbors
min_samples = 5  # Minimum number of points required to form a cluster

# Create an instance of DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)

# Fit the DBSCAN model to the point cloud data
labels = dbscan.fit_predict(class_6_points)

# Retrieve the unique cluster labels
unique_labels = np.unique(labels)

# Print the number of clusters found
num_clusters = len(unique_labels)
z_differences= dict()
print("Number of clusters found:", num_clusters)
for label in unique_labels:
    cluster_points = class_6_points[labels == label]
    cluster_with_ground= cluster_points
    # Iterate through each point with class 6
    for cluster_point in cluster_points:
        # Extract x, y, z coordinates of the current class 6 point
        x, y, z = cluster_point[:3]

        # Calculate distances between the current class 6 point and all points with class 4
        distances = cdist(np.array([[x, y, z]]), class_4_points[:, :3])

        # Find the index of the nearest point with class 4
        nearest_class_4_index = np.argmin(distances)

        # Get the coordinates and class of the nearest point with class 4
        nearest_class_4_point = class_4_points[nearest_class_4_index]

        # Add the nearest point with class 4 to the original point cloud
        cluster_with_ground = np.vstack([cluster_with_ground, nearest_class_4_point])
    max_z = np.max(cluster_with_ground[:, 2])  # Index 2 represents the z coordinate
    min_z = np.min(cluster_with_ground[:, 2])  # Index 2 represents the z coordinate
    z_difference = max_z - min_z
    z_differences[str(label)]= z_difference

points_with_cluster= np.c_[ class_6_points, labels]

np.save("clusters_grid_0_aerial", points_with_cluster)
with open("clusters_grid_0_aerial.json", 'w') as json_file:
    json.dump(z_differences, json_file, indent=2)

