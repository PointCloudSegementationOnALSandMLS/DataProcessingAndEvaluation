## Visualization

In this folder we provide different visualization scipts. The vis_class_distributions.py script, plots a bar chart showing the amount of points in the different classes. The vis_confusion_matrices.py shows the confusion_matrices. 

All other scripts visualize the point clouds. The vis_cluster script was used to visualize the calculated clusters stored in the clusters folder. The other scripst to either show the raw dataset vis_pointcloud_npy.py or with the predictions from RandLA-Net  vis_pointcloud_json.py or KPConv vis_pointcloud_txt.py. 
For all these scripts it is important to set the classes correct. When using an incorrect number of classes in the v.set_lut part, the model assigns wrong colors to the classes.