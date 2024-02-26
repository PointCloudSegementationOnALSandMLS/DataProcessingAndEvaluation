# DataProcessingAndEvaluation

This repository contains a collection of scripts that wer used for the thesis: Point Cloud Segmentation of Urban Architectural Structures from Aerial and Mobile LiDAR Scans using Neural Networks. For further information, please refer to the organisation [PointCloudSegementationOnALSandMLS](https://github.com/PointCloudSegementationOnALSandMLS). In this repository we provide differnt data preperation, visualiztion and scripts. Using, this data still needs adjusting data paths and configurations for the differnt models. Further, it currently misses some documentation, which will be added soon.

### This repository consits of the following folders and scripts:
- PALMA-II: Since the most computation were conducted on th PALMA-II HPC cluster of the University of Münster. These scipts mainly bash scripts, which allow to start the calculations on the cluster. Further, it consists off the Apptainer/Singularity image used on the Cluster.
- change_class_divisions: As we tested differen class divisions inside our training data, we here provide the scipts, which restructured the class labeling in the point clouds
- clusters: In this folder we provide a method to cluster vertical objects in the point clouds. Further we provide the results in this folder. For further information, please read the Readme inside this folder.
- evaluation: The scripts in this folder are used to evaluate the model performances. Please read the Readme inside the folder for further information.
- helperFunctions: In this folder we just provide one script with different helper and test functions we used. 
- preprocessing: In order to be able to start the training data, we had to perform several pre processing steps. Please read the Readme in this folder for further information.
- visualization: This folder contains different visualization scripts. For more information please stick to te Readme in this folder