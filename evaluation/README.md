## Evaluation
This folder contains the scripts to evaluat the performances by comparing the predictions with the ground trut labels. 
The calculation of the metrics is in all files the same.
In the evaluation_randlanet.py script, the script only evaluates the performance of one preditcion grid. The KPConv grid loops this script and evaluates the performance of all test and validation scripts. Further it calculates the mean between all validation grids. 
However for these scripts it is important to adjust the number of classes that are inside the dataset.

For the DALES and Toronto3D dataset, before calculating the performances, it is necessary to map the classes to our classes. The predicted classes may have different labels than our dataset and to enable the compariosn this is necessary