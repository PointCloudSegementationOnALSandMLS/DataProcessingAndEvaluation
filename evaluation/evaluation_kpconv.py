from email.mime import base
from plistlib import load
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
from Open3DML.ml3d.datasets import essen
import open3d as o3d
import numpy as np
import json
import os
from sklearn.metrics import confusion_matrix
from scipy.spatial import cKDTree

def loadData(dir_path, dir, grid):
    
    data = np.load(os.path.join(dir_path, dir, grid))

    points = np.array(data[:, :3], dtype=np.float32) 
    kdtree = cKDTree(points)

    num_points = len(points)
    isolated_points = []

    for i in range(num_points):
        # Query the KD-tree to find the index of the nearest neighbor
        _, neighbor_index = kdtree.query(points[i], k=2)
        
        # Skip the first neighbor (itself), take the second neighbor
        point1 = points[i]
        point2 = points[neighbor_index[1]]
        
        distance = np.linalg.norm(point2 - point1)
        if(distance>2.5):
            print(point1)
            isolated_points.append(i)

            # Set a threshold distance to define isolation (you can adjust this)

    data = np.delete(data, isolated_points, axis=0)
    base_name = os.path.basename(grid)
    pred_path= os.path.join(pred_path, base_name)

# Append ".txt" to the base name
    new_file_name = os.path.splitext(pred_path)[0] + ".txt"
    with open(new_file_name, 'r') as file:
# Read the lines of the file and store them in an array
            pred = file.readlines()
            pred = np.array(pred).astype(np.int32)
            
    

    labels = np.array(data[:, 3], dtype=np.int32)
    print(len(points))
    print(len(labels))
    #feat = data[:, 4:] if data.shape[1] > 4 else None

    data = {'point': points, 'pred': pred, 'label': labels}
    #print(data)
    return data


def calcAcccuricies(dir_path, dir, grid, output_json, predPath):


    data = loadData(dir_path, dir, grid, predPath)
    print(data.keys())
    gt_array = np.array(data['label'])
    pred_array = np.array(data['pred'])
    #print(gt_array)
    #print(pred_array)
    # Mask for ignoring label 0
    ignore_mask = (gt_array != 0)
    #print(ignore_mask)
    # Remove label 0 from both arrays
    gt_array_filtered = gt_array[ignore_mask]
    pred_array_filtered = pred_array[ignore_mask]
    unique, counts = np.unique(gt_array_filtered, return_counts=True)
    # Create a dictionary
    points_per_class = dict(zip(map(str, unique), map(int, counts)))

    # Calculate accuracy
    coorectPredictions = np.sum(pred_array_filtered == gt_array_filtered)
    points = pred_array_filtered.size
    accuracy = coorectPredictions / points

    # Calculate overall weighted accuracy
    accuracy_per_class = [0]
    for i in range(1,8):
        gt_mask = (gt_array_filtered == i)
        pred_mask = (pred_array_filtered == i)
        correct_predictions = np.sum(np.logical_and(gt_mask, pred_mask))
        total_predictions = np.sum(gt_mask)
        accuracy_class = correct_predictions / total_predictions if total_predictions > 0 else 0
        accuracy_per_class.append(accuracy_class)


    
    intersection = np.sum(np.logical_and(pred_array_filtered, gt_array_filtered))
    union = np.sum(np.logical_or(pred_array_filtered, gt_array_filtered))
    iou = intersection / union

    # Calculate IoU for each class (excluding label 0)
    iou_per_class = []
    for j in range(1, 8):
        print(j)
        gt_mask = (gt_array_filtered == j)
        pred_mask = (pred_array_filtered == j)
        intersection = np.sum(np.logical_and(gt_mask, pred_mask))
        union = np.sum(np.logical_or(gt_mask, pred_mask))
        class_iou = intersection / union if union > 0 else 0
        iou_per_class.append(class_iou)
       
    
    mean_iou = np.mean(iou_per_class)
    
    conf_matrix = confusion_matrix(data["label"], data["pred"])
    print(conf_matrix)
    # Convert to percentages
    conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
    conf_matrix_percent_excluding_class0 = conf_matrix_percent[1:, 1:]
    
    result_dict = {
        "points_per_class": points_per_class,
        #"weights": normalized_weights.tolist(),
        "accuracy" : accuracy,
        "accuracy_per_class" : accuracy_per_class,
        #"weighted_accuracy" : weighted_accuracy,
        "iou" : iou,
        "iou_per_class": iou_per_class,
        #"weighted_iou": weighted_iou,
        "mean_iou": mean_iou,
        "matrix" : conf_matrix_percent_excluding_class0.tolist()
    }

    # Save the dictionary to a JSON file
    with open(output_json + "val.json", 'w') as json_file:
        json.dump(result_dict, json_file, indent=4)
        
    return result_dict





    
dirPath = "als/street-seperate"
predPath = "als/KPConv/street-seperate"
test_path = os.path.join(dirPath, "test")
val_path = os.path.join(dirPath, "validation")
val_results = []
val_matrix = []
val_result_complete = dict()
for files in os.listdir(test_path):
    result = calcAcccuricies(dirPath, "test", files, os.path.join(dirPath, files), predPath)
for files in os.listdir(val_path):
    result = calcAcccuricies(dirPath, "validation", files, os.path.join(dirPath, files), predPath)
    if(result):
        val_results.append(result)
for key in ["accuracy", "iou",]:
    val_result_complete[key] = sum(result_dict[key] for result_dict in val_results) / len(val_results)
for key in ["accuracy_per_class", "iou_per_class"]:
    val_result_complete[key] = []
    for i in range(len(val_results[0][key])):
        val_result_complete[key].append(sum(result_dict[key][i] for result_dict in val_results) / len(val_results))
for result_dict in val_results:
        val_matrix.append(np.array(result_dict["matrix"])[:6, :6])

mean_matrix = np.mean(val_matrix, axis=0)
val_result_complete["matrix"] = mean_matrix.tolist()
print(mean_matrix)
with open(os.path.join(dirPath, files) + "val_results.json", 'w') as json_file:
    json.dump(val_result_complete, json_file, indent=4)

    
