import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
from Open3DML.ml3d.datasets import essen
import open3d as o3d
import numpy as np
import json
import os
from sklearn.metrics import confusion_matrix

def map_classes(original_classes):
    class_mapping = {
        0: 0,
        1: 4,  # Road
        2: 4,  # Road mrk.
        3: 1,  # Natural
        4: 2,  # Bldg
        5: 3,  # Util. line
        6: 3,  # Pole
        7: 5,  # Car
        8: 6   # Fence
    }

    mapped_classes = [class_mapping[class_num] for class_num in original_classes]
    return mapped_classes


def calcAcccuricies(data, class_weights,  output_json):

    #data = json.load( open( "predicted_essen_mobile_200.json" ) )  
    # Sample ground truth and predicted arrays
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

    mean_accuracy = np.mean(accuracy_per_class)

    
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
        "accuracy" : accuracy,
        "accuracy_per_class" : accuracy_per_class,
        "iou" : iou,
        "iou_per_class": iou_per_class,
        "mean_iou": mean_iou,
        "mean_accuracy": mean_accuracy,
        "matrix" : conf_matrix_percent_excluding_class0.tolist()
        
    }
    
        # Save the dictionary to a JSON file
    with open(output_json, 'w') as json_file:
        json.dump(result_dict, json_file, indent=4)
        
    return result_dict


# Example usage
predicted_json= json.load(open("predictedMobileTorontoKPConv.json"))
original_array = predicted_json["pred"]
mapped_array = map_classes(original_array)

calcAcccuricies(predicted_json, "essen_toronto_result_kpconv.json")
