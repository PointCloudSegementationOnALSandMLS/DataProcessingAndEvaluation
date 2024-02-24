import numpy as np
import json
data = json.load( open( "essen_aerial_predicted.json" ) )  
    # Sample ground truth and predicted arrays
#print(data.keys())
gt_array = np.array(data['labels'])
pred_array = np.array(data['pred'])
#print(gt_array)
#print(pred_array)
# Mask for ignoring label 0
ignore_mask = (gt_array != 0)
print(ignore_mask)
# Remove label 0 from both arrays
gt_array_filtered = gt_array[ignore_mask]
pred_array_filtered = pred_array[ignore_mask]
print(gt_array_filtered)
print(pred_array_filtered)
unique, counts = np.unique(gt_array_filtered, return_counts=True)
points_per_class = dict(zip(unique, counts))
print(points_per_class)
weight = counts / float(sum(counts))
ce_label_weights = 1 / (weight + 0.02)
normalized_weights = np.array(ce_label_weights) / np.sum(ce_label_weights)
normalized_weights = np.insert(normalized_weights, 0, 0)
ce_label_weights = np.insert(ce_label_weights, 0, 0)

# Calculate accuracy
coorectPredictions = np.sum(pred_array_filtered == gt_array_filtered)
points = pred_array_filtered.size
accuracy = coorectPredictions / points

# Calculate overall weighted accuracy
accuracy_per_class = [0]
weighted_accuracy_per_class = [0]
print(max(gt_array_filtered))
for i in range(1,max(gt_array_filtered)+1):
    gt_mask = (gt_array_filtered == i)
    pred_mask = (pred_array_filtered == i)
    correct_predictions = np.sum(np.logical_and(gt_mask, pred_mask))
    total_predictions = np.sum(gt_mask)
    accuracy_class = correct_predictions / total_predictions if total_predictions > 0 else 0
    accuracy_per_class.append(accuracy_class)
    weighted_accuracy_per_class.append(accuracy_class * ce_label_weights[i])


weighted_accuracy_per_class2 = (accuracy_per_class * normalized_weights )


weighted_accuracy = np.sum(weighted_accuracy_per_class) / np.sum(ce_label_weights)
weighted_accuracy2 = np.sum(weighted_accuracy_per_class2)
print("accuracy:", accuracy_per_class, ce_label_weights, weighted_accuracy, weighted_accuracy2)

intersection = np.sum(np.logical_and(pred_array_filtered, gt_array_filtered))
union = np.sum(np.logical_or(pred_array_filtered, gt_array_filtered))
iou = intersection / union

# Calculate IoU for each class (excluding label 0)
iou_per_class = [0]
weighted_iou_per_class= [0]
for j in range(1, max(gt_array_filtered)+1):
    print(j)
    gt_mask = (gt_array_filtered == j)
    pred_mask = (pred_array_filtered == j)
    intersection = np.sum(np.logical_and(gt_mask, pred_mask))
    union = np.sum(np.logical_or(gt_mask, pred_mask))
    class_iou = intersection / union if union > 0 else 0
    iou_per_class.append(class_iou)
    
print(iou_per_class, ce_label_weights)
weighted_iou = np.sum(np.array(iou_per_class) * normalized_weights)

result_dict = {
    "points_per_class": points_per_class,
    "weights": normalized_weights,
    "accuracy" : accuracy,
    "accuracy_per_class" : accuracy_per_class,
    "weighted_accuracy" : weighted_accuracy,
    "iou" : iou,
    "iou_per_class": iou_per_class,
    "weighted_iou": weighted_iou
}

# Save the dictionary to a JSON file
with open("accuracies.json", 'w') as json_file:
    json.dump(result_dict, json_file, indent=4)