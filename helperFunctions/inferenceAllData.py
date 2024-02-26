import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
from Open3DML.ml3d.datasets import essen
import open3d as o3d
import numpy as np
import json
import os
    
def loadData(dataPath):
    try:
        data = np.load(dataPath)
    except:
        return False
    points = np.array(data[:, :3], dtype=np.float32)

    labels = np.array(data[:, 3], dtype=np.int32)
    feat = data[:, 4:] if data.shape[1] > 4 else None

    data = {'point': points, 'feat': feat, 'label': labels}

    return data

def run_inference(dataPath, config_file, ckpt_path, output_path):
    # Load configuration
    print(output_path)
    config = _ml3d.utils.Config.load_from_file(config_file)

    # Load dataset, model, and pipeline
    dataset = essen.Essen(**config.dataset)
    framework = "torch"
    Model = _ml3d.utils.get_module("model", config.model.name, framework)
    model = Model(**config.model)
    Pipeline = _ml3d.utils.get_module("pipeline", config.pipeline.name, framework)
    pipeline = Pipeline(model, dataset, **config.pipeline)

    # Load checkpoint
    pipeline.load_ckpt(ckpt_path)
    data = loadData(dataPath)

    # Run inference
    if(not data):
        return None
    results = pipeline.run_inference(data)
    pred_label = (results['predict_labels'] + 1).astype(np.int32)
    pred_label[0] = 0  # Fill "unlabeled" value because predictions have no 0 values.
    
    data = {'pred': pred_label.tolist(), 'label': data["label"].tolist()}

    with open(output_path +"_pred.json", 'w') as json_file:
        json.dump(data, json_file)
    
    return calcAcccuricies(data, config.dataset.class_weights, output_path + "_result.json")

        
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
    weight = np.array(class_weights) / float(sum(class_weights))
    ce_label_weights = 1 / (weight + 0.02)
    print(ce_label_weights)
    ce_label_weights[0] = 0
    normalized_weights = np.array(ce_label_weights) / np.sum(ce_label_weights)


    # Calculate accuracy
    coorectPredictions = np.sum(pred_array_filtered == gt_array_filtered)
    points = pred_array_filtered.size
    accuracy = coorectPredictions / points

    # Calculate overall weighted accuracy
    accuracy_per_class = [0]
    weighted_accuracy_per_class = [0]
    for i in range(1,len(normalized_weights)):
        gt_mask = (gt_array_filtered == i)
        pred_mask = (pred_array_filtered == i)
        correct_predictions = np.sum(np.logical_and(gt_mask, pred_mask))
        total_predictions = np.sum(gt_mask)
        accuracy_class = correct_predictions / total_predictions if total_predictions > 0 else 0
        accuracy_per_class.append(accuracy_class)

    weighted_accuracy_per_class = (accuracy_per_class * normalized_weights )
    print("accuracy:", accuracy_per_class, normalized_weights, weighted_accuracy_per_class)
    weighted_accuracy = np.sum(weighted_accuracy_per_class)
    
    intersection = np.sum(np.logical_and(pred_array_filtered, gt_array_filtered))
    union = np.sum(np.logical_or(pred_array_filtered, gt_array_filtered))
    iou = intersection / union

    # Calculate IoU for each class (excluding label 0)
    iou_per_class = [0]
    weighted_iou_per_class= [0]
    for j in range(1, len(normalized_weights)):
        print(j)
        gt_mask = (gt_array_filtered == j)
        pred_mask = (pred_array_filtered == j)
        intersection = np.sum(np.logical_and(gt_mask, pred_mask))
        union = np.sum(np.logical_or(gt_mask, pred_mask))
        class_iou = intersection / union if union > 0 else 0
        iou_per_class.append(class_iou)
       
    weighted_iou = np.sum(np.array(iou_per_class) * normalized_weights)
    
    result_dict = {
        "points_per_class": points_per_class,
        "weights": normalized_weights.tolist(),
        "accuracy" : accuracy,
        "accuracy_per_class" : accuracy_per_class,
        "weighted_accuracy" : weighted_accuracy,
        "iou" : iou,
        "iou_per_class": iou_per_class,
        "weighted_iou": weighted_iou
    }

    # Save the dictionary to a JSON file
    with open(output_json, 'w') as json_file:
        json.dump(result_dict, json_file, indent=4)
        
    return result_dict
    



    
    
    
for mode in ["als", "mls"]:
    path= os.path.join("", mode)
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for dirs in directories:
        dirPath= os.path.join(path, dirs)
        ckpt_path = os.path.join(dirPath, "checkpoints", "ckpt_00250.pth")
        config_file_path = os.path.join(dirPath, 'config', 'config.yml')
        test_path = os.path.join(dirPath, "test")
        val_path = os.path.join(dirPath, "validation")
        val_results = []
        val_result_complete = dict()
        for files in os.listdir(test_path):
            result = run_inference(os.path.join(test_path, files), config_file_path, ckpt_path, os.path.join(dirPath, files))
        for files in os.listdir(val_path):
            result =run_inference(os.path.join(val_path, files), config_file_path, ckpt_path, os.path.join(dirPath, files))
            if(result):
                val_results.append(result)
        for key in ["accuracy", "weighted_accuracy", "iou", "weighted_iou"]:
            val_result_complete[key] = sum(result_dict[key] for result_dict in val_results) / len(val_results)
        for key in ["accuracy_per_class", "iou_per_class"]:
            val_result_complete[key] = []
            for i in range(len(val_results[0][key])):
                val_result_complete[key].append(sum(result_dict[key][i] for result_dict in val_results) / len(val_results))
        with open(os.path.join(dirPath, files) + "val_results", 'w') as json_file:
            json.dump(val_result_complete, json_file, indent=4)
        
        