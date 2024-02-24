from collections import Counter
import os
import numpy as np
ar = []
train_label_counts = np.zeros(8)
classes=[]
for f in os.listdir('dataset/train'):
    point_data = np.load('dataset/train/'+f)
    class_data = point_data[:,3]
    classes= np.append(classes, class_data)

unique_values, train_label_counts = np.unique(classes, return_counts=True)
print(unique_values, train_label_counts)
val_label_counts = np.zeros(8)
classes=[]
for f in os.listdir('dataset/validation'):
    point_data = np.load('dataset/validation/'+f)
    class_data = point_data[:,3]
    classes= np.append(classes, class_data)

    # Use Counter to count occurrences of each label
unique_values, val_label_counts = np.unique(classes, return_counts=True)
print(unique_values, val_label_counts)

test_label_counts = np.zeros(8)
classes=[]
for f in os.listdir('datset/test'):
    point_data = np.load('dataset/test/'+f)
    class_data = point_data[:,3]
    classes= np.append(classes, class_data)

    # Use Counter to count occurrences of each label
unique_values, test_label_counts = np.unique(classes, return_counts=True)
sum_label_counts = val_label_counts + train_label_counts + test_label_counts
print(sum_label_counts.astype(np.int32).tolist())
print(unique_values, test_label_counts)
print(np.sum(train_label_counts)+ np.sum(val_label_counts) )