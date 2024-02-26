from sklearn.metrics import confusion_matrix
import numpy as np
import json
import seaborn as sns
import os
import matplotlib.pyplot as plt

matrix= np.load("Essen/aerial/hedge_ground/matrix_val.npy")
plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt=".1f", cmap="Blues", cbar=False, xticklabels=["Natural", "Building", "Pole", "Ground", "Car", "Vertical", ], yticklabels=["Natural", "Building", "Pole", "Ground", "Car", "Vertical"])
plt.xlabel("Predicted as")
plt.ylabel("Label")
plt.title("Confusion Matrix Aerial Vertical Ground - Percentages")
plt.show()