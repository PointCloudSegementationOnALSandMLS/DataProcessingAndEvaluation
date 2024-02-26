import matplotlib.pyplot as plt
import numpy as np

# Given arrays representing classes for each set
train_set = np.array([ 23160440,  1211760,   246990, 25208374,  1626989,   561396,   110128,
   182935])
validation_set = np.array([ 387431, 59345, 259, 110366, 2313, 518, 289, 1923])
test_set = np.array([ 306918, 26364, 158, 37648, 1942, 554, 634, 274])

# Class names
class_names = ['Natural', 'Building', 'Pole', 'Street', 'Car', 'Wall', 'Fence', 'Hedge']



# Plotting the stacked bar chart
fig, ax = plt.subplots()
ax.bar(class_names, train_set, label='Train')
ax.bar(class_names, validation_set, bottom=train_set, label='Validation')
ax.bar(class_names, test_set, bottom=[i+j for i, j in zip(train_set, validation_set)], label='Test')

#ax.set_yscale('log')
# Adding labels to the bars
for i, v1, v2, v3 in zip(range(len(class_names)), train_set, validation_set, test_set):
    ax.text(i, v1/2, f'{v1/1000:.1f}k', ha='center', va='center')
    ax.text(i, v1 + v2/2, f'{v2/1000:.1f}k', ha='center', va='center')
    ax.text(i, v1 + v2 + v3/2, f'{v3/1000:.1f}k', ha='center', va='center')

# Adding legend and labels
ax.legend()
ax.set_xlabel('Categories')
ax.set_ylabel('Number of Points (log scale)')
ax.set_title('ALS point distribution across object categories')

# Display the chart
plt.show()