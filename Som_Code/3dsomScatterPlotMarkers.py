import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math
import Minisom3D

from mpl_toolkits.mplot3d import Axes3D

# import warnings library
import warnings

# ignore all warnings
from som_implementation_3D import MySom3D
from utils import Utils

warnings.filterwarnings('ignore')

### Read and Analyse Data
samples = pd.read_csv('./breast_cancer_data.csv')  # returns data frame

# feature names as a list
col = samples.columns  # .columns gives columns names in data

# Remember:
# 1) There is an id that cannot be used for classification
# 2) Diagnosis is our class label

list = ['id', 'diagnosis']
x = samples.drop(list, axis=1)

### transform to numpy matrix
data = x.to_numpy()

### compute size of map

size = int(round((round(5 * math.sqrt(data.shape[0]))) ** (1. / 3.)))

###compute number of features
number_features = data.shape[1]

print(size)

#som = Minisom3D.MiniSom3D(size, size, size, number_features, sigma=3.0, learning_rate=0.5)
som = MySom3D(size, size, size, number_features, sigma=3.0, learning_rate=0.5)
som.train(data, 100)

distanceMap = som.distance_map().T

labels = samples['diagnosis']
c = labels.astype('category')
labels = c.cat.codes

markers = ['o', 's']
colors = ['r', 'g']

Nx, Ny, Nz = size, size, size
X, Y, Z = np.arange(Nx), np.arange(Ny), -np.arange(Nz)

fig = plt.figure(figsize=(size, size))
ax = Axes3D(fig)

# Add x, y gridlines
ax.grid(b=True, color='red',
        linestyle='-.', linewidth=0.3,
        alpha=0.2)

for cnt, xx in enumerate(data):
    w = som.find_BMU(xx)
    print(w)
    #plt.plot(w[0] + .5, w[1] + .5, markers[labels[cnt]], markersize=12, markerfacecolor=colors[labels[cnt]],markeredgecolor='k')
    ax.scatter3D(w[0], w[1], w[2], color=colors[labels[cnt]], marker=markers[labels[cnt]])

# Show Figure
plt.show()

threshold = som.find_threshold(data)
no_clusters, bmu_array, samples_with_clusters_array = som.find_clusters_with_min_dist(data, 5)
samples_with_symbols_array = Utils.assign_symbols(samples_with_clusters_array)
print(samples_with_symbols_array)


fig = plt.figure(figsize=(size, size))
ax = Axes3D(fig)
ax.grid(b=True, color='red',
        linestyle='-.', linewidth=0.3,
        alpha=0.2)

"""
all_markers = ['o', '*', '.', ',', 'x', 'X', '+', 'P', 's', 'D', 'd', 'p', 'H', 'h', 'v', '^', '<', '>', '1', '2', '3', '4', '|', '_']
all_colors = ['b', 'c', 'g', 'k', 'm', 'r', 'w', 'y']

markers_and_colors = []
for i in range(no_clusters):
    markers_and_colors.append((i, all_markers[i % len(all_markers)], all_colors[i % len(all_colors)]))
"""
markers_and_colors = Utils.assign_markers_and_colors(no_clusters)

for cnt, xx in enumerate(data):
    w = som.find_BMU(xx)
    print('W ', w)
    cluster = 0
    for bmu in bmu_array:
        if w == bmu[0]:
            cluster = bmu[1]
            break
    marker = '_'
    color = 'y'
    for x in markers_and_colors:
        if x[0] == cluster:
            marker = x[1]
            color = x[2]
    ax.scatter3D(w[0], w[1], w[2], color=color, marker=marker)

# Show Figure
plt.show()
"""
bmu_array = []
no_clusters = 0
sample_array = []
for cnt, sample in enumerate(data):
    w = som.find_BMU(sample)
    sample_tuple = (sample, no_clusters)
    for bmu in bmu_array:
        difference = np.subtract(w, bmu[0])
        squared = np.square(difference)
        dist = np.sqrt(np.sum(squared, axis=-1))
        print("Dist ", dist)
        if dist < 7:
            sample_tuple = (sample, bmu[1])
            break
    if sample_tuple[1] == no_clusters:
        no_clusters += 1
    sample_array.append(sample_tuple)
    bmu_array.append((w, sample_tuple[1]))
print('No of clusters', no_clusters)
"""

colors_sequence = Utils.get_colors_array(data, som)
print(colors_sequence)
plt.imshow([colors_sequence],aspect='auto')
plt.axis('off')
plt.show()




