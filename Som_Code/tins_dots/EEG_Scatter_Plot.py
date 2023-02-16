import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from EEG_DataProcessor import EEG_DataProcessor
from RawEEGSignalParser import RawEEGSignalParser
from som_implementation_3D import MySom3D
from utils import Utils

DATASET_PATH = "./data/Dots_30_001/"
parser = RawEEGSignalParser(DATASET_PATH)

full_data = parser.load_all_channels()

event_timestamps = parser.load_event_timestamps()
event_codes = parser.load_event_codes()

eegDataProcessor = EEG_DataProcessor(DATASET_PATH, full_data, event_timestamps, event_codes)
eegDataProcessor.create_trials(save=False)
eegDataProcessor.link_trials(save=False)

size = 4
no_features = 128
no_iterations = 1

print(eegDataProcessor.processed_data.shape)

som =MySom3D(size, size, size, no_features, sigma=0.3, learning_rate=0.5)
som.train(eegDataProcessor.processed_data, no_iterations)

distance_map = som.distance_map().T

Nx, Ny, Nz = size, size, size
X, Y, Z = np.arange(Nx), np.arange(Ny), -np.arange(Nz)

fig = plt.figure(figsize=(size, size))
ax = Axes3D(fig)

# Add x, y gridlines
ax.grid(b=True, color='red',
        linestyle='-.', linewidth=0.3,
        alpha=0.2)

kw = {
    'cmap': 'Blues'
}

for i in range(0, size):
    for j in range(0, size):
        for k in range(0, size):
            ax.scatter3D(X[i], Y[j], Z[k], c=str(distance_map[i][j][k]), s=100, cmap=plt.get_cmap('jet'))

# Show Figure


plt.show()

print("HERE")
threshold = som.find_threshold(eegDataProcessor.processed_data)
print('Max dist ', threshold)
no_clusters, bmu_array, samples_with_clusters_array = som.find_clusters_with_min_dist(eegDataProcessor.processed_data, 0.3, threshold)
print('No clusters ', no_clusters)
samples_with_symbols_array = Utils.assign_symbols(samples_with_clusters_array)



fig = plt.figure(figsize=(size, size))
ax = Axes3D(fig)
ax.grid(b=True, color='red',
        linestyle='-.', linewidth=0.3,
        alpha=0.2)

markers_and_colors = Utils.assign_markers_and_colors(no_clusters)

for cnt, xx in enumerate(eegDataProcessor.processed_data):
    w = som.find_BMU(xx)
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





