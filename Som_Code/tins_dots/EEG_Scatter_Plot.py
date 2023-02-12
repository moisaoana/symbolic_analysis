import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from EEG_DataProcessor import EEG_DataProcessor
from RawEEGSignalParser import RawEEGSignalParser
from som_implementation_3D import MySom3D

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

for i in range(0, 4):
    for j in range(0, 4):
        for k in range(0, 4):
            ax.scatter3D(X[i], Y[j], Z[k], c=str(distance_map[i][j][k]), s=100, cmap=plt.get_cmap('jet'))

# Show Figure


plt.show()





