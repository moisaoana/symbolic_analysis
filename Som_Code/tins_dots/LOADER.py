import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from EEG_DataProcessor import EEG_DataProcessor
from RawEEGSignalParser import RawEEGSignalParser
from som_implementation_3D import MySom3D
from mayavi import mlab

DATASET_PATH = "./data/Dots_30_001/"
parser = RawEEGSignalParser(DATASET_PATH)
#print("******************************************")

#channel_data = parser.load_channel_data(0)
#print(channel_data.shape)

full_data = parser.load_all_channels()
#print("full data shape: ",full_data.shape)
#print("full data: ", full_data)
#print("*****************************************")

#
event_timestamps = parser.load_event_timestamps()
event_codes = parser.load_event_codes()
#print(event_timestamps)
#print(len(event_timestamps))
#print(event_codes)
#print(len(event_codes))
# #
#
eegDataProcessor = EEG_DataProcessor(DATASET_PATH, full_data, event_timestamps, event_codes)
eegDataProcessor.create_trials(save=False)
eegDataProcessor.link_trials(save=False)

som =MySom3D(4, 4, 4, 128, sigma=0.3, learning_rate=0.5)
som.train(eegDataProcessor.processed_data, 10)

distance_map = som.distance_map().T

#Nx, Ny, Nz = 4, 4, 4
#X, Y, Z = np.arange(Nx), np.arange(Ny), -np.arange(Nz)

#fig = plt.figure(figsize=(4, 4))
#ax = Axes3D(fig)

# Add x, y gridlines
#ax.grid(b=True, color='red',
#        linestyle='-.', linewidth=0.3,
#        alpha=0.2)

#kw = {
#    'cmap': 'Blues'
#}

#for i in range(0, 4):
#    for j in range(0, 4):
#        for k in range(0, 4):
#            ax.scatter3D(X[i], Y[j], Z[k], c=str(distanceMap[i][j][k]), s=100, cmap=plt.get_cmap('jet'))

# Show Figure


#plt.show()

volume_slice_x = mlab.volume_slice(distance_map,plane_orientation='x_axes')
volume_slice_y = mlab.volume_slice(distance_map, plane_orientation='y_axes')
volume_slice_z = mlab.volume_slice(distance_map, plane_orientation='z_axes')



outline = mlab.outline(volume_slice_x)

colorbar = mlab.colorbar(object=volume_slice_x, title='Data values')

# Show the plot
mlab.show()




