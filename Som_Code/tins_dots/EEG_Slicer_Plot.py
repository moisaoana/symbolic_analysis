import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from EEG_DataProcessor import EEG_DataProcessor
from RawEEGSignalParser import RawEEGSignalParser
from som_implementation_3D import MySom3D
from mayavi import mlab

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
no_iterations = 10

som =MySom3D(size, size, size, no_features, sigma=0.3, learning_rate=0.5)
som.train(eegDataProcessor.processed_data, no_iterations)

distance_map = som.distance_map().T

volume_slice_x = mlab.volume_slice(distance_map,plane_orientation='x_axes')
volume_slice_y = mlab.volume_slice(distance_map, plane_orientation='y_axes')
volume_slice_z = mlab.volume_slice(distance_map, plane_orientation='z_axes')

outline = mlab.outline(volume_slice_x)

colorbar = mlab.colorbar(object=volume_slice_x, title='Data values')

mlab.show()




