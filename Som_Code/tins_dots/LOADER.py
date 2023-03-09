import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from EEG_DataProcessor import EEG_DataProcessor
from RawEEGSignalParser import RawEEGSignalParser

from mayavi import mlab

from Som_Code.som_implementation_3D import MySom3D

DATASET_PATH = "./data/Dots_30_001/"
parser = RawEEGSignalParser(DATASET_PATH)

#channel_data = parser.load_channel_data(0)
#print(channel_data.shape)

full_data = parser.load_all_channels()
#print("full data shape: ",full_data.shape)
#print("full data: ", full_data)

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





