import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from EEG_DataProcessor import EEG_DataProcessor
from RawEEGSignalParser import RawEEGSignalParser
import plotly.graph_objs as go
from Som_Code.som_implementation_3D import MySom3D
from Plots_Generator import PlotsGenerator, GroupingMethod, Alignment, Method
from Som_Code.readerUtils import ReaderUtils
from Som_Code.tins_dots.EEG_MainHelper import EEG_MainHelper
from Som_Code.utils import Utils

DATASET_PATH = "./data/Dots_30_001/"
parser = RawEEGSignalParser(DATASET_PATH)

full_data = parser.load_all_channels()
#full_data = parser.load_A_channels()
#full_data = parser.load_B_and_D_channels()

event_timestamps = parser.load_event_timestamps()
event_codes = parser.load_event_codes()

eegDataProcessor = EEG_DataProcessor(DATASET_PATH, full_data, event_timestamps, event_codes)
eegDataProcessor.create_trials(save=False)
eegDataProcessor.link_trials(save=False)

print(eegDataProcessor.processed_data.shape)
rank = np.linalg.matrix_rank(
    np.matmul(eegDataProcessor.trials[0].trial_data, np.transpose(eegDataProcessor.trials[0].trial_data)))
print(rank)

#eegDataProcessor.apply_pca(5)
#eegDataProcessor.apply_ica(rank, eegDataProcessor.trials[0].trial_data, parser.CHANNEL_NAMES, parser.SAMPLING_FREQUENCY)
#eegDataProcessor.apply_ica_infomax(rank, eegDataProcessor.processed_data)
#eegDataProcessor.apply_fastica(eegDataProcessor.processed_data, 99)
#eegDataProcessor.apply_ica_on_each_trial(rank)
#eegDataProcessor.reconstruct_trials()

size = 10
no_features = eegDataProcessor.processed_data.shape[1]
no_iterations = 2
sigma = 2
learning_rate = 1
no_samples = eegDataProcessor.processed_data.shape[0]

print(eegDataProcessor.processed_data.shape)

print(len(eegDataProcessor.trials))
som = MySom3D(size, size, size, no_features, sigma=sigma, learning_rate=learning_rate)
som.pca_init(eegDataProcessor.processed_data)
som.train(eegDataProcessor.processed_data, no_iterations)

distance_map = som.distance_map().T
# ReaderUtils.writeDistanceMap(distance_map_train)
# distance_map = ReaderUtils.readDistanceMap()

# weights_train = som.getWeights()
# ReaderUtils.writeWeights(weights_train)
# som.setWeights(ReaderUtils.readWeights())

# PlotsGenerator.generateScatterPlotForDistanceMapPlotly(size, distance_map)


# CLUSTERING
"""
samples_with_clusters_array_train, markers_and_colors_train = PlotsGenerator.generateScatterPlotForClustersPlotly(som, eegDataProcessor.processed_data)
ReaderUtils.writeSamplesWithClusters(samples_with_clusters_array_train)
ReaderUtils.writeMarkersAndColors(markers_and_colors_train)
samples_with_clusters_array = ReaderUtils.readSamplesWithClusters()
markers_and_colors = ReaderUtils.readMarkersAndColors()
"""

pathLeft = "color_seq_plots/updated_som/all_channels/no_pca_no_ica/rgb/rularePsi1W_coeff2/left/rulareSOM10/"
pathRight = "color_seq_plots/updated_som/all_channels/no_pca_no_ica/rgb/rularePsi1W_coeff1/right/rulareSOM10/"
pathWindowStart = "color_seq_plots/updated_som/all_channels/no_pca_no_ica/rgb/rularePsi1W_coeff1/window_start/rulareSOM10/"
pathWindowEnd = "color_seq_plots/updated_som/all_channels/no_pca_no_ica/rgb/rularePsi1W_coeff1/window_end/rulareSOM10/"
params = "size: " + str(size) + " ep: " + str(no_iterations) + " feat: " + str(no_features) + " sigma: " + str(
    sigma) + " lr: " + str(learning_rate)

coeff = 2
EEG_MainHelper.full_pipeline(eegDataProcessor, som, pathLeft, pathRight, pathWindowStart, pathWindowEnd, params, coeff, no_samples)

"""
PlotsGenerator.generateSlicerPlotMayavi(distance_map)
"""