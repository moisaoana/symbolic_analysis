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

# print(full_data.shape)
# print(np.where(event_codes == 129))
# print(np.where(event_codes == 131))


eegDataProcessor = EEG_DataProcessor(DATASET_PATH, full_data, event_timestamps, event_codes)
eegDataProcessor.create_trials(save=False)
eegDataProcessor.link_trials(save=False)

print(eegDataProcessor.processed_data.shape)
rank = np.linalg.matrix_rank(
    np.matmul(eegDataProcessor.trials[0].trial_data, np.transpose(eegDataProcessor.trials[0].trial_data)))
print(rank)



#cov_matrix = np.cov(eegDataProcessor.processed_data, bias=True)
#rank = np.linalg.matrix_rank(cov_matrix)
#print("Rank: "+ str(rank))

#eegDataProcessor.apply_pca(5)
#eegDataProcessor.apply_ica(rank, eegDataProcessor.trials[0].trial_data, parser.CHANNEL_NAMES, parser.SAMPLING_FREQUENCY)
#eegDataProcessor.apply_ica_infomax(99, eegDataProcessor.trials[0].trial_data)
#eegDataProcessor.reconstruct_trials()

size = 5
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

# ---------------- do not uncomment ---------------
# PlotsGenerator.generateColorSeguenceForAllTrials(len(eegDataProcessor.trials), eegDataProcessor.trials, som)

# PlotsGenerator.generateColorSeguenceForAllTrialsInPDF(len(eegDataProcessor.trials), eegDataProcessor.trials, som)

# figure_data_array = PlotsGenerator.getTrialSequencesArray(eegDataProcessor.trials, som)

# PlotsGenerator.groupByStimulusVisibility(figure_data_array)
# PlotsGenerator.groupByResponse(figure_data_array)
# PlotsGenerator.groupByStimulus(figure_data_array)

# -----------

# CLUSTERING
"""
samples_with_clusters_array_train, markers_and_colors_train = PlotsGenerator.generateScatterPlotForClustersPlotly(som, eegDataProcessor.processed_data)
ReaderUtils.writeSamplesWithClusters(samples_with_clusters_array_train)
ReaderUtils.writeMarkersAndColors(markers_and_colors_train)
samples_with_clusters_array = ReaderUtils.readSamplesWithClusters()
markers_and_colors = ReaderUtils.readMarkersAndColors()
"""

pathLeft = "color_seq_plots/updated_som/all_channels/no_pca_no_ica/rgb/rularePsi1W_coeff1.5/left/rulareSOM5/"
pathRight = "color_seq_plots/updated_som/all_channels/no_pca_no_ica/rgb/rularePsi1W_coeff1.5/right/rulareSOM5/"
pathWindow = "color_seq_plots/updated_som/all_channels/no_pca_no_ica/rgb/rularePsi1W_coeff1.5/window_end/rulareSOM5/"
params = "size: " + str(size) + " ep: " + str(no_iterations) + " feat: " + str(no_features) + " sigma: " + str(
    sigma) + " lr: " + str(learning_rate)
# response_psi_threshold = 0.09
# visibility_psi_threshold = 0.15
# stimulus_psi_threshold = 0.05

# psi_threshold = 0.0009


# lista de liste (lista pt nothing, lista pt smth, lista pt identified), fiecare lista contine freq matrix pt each trial
"""
list_freq_by_response = PlotsGenerator.groupByResponseV2(eegDataProcessor.trials, som, pathLeft, params,
                                                         alignment=Alignment.LEFT)
PlotsGenerator.groupByResponseV2(eegDataProcessor.trials, som, pathRight, params, alignment=Alignment.RIGHT)

list_freq_by_stimulus = PlotsGenerator.groupByStimulusV2(eegDataProcessor.trials, som, pathLeft, params,
                                                         alignment=Alignment.LEFT)
PlotsGenerator.groupByStimulusV2(eegDataProcessor.trials, som, pathRight, params, alignment=Alignment.RIGHT)

list_freq_by_visibility = PlotsGenerator.groupByVisibilityV2(eegDataProcessor.trials, som, pathLeft, params,
                                                             alignment=Alignment.LEFT)
PlotsGenerator.groupByVisibilityV2(eegDataProcessor.trials, som, pathRight, params, alignment=Alignment.RIGHT)
"""
#WINDOW-----------------
EEG_MainHelper.take_minimum_window_from_trials_end(eegDataProcessor.trials, eegDataProcessor.trials_lengths)
list_freq_by_response = PlotsGenerator.groupByResponseV2(eegDataProcessor.trials, som, pathWindow, params, alignment=Alignment.LEFT, method=Method.BMU)

list_freq_by_stimulus = PlotsGenerator.groupByStimulusV2(eegDataProcessor.trials, som, pathWindow, params, alignment=Alignment.LEFT, method=Method.BMU)

list_freq_by_visibility = PlotsGenerator.groupByVisibilityV2(eegDataProcessor.trials, som, pathWindow, params, alignment=Alignment.LEFT, method=Method.BMU)

#LEFT, RIGHT------------------------
"""
list_freq_by_response = PlotsGenerator.groupByResponseV2(eegDataProcessor.trials, som, pathLeft, params, alignment=Alignment.LEFT, method=Method.BMU)
PlotsGenerator.groupByResponseV2(eegDataProcessor.trials, som, pathRight, params, alignment=Alignment.RIGHT, method=Method.BMU)

list_freq_by_stimulus = PlotsGenerator.groupByStimulusV2(eegDataProcessor.trials, som, pathLeft, params, alignment=Alignment.LEFT, method=Method.BMU)
PlotsGenerator.groupByStimulusV2(eegDataProcessor.trials, som, pathRight, params, alignment=Alignment.RIGHT, method=Method.BMU)

list_freq_by_visibility = PlotsGenerator.groupByVisibilityV2(eegDataProcessor.trials, som, pathLeft, params, alignment=Alignment.LEFT, method=Method.BMU)
PlotsGenerator.groupByVisibilityV2(eegDataProcessor.trials, som, pathRight, params, alignment=Alignment.RIGHT, method=Method.BMU)
"""
# ---------------------------------
coeff = 1.5
EEG_MainHelper.main_with_psi1(list_freq_by_response, list_freq_by_stimulus, list_freq_by_visibility, som, eegDataProcessor, pathLeft, pathRight, pathWindow, params, no_samples, coeff, weighted=True, window=True)

# ---------------------------------

#EEG_MainHelper.main_with_psi2(list_freq_by_response, list_freq_by_stimulus, list_freq_by_visibility, som, eegDataProcessor, pathLeft, pathRight, pathWindow, params, no_samples, coeff, window=True)


""""
visibility_PSIs_for_all_colors_matrix_array = PlotsGenerator.computePSIByGroup(color_freq_for_each_trial, group= GroupingMethod.BY_VISIBILITY)
print("Visibility PSIs: ")
print(visibility_PSIs_for_all_colors_matrix_array)

PlotsGenerator.groupByVisibilityWithPsiUsingBMU(eegDataProcessor.trials,
                                                    som,
                                                    visibility_PSIs_for_all_colors_matrix_array,
                                                    visibility_psi_threshold,
                                                    path,
                                                    params,
                                                    align=0)


stimulus_PSIs_for_all_colors_matrix_array = PlotsGenerator.computePSIByGroup(color_freq_for_each_trial, group= GroupingMethod.BY_STIMULUS)
print("Stimulus PSIs: ")
print(stimulus_PSIs_for_all_colors_matrix_array)
PlotsGenerator.groupByStimulusWithPsiUsingBMU(eegDataProcessor.trials,
                                                    som,
                                                    stimulus_PSIs_for_all_colors_matrix_array,
                                                    stimulus_psi_threshold,
                                                    path,
                                                    params,
                                                    align=0)

PlotsGenerator.generateSlicerPlotMayavi(distance_map)
"""