import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from EEG_DataProcessor import EEG_DataProcessor
from RawEEGSignalParser import RawEEGSignalParser

import plotly.graph_objs as go

from som_implementation_3D import MySom3D
from Plots_Generator import PlotsGenerator, GroupingMethod, Alignment
from readerUtils import ReaderUtils
from tins_dots.EEG_MainHelper import EEG_MainHelper
from utils import Utils

DATASET_PATH = "./data/Dots_30_001/"
parser = RawEEGSignalParser(DATASET_PATH)

# full_data = parser.load_all_channels()
# full_data = parser.load_A_channels()
full_data = parser.load_B_and_D_channels()

event_timestamps = parser.load_event_timestamps()
event_codes = parser.load_event_codes()

eegDataProcessor = EEG_DataProcessor(DATASET_PATH, full_data, event_timestamps, event_codes)
eegDataProcessor.create_trials(save=False)
eegDataProcessor.link_trials(save=False)

# cov_matrix = np.cov(eegDataProcessor.processed_data, bias=True)
# rank = np.linalg.matrix_rank(cov_matrix)
# print("Rank: "+ str(rank))

eegDataProcessor.apply_pca(5)
eegDataProcessor.apply_ica()
eegDataProcessor.reconstruct_trials()

size = 10
no_features = eegDataProcessor.processed_data.shape[1]
no_iterations = 1
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
# samples_with_clusters_array_train, markers_and_colors_train = PlotsGenerator.generateScatterPlotForClustersPlotly(som, eegDataProcessor.processed_data)
# ReaderUtils.writeSamplesWithClusters(samples_with_clusters_array_train)
# ReaderUtils.writeMarkersAndColors(markers_and_colors_train)
# samples_with_clusters_array = ReaderUtils.readSamplesWithClusters()
# markers_and_colors = ReaderUtils.readMarkersAndColors()

pathLeft = "color_seq_plots/updated_som/B_and_D_channels/pca+ica_5comp/rgb/left/rularePsi1W/"
pathRight = "color_seq_plots/updated_som/B_and_D_channels/pca+ica_5comp/rgb/right/rularePsi1W/"
params = "size: " + str(size) + " ep: " + str(no_iterations) + " feat: " + str(no_features) + " sigma: " + str(
    sigma) + " lr: " + str(learning_rate)
# response_psi_threshold = 0.09
# visibility_psi_threshold = 0.15
# stimulus_psi_threshold = 0.05

# psi_threshold = 0.0009


# lista de liste (lista pt nothing, lista pt smth, lista pt identified), fiecare lista contine freq matrix pt each trial
list_freq_by_response = PlotsGenerator.groupByResponseV2(eegDataProcessor.trials, som, pathLeft, params,
                                                         alignment=Alignment.LEFT)
PlotsGenerator.groupByResponseV2(eegDataProcessor.trials, som, pathRight, params, alignment=Alignment.RIGHT)

list_freq_by_stimulus = PlotsGenerator.groupByStimulusV2(eegDataProcessor.trials, som, pathLeft, params,
                                                         alignment=Alignment.LEFT)
PlotsGenerator.groupByStimulusV2(eegDataProcessor.trials, som, pathRight, params, alignment=Alignment.RIGHT)

list_freq_by_visibility = PlotsGenerator.groupByVisibilityV2(eegDataProcessor.trials, som, pathLeft, params,
                                                             alignment=Alignment.LEFT)
PlotsGenerator.groupByVisibilityV2(eegDataProcessor.trials, som, pathRight, params, alignment=Alignment.RIGHT)

# ---------------------------------

EEG_MainHelper.main_with_psi1(list_freq_by_response, list_freq_by_stimulus, list_freq_by_visibility, som, eegDataProcessor, pathLeft, pathRight, params, no_samples, weighted=True)

# ---------------------------------

# EEG_MainHelper.main_with_psi2(list_freq_by_response, list_freq_by_stimulus, list_freq_by_visibility, som, eegDataProcessor, pathLeft, pathRight, params, no_samples)


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