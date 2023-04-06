import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from EEG_DataProcessor import EEG_DataProcessor
from RawEEGSignalParser import RawEEGSignalParser

import plotly.graph_objs as go


from som_implementation_3D import MySom3D
from Plots_Generator import PlotsGenerator, GroupingMethod
from readerUtils import ReaderUtils
from utils import Utils

DATASET_PATH = "./data/Dots_30_001/"
parser = RawEEGSignalParser(DATASET_PATH)

full_data = parser.load_all_channels()

event_timestamps = parser.load_event_timestamps()
event_codes = parser.load_event_codes()

eegDataProcessor = EEG_DataProcessor(DATASET_PATH, full_data, event_timestamps, event_codes)
eegDataProcessor.create_trials(save=False)
eegDataProcessor.link_trials(save=False)

#cov_matrix = np.cov(eegDataProcessor.processed_data, bias=True)
#rank = np.linalg.matrix_rank(cov_matrix)
#print("Rank: "+ str(rank))

eegDataProcessor.apply_pca(5)
eegDataProcessor.apply_ica()
eegDataProcessor.reconstruct_trials()


size = 10
no_features = eegDataProcessor.processed_data.shape[1]
no_iterations = 1
sigma = 2
learning_rate = 1

print(eegDataProcessor.processed_data.shape)

print(len(eegDataProcessor.trials))
som = MySom3D(size, size, size, no_features, sigma=sigma, learning_rate=learning_rate)
som.train(eegDataProcessor.processed_data, no_iterations)

distance_map_train = som.distance_map().T
ReaderUtils.writeDistanceMap(distance_map_train)
distance_map = ReaderUtils.readDistanceMap()

weights_train = som.getWeights()
ReaderUtils.writeWeights(weights_train)
som.setWeights(ReaderUtils.readWeights())


PlotsGenerator.generateScatterPlotForDistanceMapPlotly(size, distance_map)

#PlotsGenerator.generateColorSeguenceForAllTrials(len(eegDataProcessor.trials), eegDataProcessor.trials, som)

#PlotsGenerator.generateColorSeguenceForAllTrialsInPDF(len(eegDataProcessor.trials), eegDataProcessor.trials, som)

#figure_data_array = PlotsGenerator.getTrialSequencesArray(eegDataProcessor.trials, som)

#PlotsGenerator.groupByStimulusVisibility(figure_data_array)
#PlotsGenerator.groupByResponse(figure_data_array)
#PlotsGenerator.groupByStimulus(figure_data_array)

#CLUSTERING
samples_with_clusters_array_train, markers_and_colors_train = PlotsGenerator.generateScatterPlotForClustersPlotly(som, eegDataProcessor.processed_data)
ReaderUtils.writeSamplesWithClusters(samples_with_clusters_array_train)
ReaderUtils.writeMarkersAndColors(markers_and_colors_train)
samples_with_clusters_array = ReaderUtils.readSamplesWithClusters()
markers_and_colors = ReaderUtils.readMarkersAndColors()

path = "color_seq_plots/updated_som/all_channels/pca+ica_5comp/clusters/left/"
params = "size: " + str(size) +" ep: " + str(no_iterations) +" feat: " + str(no_features) +" sigma: " + str(sigma) +" lr: " + str(learning_rate)
response_psi_threshold = 0.5
visibility_psi_threshold = 0.15
stimulus_psi_threshold = 0.05

figure_data_array, color_freq_for_each_trial = PlotsGenerator.getTrialSequencesArrayUsingClustersLeftAlignment(eegDataProcessor.trials, markers_and_colors, samples_with_clusters_array)
PlotsGenerator.groupByStimulusVisibility(figure_data_array, path, params)
PlotsGenerator.groupByResponse(figure_data_array, path, params)
PlotsGenerator.groupByStimulus(figure_data_array, path, params)




response_PSIs_for_all_colors_matrix_array = PlotsGenerator.computePSIByGroup(color_freq_for_each_trial, group= GroupingMethod.BY_RESPONSE)
print("Response PSIs: ")
print(response_PSIs_for_all_colors_matrix_array)
PlotsGenerator.groupByResponseWithPsiUsingBMU(eegDataProcessor.trials,
                                                som,
                                                response_PSIs_for_all_colors_matrix_array,
                                                response_psi_threshold,
                                                path,
                                                params,
                                                align=0)


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
