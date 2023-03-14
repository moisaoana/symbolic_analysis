import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from EEG_DataProcessor import EEG_DataProcessor
from RawEEGSignalParser import RawEEGSignalParser

import plotly.graph_objs as go


from Som_Code.som_implementation_3D import MySom3D
from Som_Code.tins_dots.Plots_Generator import PlotsGenerator
from Som_Code.utils import Utils

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

print(len(eegDataProcessor.trials))

som = MySom3D(size, size, size, no_features, sigma=0.3, learning_rate=0.5)
som.train(eegDataProcessor.processed_data, no_iterations)

distance_map = som.distance_map().T

PlotsGenerator.generateScatterPlotForDistanceMapPlotly(size, distance_map)

#PlotsGenerator.generateColorSeguenceForAllTrials(len(eegDataProcessor.trials), eegDataProcessor.trials, som)

#PlotsGenerator.generateColorSeguenceForAllTrialsInPDF(len(eegDataProcessor.trials), eegDataProcessor.trials, som)

#figure_data_array = PlotsGenerator.getTrialSequencesArray(eegDataProcessor.trials, som)

#PlotsGenerator.groupByStimulusVisibility(figure_data_array)
#PlotsGenerator.groupByResponse(figure_data_array)
#PlotsGenerator.groupByStimulus(figure_data_array)

samples_with_clusters_array, markers_and_colors = PlotsGenerator.generateScatterPlotForClustersPlotly(som, eegDataProcessor)

figure_data_array = PlotsGenerator.getTrialSequencesArrayUsingClusters(eegDataProcessor.trials, markers_and_colors,samples_with_clusters_array)
PlotsGenerator.groupByStimulusVisibility(figure_data_array)
PlotsGenerator.groupByResponse(figure_data_array)
PlotsGenerator.groupByStimulus(figure_data_array)

PlotsGenerator.generateSlicerPlotMayavi(distance_map)
