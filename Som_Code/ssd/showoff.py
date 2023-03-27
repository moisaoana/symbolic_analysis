import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from Minisom3D import MiniSom3D
from som_implementation_3D import MySom3D
from ssd.Convolver import Convolver
from tins_dots.Plots_Generator import PlotsGenerator

DATASET_NAME = 'M045_SRCS'
DATASET_PATH = f'./data/1503/'

convolver = Convolver(DATASET_PATH, ELECTRODE='El_01')
raster = convolver.build_raster()

# SEPARATE RASTER
trial_rasters = convolver.split_raster_by_trials(raster, downsample=True)

trial_rasters = convolver.convolve_all_trial_rasters(trial_rasters)
trial_rasters_by_condition = convolver.get_trials_separated_by_conditions(trial_rasters)

# print(trial_rasters_by_condition)
print(np.vstack(np.array(trial_rasters_by_condition)).shape)

print("TRIALS BY CONDITION SHAPE: ", np.array(trial_rasters_by_condition).shape)

all_trials_ordered = np.vstack(trial_rasters_by_condition)

trials_transposed = np.swapaxes(all_trials_ordered, 1, 2)
processed_data = np.vstack(trials_transposed)

# compute transpose

# transpose_trials = []
# for trial in all_trials_ordered:
#     transpose_trials.append(trial.T)
#
# print("TEST: ", transpose_trials[0].shape)
#
# def link_trials(trials):
#     processed_data = []
#     for trial in trials:
#         processed_data.append(trial)
#     processed_data = np.vstack(processed_data)
#     return processed_data
#
# processed_data = link_trials(transpose_trials)
#
# print("TEST: ", processed_data.shape)


size = 4
no_features = processed_data.shape[1]
no_iterations = 1
sigma = 0.3
learning_rate = 0.5


som = MySom3D(size, size, size, no_features, sigma=sigma, learning_rate=learning_rate)
som.train(processed_data, no_iterations)

distance_map = som.distance_map().T

PlotsGenerator.generateScatterPlotForDistanceMapPlotly(size, distance_map)
samples_with_clusters_array, markers_and_colors = PlotsGenerator.generateScatterPlotForClustersPlotly(som,
                                                                                                      processed_data)

figure_data_array = PlotsGenerator.getTrialSequencesArrayUsingBMULeftAlignment(trials_transposed, som, True)

""""
som = MiniSom3D(size, size, size, no_features, sigma=0.3, learning_rate=0.5)

som.train(processed_data, len(processed_data), verbose=True)

distance_map = som.distance_map().T

figure_data_array = PlotsGenerator.getTrialSequencesArrayUsingBMULeftAlignmentMINISOM(trials_transposed, som, True)
"""

def groupByCondition(figure_array, path, params):
    fig, ax = PlotsGenerator.generateGridWithColorSequences(figure_array[0:10], n_rows=10, n_cols=1)
    plt.suptitle("Condition 1, " + params)
    fig.set_size_inches(6, 4)
    plt.savefig(path + "c1.png", dpi=300)

    fig, ax = PlotsGenerator.generateGridWithColorSequences(figure_array[10:20], n_rows=10, n_cols=1)
    plt.suptitle("Condition 2, " + params)
    fig.set_size_inches(6, 4)
    plt.savefig(path + "c2.png", dpi=300)

    fig, ax = PlotsGenerator.generateGridWithColorSequences(figure_array[20:30], n_rows=10, n_cols=1)
    plt.suptitle("Condition 3, " + params)
    fig.set_size_inches(6, 4)
    plt.savefig(path + "c3.png", dpi=300)

    fig, ax = PlotsGenerator.generateGridWithColorSequences(figure_array[30:40], n_rows=10, n_cols=1)
    plt.suptitle("Condition 4, " + params)
    fig.set_size_inches(6, 4)
    plt.savefig(path + "c4.png", dpi=300)

    fig, ax = PlotsGenerator.generateGridWithColorSequences(figure_array[40:50], n_rows=10, n_cols=1)
    plt.suptitle("Condition 5, " + params)
    fig.set_size_inches(6, 4)
    plt.savefig(path + "c5.png", dpi=300)

    fig, ax = PlotsGenerator.generateGridWithColorSequences(figure_array[50:60], n_rows=10, n_cols=1)
    plt.suptitle("Condition 6, " + params)
    fig.set_size_inches(6, 4)
    plt.savefig(path + "c6.png", dpi=300)

    fig, ax = PlotsGenerator.generateGridWithColorSequences(figure_array[60:70], n_rows=10, n_cols=1)
    plt.suptitle("Condition 7, " + params)
    fig.set_size_inches(6, 4)
    plt.savefig(path + "c7.png", dpi=300)

    fig, ax = PlotsGenerator.generateGridWithColorSequences(figure_array[70:80], n_rows=10, n_cols=1)
    plt.suptitle("Condition 8, " + params)
    fig.set_size_inches(6, 4)
    plt.savefig(path + "c8.png", dpi=300)
    plt.show()


groupByCondition(figure_data_array, "color_seq_plots_v2/mysom/rgb_EL_01_4_noT/",
                 "size: " + str(size) +
                 " ep: " + str(no_iterations) +
                 " feat: " + str(no_features) +
                 " sigma: " + str(sigma) +
                 " lr: " + str(learning_rate) +
                 " el: El_01 ")

"""

test_raster = trial_rasters[0]
print(len(trial_rasters))
print(test_raster.shape)
# SHOW RASTER FROM A TRIAL
matrix = test_raster[:, 1450:1550]
print(matrix.shape)
sns.set(rc={'figure.figsize':(20,4)})
ax = sns.heatmap(matrix)
plt.show()

matrix = test_raster
print(matrix.shape)
sns.set(rc={'figure.figsize':(20,4)})
ax = sns.heatmap(matrix)
plt.show()
"""
