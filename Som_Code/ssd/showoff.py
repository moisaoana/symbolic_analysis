import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


from Convolver import Convolver
from som_implementation_3D import MySom3D
from tins_dots.Plots_Generator import PlotsGenerator

DATASET_NAME = 'M045_SRCS'
DATASET_PATH = f'./data/1503/'

convolver = Convolver(DATASET_PATH)
raster = convolver.build_raster()

# SEPARATE RASTER
trial_rasters = convolver.split_raster_by_trials(raster, downsample=True)

trial_rasters = convolver.convolve_all_rasters(trial_rasters)



def find_csv_filenames(path_to_dir, suffix=".csv" ):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]

csvs = find_csv_filenames(DATASET_PATH)
condition_file = csvs[0]

condition_table = pd.read_csv(DATASET_PATH + condition_file, skiprows=2)
condition_numpy = condition_table.to_numpy()[:, :3]
print(condition_numpy)



def separate_trials_by_conditions(trial_rasters, condition_numpy):
    trial_rasters_by_condition = []
    for condition in range(1, 9): # 8 conditions

        trial_rasters_condition = np.array(trial_rasters)[condition_numpy[:, 1] == condition]
        trial_rasters_by_condition.append(trial_rasters_condition)

    return trial_rasters_by_condition


trial_rasters_by_condition = separate_trials_by_conditions(trial_rasters, condition_numpy)

def get_data_by_condition(data, condition_numpy, condition):
    return np.array(data)[condition_numpy[:, 1] == condition]

print(trial_rasters_by_condition)
print(np.vstack(np.array(trial_rasters_by_condition)).shape)

print("TRIALS BY CONDITION SHAPE: ", np.array(trial_rasters_by_condition).shape)

all_trials_ordered = np.vstack(np.array(trial_rasters_by_condition))


#compute transpose

transpose_trials = []
for trial in all_trials_ordered:
    transpose_trials.append(trial.T)

print(transpose_trials[0].shape)

def link_trials(trials):
    processed_data = []
    for trial in trials:
        processed_data.append(trial)
    processed_data = np.vstack(processed_data)
    return processed_data

processed_data = link_trials(transpose_trials)

print(processed_data.shape)


size = 4
no_features = 9
no_iterations = 1


som = MySom3D(size, size, size, no_features, sigma=0.3, learning_rate=0.5)
som.train(processed_data, no_iterations)

distance_map = som.distance_map().T

PlotsGenerator.generateScatterPlotForDistanceMapPlotly(size, distance_map)

samples_with_clusters_array, markers_and_colors = PlotsGenerator.generateScatterPlotForClustersPlotly(som, processed_data)


figure_data_array = PlotsGenerator.getTrialSequencesArrayUsingBMULeftAlignment(transpose_trials, som, True)


def groupByCondition(figure_array, path):
    fig, ax = PlotsGenerator.generateGridWithColorSequences(figure_array[0:10], n_rows=10, n_cols=1)
    plt.suptitle("Condition 1")
    fig.set_size_inches(6, 4)
    plt.savefig(path + "c1.png", dpi=300)

    fig, ax = PlotsGenerator.generateGridWithColorSequences(figure_array[10:20], n_rows=10, n_cols=1)
    plt.suptitle("Condition 2")
    fig.set_size_inches(6, 4)
    plt.savefig(path + "c2.png", dpi=300)

    fig, ax = PlotsGenerator.generateGridWithColorSequences(figure_array[20:30], n_rows=10, n_cols=1)
    plt.suptitle("Condition 3")
    fig.set_size_inches(6, 4)
    plt.savefig(path + "c3.png", dpi=300)

    fig, ax = PlotsGenerator.generateGridWithColorSequences(figure_array[30:40], n_rows=10, n_cols=1)
    plt.suptitle("Condition 4")
    fig.set_size_inches(6, 4)
    plt.savefig(path + "c4.png", dpi=300)

    fig, ax = PlotsGenerator.generateGridWithColorSequences(figure_array[40:50], n_rows=10, n_cols=1)
    plt.suptitle("Condition 5")
    fig.set_size_inches(6, 4)
    plt.savefig(path + "c5.png", dpi=300)

    fig, ax = PlotsGenerator.generateGridWithColorSequences(figure_array[50:60], n_rows=10, n_cols=1)
    plt.suptitle("Condition 6")
    fig.set_size_inches(6, 4)
    plt.savefig(path + "c6.png", dpi=300)

    fig, ax = PlotsGenerator.generateGridWithColorSequences(figure_array[60:70], n_rows=10, n_cols=1)
    plt.suptitle("Condition 7")
    fig.set_size_inches(6, 4)
    plt.savefig(path + "c7.png", dpi=300)

    fig, ax = PlotsGenerator.generateGridWithColorSequences(figure_array[70:80], n_rows=10, n_cols=1)
    plt.suptitle("Condition 8")
    fig.set_size_inches(6, 4)
    plt.savefig(path + "c8.png", dpi=300)
    plt.show()


groupByCondition(figure_data_array, "color_seq_plots/rgb/left/")


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


