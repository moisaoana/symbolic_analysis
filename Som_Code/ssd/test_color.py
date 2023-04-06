import os

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt



# DATASET_NAME = 'M045_SRCS'
from Minisom3D import MiniSom3D
from ssd.Convolver import Convolver

DATASET_NAME = 'M045_0009'
DATASET_PATH = f"../../../../../DATA/TINS/{DATASET_NAME}/"

convolver = Convolver(DATASET_PATH)
raster = convolver.build_raster(ELECTRODE='El_01')

# SEPARATE RASTER
trial_rasters = convolver.split_raster_by_trials(raster, downsample=True)
print("TRIAL RASTERS SHAPE:", np.array(trial_rasters).shape)

trial_rasters = convolver.convolve_all_trial_rasters(trial_rasters)
print("CONVOLVED RASTERS SHAPE:", np.array(trial_rasters).shape)
trial_rasters_by_condition = convolver.get_trials_separated_by_conditions(trial_rasters)

# print(len(trial_rasters_by_condition))


print("TRIALS BY CONDITION SHAPE: ", np.array(trial_rasters_by_condition).shape)

all_trials_ordered = np.vstack(trial_rasters_by_condition)
print("STACKED TRIALS SHAPE: ", all_trials_ordered.shape)

trials_transposed = np.swapaxes(all_trials_ordered, 1, 2)
print("SWAPPED SHAPE: ", trials_transposed.shape)
processed_data = np.vstack(trials_transposed)
print("STACKED AGAIN SHAPE: ", processed_data.shape)

size = 10
no_features = processed_data.shape[1]
no_iterations = 1

# som = MySom3D(size, size, size, no_features, sigma=0.3, learning_rate=0.5)
# som.train(processed_data, no_iterations)

som = MiniSom3D(size, size, size, no_features, sigma=0.3, learning_rate=0.5)
som.train(processed_data, len(processed_data), verbose=True)


test = [som.find_BMU(sample) for sample in processed_data]
test = np.array(test)
print(np.unique(test, axis=0, return_counts=True))
print(len(np.unique(test, axis=0)))

def get_colours(som, condition_rasters):
    condition_colours = []
    for trial in condition_rasters:
        trial_colours = []
        for sample in trial.T:
            bmu = som.find_BMU(sample)
            colour = [float(bmu[0]) / float(som.getX()),
                      float(bmu[1]) / float(som.getY()),
                      float(bmu[2]) / float(som.getZ())]
            trial_colours.append(colour)
        condition_colours.append(trial_colours)

    condition_colours = np.array(condition_colours)
    print(condition_colours.shape)

    return condition_colours


def plot_results(som, trial_rasters_by_condition):
    fig, axs = plt.subplots(len(trial_rasters_by_condition), 1,
                                   sharex=True,
                                   # gridspec_kw={"height_ratios": [nr_conditions, 1]},
                                   figsize=(12, 8))

    for cond_nr, condition_rasters in enumerate(trial_rasters_by_condition):
        condition_colours = get_colours(som, condition_rasters)
        #axs[cond_nr].set_title(f"Condition {cond_nr}")
        axs[cond_nr].imshow(condition_colours, aspect='auto', interpolation='none')

    plt.show()

print("TRIALS BY CONDITION SHAPE: ", np.array(trial_rasters_by_condition).shape)
plot_results(som, np.array(trial_rasters_by_condition))




# def get_colours(som, trials_transposed, cond_nr):
#     condition_colours = []
#     for trial in trials_transposed[cond_nr * 10: (cond_nr+1)*10]:
#         trial_colours = []
#         for sample in trial:
#             bmu = som.find_BMU(sample)
#             colour = [float(bmu[0]) / float(som.getX()),
#                       float(bmu[1]) / float(som.getY()),
#                       float(bmu[2]) / float(som.getZ())]
#             trial_colours.append(colour)
#         condition_colours.append(trial_colours)
#
#     condition_colours = np.array(condition_colours)
#     # print(condition_colours.shape)
#
#     return condition_colours
#
#
# def plot_results(som, trials_transposed):
#     nr_conditions = 8
#     fig, axs = plt.subplots(nr_conditions, 1,
#                                    sharex=True,
#                                    # gridspec_kw={"height_ratios": [nr_conditions, 1]},
#                                    figsize=(12, 8))
#
#     for cond_nr in range(nr_conditions):
#         condition_colours = get_colours(som, trials_transposed, cond_nr)
#         #axs[cond_nr].set_title(f"Condition {cond_nr}")
#         axs[cond_nr].set_ylabel(f"{cond_nr*45}")
#         axs[cond_nr].imshow(condition_colours, aspect='auto', interpolation='none')
#
#     plt.show()

# plot_results(som, trials_transposed)





