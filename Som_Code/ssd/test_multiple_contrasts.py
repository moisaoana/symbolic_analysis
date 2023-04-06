import os

import numpy as np

from matplotlib import pyplot as plt

from som_implementation_3D import MySom3D
from ssd.Convolver import Convolver


def load_data_fast(DATASET_PATH):
    convolver = Convolver(DATASET_PATH, TRIAL_START=128, STIMULUS_ON=129, STIMULUS_OFF=150, TRIAL_END=192)
    # raster = convolver.build_raster(ELECTRODE='El_01')

    all_electrodes_raster = []
    for electrodes in convolver.labels_by_channels:
        raster = convolver.build_raster(ELECTRODE=electrodes)
        all_electrodes_raster.append(raster)
    raster = np.vstack(all_electrodes_raster)

    # SEPARATE RASTER
    trial_rasters = convolver.split_raster_by_trials(raster, downsample=True)
    print("TRIAL RASTERS SHAPE:", trial_rasters.shape)

    trial_rasters = convolver.convolve_all_trial_rasters(trial_rasters)
    print("CONVOLVED RASTERS SHAPE:", np.array(trial_rasters).shape)


    trial_rasters_by_condition = convolver.get_trials_separated_by_conditions(trial_rasters)
    print("TRIALS BY CONDITION SHAPE: ", trial_rasters_by_condition.shape)

    trial_rasters_by_condition = trial_rasters_by_condition[::3]
    print("TRIALS BY CONDITION SHAPE: ", np.array(trial_rasters_by_condition).shape)

    all_trials_ordered = np.vstack(trial_rasters_by_condition)
    print("STACKED TRIALS SHAPE: ", all_trials_ordered.shape)

    trials_transposed = np.swapaxes(all_trials_ordered, 1, 2)
    print("SWAPPED SHAPE: ", trials_transposed.shape)

    processed_data = np.vstack(trials_transposed)
    print("STACKED AGAIN SHAPE: ", processed_data.shape)

    return processed_data, trial_rasters_by_condition

def load_data_optimized_memory(DATASET_PATH):
    # REM optimize memory
    convolver = Convolver(DATASET_PATH, TRIAL_START=128, STIMULUS_ON=129, STIMULUS_OFF=150, TRIAL_END=192)

    all_electrodes_raster = np.zeros(shape=(convolver.NR_UNITS, convolver.RECORDING_LENGTH))
    counter = 0
    for electrode in convolver.labels_by_channels:
        raster = convolver.build_raster(ELECTRODE=electrode)
        all_electrodes_raster[counter:counter+raster.shape[0]] = raster
        counter += raster.shape[0]
    raster = all_electrodes_raster

    print("RASTER SHAPE:", raster.shape)

    # SEPARATE RASTER
    trial_rasters = convolver.split_raster_by_trials(raster, downsample=True)
    print("TRIAL RASTERS SHAPE:", trial_rasters.shape)

    trial_rasters = convolver.convolve_all_trial_rasters(trial_rasters)
    print("CONVOLVED RASTERS SHAPE:", trial_rasters.shape)



    trial_rasters_by_condition = convolver.get_trials_separated_by_conditions(trial_rasters)
    print("TRIALS BY CONDITION SHAPE: ", trial_rasters_by_condition.shape)

    trial_rasters_by_condition = trial_rasters_by_condition[::3]
    print("TRIALS BY CONDITION SHAPE: ", np.array(trial_rasters_by_condition).shape)

    all_trials_ordered = np.vstack(trial_rasters_by_condition)
    print("STACKED TRIALS SHAPE: ", all_trials_ordered.shape)

    trials_transposed = np.swapaxes(all_trials_ordered, 1, 2)
    print("SWAPPED SHAPE: ", trials_transposed.shape)

    processed_data = np.vstack(trials_transposed)
    print("STACKED AGAIN SHAPE: ", processed_data.shape)

    return processed_data, trial_rasters_by_condition


def load_data_optimized_memory_slow(DATASET_PATH):
    # REM optimize memory
    convolver = Convolver(DATASET_PATH, TRIAL_START=128, STIMULUS_ON=129, STIMULUS_OFF=150, TRIAL_END=192)

    all_electrodes_raster = []
    for electrode in convolver.labels_by_channels:
        raster = convolver.build_raster(ELECTRODE=electrode)
        trial_rasters = convolver.split_raster_by_trials(raster, downsample=True)
        trial_rasters = convolver.convolve_all_trial_rasters(trial_rasters)
        all_electrodes_raster.append(trial_rasters)
        print(f"RASTER of {electrode} ", trial_rasters.shape)

    trial_rasters = np.concatenate(all_electrodes_raster, axis=1)
    print("CONVOLVED RASTERS SHAPE:", trial_rasters.shape)



    trial_rasters_by_condition = convolver.get_trials_separated_by_conditions(trial_rasters)
    print("TRIALS BY CONDITION SHAPE: ", trial_rasters_by_condition.shape)


    # 24 conditions, 8 by orientation, 3 by contrast levels
    # condition 1, 4, ... are the 8 orientations for contrast level 100
    # condition 2, 5, ... are the 8 orientations for contrast level 50
    # As such, we jump 3 by 3 to get the conditions we want for contrast 100, for 50 it would be [1::3] - indexes are decremented by 1
    trial_rasters_by_condition = trial_rasters_by_condition[::3]
    print("TRIALS BY CONDITION SHAPE: ", np.array(trial_rasters_by_condition).shape)

    all_trials_ordered = np.vstack(trial_rasters_by_condition)
    print("STACKED TRIALS SHAPE: ", all_trials_ordered.shape)

    trials_transposed = np.swapaxes(all_trials_ordered, 1, 2)
    print("SWAPPED SHAPE: ", trials_transposed.shape)

    processed_data = np.vstack(trials_transposed)
    print("STACKED AGAIN SHAPE: ", processed_data.shape)

    return processed_data, trial_rasters_by_condition






DATASET_NAME = 'M017_0002_sorted_full'
DATASET_PATH = f"./data/{DATASET_NAME}/"

# processed_data, trial_rasters_by_condition = load_data_fast(DATASET_PATH)                       # 20 GB RAM
# processed_data, trial_rasters_by_condition = load_data_optimized_memory(DATASET_PATH)           # 12 GB RAM
processed_data, trial_rasters_by_condition = load_data_optimized_memory_slow(DATASET_PATH)        # 4 GB RAM

size = 10
no_features = processed_data.shape[1]
no_iterations = 1

som = MySom3D(size, size, size, no_features, sigma=2, learning_rate=1)
som.train(processed_data, no_iterations)

# som = MiniSom3D(size, size, size, no_features, sigma=0.3, learning_rate=0.5)
# som.train(processed_data, len(processed_data), verbose=True)

test = [som.find_BMU(sample) for sample in processed_data]
test = np.array(test)
print(np.unique(test, axis=0, return_counts=True))
print(len(np.unique(test, axis=0)))


def get_colours(som, condition_rasters):
    # condition_colours = []
    # for trial in condition_rasters:
    #     trial_colours = []
    #     for sample in trial.T:
    #         bmu = som.find_BMU(sample)
    #         colour = [float(bmu[0]) / float(som.getX()),
    #                   float(bmu[1]) / float(som.getY()),
    #                   float(bmu[2]) / float(som.getZ())]
    #         trial_colours.append(colour)
    #         # print(colour)
    #     # condition_colours.append(trial_colours)
    #
    # condition_colours = np.array(condition_colours)
    # print(condition_colours.shape)

    # REM optimize memory
    condition_colours = np.zeros(shape=(condition_rasters.shape[0], condition_rasters.shape[2], 3))
    for trial_id, trial in enumerate(condition_rasters):
        for sample_id, sample in enumerate(trial.T):
            bmu = som.find_BMU(sample)
            condition_colours[trial_id, sample_id, 0] = float(bmu[0]) / float(som.getX())
            condition_colours[trial_id, sample_id, 1] = float(bmu[1]) / float(som.getY())
            condition_colours[trial_id, sample_id, 2] = float(bmu[2]) / float(som.getZ())

    condition_colours = np.array(condition_colours)
    print(condition_colours.shape)

    return condition_colours


def plot_results(som, trial_rasters_by_condition):
    fig, axs = plt.subplots(len(trial_rasters_by_condition), 1,
                                   sharex=True,
                                   # gridspec_kw={"height_ratios": [nr_conditions, 1]},
                                   figsize=(12, 8))
    fig.suptitle("Contrast 100", fontsize=15)
    for cond_nr, condition_rasters in enumerate(trial_rasters_by_condition):
        condition_colours = get_colours(som, condition_rasters)
        #axs[cond_nr].set_title(f"Condition {cond_nr}")
        axs[cond_nr].set_ylabel(f"{cond_nr*45}", rotation=0)
        axs[cond_nr].imshow(condition_colours, aspect='auto', interpolation='none')

    plt.show()

print("TRIALS BY CONDITION SHAPE: ", np.array(trial_rasters_by_condition).shape)
plot_results(som, np.array(trial_rasters_by_condition))




