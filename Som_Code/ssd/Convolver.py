import os
import struct

import numpy as np
import matplotlib.pyplot as plt

from SsdParser import SsdParser


class Convolver():
    def __init__(self, DATASET_PATH, THETA=20, show=False):
        self.DATASET_PATH = DATASET_PATH
        self.show = show
        self.THETA = THETA

        parser = SsdParser(DATASET_PATH)

        units_by_channels, labels_by_channels, timestamps_by_channels = parser.units_by_channel, parser.labels, parser.timestamps_by_channel
        event_codes, event_timestamps = parser.event_codes, parser.event_timestamps
        self.RECORDING_LENGTH = parser.RECORDING_LENGTH
        self.SAMPLING_FREQUENCY = parser.spike_sampling_frequency

        if self.RECORDING_LENGTH == 0:
            raise Exception("NU MERE")

        ELECTRODE = 'El_15'
        self.X = np.array(units_by_channels[ELECTRODE])
        self.t = np.array(timestamps_by_channels[ELECTRODE])
        self.y = np.array(labels_by_channels[ELECTRODE])

        groups = parser.split_event_codes()
        self.timestamp_intervals = parser.split_event_timestamps_by_codes()
        # print(groups)
        # print(timestamp_intervals)
        print(len(self.timestamp_intervals))

        if self.show == True:
            print(self.RECORDING_LENGTH)
            print(self.SAMPLING_FREQUENCY)

            print(self.t.shape)
            print(self.X.shape)
            print(self.y.shape)

            print(event_codes.shape)
            print(event_timestamps.shape)

    def build_raster(self):
        unique_y = np.unique(self.y)
        raster = np.zeros((len(unique_y), self.RECORDING_LENGTH))
        # print(raster.shape)
        for unique_neuron in unique_y:
            timestamps_of_unit = self.t[self.y == unique_neuron]
            # print(timestamps_of_unit[0])
            for timestamp in timestamps_of_unit:
                raster[unique_neuron - 1, timestamp] = 1
            # print(t[y==unique_label].shape)
            # print(np.count_nonzero(raster[unique_label-1]))
        return raster

    def split_raster_by_trials(self, raster, downsample=True):
        trial_rasters = []
        for timestamp_interval in self.timestamp_intervals:
            trial_raster = raster[:, timestamp_interval[0]:timestamp_interval[1]]
            if downsample == True:
                trial_time = timestamp_interval[1] - timestamp_interval[0]
                # downsampling such that each sample now incorporates whether a spike occured in that ms
                trial_ms_time = trial_time / self.SAMPLING_FREQUENCY * 1000
                samples_in_1ms = int(trial_time / trial_ms_time)  # 32 samples

                test = []
                for index in range(0, trial_raster.shape[1], samples_in_1ms):
                    spikes = np.count_nonzero(trial_raster[:, index:index + samples_in_1ms], axis=1)  # might have 2 spikes in a single window of 32 samples
                    spikes[spikes == 0] = 0
                    spikes[spikes >= 1] = 1

                    test.append(spikes)

                trial_raster = np.array(test, dtype=float).T

            trial_rasters.append(trial_raster)

        if self.show == True:
            print(len(trial_rasters))
            print(trial_rasters[0].shape)

        return trial_rasters

    def convolve_one_raster(self, trial_raster):
        for neuron in range(trial_raster.shape[0]):
            for time in range(trial_raster.shape[1]):
                if trial_raster[neuron, time] == 1:  # had spike at time
                    trial_raster[neuron, time] = trial_raster[neuron, time - 1] + 1
                else:
                    trial_raster[neuron, time] = trial_raster[neuron, time - 1] * np.exp(-1 / self.THETA)

        return trial_raster

    def convolve_all_rasters(self, trial_rasters):
        for trial_raster in trial_rasters:
            trial_raster = self.convolve_one_raster(trial_raster)

        if self.show == True:
            print(trial_rasters[0].shape)
            print("TRIAL RASTERS SHAPE: ", np.array(trial_rasters).shape)

        return trial_rasters