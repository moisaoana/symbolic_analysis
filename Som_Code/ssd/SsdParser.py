import os
import struct

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from ssd.TSCParser import TSCParser


class SsdParser(TSCParser):
    def __init__(self, dir_name, show=False, TRIAL_START=1, STIMULUS_ON=2, STIMULUS_OFF=4, TRIAL_END=8):
        super().__init__()
        self.DATASET_PATH = dir_name
        self.show = show
        self.TRIAL_START = TRIAL_START
        self.STIMULUS_ON = STIMULUS_ON
        self.STIMULUS_OFF = STIMULUS_OFF
        self.TRIAL_END = TRIAL_END

        self.parse_ssd_file()
        if self.show == True:
            print(self.spike_count_per_unit)
            # print(self.unit_electrode)
        self.get_data()

    def parse_ssd_file(self):
        """
        Function that parses the .spktwe file from the directory and returns an array of length=nr of channels and each value
        represents the number of spikes in each channel
        @param dir_name: Path to directory that contains the files
        @return: spikes_per_channel: an array of length=nr of channels and each value is the number of spikes on that channel
        """
        string_nr_units = 'Number of units:'
        string_unit_names= 'List with the names of units:'
        string_channel_names = 'List with the recording elements (channels, tetrodes, etc) where each unit originates:'
        string_spike_counts = 'Number of spikes in each unit:'
        string_waveform_length = 'Waveform length in samples:'
        string_align = 'Waveform spike align offset - the sample in waveform that is aligned to the spike (first spike in multitrodes):'
        string_align2 = 'Waveform spike align offset - the sample in waveform that is aligned to the spike:'
        string_electrodes_per_multitrode = 'Number of multitrode electrodes that were used for sorting (single electrodes = 1; tetrodes = 4; etc.):'
        string_length = 'Original recording length (in samples of spike times):'
        string_spike_sampling_frequency = 'Spike times and event sampling frequency [Hz]:'
        string_waveform_sampling_frequency = 'Waveform internal sampling frequency [Hz] (can be different than the sampling of the spike times):'


        for file_name in os.listdir(self.DATASET_PATH):
            full_file_name = self.DATASET_PATH + file_name
            if full_file_name.endswith(".ssd"):
                file = open(full_file_name, "r")
                lines = file.readlines()
                lines = [line.rstrip() for line in lines]
                lines = np.array(lines)


                index = self.get_index_line(lines, string_nr_units)
                self.NR_UNITS = lines[index].astype(int)

                index = self.get_index_line(lines, string_unit_names)
                self.unit_names = lines[index: index + self.NR_UNITS]

                index = self.get_index_line(lines, string_channel_names)
                self.channel_names = lines[index: index + self.NR_UNITS]
                self.unique_channel_names = np.unique(self.channel_names)
                self.NR_CHANNELS = len(np.unique(self.channel_names))

                index = self.get_index_line(lines, string_spike_counts)
                self.spike_count_per_unit = lines[index: index + self.NR_UNITS].astype(int)

                # index = np.where(lines == 'Number of spikes in each unit:')
                # count = 1
                # while str(lines[index[0][0] + count]).isdigit():
                #     count += 1
                # spikes_per_unit = lines[index[0][0] + 1:index[0][0] + count]

                unit_electrode = [i.strip('El_') for i in lines if str(i).startswith('El_')]
                self.unit_electrode = np.array(unit_electrode).astype(int)

                index = self.get_index_line(lines, string_spike_sampling_frequency)
                self.spike_sampling_frequency = lines[index].astype(float).astype(int)

                index = self.get_index_line(lines, string_waveform_sampling_frequency)
                self.waveform_sampling_frequency = lines[index].astype(float).astype(int)



                try:
                    index = self.get_index_line(lines, string_electrodes_per_multitrode)
                    self.NR_ELECTRODES_PER_MULTITRODE = lines[index].astype(int)
                except IndexError:
                    self.NR_ELECTRODES_PER_MULTITRODE = 1

                index = self.get_index_line(lines, string_waveform_length)
                self.FULL_WAVEFORM_LENGTH = lines[index].astype(int)
                self.WAVEFORM_LENGTH = self.FULL_WAVEFORM_LENGTH // self.NR_ELECTRODES_PER_MULTITRODE

                try:
                    index = self.get_index_line(lines, string_align)
                    self.WAVEFORM_ALIGNMENT = lines[index].astype(int)
                except IndexError:
                    try:
                        index = self.get_index_line(lines, string_align2)
                        self.WAVEFORM_ALIGNMENT = lines[index].astype(int)
                    except IndexError:
                        self.WAVEFORM_ALIGNMENT = 0



                try:
                    index = self.get_index_line(lines, string_length)
                    self.RECORDING_LENGTH = lines[index].astype(int)
                except IndexError:
                    self.RECORDING_LENGTH = 0

                if self.show == True:
                    print("UNITS:", self.NR_UNITS)
                    print(self.unit_names)
                    print(self.channel_names)
                    print(self.spike_count_per_unit)
                    print(self.unit_electrode)

                    print("NR_ELECTRODES_PER_MULTITRODE:", self.NR_ELECTRODES_PER_MULTITRODE)

                    print("FULL_WAVEFORM_LENGTH:", self.FULL_WAVEFORM_LENGTH)
                    print("WAVEFORM_LENGTH:", self.WAVEFORM_LENGTH)

                    print("WAVEFORM_ALIGNMENT:", self.WAVEFORM_ALIGNMENT)

                    print("RECORDING_LENGTH:", self.RECORDING_LENGTH)

                # return spikes_per_unit.astype('int'), unit_electrode.astype('int')

    def find_ssd_files(self, DATASET_PATH):
        """
        Searches in a folder for certain file formats and returns them
        :param DATASET_PATH: folder that contains files, looks for files that contain the data
        :return: returns the names of the files that contains data
        """
        # timestamp_file = None
        timestamp_file = None
        waveform_file = None
        event_timestamps_filename = None
        event_codes_filename = None
        unit_statistics_filename = None
        amplitudes_file = None
        widths_file = None
        muie_file = None

        for file_name in os.listdir(DATASET_PATH):
            # File holding spike amplitudes; binary file containing the amplitudes of spikes (matched 1:1 with the file holding spike timestamps); 32 bit IEEE 754-1985, single precision floating point file:
            if file_name.endswith(".ssdsa"):
                amplitudes_file = DATASET_PATH + file_name
            #
            # File holding spike widths; binary file containing the widths (in samples) of spikes (matched 1:1 with the file holding spike timestamps); 32 bit IEEE 754-1985, single precision floating point file:
            if file_name.endswith(".ssdsw"):
                widths_file = DATASET_PATH + file_name

            # File holding spike timestamps for the units; binary file containing the time in samples from the beginning of the experiment (spikes of each unit are stored continuously); 32 bit signed integer file:
            if file_name.endswith(".ssdst"):
                timestamp_file = DATASET_PATH + file_name

            # File holding spike waveforms for the units; binary file containing the waveforms for all the units (waveforms for each unit are stored continuously);  32 bit IEEE 754-1985, single precision floating point file:
            if file_name.endswith(".ssduw"):
                waveform_file = DATASET_PATH + file_name
            # else:
            #     waveform_file = None

            # File holding event timestamps; timestamp is in samples; (32 bit signed integer file):
            if file_name.endswith(".ssdet"):
                event_timestamps_filename = DATASET_PATH + file_name

            # File holding codes of events corresponding to the event timestamps file; timestamp is in samples; (32 bit signed integer file):
            if file_name.endswith(".ssdec"):
                event_codes_filename = DATASET_PATH + file_name

            # File holding unit statistics (average & SD spike waveforms, spike widths); binary file with average waveforms stored one after another for each unit, SD waveforms for each unit, followed by the other statistics (spike width, etc); 32 bit IEEE 754-1985, single precision floating point:
            if file_name.endswith(".ssdus"):
                unit_statistics_filename = DATASET_PATH + file_name

            if file_name.endswith(".ssmpf"):
                muie_file = DATASET_PATH + file_name

        return timestamp_file, waveform_file, event_timestamps_filename, event_codes_filename, unit_statistics_filename

    def get_data(self):
        timestamp_file, waveform_file, event_timestamps_file, event_codes_file, unit_statistics_file = self.find_ssd_files(self.DATASET_PATH)

        self.timestamps = self.FileReader.read_timestamps(timestamp_file)
        self.timestamps_by_unit = self.separate_by_unit(self.timestamps, self.TIMESTAMP_LENGTH)
        self.timestamps_by_channel, self.labels = self.separate_units_by_channel_dict(self.timestamps_by_unit, self.TIMESTAMP_LENGTH)

        if waveform_file != None:
            self.waveforms = self.FileReader.read_waveforms(waveform_file)
            self.waveforms_by_unit = self.separate_by_unit(self.waveforms, self.FULL_WAVEFORM_LENGTH)
            self.units_by_channel, self.labels = self.separate_units_by_channel_dict(self.waveforms_by_unit, self.FULL_WAVEFORM_LENGTH)


        self.event_timestamps = self.FileReader.read_event_timestamps(event_timestamps_file)
        self.event_codes = self.FileReader.read_event_codes(event_codes_file)
        self.unit_statistics = self.FileReader.read_unit_statistics(unit_statistics_file)



    def separate_by_unit(self, data, length):
        """
        Separates a data by spikes_per_unit, knowing that data are put one after another and unit after unit
        :param spikes_per_unit: list of lists - returned by parse_ssd_file
        :param data: timestamps / waveforms
        :param length: 1 for timestamps and 58 for waveforms
        :return:
        """
        separated_data = []
        sum = 0
        for spikes_in_unit in self.spike_count_per_unit:
            separated_data.append(data[sum * length: (sum + spikes_in_unit) * length])
            sum += spikes_in_unit

        return separated_data


    def get_data_from_unit(self, data_by_unit, unit, length):
        """
        Selects data by chosen unit
        :param data_by_channel: all the data of a type (all timestamps / all waveforms from all units)
        :param unit: receives inputs from 1 to NR_UNITS, stored in list with start index 0 (so its channel -1)
        :param length: 1 for timestamps and 58 for waveforms
        :return:
        """
        data_on_unit = data_by_unit[unit]
        data_on_unit = np.reshape(data_on_unit, (-1, length))

        return data_on_unit

    def separate_units_by_channel_dict(self, data, data_length):
        units_in_channels = {}
        labels = {}
        for channel_name in self.unique_channel_names:
            units_in_channels[channel_name] = []
            labels[channel_name] = []

        for unit, channel_name in enumerate(self.channel_names):
            waveforms_on_unit = self.get_data_from_unit(data, unit, data_length)
            units_in_channels[channel_name].extend(waveforms_on_unit.tolist())
            labels[channel_name].extend(list(np.full((len(waveforms_on_unit),), unit)))

        for channel_name in labels.keys():
            if labels[channel_name] != []:
                label_set = labels[channel_name]
                label_set = np.array(label_set)
                min_label = np.amin(label_set)
                label_set = label_set - min_label + 1
                labels[channel_name] = label_set.tolist()


        return units_in_channels, labels

    def separate_units_by_channel(self, data, data_length):
        units_in_channels = []
        labels = []
        for i in range(self.NR_CHANNELS):
            units_in_channels.insert(0, [])
            labels.insert(0, [])

        for unit, channel in enumerate(self.unit_electrode):
            waveforms_on_unit = self.get_data_from_unit(data, unit, data_length)
            units_in_channels[channel - 1].extend(waveforms_on_unit.tolist())
            labels[channel - 1].extend(list(np.full((len(waveforms_on_unit),), unit)))

        reset_labels = []
        for label_set in labels:
            if label_set != []:
                label_set = np.array(label_set)
                min_label = np.amin(label_set)
                label_set = label_set - min_label + 1
                reset_labels.append(label_set.tolist())
            else:
                reset_labels.append([])

        return units_in_channels, reset_labels

    def split_multitrode(self):
        units_by_electrodes = []
        for channel_name in self.units_by_channel.keys():
            units = self.units_by_channel[channel_name]
            if len(units) != 0:
                units = np.array(units)
                units_by_electrode = []
                for step in range(0, self.FULL_WAVEFORM_LENGTH, self.WAVEFORM_LENGTH):
                    units_by_electrode.append(units[:, step:step + self.WAVEFORM_LENGTH])

                units_by_electrodes.append(units_by_electrode)
            else:
                units_by_electrodes.append([])

        return np.array(units_by_electrodes)

    def split_multitrode_dict(self):
        units_by_electrodes = {}
        for channel_name in self.units_by_channel.keys():
            units = self.units_by_channel[channel_name]
            if len(units) != 0:
                units = np.array(units)
                units_by_electrode = []
                for step in range(0, self.FULL_WAVEFORM_LENGTH, self.WAVEFORM_LENGTH):
                    units_by_electrode.append(units[:, step:step + self.WAVEFORM_LENGTH])

                units_by_electrodes[channel_name] = units_by_electrode
            else:
                units_by_electrodes[channel_name] = []

        return units_by_electrodes


    def split_event_codes(self):
        groups = []

        group = []
        # print(np.unique(self.event_codes))
        for id, event_code in enumerate(self.event_codes):
            if event_code == self.TRIAL_START:
                group = []
                group.append(id)
            elif len(group) == 1 and event_code == self.STIMULUS_ON:
                group.append(id)
            elif len(group) == 2 and event_code == self.STIMULUS_OFF:
                group.append(id)
            elif len(group) == 3 and event_code == self.TRIAL_END:
                group.append(id)
                groups.append(group)
                group = []

        self.groups = np.array(groups)

        return self.groups

    def split_event_timestamps_by_codes(self):
        groups = self.split_event_codes()

        timestamp_intervals = []
        for group in groups:
            # print(self.event_timestamps[group[0]], self.event_timestamps[group[1]] - self.event_timestamps[group[0]], self.event_timestamps[group[3]] - self.event_timestamps[group[2]],self.event_timestamps[group[3]])
            timestamps_of_interest = [self.event_timestamps[group[0]], self.event_timestamps[group[-1]]]
            timestamp_intervals.append(timestamps_of_interest)

        self.timestamp_trial_intervals = np.array(timestamp_intervals)
        return self.timestamp_trial_intervals


    def assert_correctness(self):
        print(f"Number of Units: {self.spike_count_per_unit.shape}")
        print(f"Number of Units: {len(self.unit_electrode)}")
        print(f"Number of Spikes in all Units: {np.sum(self.spike_count_per_unit)}")
        print(f"Unit - Electrode Assignment: {self.unit_electrode}")
        print("--------------------------------------------")

        print(f"DATASET is in folder: {self.DATASET_PATH}")
        timestamp_file, waveform_file, _, _ = self.find_ssd_files(self.DATASET_PATH)
        print(f"TIMESTAMP file found: {timestamp_file}")
        print(f"WAVEFORM file found: {waveform_file}")
        print("--------------------------------------------")

        timestamps = self.FileReader.read_timestamps(timestamp_file)
        print(f"Timestamps found in file: {timestamps.shape}")
        print(f"Number of spikes in all channels should be equal: {np.sum(self.spike_count_per_unit)}")
        print(f"Assert equality: {len(timestamps) == np.sum(self.spike_count_per_unit)}")

        timestamps_by_unit = self.separate_by_unit(timestamps, self.TIMESTAMP_LENGTH)
        print(f"Spikes per channel parsed from file: {self.spike_count_per_unit}")
        print(f"Timestamps per channel should be equal: {list(map(len, timestamps_by_unit))}")
        print(f"Assert equality: {list(self.spike_count_per_unit) == list(map(len, timestamps_by_unit))}")
        print("--------------------------------------------")

        waveforms = self.FileReader.read_waveforms(waveform_file)
        print(f"Waveforms found in file: {waveforms.shape}")
        print(f"Waveforms should be Timestamps*{self.WAVEFORM_LENGTH}: {len(timestamps) * self.WAVEFORM_LENGTH}")
        print(f"Assert equality: {len(timestamps) * self.WAVEFORM_LENGTH == len(waveforms)}")
        waveforms_by_unit = self.separate_by_unit(waveforms, self.WAVEFORM_LENGTH)
        print(f"Waveforms per channel: {list(map(len, waveforms_by_unit))}")
        print(f"Spikes per channel parsed from file: {self.spike_count_per_unit}")
        waveform_lens = list(map(len, waveforms_by_unit))
        print(f"Waveforms/{self.WAVEFORM_LENGTH} per channel should be equal: {[i // self.WAVEFORM_LENGTH for i in waveform_lens]}")
        print(f"Assert equality: {list(self.spike_count_per_unit) == [i // self.WAVEFORM_LENGTH for i in waveform_lens]}")
        print(f"Sum of lengths equal to total: {len(waveforms) == np.sum(np.array(waveform_lens))}")
        print("--------------------------------------------")



    def read_kampff_channel(self, key):
        units_in_channels, labels = self.separate_units_by_channel_dict(self.waveforms_by_unit, data_length=self.WAVEFORM_LENGTH)
        intracellular_labels = self.get_intracellular_labels()

        return units_in_channels[key], labels[key], intracellular_labels

    def plot_spikes_on_unit(self, unit, show=False):
        waveforms_on_unit = self.get_data_from_unit(self.waveforms_by_unit, unit, self.WAVEFORM_LENGTH)
        plt.figure()
        plt.title(f"Spikes ({len(waveforms_on_unit)}) on unit {unit}")
        for i in range(0, len(waveforms_on_unit)):
            plt.plot(np.arange(len(waveforms_on_unit[i])), waveforms_on_unit[i])

        if show:
            plt.show()

    def plot_sorted_data_all_available_channels(self):
        for channel in range(self.NR_CHANNELS):
            if self.units_by_channel[channel] != [] and self.labels[channel] != []:
                self.plot_data_pca(f"Units in Channel {channel + 1}", self.units_by_channel[channel], self.labels[channel])
        plt.show()

    def plot_sorted_data_all_available_channels_dict(self):
        for channel_name in self.channel_names:
            if self.units_by_channel[channel_name] != [] and self.labels[channel_name] != []:
                self.plot_data_pca(f"Units in Channel {channel_name}", self.units_by_channel[channel_name], self.labels[channel_name])
        plt.show()

    def plot_multitrode(self, data, labels, channel, nr_dim=2):
        for nr in range(self.NR_ELECTRODES_PER_MULTITRODE):
            self.plot_data_pca(f'Multitrode {channel} - Electrode {nr + 1}', data[channel][nr], labels[channel], nr_dim=nr_dim, show=True)

    def plot_multitrodes(self, data, labels, nr_dim):
        for channel_name in data.keys():
            self.plot_multitrode(data, labels, channel_name, nr_dim)
