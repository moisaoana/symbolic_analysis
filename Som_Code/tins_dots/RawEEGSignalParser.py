import os
import numpy as np

from TinsParser import TinsParser


class RawEEGSignalParser(TinsParser):
    def __init__(self, dir_name):
        TinsParser.__init__(self)
        self.DATASET_PATH = dir_name

        self.parse_epd_file(self.DATASET_PATH)


    def parse_epd_file(self, dir_name):
        """
        Function that parses the .spktwe file from the directory and returns an array of length=nr of channels and each value
        represents the number of spikes in each channel
        @param dir_name: Path to directory that contains the files
        @return: spikes_per_channel: an array of length=nr of channels and each value is the number of spikes on that channel
        """
        string_nr_channels = 'Number of EEG channels:'
        string_nr_samples = 'Total number of samples:'
        string_filenames = 'List with filenames that hold individual channel samples (32 bit IEEE 754-1985, single precision floating point; amplitudes are measured in uV):'
        string_channel_names = 'List with labels of EEG channels:'
        string_sampling_frequency = 'Sampling frequency (Hz):'
        string_event_timestamp_filename = 'File holding event timestamps; timestamp is in samples; (32 bit signed integer file):'
        string_event_codes_filename = 'File holding codes of events corresponding to the event timestamps file; timestamp is in samples; (32 bit signed integer file):'

        for file_name in os.listdir(dir_name):
            full_file_name = dir_name + file_name
            if full_file_name.endswith(".epd"):
                file = open(full_file_name, "r")
                lines = file.readlines()
                lines = [line.rstrip() for line in lines]
                lines = np.array(lines)

                index = self.get_index_line(lines, string_nr_channels)
                self.NR_CHANNELS = lines[index].astype(int)
                print("NR_CHANNELS:", self.NR_CHANNELS)

                index = self.get_index_line(lines, string_nr_samples)
                self.NR_SAMPLES = lines[index].astype(int)
                print("NR_SAMPLES:", self.NR_SAMPLES)

                index = self.get_index_line(lines, string_sampling_frequency)
                self.SAMPLING_FREQUENCY = lines[index].astype(float)
                print("SAMPLING_FREQUENCY:", self.SAMPLING_FREQUENCY)

                index = self.get_index_line(lines, string_filenames)
                self.FILENAMES = lines[index: index + self.NR_CHANNELS]
                print(self.FILENAMES)

                index = self.get_index_line(lines, string_channel_names)
                self.CHANNEL_NAMES = lines[index: index + self.NR_CHANNELS]
                print(self.CHANNEL_NAMES)

                index = self.get_index_line(lines, string_event_timestamp_filename)
                self.FILENAME_EVENT_TIMESTAMPS = lines[index]
                print(self.CHANNEL_NAMES)

                index = self.get_index_line(lines, string_event_codes_filename)
                self.FILENAME_EVENT_CODES = lines[index]
                print(self.CHANNEL_NAMES)

    def load_event_timestamps(self):
        self.event_timestamps = self.FileReader.read_event_timestamps(self.DATASET_PATH + self.FILENAME_EVENT_TIMESTAMPS)

        return self.event_timestamps

    def load_event_codes(self):
        self.event_codes = self.FileReader.read_event_codes(self.DATASET_PATH + self.FILENAME_EVENT_CODES)

        return self.event_codes


    def load_channel_data(self, channel):
        print("-----> Loading:", self.FILENAMES[channel])
        self.data_channel = self.FileReader.read_signal(self.DATASET_PATH + self.FILENAMES[channel])
        print("-----> Finished loading:", self.FILENAMES[channel])

        return self.data_channel


    def load_all_channels(self):
        data_all_channels = []
        for chn_id in range(self.NR_CHANNELS):
            print(chn_id)
            data_channel = self.FileReader.read_signal(self.DATASET_PATH + self.FILENAMES[chn_id])
            data_all_channels.append(data_channel)

        self.data_all_channels = np.vstack(data_all_channels).T

        return self.data_all_channels

    def load_A_channels(self):
        data_all_channels = []
        for chn_id in range(32):
            print(chn_id)
            data_channel = self.FileReader.read_signal(self.DATASET_PATH + self.FILENAMES[chn_id])
            data_all_channels.append(data_channel)

        self.data_all_channels = np.vstack(data_all_channels).T

        return self.data_all_channels

    def load_B_and_D_channels(self):
        data_all_channels = []

        for chn_id in range(32, 64):
            print(chn_id)
            data_channel = self.FileReader.read_signal(self.DATASET_PATH + self.FILENAMES[chn_id])
            data_all_channels.append(data_channel)

        for chn_id in range(96,128):
            print(chn_id)
            data_channel = self.FileReader.read_signal(self.DATASET_PATH + self.FILENAMES[chn_id])
            data_all_channels.append(data_channel)

        self.data_all_channels = np.vstack(data_all_channels).T

        return self.data_all_channels