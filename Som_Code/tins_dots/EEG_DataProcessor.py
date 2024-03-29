import os

import numpy as np


class EEG_Trial:
    def __init__(self, start_timestamp, end_timestamp, trial_data):
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.trial_data = trial_data


class EEG_DataProcessor:
    def __init__(self, DATASET_PATH, data, event_timestamps, event_codes):
        self.DATASET_PATH = DATASET_PATH
        self.data = data
        self.event_timestamps = event_timestamps
        self.event_codes = event_codes

        self.split_event_timestamps_by_codes()

    def split_event_codes(self):
        groups = []

        group = []
        for id, event_code in enumerate(self.event_codes):
            if event_code == 128:
                group = []
                group.append(id)
            elif len(group) == 1 and event_code == 129:
                group.append(id)
            elif len(group) == 2 and (event_code == 1 or event_code == 2 or event_code == 3):
                group.append(id)
            elif len(group) == 3 and event_code == 131:
                group.append(id)
                groups.append(group)
                group = []

        self.groups = np.array(groups)

        return self.groups

    def split_event_timestamps_by_codes(self):
        groups = self.split_event_codes()

        timestamp_intervals = []
        for group in groups:
            timestamps_of_interest = [self.event_timestamps[group[0]], self.event_timestamps[group[-1]]]
            timestamp_intervals.append(timestamps_of_interest)

        self.timestamp_intervals = np.array(timestamp_intervals)
        return self.timestamp_intervals

    def create_trials(self, save=True):
        self.trials = []
        for trial_id, timestamp in enumerate(self.timestamp_intervals):
            self.trials.append(
                EEG_Trial(
                    start_timestamp=timestamp[0],
                    end_timestamp=timestamp[1],
                    trial_data=self.data[timestamp[0]:timestamp[1]]
                )
            )
            if save==True:
                if not os.path.exists(self.DATASET_PATH+"/processed/"):
                    os.makedirs(self.DATASET_PATH+"/processed/")
                np.savetxt(self.DATASET_PATH + "/processed/" + "trial{trial_id}-{timestamp[0]}-{timestamp[1]}.csv", self.data[timestamp[0]:timestamp[1]], delimiter=",")


    def link_trials(self, save=True):
        processed_data = []
        for trial in self.trials:
            #print(trial.trial_data)
            processed_data.append(trial.trial_data)

        #print("processed data before: ", processed_data)
        self.processed_data = np.vstack(processed_data)
        #print("processed data after: ", self.processed_data)

        if save == True:
            if not os.path.exists(self.DATASET_PATH + "/processed/"):
                os.makedirs(self.DATASET_PATH + "/processed/")
            np.savetxt(self.DATASET_PATH + "/processed/" + "trial-all.csv", self.processed_data, delimiter=",")

        return self.processed_data