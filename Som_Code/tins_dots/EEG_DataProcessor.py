import os

import mne
import numpy as np
from sklearn.decomposition import PCA
from mne.preprocessing import ICA
from sklearn.decomposition import FastICA



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
        self.trials_lengths = []
        self.split_event_timestamps_by_codes()

    def split_event_codes(self):
        groups = []
        group = []
        for id, event_code in enumerate(self.event_codes):
            if event_code == 129:
                group = []
                group.append(id)
            elif len(group) == 1 and (event_code == 1 or event_code == 2 or event_code == 3):
                group.append(id)
            elif len(group) == 2 and event_code == 131:
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
            if save == True:
                if not os.path.exists(self.DATASET_PATH + "/processed/"):
                    os.makedirs(self.DATASET_PATH + "/processed/")
                np.savetxt(self.DATASET_PATH + "/processed/" + f"trial{trial_id}-{timestamp[0]}-{timestamp[1]}.csv",
                           self.data[timestamp[0]:timestamp[1]], delimiter=",")

    def link_trials(self, save=True):
        processed_data = []
        for trial in self.trials:
            # print(trial.trial_data)
            self.trials_lengths.append(len(trial.trial_data))
            processed_data.append(trial.trial_data)

        # print("processed data before: ", processed_data)
        self.processed_data = np.vstack(processed_data)
        # print("processed data after: ", self.processed_data)

        if save == True:
            if not os.path.exists(self.DATASET_PATH + "/processed/"):
                os.makedirs(self.DATASET_PATH + "/processed/")
            np.savetxt(self.DATASET_PATH + "/processed/" + "trial-all.csv", self.processed_data, delimiter=",")

        return self.processed_data

    def apply_pca(self, no_components):
        pca = PCA(n_components=no_components)
        self.processed_data = pca.fit_transform(self.processed_data)
        print(sum(pca.explained_variance_ratio_))

    def reconstruct_trials(self):
        start_index = 0
        for cnt, trial in enumerate(self.trials):
            last_index = start_index + self.trials_lengths[cnt]
            trial.trial_data = self.processed_data[start_index:last_index]
            start_index = start_index + self.trials_lengths[cnt]

    def apply_ica(self, n_comp, all_trials, channel_names, sfreq):
        #transformer = FastICA(n_comp, whiten='unit-variance', max_iter=1000)
        #self.processed_data = transformer.fit_transform(self.processed_data)

        ica = ICA(n_components=n_comp, method='infomax', random_state=23, verbose='INFO')
        ch_types = ['eeg' for _ in range(128)]
        info = mne.create_info(ch_names=channel_names.tolist(), ch_types=ch_types, sfreq=sfreq)
        data = mne.io.RawArray(all_trials.T, info)
        data.filter(1, 40, fir_design='firwin')
        ica.fit(data)
        #ica.plot_components()
        clean_data = ica.apply(data).get_data()
        print(clean_data.shape)

    def apply_ica_infomax(self, n_comp, all_trials):
        unmixing_matrix, n_it = mne.preprocessing.infomax(all_trials, n_subgauss=n_comp,return_n_iter=True, verbose='INFO')
        print(n_it)
        print(unmixing_matrix)

