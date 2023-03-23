import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from ssd.BinaryFileReader import BinaryFileReader


class TSCParser:
    def __init__(self):
        self.FileReader = BinaryFileReader()
        self.TIMESTAMP_LENGTH = 1


    def get_index_line(self, lines, const_string):
        index_line = np.where(lines == const_string)
        return index_line[0][0] + 1

    def get_intracellular_labels(self):
        intracellular_labels = np.zeros((len(self.timestamps)))
        # given_index = np.zeros((len(event_timestamps[event_codes == 1])))
        # for index, timestamp in enumerate(timestamps):
        #     for index2, event_timestamp in enumerate(event_timestamps[event_codes == 1]):
        #         if event_timestamp - 1000 < timestamp < event_timestamp + 1000 and given_index[index2] == 0:
        #             given_index[index2] = 1
        #             intracellular_labels[index] = 1
        #             break

        for index2, event_timestamp in enumerate(self.event_timestamps[self.event_codes == 1]):
            indexes = []
            for index, timestamp in enumerate(self.timestamps):
                if event_timestamp - self.WAVEFORM_LENGTH < timestamp < event_timestamp + self.WAVEFORM_LENGTH:
                    # given_index[index2] = 1
                    indexes.append(index)

            if indexes != []:
                min = indexes[0]
                for i in range(1, len(indexes)):
                    if self.timestamps[indexes[i]] < self.timestamps[min]:
                        min = indexes[i]
                intracellular_labels[min] = 1

        return intracellular_labels


""""
    def plot_data_pca(self, title, data, labels, nr_dim=2, show=False):
        data = np.array(data)
        pca_ = PCA(n_components=nr_dim)
        data_pca = pca_.fit_transform(data)
        sp.plot(title, data_pca, labels)

        if show == True:
            plt.show()
"""