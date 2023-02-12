import struct
import numpy as np


class BinaryFileReader:

    def __init__(self):
        pass

    def read_signal(self, channel_filename):
        return self.read_data_file(channel_filename, 'f')

    def read_event_timestamps(self, event_timestamps_filename):
        return self.read_data_file(event_timestamps_filename, 'i')

    def read_event_codes(self, event_codes_filename):
        return self.read_data_file(event_codes_filename, 'i')

    def read_data_file(self, filename, data_type):
        """
        General reading method that will be called on more specific functions
        :param filename: name of the file
        :param data_type: usually int/float chosen by the file format (int/float - mentioned in epd/spktwe/ssd)
        :return: data: data read from file
        """

        with open(filename, 'rb') as file:
            data = []
            read_val = file.read(4)
            data.append(struct.unpack(data_type, read_val)[0])

            while read_val:
                read_val = file.read(4)
                try:
                    data.append(struct.unpack(data_type, read_val)[0])
                except struct.error:
                    break

            return np.array(data)