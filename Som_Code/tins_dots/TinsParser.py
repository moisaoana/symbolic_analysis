import numpy as np
from BinaryFileReader import BinaryFileReader

class TinsParser:
    def __init__(self):
        self.FileReader = BinaryFileReader()
        self.TIMESTAMP_LENGTH = 1


    def get_index_line(self, lines, const_string):
        index_line = np.where(lines == const_string)
        return index_line[0][0] + 1