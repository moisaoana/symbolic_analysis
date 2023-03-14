import numpy as np
import ast


class ReaderUtils:

    @staticmethod
    def writeDistanceMap(distance_map):
        with open('distance_map.txt', 'w') as file:
            array_string = np.array2string(distance_map, separator=', ')
            file.write(array_string)

    @staticmethod
    def readDistanceMap(size):
        return np.loadtxt('distance_map.txt', delimiter=',').reshape((size, size, size))

    @staticmethod
    def writeSamplesWithClusters(samples_with_clusters):
        with open('samples_with_clusters.txt', 'w') as file:
            for item in samples_with_clusters:
                line = "{} {}\n".format(item[0], item[1])
                file.write(line)

    @staticmethod
    def readSamplesWithClusters():
        samples_with_clusters = []
        with open('samples_with_clusters.txt', 'r') as file:
            for line in file:
                # Parse the line into a tuple
                line = line.strip()
                arr_str, num_str = line.split(' ')
                arr = ast.literal_eval(arr_str)
                num = int(num_str)
                item = (arr, num)
                # Append the tuple to the data array
                samples_with_clusters.append(item)
        return samples_with_clusters

    @staticmethod
    def writeMarkersAndColors(markers_and_colors):
        with open('markers_colors.txt', 'w') as file:
            for item in markers_and_colors:
                line = "{} {} {} {}\n".format(item[0], item[1], item[2], item[3])
                file.write(line)

    @staticmethod
    def readMarkersAndColors():
        markers_colors = []
        with open('markers_colors.txt', 'r') as file:
            contents = file.read()
            lines = contents.split('\n')
            for line in lines:
                # Parse the line into a tuple
                line = line.strip()
                if line == "":
                    continue
                num_str, str1, str2, arr_str = line.split(' ')
                num = int(num_str)
                arr = ast.literal_eval(arr_str)
                item = (num, str1, str2, arr)

                # Append the tuple to the data array
                markers_colors.append(item)
        return markers_colors