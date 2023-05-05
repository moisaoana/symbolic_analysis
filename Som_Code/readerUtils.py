import numpy as np
import ast
import json


class ReaderUtils:

    @staticmethod
    def writeDistanceMap(distance_map):
        with open('distance_map.bin', 'wb') as f:
            np.save(f, distance_map)

    @staticmethod
    def readDistanceMap():
        with open('distance_map.bin', 'rb') as f:
            arr = np.load(f)
        return arr

    @staticmethod
    def writeWeights(weights):
        with open('weights.bin', 'wb') as f:
            np.save(f, weights)

    @staticmethod
    def readWeights():
        with open('weights.bin', 'rb') as f:
            arr = np.load(f)
        return arr

    @staticmethod
    def writeSamplesWithClusters(samples_with_clusters):
        with open('samples_with_clusters.txt', 'w') as file:
            for item in samples_with_clusters:
                values_str = ','.join(str(x) for x in item[0])
                line = "{} {}\n".format('[' + values_str + ']', item[1])
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
                item = (np.array(arr), num)
                # Append the tuple to the data array
                samples_with_clusters.append(item)
        return samples_with_clusters

    @staticmethod
    def writeMarkersAndColors(markers_and_colors):
        with open('markers_colors.txt', 'w') as file:
            for item in markers_and_colors:
                line = "{} {} {} {}\n".format(item[0], item[1], item[2], str(item[3]).replace(" ", ""))
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

    @staticmethod
    def write_data_to_file(data, filename):
        data_json = json.dumps(data)
        with open(filename, 'w') as f:
            f.write(data_json)

    @staticmethod
    def write_matrix_to_file(data, filename):
        # Open a file for writing
        with open(filename, 'w') as f:
            # Write each row of the matrix to the file
            for row in data:
                f.write(' '.join([str(elem) for elem in row]))
                f.write('\n')
