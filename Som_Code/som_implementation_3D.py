import sys

import numpy as np
from numpy import (random, subtract, nan, zeros, transpose, cov, argsort, linspace)
from scipy import linalg


class MySom3D(object):
    def __init__(self, x, y, z, input_len, sigma=1.0, learning_rate=0.5, learning_rate_decay=.1, sigma_decay=.1,
                 neighborhood_function='gaussian',
                 activation_distance='euclidean', random_seed=None, sigma_threshold=1e-4):
        self._x = x
        self._y = y
        self._z = z
        self._learning_rate = learning_rate
        self._sigma = sigma
        self._learning_rate_decay = learning_rate_decay
        self._sigma_decay = sigma_decay
        self._input_len = input_len
        self._random_generator = random.RandomState(random_seed)
        self._weights = self._random_generator.rand(x, y, z, input_len)
        neig_functions = {'gaussian': self._gaussian}
        self._neighborhood = neig_functions[neighborhood_function]
        distance_functions = {'euclidean': self._euclidean_distance}
        self._activation_distance = distance_functions[activation_distance]
        self._sigma_threshold = sigma_threshold

    def getWeights(self):
        return self._weights

    def setWeights(self,weights):
        self._weights = weights

    def getX(self):
        return self._x

    def getY(self):
        return self._y

    def getZ(self):
        return self._z

    def train(self, input_data, epochs):
        for epoch in np.arange(0, epochs):
            print("Epoch: ", epoch)
            for sample in input_data:
                sample = input_data[np.random.randint(0, len(input_data))]
               # print("TYPE ", type(sample))
                x_BMU, y_BMU, z_BMU = self.find_BMU(sample)
                self._update_weights(sample, (x_BMU, y_BMU, z_BMU))
                self._learning_rate, self._sigma = self._decay(epoch)

    def _decay(self, epoch):
        new_learning_rate = self._learning_rate * np.exp(-epoch * self._learning_rate_decay)
        new_sigma = self._sigma * np.exp(-epoch * self._sigma_decay)
        return new_learning_rate, new_sigma

    def find_BMU(self, sample):
        distance = self._euclidean_distance(sample, self._weights)
        return np.unravel_index(np.argmin(distance, axis=None), distance.shape)  # returns coords of min

    def _update_weights(self, sample, BMU_coord):
        x_BMU, y_BMU, z_BMU = BMU_coord
        # if radius is close to zero then only BMU is changed
        if self._sigma < self._sigma_threshold:
            self._weights[x_BMU, y_BMU, z_BMU, :] += self._learning_rate * (
                    sample - self._weights[x_BMU, y_BMU, z_BMU, :])
        else:
            # Change all cells in a small neighborhood of BMU
            for i in range(self._weights.shape[0]):
                for j in range(self._weights.shape[1]):
                    for k in range(self._weights.shape[2]):
                        if self._euclidean_distance(np.array([i, j, k]), BMU_coord) < 3 * self._sigma:
                            # formula pentru limitarea modificarilor, problematic ii ca tot itereaza prin toate weighturile
                            self._weights[i, j, k, :] += self._learning_rate * self._neighborhood(i, j, k, x_BMU, y_BMU,
                                                                                                  z_BMU) * (
                                                                 sample - self._weights[i, j, k, :])

    def _gaussian(self, x_neighb, y_neighb, z_neighb, x_BMU, y_BMU, z_BMU):
        dist_sq = np.square(x_neighb - x_BMU) + np.square(y_neighb - y_BMU) + np.square(z_neighb - z_BMU)
        dist_func = np.exp(-dist_sq / (2 * self._sigma * self._sigma))
        return dist_func

    def _euclidean_distance(self, x, w):
        difference = np.subtract(x, w)
        squared = np.square(difference)
        dist = np.sqrt(np.sum(squared, axis=-1))
        return dist

    def distance_map(self):
        distance_map = np.zeros((self._x, self._y, self._z))
        for x in range(self._weights.shape[0]):
            for y in range(self._weights.shape[1]):
                for z in range(self._weights.shape[2]):
                    current_weights = self._weights[x, y, z]
                    neighbours = get_valid_neighbours(np.array([x, y, z]), self._weights.shape[:-1])
                    for neighbour in neighbours:
                        neighb_weights = self._weights[tuple(neighbour)]
                        distance_map[x, y, z] += self._euclidean_distance(current_weights, neighb_weights)

        return distance_map / np.amax(distance_map)

    def random_sampling_init(self, input_data):
        random.shuffle(input_data)
        index_input_data = 0
        for i in range(self._x):
            for j in range(self._y):
                for k in range(self._z):
                    self._weights[i, j, k] = input_data[index_input_data]
                    index_input_data += 1

    def pca_init(self, input_data):
        eigenvalues, eigenvectors = linalg.eig(cov(transpose(input_data)))
        eigenvalues_order = argsort(-eigenvalues)
        for i, c1 in enumerate(linspace(-1, 1, self._x)):
            for j, c2 in enumerate(linspace(-1, 1, self._y)):
                for k, c3 in enumerate(linspace(-1, 1, self._z)):
                    self._weights[i, j, k] = c1 * eigenvectors[eigenvalues_order[0]] + c2 * eigenvectors[
                        eigenvalues_order[1]] + c3 * eigenvectors[eigenvalues_order[3]]

    def find_clusters(self, samples_data, threshold):
        bmu_array = []
        no_clusters = 0
        sample_array = []
        for cnt, sample in enumerate(samples_data):
            w = self.find_BMU(sample)
            sample_tuple = (sample, no_clusters)
            for bmu in bmu_array:
                difference = np.subtract(w, bmu[0])
                squared = np.square(difference)
                dist = np.sqrt(np.sum(squared, axis=-1))
                if dist < threshold:
                    sample_tuple = (sample, bmu[1])
                    break
            if sample_tuple[1] == no_clusters:
                no_clusters += 1
            sample_array.append(sample_tuple)
            if (w, sample_tuple[1]) not in bmu_array:
                bmu_array.append((w, sample_tuple[1]))
        print('No of clusters', no_clusters)
        return no_clusters, bmu_array, sample_array

    def find_clusters_with_min_dist(self, samples_data, threshold, max_distance):
        bmu_array = []
        no_clusters = 0
        sample_array = []
        for cnt, sample in enumerate(samples_data):
            print('Sample ',cnt)
            w = self.find_BMU(sample)
            sample_tuple = (sample, no_clusters)
            min_dist = sys.maxsize
            min_cluster = no_clusters
            for bmu in bmu_array:
                difference = np.subtract(w, bmu[0])
                squared = np.square(difference)
                dist = np.sqrt(np.sum(squared, axis=-1))
                if dist/max_distance < threshold and dist < min_dist:
                    min_dist = dist
                    min_cluster = bmu[1]
            sample_tuple = (sample, min_cluster)
            if sample_tuple[1] == no_clusters:
                no_clusters += 1
            sample_array.append(sample_tuple)
            if (w, sample_tuple[1]) not in bmu_array:
                bmu_array.append((w, sample_tuple[1]))
        print('No of clusters', no_clusters)
        return no_clusters, bmu_array, sample_array

    def find_threshold(self, samples_data):
        bmu_array = []
        max_threshold = 0
        i = 0
        for sample in samples_data:
            print('Sample ', i)
            w = self.find_BMU(sample)
            for bmu in bmu_array:
                difference = np.subtract(w, bmu)
                squared = np.square(difference)
                dist = np.sqrt(np.sum(squared, axis=-1))
                if dist > max_threshold:
                    max_threshold = dist
            if w not in bmu_array:
                bmu_array.append(w)
            i+=1
        return max_threshold


def get_valid_neighbours(point, shape):
    neighbours = get_neighbours(point)

    neighbours = validate_neighbours(neighbours, shape)

    return neighbours


def get_neighbours(point):
    """
    Get all the coordinates of the neighbours of a point
    :param point: vector - the coordinates of the chunk we are looking at

    :returns neighbours: array - vector of coordinates of the neighbours
    """
    # ndim = the number of dimensions of a point=chunk
    ndim = len(point)

    # offsetIndexes gives all the possible neighbours ( (0,0)...(2,2) ) of an unknown point in n-dimensions
    offsetIndexes = np.indices((3,) * ndim).reshape(ndim, -1).T

    # np.r_ does row-wise merging (basically concatenate), this instructions is equivalent to offsets=np.array([-1, 0, 1]).take(offsetIndexes)
    offsets = np.r_[-1, 0, 1].take(offsetIndexes)

    # remove the point itself (0,0) from the offsets (np.any will give False only for the point that contains only 0 on all dimensions)
    offsets = offsets[np.any(offsets, axis=1)]

    # calculate the coordinates of the neighbours of the point using the offsets
    neighbours = point + offsets

    return neighbours


def validate_neighbours(neighbours, shape):
    # validate the neighbours so they do not go out of the array
    valid = np.all((neighbours < np.array(shape)) & (neighbours >= 0), axis=1)

    neighbours = neighbours[valid]

    return neighbours
