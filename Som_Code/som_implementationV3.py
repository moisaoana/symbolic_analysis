import numpy as np
from numpy import (random, subtract, nan, zeros, transpose, cov, argsort, linspace, meshgrid, arange)
from scipy import linalg


class MySom(object):
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5, learning_rate_decay=.1, sigma_decay=.1,
                 neighborhood_function='gaussian',
                 activation_distance='euclidean', random_seed=None, sigma_threshold=1e-4):
        self._x = x
        self._y = y
        self._learning_rate = learning_rate
        self._sigma = sigma
        self._learning_rate_decay = learning_rate_decay
        self._sigma_decay = sigma_decay
        self._input_len = input_len
        self._random_generator = random.RandomState(random_seed)
        self._weights = self._random_generator.rand(x, y, input_len)
        neig_functions = {'gaussian': self._gaussian}
        self._neighborhood = neig_functions[neighborhood_function]
        distance_functions = {'euclidean': self._euclidean_distance}
        self._activation_distance = distance_functions[activation_distance]
        self._sigma_threshold = sigma_threshold

        self._neigx = arange(x)
        self._neigy = arange(y)  # used to evaluate the neighborhood function

        self._xx, self._yy = meshgrid(self._neigx, self._neigy)

    def getWeights(self):
        return self._weights

    def train(self, input_data, epochs):
        for epoch in np.arange(0, epochs):
            #random.shuffle(input_data)
            for sample in input_data:
                sample = input_data[np.random.randint(0, len(input_data))]

                x_BMU, y_BMU = self.find_BMU(sample)
                # print(x_BMU, y_BMU)
                self._update_weights(sample, (x_BMU, y_BMU))
                self._decay(epoch)

    def _decay(self, epoch):
        self._learning_rate = self._learning_rate * np.exp(-epoch * self._learning_rate_decay)
        self._sigma = self._sigma * np.exp(-epoch * self._sigma_decay)


    def find_BMU(self, sample):
        distance = self._euclidean_distance(sample, self._weights)
        return np.unravel_index(np.argmin(distance, axis=None), distance.shape)  # returns coords of min

    def _update_weights(self, sample, BMU_coord):
        x_BMU, y_BMU = BMU_coord

        # if radius is close to zero then only BMU is changed
        if self._sigma < self._sigma_threshold:
            self._weights[x_BMU, y_BMU, :] += self._learning_rate * (sample - self._weights[x_BMU, y_BMU, :])
        else:
            self._weights += np.einsum('ij, ijk->ijk', self._learning_rate * self._gaussian(BMU_coord), sample - self._weights)


    def _gaussian(self, BMU_coord):
        d = (2 * self._sigma * self._sigma)
        # print((x_BMU, y_BMU))
        ax = np.exp(-np.power(self._xx - self._xx.T[BMU_coord], 2) / d)
        ay = np.exp(-np.power(self._yy - self._yy.T[BMU_coord], 2) / d)

        return (ax * ay).T


    def _euclidean_distance(self, x, w):
        difference = np.subtract(x, w)
        squared = np.square(difference)
        dist = np.sqrt(np.sum(squared, axis=-1))
        return dist

    def distance_map(self):
        distance_map = np.zeros((self._x, self._y))
        for x in range(self._weights.shape[0]):
            for y in range(self._weights.shape[1]):
                current_weights = self._weights[x, y]

                neighbours = get_valid_neighbours(np.array([x, y]), self._weights.shape[:-1])
                for neighbour in neighbours:
                    neighb_weights = self._weights[tuple(neighbour)]
                    distance_map[x, y] += self._euclidean_distance(current_weights, neighb_weights)

        return distance_map / np.amax(distance_map)

    def random_sampling_init(self, input_data):
        for i in range(self._x):
            for j in range(self._y):
                self._weights[i, j] = input_data[np.random.randint(0, len(input_data))]

    def pca_init(self, input_data):
        eigenvalues, eigenvectors = linalg.eig(cov(transpose(input_data)))
        eigenvalues_order = argsort(-eigenvalues)
        for i, c1 in enumerate(linspace(-1, 1, self._x)):
            for j, c2 in enumerate(linspace(-1, 1, self._y)):
                self._weights[i, j] = c1 * eigenvectors[eigenvalues_order[0]] + c2 * eigenvectors[eigenvalues_order[1]]



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