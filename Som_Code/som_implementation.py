import numpy as np
from numpy import (random, subtract, nan, zeros, transpose, cov, argsort, linspace)
from scipy import linalg


class MySom(object):
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5, learning_rate_decay=.1, sigma_decay=.1,
                 neighborhood_function='gaussian',
                 activation_distance='euclidean', random_seed=None):
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

    def getWeights(self):
        return self._weights

    def train(self, input_data, epochs):
        for epoch in np.arange(0, epochs):
            random.shuffle(input_data)
            for sample in input_data:
                x_BMU, y_BMU = self.find_BMU(sample)
                self._update_weights(sample, (x_BMU, y_BMU))
                self._learning_rate, self._sigma = self._decay(epoch)

    def _decay(self, epoch):
        new_learning_rate = self._learning_rate * np.exp(-epoch * self._learning_rate_decay)
        new_sigma = self._sigma * np.exp(-epoch * self._sigma_decay)
        return new_learning_rate, new_sigma

    def find_BMU(self, sample):
        # distance = (np.sqrt(np.square(self._weights - sample))).sum(axis=1)
        distance = self._euclidean_distance(sample, self._weights)
        # print(distance)
        return np.unravel_index(np.argmin(distance, axis=None), distance.shape)  # returns coords of min

    def _update_weights(self, sample, BMU_coord):
        x_BMU, y_BMU = BMU_coord
        # if radius is close to zero then only BMU is changed
        if self._sigma < 1e-4:
            self._weights[x_BMU, y_BMU, :] += self._learning_rate * (sample - self._weights[x_BMU, y_BMU, :])
            return
        # Change all cells in a small neighborhood of BMU
        for i in range(self._weights.shape[0]):
            for j in range(self._weights.shape[1]):
                if self._neighborhood(i, j, x_BMU, y_BMU) < self._sigma:
                    self._weights[i, j, :] += self._learning_rate * self._neighborhood(i, j, x_BMU, y_BMU) * (
                            sample - self._weights[i, j, :])

    def _gaussian(self, x_neighb, y_neighb, x_BMU, y_BMU):
        d = self._sigma * self._sigma
        dist_sq = np.square(x_neighb - x_BMU) + np.square(y_neighb - y_BMU)
        dist_func = np.exp(-dist_sq / 2 / d)
        return dist_func

    def _euclidean_distance(self, x, w):
        return linalg.norm(subtract(x, w), axis=-1)

    def distance_map(self):
        distance_map = np.zeros((self._x, self._y))
        ii = [0, -1, -1, -1, 0, 1, 1, 1]
        jj = [-1, -1, 0, 1, 1, 1, 0, -1]
        for x in range(self._weights.shape[0]):
            for y in range(self._weights.shape[1]):
                current_weights = self._weights[x, y]
                distance_sum = 0
                for k in range(0, len(ii)):
                    if (0 <= x + ii[k] < self._weights.shape[0] and
                            0 <= y + jj[k] < self._weights.shape[1]):
                        neighb_weights = self._weights[x + ii[k], y + jj[k]]
                        distance_sum += self._euclidean_distance(current_weights, neighb_weights)
                distance_map[x, y] = distance_sum
        return distance_map / distance_map.max()

    def random_sampling_init(self, input_data):
        random.shuffle(input_data)
        index_input_data = 0
        for i in range(self._x):
            for j in range(self._y):
                self._weights[i, j] = input_data[index_input_data]
                index_input_data += 1

    def pca_init(self, input_data):
        eigenvalues, eigenvectors = linalg.eig(cov(transpose(input_data)))
        eigenvalues_order = argsort(-eigenvalues)
        for i, c1 in enumerate(linspace(-1, 1, self._x)):
            for j, c2 in enumerate(linspace(-1, 1, self._y)):
                self._weights[i, j] = c1 * eigenvectors[eigenvalues_order[0]] + c2 * eigenvectors[eigenvalues_order[1]]

# som = Som(3, 4, 5, sigma=0.3, learning_rate=0.5)
