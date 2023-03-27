from minisom import MiniSom, fast_norm
from numpy import (array, unravel_index, nditer, linalg, random, subtract, max,
                   power, exp, pi, zeros, ones, arange, outer, meshgrid, dot,
                   logical_and, mean, std, cov, argsort, linspace, transpose,
                   einsum, prod, nan, sqrt, hstack, diff, argmin, multiply,
                   nanmean, nansum)
from numpy.linalg import norm
from warnings import warn
from minisom import asymptotic_decay, _build_iteration_indexes



class MiniSom3D(MiniSom):
    def __init__(self, x, y, z, input_len, sigma=1.0, learning_rate=0.5, decay_function=asymptotic_decay,
                 neighborhood_function='gaussian', topology='rectangular', activation_distance='euclidean',
                 random_seed=None):
        """Initializes a Self Organizing Maps.

        A rule of thumb to set the size of the grid for a dimensionality
        reduction task is that it should contain 5*sqrt(N) neurons
        where N is the number of samples in the dataset to analyze.

        E.g. if your dataset has 150 samples, 5*sqrt(150) = 61.23
        hence a map 8-by-8 should perform well.

        Parameters
        ----------
        x : int
            x dimension of the SOM.

        y : int
            y dimension of the SOM.

        input_len : int
            Number of the elements of the vectors in input.

        sigma : float, optional (default=1.0)
            Spread of the neighborhood function, needs to be adequate
            to the dimensions of the map.
            (at the iteration t we have sigma(t) = sigma / (1 + t/T)
            where T is #num_iteration/2)
        learning_rate : initial learning rate
            (at the iteration t we have
            learning_rate(t) = learning_rate / (1 + t/T)
            where T is #num_iteration/2)

        decay_function : function (default=None)
            Function that reduces learning_rate and sigma at each iteration
            the default function is:
                        learning_rate / (1+t/(max_iterarations/2))

            A custom decay function will need to to take in input
            three parameters in the following order:

            1. learning rate
            2. current iteration
            3. maximum number of iterations allowed


            Note that if a lambda function is used to define the decay
            MiniSom will not be pickable anymore.

        neighborhood_function : string, optional (default='gaussian')
            Function that weights the neighborhood of a position in the map.
            Possible values: 'gaussian', 'mexican_hat', 'bubble', 'triangle'

        topology : string, optional (default='rectangular')
            Topology of the map.
            Possible values: 'rectangular', 'hexagonal'

        activation_distance : string, callable optional (default='euclidean')
            Distance used to activate the map.
            Possible values: 'euclidean', 'cosine', 'manhattan', 'chebyshev'

            Example of callable that can be passed:

            def euclidean(x, w):
                return linalg.norm(subtract(x, w), axis=-1)

        random_seed : int, optional (default=None)
            Random seed to use.
        """

        super(MiniSom3D, self).__init__(x, y, input_len, sigma, learning_rate, decay_function, neighborhood_function, topology,
                         activation_distance, random_seed)
        self.shape = (x, y, z)

        if sigma >= x or sigma >= y or sigma >= z:
            warn('Warning: sigma is too high for the dimension of the map.')

        self._random_generator = random.RandomState(random_seed)

        self._learning_rate = learning_rate
        self._sigma = sigma
        self._input_len = input_len
        # random initialization

        self._weights = self._random_generator.rand(x, y, z, input_len) * 2 - 1


        self._weights /= linalg.norm(self._weights, axis=-1, keepdims=True)


        self._activation_map = zeros((x, y, z))

        self._neigx = arange(x)
        self._neigy = arange(y)  # used to evaluate the neighborhood function
        self._neigz = arange(z)  # used to evaluate the neighborhood function

        if topology not in ['hexagonal', 'rectangular']:
            msg = '%s not supported only hexagonal and rectangular available'
            raise ValueError(msg % topology)
        self.topology = topology

        self._xx, self._yy, self._zz = meshgrid(self._neigx, self._neigy, self._neigz)

        self._xx = self._xx.astype(float)
        self._yy = self._yy.astype(float)

        self._zz = self._zz.astype(float)

        if topology == 'hexagonal':
            self._xx[::-2] -= 0.5
            if neighborhood_function in ['triangle']:
                warn('triangle neighborhood function does not ' +
                     'take in account hexagonal topology')

        self._decay_function = decay_function


        neig_functions = {'gaussian': self._gaussian3D,
                          'mexican_hat': self._mexican_hat3D,
                          'bubble': self._bubble3D,
                          'triangle': self._triangle3D
                          }

        if neighborhood_function not in neig_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (neighborhood_function,
                                    ', '.join(neig_functions.keys())))

        if neighborhood_function in ['triangle',
                                     'bubble'] and (divmod(sigma, 1)[1] != 0
                                                    or sigma < 1):
            warn('sigma should be an integer >=1 when triangle or bubble' +
                 'are used as neighborhood function')

        self.neighborhood = neig_functions[neighborhood_function]

        distance_functions = {'euclidean': self._euclidean_distance,
                              'cosine': self._cosine_distance,
                              'manhattan': self._manhattan_distance,
                              'chebyshev': self._chebyshev_distance}

        if isinstance(activation_distance, str):
            if activation_distance not in distance_functions:
                msg = '%s not supported. Distances available: %s'
                raise ValueError(msg % (activation_distance,
                                        ', '.join(distance_functions.keys())))

            self._activation_distance = distance_functions[activation_distance]
        elif callable(activation_distance):
            self._activation_distance = activation_distance

    def _gaussian3D(self, c, sigma):
        """Returns a Gaussian centered in c."""
        d = 2 * sigma * sigma
        ax = exp(-power(self._xx - self._xx.T[c], 2) / d)
        ay = exp(-power(self._yy - self._yy.T[c], 2) / d)
        az = exp(-power(self._zz - self._zz.T[c], 2) / d)
        return (ax * ay * az).T  # the external product gives a matrix

    def _mexican_hat3D(self, c, sigma):
        """Mexican hat centered in c."""
        p = power(self._xx-self._xx.T[c], 2) + power(self._yy-self._yy.T[c], 2)  + power(self._zz-self._zz.T[c], 2)
        d = 2*sigma*sigma
        return (exp(-p/d)*(1-2/d*p)).T

    def _bubble3D(self, c, sigma):
        """Constant function centered in c with spread sigma.
        sigma should be an odd value.
        """
        ax = logical_and(self._neigx > c[0]-sigma,
                         self._neigx < c[0]+sigma)
        ay = logical_and(self._neigy > c[1]-sigma,
                         self._neigy < c[1]+sigma)
        az = logical_and(self._neigz > c[2]-sigma,
                         self._neigz < c[2]+sigma)
        return outer(outer(ax, ay), az)*1.

    def _triangle3D(self, c, sigma):
        """Triangular function centered in c with spread sigma."""
        triangle_x = (-abs(c[0] - self._neigx)) + sigma
        triangle_y = (-abs(c[1] - self._neigy)) + sigma
        triangle_z = (-abs(c[1] - self._neigz)) + sigma
        triangle_x[triangle_x < 0] = 0.
        triangle_y[triangle_y < 0] = 0.
        triangle_z[triangle_z < 0] = 0.
        inter = outer(triangle_x, triangle_y)
        return outer(inter, triangle_z)

    def pca_weights_init(self, data):
        """Initializes the weights to span the first two principal components.

        This initialization doesn't depend on random processes and
        makes the training process converge faster.

        It is strongly reccomended to normalize the data before initializing
        the weights and use the same normalization for the training data.
        """
        if self._input_len == 1:
            msg = 'The data needs at least 2 features for pca initialization'
            raise ValueError(msg)
        self._check_input_len(data)
        if len(self._neigx) == 1 or len(self._neigy) == 1 or len(self._neigz) == 1:
            msg = 'PCA initialization inappropriate:' + \
                  'One of the dimensions of the map is 1.'
            warn(msg)
        pc_length, pc = linalg.eig(cov(transpose(data)))
        pc_order = argsort(-pc_length)
        for i, c1 in enumerate(linspace(-1, 1, len(self._neigx))):
            for j, c2 in enumerate(linspace(-1, 1, len(self._neigy))):
                for k, c3 in enumerate(linspace(-1, 1, len(self._neigz))):
                    self._weights[i, j, k] = c1*pc[pc_order[0]] + c2*pc[pc_order[1]] + c3*pc[pc_order[2]]

    def update(self, x, win, t, max_iteration):
        """Updates the weights of the neurons.

        Parameters
        ----------
        x : np.array
            Current pattern to learn.
        win : tuple
            Position of the winning neuron for x (array or tuple).
        t : int
            Iteration index
        max_iteration : int
            Maximum number of training itarations.
        """
        eta = self._decay_function(self._learning_rate, t, max_iteration)
        # sigma and learning rate decrease with the same rule
        sig = self._decay_function(self._sigma, t, max_iteration)
        # improves the performances
        g = self.neighborhood(win, sig) * eta
        # w_new = eta * neighborhood_function * (x-w)
        self._weights += einsum('ijk, ijkl->ijkl', g, x - self._weights)

    def train(self, data, num_iteration, random_order=False, verbose=False):
        """Trains the SOM.

        Parameters
        ----------
        data : np.array or list
            Data matrix.

        num_iteration : int
            Maximum number of iterations (one iteration per sample).
        random_order : bool (default=False)
            If True, samples are picked in random order.
            Otherwise the samples are picked sequentially.

        verbose : bool (default=False)
            If True the status of the training
            will be printed at each iteration.
        """
        self._check_iteration_number(num_iteration)
        self._check_input_len(data)
        random_generator = None
        if random_order:
            random_generator = self._random_generator
        iterations = _build_iteration_indexes(len(data), num_iteration,
                                              verbose, random_generator)
        for t, iteration in enumerate(iterations):
            self.update(data[iteration], self.winner(data[iteration]),
                          t, num_iteration)
        if verbose:
            print('\n quantization error:', self.quantization_error(data))
            print('\n topographic error:', self.topographic_error(data))

    def get_euclidean_coordinates(self):
        """Returns the position of the neurons on an euclidean
        plane that reflects the chosen topology in two meshgrids xx and yy.
        Neuron with map coordinates (1, 4) has coordinate (xx[1, 4], yy[1, 4])
        in the euclidean plane.

        Only useful if the topology chosen is not rectangular.
        """
        return self._xx.T, self._yy.T, self._zz.T

    def convert_map_to_euclidean(self, xyz):
        """Converts map coordinates into euclidean coordinates
        that reflects the chosen topology.

        Only useful if the topology chosen is not rectangular.
        """
        return self._xx.T[xyz], self._yy.T[xyz], self._zz.T[xyz]

    def distance_map(self, scaling='sum'):
        """Returns the distance map of the weights.
        If scaling is 'sum' (default), each cell is the normalised sum of
        the distances between a neuron and its neighbours. Note that this
        method uses the euclidean distance.

        Parameters
        ----------
        scaling : string (default='sum')
            If set to 'mean', each cell will be the normalized
            by the average of the distances of the neighbours.
            If set to 'sum', the normalization is done
            by the sum of the distances.
        """

        if scaling not in ['sum', 'mean']:
            raise ValueError('scaling should be either "sum" or "mean" ('
                             '"{scaling}" not valid)')

        um = nan * zeros((self._weights.shape[0],
                          self._weights.shape[1],
                          self._weights.shape[2],
                          26))  # 2 spots more for hexagonal topology

        # ii = [[0, -1, -1, -1, 0, 1, 1, 1]] * 2
        # jj = [[-1, -1, 0, 1, 1, 1, 0, -1]] * 2

        ii = [[0, -1, -1, -1, 0, 1, 1, 1, 0,
               0, -1, -1, -1, 0, 1, 1, 1,
               0, -1, -1, -1, 0, 1, 1, 1, 0]]*2
        jj = [[-1, -1, 0, 1, 1, 1, 0, -1, 0
               -1, -1, 0, 1, 1, 1, 0, -1,
               -1, -1, 0, 1, 1, 1, 0, -1, 0]]*2
        kk = [[ -1, -1, -1, -1, -1, -1, -1, -1, -1,
                0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1, 1]] * 2

        if self.topology == 'hexagonal': # HAS TO BE REMADE for 3D
            ii = [[1, 1, 1, 0, -1, 0], [0, 1, 0, -1, -1, -1]]
            jj = [[1, 0, -1, -1, 0, 1], [1, 0, -1, -1, 0, 1]]

        for x in range(self._weights.shape[0]):
            for y in range(self._weights.shape[1]):
                for z in range(self._weights.shape[2]):
                    w_2 = self._weights[x, y, z]
                    e = y % 2 == 0   # only used on hexagonal topology
                    for a, (i, j, k) in enumerate(zip(ii[e], jj[e], kk[e])):
                        if (x+i >= 0 and x+i < self._weights.shape[0]
                            and y+j >= 0 and y+j < self._weights.shape[1]
                            and z+k >= 0 and z+k < self._weights.shape[2]):
                            w_1 = self._weights[x+i, y+j, z+k]
                            um[x, y, z, a] = fast_norm(w_2-w_1)

        if scaling == 'mean':
            um = nanmean(um, axis=3)
        if scaling == 'sum':
            um = nansum(um, axis=3)

        return um/um.max()

    def activation_response(self, data):
        """
            Returns a matrix where the element i,j is the number of times
            that the neuron i,j have been winner.
        """
        self._check_input_len(data)
        a = zeros((self._weights.shape[0], self._weights.shape[1], self._weights.shape[2]))
        for x in data:
            a[self.winner(x)] += 1
        return a

    def _distance_from_weights(self, data):
        """Returns a matrix d where d[i,j] is the euclidean distance between
        data[i] and the j-th weight.
        """
        input_data = array(data)
        weights_flat = self._weights.reshape(-1, self._weights.shape[-1])
        input_data_sq = power(input_data, 2).sum(axis=1, keepdims=True)
        weights_flat_sq = power(weights_flat, 2).sum(axis=1, keepdims=True)
        cross_term = dot(input_data, weights_flat.T)
        return sqrt(-2 * cross_term + input_data_sq + weights_flat_sq.T)


    def quantization(self, data):
        """Assigns a code book (weights vector of the winning neuron)
        to each sample in data."""
        self._check_input_len(data)
        winners_coords = argmin(self._distance_from_weights(data), axis=1)
        return self._weights[unravel_index(winners_coords, self._weights.shape[:-1])]

    def topographic_error(self, data):
        """Returns the topographic error computed by finding
        the best-matching and second-best-matching neuron in the map
        for each input and then evaluating the positions.

        A sample for which these two nodes are not adjacent counts as
        an error. The topographic error is given by the
        the total number of errors divided by the total of samples.

        If the topographic error is 0, no error occurred.
        If 1, the topology was not preserved for any of the samples."""
        self._check_input_len(data)
        if self.topology == 'hexagonal':
            msg = 'Topographic error not implemented for hexagonal topology.'
            raise NotImplementedError(msg)
        total_neurons = prod(self._activation_map.shape)
        if total_neurons == 1:
            warn('The topographic error is not defined for a 1-by-1 map.')
            return nan

        t = 1.42
        # b2mu: best 2 matching units
        b2mu_inds = argsort(self._distance_from_weights(data), axis=1)[:, :2]
        b2my_xyz = unravel_index(b2mu_inds, self._weights.shape[:-1])
        b2mu_x, b2mu_y, b2mu_z = b2my_xyz[0], b2my_xyz[1], b2my_xyz[2]
        dxdydz = hstack([diff(b2mu_x), diff(b2mu_y), diff(b2mu_z)])
        distance = norm(dxdydz, axis=1) # matrix or vector norm, might not work for nd array
        return (distance > t).mean()