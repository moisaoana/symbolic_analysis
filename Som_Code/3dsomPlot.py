import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import math

from mayavi import mlab

# import warnings library
import warnings

# ignore all warnings
import Minisom3D
from som_implementation_3D import MySom3D

warnings.filterwarnings('ignore')

### Read and Analyse Data
samples = pd.read_csv('./breast_cancer_data.csv')  # returns data frame

# feature names as a list
col = samples.columns  # .columns gives columns names in data

# Remember:
# 1) There is an id that cannot be used for classification
# 2) Diagnosis is our class label

list = ['id', 'diagnosis']
x = samples.drop(list, axis=1)

### transform to numpy matrix
data = x.to_numpy()

### compute size of map

size = int(round((round(5 * math.sqrt(data.shape[0]))) ** (1. / 3.)))

###compute number of features
number_features = data.shape[1]

print(size)

som = Minisom3D.MiniSom3D(size, size, size, number_features, sigma=0.3, learning_rate=0.5)
# som = Minisom3D.MiniSom3D(size, size, size, number_features, sigma=0.3, learning_rate=0.5)
som.train(data, 100)

distance_map = som.distance_map().T

# som_shape = distance_map.shape

# Reshape the distance map to match the shape of the SOM
# distance_map = distance_map.reshape(som_shape)


# Use mlab.pipeline.volume to create a volume rendering of the distance map
# mlab.pipeline.volume(mlab.pipeline.scalar_field(distance_map))
# mlab.show()
# mlab.clf()
# mlab.volume_slice(distance_map)

# mlab.show()distance_map_shape = distance_map.shape


volume_slice_x = mlab.volume_slice(distance_map,plane_orientation='x_axes')
volume_slice_y = mlab.volume_slice(distance_map, plane_orientation='y_axes')
volume_slice_z = mlab.volume_slice(distance_map, plane_orientation='z_axes')



outline = mlab.outline(volume_slice_x)

colorbar = mlab.colorbar(object=volume_slice_x, title='Data values')

# Show the plot
mlab.show()
