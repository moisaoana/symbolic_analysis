import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math
import Minisom3D

from mpl_toolkits.mplot3d import Axes3D

# import warnings library
import warnings

# ignore all warnings
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
som.train(data, 1000)

distanceMap = som.distance_map().T

Nx, Ny, Nz = size, size, size
X, Y, Z = np.arange(Nx), np.arange(Ny), -np.arange(Nz)

fig = plt.figure(figsize=(size, size))
ax = Axes3D(fig)

# Add x, y gridlines
ax.grid(b=True, color='red',
        linestyle='-.', linewidth=0.3,
        alpha=0.2)

kw = {
    'cmap': 'Blues'
}

for i in range(0, size):
    for j in range(0, size):
        for k in range(0, size):
            ax.scatter3D(X[i], Y[j], Z[k], c=str(distanceMap[i][j][k]), s=100, cmap=plt.get_cmap('jet'))

# Show Figure


plt.show()
