
import matplotlib
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import math
import Minisom3D
from mpl_toolkits.mplot3d import Axes3D
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
som.train(data, 100)

distanceMap = som.distance_map().T

Nx, Ny, Nz = size, size, size
X, Y, Z = np.meshgrid(np.arange(Nx), np.arange(Ny), -np.arange(Nz))

# np.arrange(5) => [0,1,2,3,4]

kw = {
    'vmin': distanceMap.min(),
    'vmax': distanceMap.max(),
    'levels': np.linspace(distanceMap.min(), distanceMap.max(), 10),  # for color bar
    'cmap': 'Blues'
}

# Create a figure with 3D ax
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111, projection='3d')

print(distanceMap)
print("------------------------")

print(X[:, :, 0])
print(Y[:, :, 0])
print(distanceMap[:, :, 0])
# Plot contour surfaces
_ = ax.contourf(
    X[:, :, 0], Y[:, :, 0], distanceMap[:, :, 0],
    zdir='z', offset=0, **kw
)
_ = ax.contourf(
    X[0, :, :], distanceMap[0, :, :], Z[0, :, :],
    zdir='y', offset=0, **kw
)
C = ax.contourf(
    distanceMap[:, -1, :], Y[:, -1, :], Z[:, -1, :],
    zdir='x', offset=X.max(), **kw
)
# --


# Set limits of the plot from coord limits
xmin, xmax = X.min(), X.max()
ymin, ymax = Y.min(), Y.max()
zmin, zmax = Z.min(), Z.max()
ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

# Plot edges
edges_kw = dict(color='0.1', linewidth=1, zorder=1e3)
ax.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
ax.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

# Set labels and zticks
ax.set(
    xlabel='X',
    ylabel='Y',
    zlabel='Z',
    zticks=[0, -150, -300, -450],
)

# Colorbar
fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label='Name [units]')

# Show Figure
plt.show()