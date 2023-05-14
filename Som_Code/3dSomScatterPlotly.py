import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import plotly.graph_objs as go

from matplotlib import pyplot as plt

import Minisom3D

from mpl_toolkits.mplot3d import Axes3D

# import warnings library
import warnings

# ignore all warnings
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
som.pca_weights_init(data)
som.train(data, 100)

distanceMap = som.distance_map().T

colors = []
X_coord = []
Y_coord = []
Z_coord = []

for i in range(0, size):
    for j in range(0, size):
        for k in range(0, size):
            X_coord.append(i)
            Y_coord.append(j)
            Z_coord.append(k)
            colors.append(distanceMap[i][j][k])

trace = go.Scatter3d(
    x=X_coord,
    y=Y_coord,
    z=Z_coord,
    mode='markers',
    marker=dict(
        size=10,
        color=colors,
        opacity=0.8,
        symbol='circle',
        colorscale='Viridis',
        showscale=True
    ),
    text=colors
)

layout = go.Layout(
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z'),
    )
)

# Create a figure with the trace and layout, and show the plot
fig1 = go.Figure(data=[trace], layout=layout)
fig1.update_layout(
    title='Distance map'
)
fig1.show()
