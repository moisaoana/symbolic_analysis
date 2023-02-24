
from mpl_toolkits.mplot3d import Axes3D
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from mpl_toolkits import mplot3d
import math
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
from matplotlib import pyplot as plt
from PIL import Image
import io
import matplotlib
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

import Minisom3D

from mpl_toolkits.mplot3d import Axes3D

# import warnings library
import warnings

# ignore all warnings
from som_implementation_3D import MySom3D
from utils import Utils

warnings.filterwarnings('ignore')
matplotlib.use('Qt5Agg')

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

# som = Minisom3D.MiniSom3D(size, size, size, number_features, sigma=3.0, learning_rate=0.5)
som = MySom3D(size, size, size, number_features, sigma=3.0, learning_rate=0.5)
som.train(data, 100)

distanceMap = som.distance_map().T

labelsText = samples['diagnosis']
labels = samples['diagnosis']
c = labels.astype('category')
labels = c.cat.codes

markers = ['circle', 'square']
colors = ['red', 'green']

BMU_X1 = []
BMU_Y1 = []
BMU_Z1 = []
C1 = []
M1 = []
L1 = []

for cnt, xx in enumerate(data):
    w = som.find_BMU(xx)
    BMU_X1.append(w[0])
    BMU_Y1.append(w[1])
    BMU_Z1.append(w[2])
    C1.append(colors[labels[cnt]])
    M1.append(markers[labels[cnt]])
    L1.append(labelsText[cnt])

trace = go.Scatter3d(
    x=BMU_X1,
    y=BMU_Y1,
    z=BMU_Z1,
    mode='markers',
    marker=dict(
        size=5,
        color=C1,
        opacity=0.8,
        symbol=M1
    ),
    text=L1,
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
    title='Clusters given by labels'
)
fig1.show()

# ----------------------------------------------

threshold = som.find_threshold(data)
no_clusters, bmu_array, samples_with_clusters_array = som.find_clusters_with_min_dist(data, 0.5, threshold)
samples_with_symbols_array = Utils.assign_symbols(samples_with_clusters_array)
print(samples_with_symbols_array)

markers_and_colors = Utils.assign_markers_and_colors(no_clusters)
print(no_clusters)
print("-------------------------------------")

BMU_X = []
BMU_Y = []
BMU_Z = []
M = []
C = []

for cnt, xx in enumerate(data):
    w = som.find_BMU(xx)
    # print('W ', w)
    cluster = 0
    for bmu in bmu_array:
        if w == bmu[0]:
            cluster = bmu[1]
            break
    marker = '_'
    color = 'y'
    for x in markers_and_colors:
        if x[0] == cluster:
            marker = x[1]
            color = x[2]
    BMU_X.append(w[0])
    BMU_Y.append(w[1])
    BMU_Z.append(w[2])
    M.append(marker)
    C.append(color)

print(M)

trace = go.Scatter3d(
    x=BMU_X,
    y=BMU_Y,
    z=BMU_Z,
    mode='markers',
    marker=dict(
        size=5,
        color=C,
        opacity=0.8,
        symbol=M
    )
)

layout = go.Layout(
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z'),
    )
)

# Create a figure with the trace and layout, and show the plot
fig = go.Figure(data=[trace], layout=layout)
fig.update_layout(
    title='Clusters given by som'
)

fig.show()

colors_sequence = Utils.get_rgb_colors_array(data, som)

fig2 = go.Figure()

# Add a rectangle trace for each segment of the barcode
for i in range(len(colors_sequence)):
    fig2.add_shape(
        type='rect',
        x0=i,
        y0=0,
        x1=i + 1,
        y1=3,
        fillcolor=colors_sequence[i],
        line=dict(width=0),
    )

# Update the layout
fig2.update_layout(
    xaxis=dict(range=[0, len(colors_sequence)], title='time', showticklabels=False),
    yaxis=dict(title='Trial no', range=[0, 3]),
    width=1200,
)

# Show the plot
fig2.show()
"""
fig3 = make_subplots(rows=2, cols=1)

for shape in fig2.layout.shapes:
    fig3.add_shape(shape, row=1, col=1)

for shape in fig2.layout.shapes:
    fig3.add_shape(shape, row=2, col=1)


# Update the layout for the first subplot
fig3.update_xaxes(title_text="X Axis Title", range=[0, len(colors_sequence)], row=1, col=1)
fig3.update_yaxes(title_text="Trial 1", range=[0, 3], row=1, col=1)

# Update the layout for the second subplot
fig3.update_xaxes(title_text="X Axis Title", range=[0, len(colors_sequence)], row=2, col=1)
fig3.update_yaxes(title_text="Trial 2", range=[0, 3], row=2, col=1)
fig3.update_layout(title='Color sequences for each trial')

fig3.show()
"""

"""
fig3 = plt.figure(figsize=(10, 10))
rows = 10
columns = 1
image_bytes = pio.to_image(fig2, format='png')
image = Image.open(io.BytesIO(image_bytes))
fig3.add_subplot(rows, columns, 1)
plt.imshow(image)
plt.axis('off')
plt.title("Trial 1")

fig3.add_subplot(rows, columns, 2)
plt.imshow(image)
plt.axis('off')
plt.title("Trial 2")

fig3.add_subplot(rows, columns, 3)
plt.imshow(image)
plt.axis('off')
plt.title("Trial 3")

fig3.add_subplot(rows, columns, 4)
plt.imshow(image)
plt.axis('off')
plt.title("Trial 4")

fig3.add_subplot(rows, columns, 5)
plt.imshow(image)
plt.axis('off')
plt.title("Trial 5")

fig3.add_subplot(rows, columns, 6)
plt.imshow(image)
plt.axis('off')
plt.title("Trial 6")

fig3.add_subplot(rows, columns, 7)
plt.imshow(image)
plt.axis('off')
plt.title("Trial 7")

fig3.add_subplot(rows, columns, 8)
plt.imshow(image)
plt.axis('off')
plt.title("Trial 8")


fig3.add_subplot(rows, columns, 9)
plt.imshow(image)
plt.axis('off')
plt.title("Trial 9")


fig3.add_subplot(rows, columns, 10)
plt.imshow(image)
plt.axis('off')
plt.title("Trial 10")




fig3.show()
"""
c = canvas.Canvas("trials.pdf", pagesize=A4)
y = 1
image_bytes = pio.to_image(fig2, format='png')
image = Image.open(io.BytesIO(image_bytes))
c.drawImage(ImageReader(image), 50, y, width=500, preserveAspectRatio=True)
y -= image.height
c.drawImage(ImageReader(image), 50, y, width=500, preserveAspectRatio=True)
c.save()