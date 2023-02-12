import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns  # data visualization library
import matplotlib.pyplot as plt
from minisom import MiniSom
import math

# import warnings library
import warnings

# ignore all warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from som_implementationV2 import MySom

warnings.filterwarnings('ignore')

samples = pd.read_csv('./Iris.csv')  # returns data frame

col = samples.columns  # .columns gives columns names in data

labels = samples['Species']
c = labels.astype('category')
labels = c.cat.codes

y = samples.Species

setosa, versicolor, virginica = y.value_counts()
print('Number of iris-setosa: ', setosa)
print('Number of iris-versicolor : ', versicolor)
print('Number of iris-virginica : ', virginica)



list = ['Id', 'Species']
x = samples.drop(list, axis=1)

dataset = pd.DataFrame(MinMaxScaler(feature_range=(0, 1)).fit_transform(x.values))

data = dataset.to_numpy()
# data, labels = shuffle(data, labels)

size = int(math.sqrt(int(5 * math.sqrt(data.shape[0]))))


SHAPE = (20, 20)
EPOCHS = 100
SIGMA = 3
LR = 0.5



number_features = data.shape[1]
print('number features', number_features)

print('nr of points', data.shape[0])
print('number of nodes ', size)

print("data_shape", data.shape)
som = MySom(SHAPE[0], SHAPE[1], number_features, sigma=SIGMA, learning_rate=LR)
# som.pca_init(data)
#som.pca_init(data)
som.train(data, EPOCHS)


# print(som.distance_map())

markers = ['o', 's', 'D']
colors = ['r', 'g', 'b']

plt.figure(figsize=SHAPE)
plt.pcolor(som.distance_map().T)

for cnt, xx in enumerate(data):
    w = som.find_BMU(xx)
    plt.plot(w[0] + .5, w[1] + .5, markers[labels[cnt]], markersize=12, markerfacecolor=colors[labels[cnt]],
             markeredgecolor='k')

plt.colorbar()
plt.title('MySom')

plt.show()

som2 = MiniSom(SHAPE[0], SHAPE[1], number_features, sigma=SIGMA, learning_rate=LR)
som2.pca_weights_init(data)
som2.train(data, EPOCHS)  # trains the SOM with 100 iterations

plt.figure(figsize=SHAPE)
plt.pcolor(som2.distance_map().T)


for cnt, xx in enumerate(data):
    w = som2.winner(xx)
    plt.plot(w[0] + .5, w[1] + .5, markers[labels[cnt]], markersize=12, markerfacecolor=colors[labels[cnt]],
             markeredgecolor='k')

plt.colorbar()
plt.title('MiniSom')

plt.show()