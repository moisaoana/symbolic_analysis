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
from som_with_print import MiniSomWithPrint

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

size = int(math.sqrt(int(5 * math.sqrt(data.shape[0]))))

number_features = data.shape[1]
print('number features', number_features)

print('nr of points', data.shape[0])
print('number of nodes ', size)

print(data.shape)
som = MySom(size, size, number_features, sigma=2.0, learning_rate=0.2)
som.pca_init(data)
som.train(data, 100)


# print(som.distance_map())

markers = ['o', 's', 'D']
colors = ['r', 'g', 'b']

plt.figure(figsize=(size, size))
plt.pcolor(som.distance_map().T)

for cnt, xx in enumerate(data):
    w = som.find_BMU(xx)
    plt.plot(w[0] + .5, w[1] + .5, markers[labels[cnt]], markersize=12, markerfacecolor=colors[labels[cnt]],
             markeredgecolor='k')

plt.colorbar()
plt.title('MySom')

plt.show()

som2 = MiniSom(size, size, number_features, sigma=2.0, learning_rate=0.2)
som2.pca_weights_init(data)
som2.train(data, 100)  # trains the SOM with 100 iterations

plt.figure(figsize=(16, 4))
plt.pcolor(som2.distance_map().T)


for cnt, xx in enumerate(data):
    w = som2.winner(xx)
    plt.plot(w[0] + .5, w[1] + .5, markers[labels[cnt]], markersize=12, markerfacecolor=colors[labels[cnt]],
             markeredgecolor='k')

plt.colorbar()
plt.title('MiniSom')

plt.show()