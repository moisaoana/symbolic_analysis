import inline as inline
import matplotlib
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns  # data visualization library
import matplotlib.pyplot as plt
from minisom import MiniSom
import math

# import warnings library
import warnings

# ignore all warnings
from som_implementationV2 import MySom

warnings.filterwarnings('ignore')

############## Data Content ###################
# ID number
# Diagnosis (M = malignant, B = benign)
# radius (mean of distances from center to points on the perimeter)
# texture (standard deviation of gray-scale values)
# perimeter
# area
# smoothness (local variation in radius lengths)
# compactness (perimeter^2 / area - 1.0)
# concavity (severity of concave portions of the contour)
# concave points (number of concave portions of the contour)
# symmetry
# fractal dimension ("coastline approximation" - 1)
# The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.
# All feature values are recoded with four significant digits.
# Missing attribute values: none
# Class distribution: 357 benign, 212 malignant

#print(np.sqrt(np.square(np.array([4, 5, 6]) - np.array([1, 2, 3]))).sum(axis=0))

### Read and Analyse Data
samples = pd.read_csv('./breast_cancer_data.csv')  # returns data frame

# Before making anything like feature selection,feature extraction and classification, firstly we start with basic data analysis. Lets look at features of data.
#print(samples.head())  # head method show only first 5 rows

# feature names as a list
col = samples.columns  # .columns gives columns names in data
#print(col)

# Remember:
# 1) There is an id that cannot be used for classification
# 2) Diagnosis is our class label

y = samples.diagnosis

B, M = y.value_counts()
#print('Number of Benign: ', B)
#print('Number of Malignant : ', M)

list = ['id', 'diagnosis']
x = samples.drop(list, axis=1)

#print(x)

### transform to numpy matrix
data = x.to_numpy()

### compute size of map
size = int(math.sqrt(int(5 * math.sqrt(data.shape[0]))))

###compute number of features
number_features = data.shape[1]

#print(size)
#print(number_features)
#print(data)

print(data.shape)
som = MySom(size, size, number_features, sigma=2.0, learning_rate=0.5)
som.pca_init(data)
som.train(data, 1000)


print(size)

plt.figure(figsize=(size, size))
plt.pcolor(som.distance_map().T)

#print(som.distance_map().T)


target = samples['diagnosis'].values
#print target

t = np.zeros(len(target), dtype=int)

t[target == 'M'] = 0
t[target == 'B'] = 1

print(t)

#print(t)

labels = samples['diagnosis']
c = labels.astype('category')
labels = c.cat.codes


markers = ['o', 's']
colors = ['r', 'g']
for cnt, xx in enumerate(data):
    w = som.find_BMU(xx)
    plt.plot(w[0] + .5, w[1] + .5, markers[labels[cnt]], markersize=12, markerfacecolor=colors[labels[cnt]],
             markeredgecolor='k')

plt.colorbar()
plt.title('MySom')

plt.show()



