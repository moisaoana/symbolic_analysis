import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import Minisom3D
from EEG_DataProcessor import EEG_DataProcessor
from RawEEGSignalParser import RawEEGSignalParser
from som_implementation_3D import MySom3D
import plotly.graph_objs as go

from tins_dots.Plots_Generator import PlotsGenerator
from utils import Utils

DATASET_PATH = "./data/Dots_30_001/"
parser = RawEEGSignalParser(DATASET_PATH)

full_data = parser.load_all_channels()

event_timestamps = parser.load_event_timestamps()
event_codes = parser.load_event_codes()

eegDataProcessor = EEG_DataProcessor(DATASET_PATH, full_data, event_timestamps, event_codes)
eegDataProcessor.create_trials(save=False)
eegDataProcessor.link_trials(save=False)

size = 4
no_features = 32
no_iterations = 1

print(eegDataProcessor.processed_data.shape)

print(len(eegDataProcessor.trials))
#for cnt, trial in enumerate(eegDataProcessor.trials):
#    print("Trial " + str(cnt))
 #   print('size '+str(len(trial.trial_data)))

som = MySom3D(size, size, size, no_features, sigma=0.3, learning_rate=0.5)
som.train(eegDataProcessor.processed_data, no_iterations)

distance_map = som.distance_map().T

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
            colors.append(distance_map[i][j][k])

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


#PlotsGenerator.generateColorSeguenceForAllTrials(len(eegDataProcessor.trials), eegDataProcessor.trials, som)

#PlotsGenerator.generateColorSeguenceForAllTrialsInPDF(len(eegDataProcessor.trials), eegDataProcessor.trials, som)

#figure_data_array = PlotsGenerator.getTrialSequencesArray(eegDataProcessor.trials, som)
#fig, axs = PlotsGenerator.generateGridWithColorSequences(figure_data_array[0:7], n_rows=len(figure_data_array[0:7]), n_cols=1)
#plt.show()
#PlotsGenerator.groupByStimulusVisibility(figure_data_array)
#PlotsGenerator.groupByResponse(figure_data_array)
#PlotsGenerator.groupByStimulus(figure_data_array)

print("HERE")
threshold = som.find_threshold(eegDataProcessor.processed_data)
print('Max dist ', threshold)
no_clusters, bmu_array, samples_with_clusters_array, samples_nparray, clusters_nparray = som.find_clusters_with_min_dist(eegDataProcessor.processed_data,
                                                                                      0.3, threshold)
print('No clusters ', no_clusters)
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
BMU = []

for cnt, xx in enumerate(eegDataProcessor.processed_data):
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
    notInBMU = True
    for x in BMU:
        if w == x:
            notInBMU = False
    if notInBMU:
        BMU.append(w)
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

print(markers_and_colors)
figure_data_array = PlotsGenerator.getTrialSequencesArrayUsingClustersV2(eegDataProcessor.trials, markers_and_colors,samples_with_clusters_array)
PlotsGenerator.groupByStimulusVisibility(figure_data_array)
PlotsGenerator.groupByResponse(figure_data_array)
PlotsGenerator.groupByStimulus(figure_data_array)

