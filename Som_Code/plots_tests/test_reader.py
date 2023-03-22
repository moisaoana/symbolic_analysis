import numpy as np

from readerUtils import ReaderUtils


samples_with_clusters = [
        (np.array([1,2,3]),0),
        (np.array([5,6,7]),1),
        (np.array([8,9,10]),2)
    ]

markers_and_colors = [
    (0,'aa','bb',[1,1,1]),
    (1,'yy','ccc',[1,0,1])
]

ReaderUtils.writeSamplesWithClusters(samples_with_clusters)
ReaderUtils.writeMarkersAndColors(markers_and_colors)

sampless = ReaderUtils.readSamplesWithClusters()
print(type(sampless))
for s in sampless:
    print(s)
    print(type(s[0]))
    print(type(s[1]))

markerss=ReaderUtils.readMarkersAndColors()
print(markerss)

"""
distance_map = [[[0.10661264,0.15743205,0.27264908,0.16293388],
  [0.18688975,0.5149894,0.43563829,0.33078474],
  [0.19218532,0.51349804,0.83168095,0.21736813],
  [0.12999019,0.23073743,0.21747862,0.15706258]],
 [[0.14716546,0.35097595,0.37951833,0.16767611],
  [0.31927563,0.464252,0.49991756,0.44668011],
  [0.46642803,0.72111441,1.,0.34654221],
  [0.1745068,0.33874931,0.34561134,0.29605253]],
 [[0.16587669,0.2709358,0.22480205,0.13476979],
  [0.27674655,0.5388106,0.4084952,0.37682183],
  [0.31526582,0.40570153,0.54166514,0.45494698],
  [0.25339959,0.29706794,0.30620627,0.28712388]],
 [[0.13742905,0.19154552,0.14557962,0.09029158],
  [0.1450068,0.26586202,0.36377306,0.26756931],
  [0.17686014,0.26459412,0.46547696,0.20367847],
  [0.15436409,0.19822016,0.23063479,0.13743822]]]

print(distance_map)
with open('array.bin', 'wb') as f:
    np.save(f, distance_map)

with open('array.bin', 'rb') as f:
    arr = np.load(f)
print(arr)
"""