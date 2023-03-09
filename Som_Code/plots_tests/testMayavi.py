import numpy as np
from mayavi import mlab

# Create a random 3D dataset with 1 million points
n_points = 1000000
x, y, z = np.random.random((3, n_points))

# Define a list of markers and colors to use
markers = ['2ddiamond', '2dcircle', '2dtriangle']
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

# Assign a marker and color to each point randomly
marker_indices = np.random.randint(len(markers), size=n_points)
color_indices = np.random.randint(len(colors), size=n_points)

# Create the Mayavi scene and add the data
fig = mlab.figure()
for i, (marker, color) in enumerate(zip(markers, colors)):
    x_m = x[(marker_indices == i) & (color_indices == i)]
    y_m = y[(marker_indices == i) & (color_indices == i)]
    z_m = z[(marker_indices == i) & (color_indices == i)]
    points = mlab.points3d(x_m, y_m, z_m, mode=marker, color=color, scale_factor=0.1, opacity=0.5)

# Add axis labels and show the plot
mlab.xlabel('X')
mlab.ylabel('Y')
mlab.zlabel('Z')
mlab.show()
