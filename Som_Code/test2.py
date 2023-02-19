import numpy as np
from mayavi import mlab
from mayavi.filters import extract_unstructured_grid

# Generate random data and markers
N = 1000000
x, y, z = np.random.randn(3, N)
markers = np.random.choice(['sphere', 'cube', 'cone'], N)

# Create figure and plot
fig = mlab.figure('1 Million Points 3D Scatter Plot')

# Create separate GlyphFactory instances for each marker type
for marker in set(markers):
    indices = np.where(markers == marker)[0]
    xyz = np.column_stack((x[indices], y[indices], z[indices]))
    factory = mlab.pipeline.glyph.GlyphFactory()
    factory.glyph_source = factory.glyph_dict[marker]()
    factory.glyph_source.glyph_source.radius = 0.1
    factory.glyph_source.glyph_source.height = 0.2
    factory.glyph_source.glyph_source.resolution = 32
    factory.add_input_data(xyz)

mlab.axes(xlabel='X-axis', ylabel='Y-axis', zlabel='Z-axis')

mlab.show()
