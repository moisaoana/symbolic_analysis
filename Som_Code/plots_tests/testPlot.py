import plotly.graph_objs as go
import numpy as np

# Generate random data
N = 699999
x, y, z = np.random.randn(3, N)

# Create trace
trace = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=2,
        color=z,                # set color to z value
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)

# Create layout
layout = go.Layout(
    title='1 Million Points 3D Scatter Plot',
    scene=dict(
        xaxis=dict(title='X-axis'),
        yaxis=dict(title='Y-axis'),
        zaxis=dict(title='Z-axis'),
    )
)

# Create figure and plot
fig = go.Figure(data=[trace], layout=layout)
fig.show()


import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Create two sample figures with shapes
fig1 = go.Figure()
fig1.add_shape(type='rect', x0=0, y0=0, x1=1, y1=1, fillcolor='red', line_color='red')
fig1.add_shape(type='rect', x0=1, y0=1, x1=2, y1=2, fillcolor='blue', line_color='blue')

fig2 = go.Figure()
fig2.add_shape(type='rect', x0=0, y0=0, x1=1, y1=1, fillcolor='green', line_color='green')
fig2.add_shape(type='rect', x0=1, y0=1, x1=2, y1=2, fillcolor='yellow', line_color='yellow')

# Create a new figure with two subplots arranged in a 1x2 grid
fig = make_subplots(rows=1, cols=2)

# Add the shapes from fig1 to the first subplot
for shape in fig1.layout.shapes:
    fig.add_shape(shape, row=1, col=1)

# Add the shapes from fig2 to the second subplot
for shape in fig2.layout.shapes:
    fig.add_shape(shape, row=1, col=2)

# Update the layout
fig.update_layout(title='Multiple Figures with Shapes in One Plot')

# Show the plot
fig.show()
