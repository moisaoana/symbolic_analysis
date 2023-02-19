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
