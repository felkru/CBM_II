import plotly.graph_objects as go
import numpy as np

# Define the function for the surface
def r(u, v):
    R = 1  # You can change this value
    x = R * np.sin(u)**2 * np.cos(v)
    y = R * np.sin(u)**2 * np.sin(v)
    z = R * np.sin(u) * np.cos(u)
    return x, y, z

# Create a grid of u and v values
u = np.linspace(0, np.pi, 50)  # u ranges from 0 to pi/2 based on sin(u)cos(u) and sin^2(u)
v = np.linspace(0, 2 * np.pi, 50)
u, v = np.meshgrid(u, v)

# Calculate the coordinates for the surface
x, y, z = r(u, v)

# Create the 3D plot
fig = go.Figure(data=[go.Surface(x=x, y=y, z=z)])

# Customize the layout
fig.update_layout(
    title='Surface plot of r(u,v)',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    )
)

# Show the plot
fig.show()