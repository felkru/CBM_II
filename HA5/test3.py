import plotly.graph_objects as go
import numpy as np

# Define the equation implicitly
# We are looking for points (x, y, z) such that:
# x^2 + y^2 + z^2 = R^2 - R^2 * z^2 / (x^2 + y^2 + z^2)

# Rearrange the equation to make it easier to plot
# (x^2 + y^2 + z^2) * (x^2 + y^2 + z^2) = (R^2 - R^2 * z^2 / (x^2 + y^2 + z^2)) * (x^2 + y^2 + z^2)
# (x^2 + y^2 + z^2)^2 = R^2 * (x^2 + y^2 + z^2) - R^2 * z^2

# Let r^2 = x^2 + y^2 + z^2. The equation becomes:
# r^4 = R^2 * r^2 - R^2 * z^2
# R^2 * z^2 = R^2 * r^2 - r^4
# z^2 = r^2 - r^4 / R^2
# z = ± sqrt(r^2 - r^4 / R^2)

# Let's consider plotting this in spherical coordinates for simplicity, where
# x = ρ * sin(φ) * cos(θ)
# y = ρ * sin(φ) * sin(θ)
# z = ρ * cos(φ)
# and ρ^2 = x^2 + y^2 + z^2

# Substituting into the rearranged equation:
# ρ^4 = R^2 * ρ^2 - R^2 * (ρ * cos(φ))^2
# ρ^4 = R^2 * ρ^2 - R^2 * ρ^2 * cos^2(φ)

# If ρ is not zero:
# ρ^2 = R^2 - R^2 * cos^2(φ)
# ρ^2 = R^2 * (1 - cos^2(φ))
# ρ^2 = R^2 * sin^2(φ)
# ρ = R * |sin(φ)|

# Since we typically use φ from 0 to π, sin(φ) is non-negative, so ρ = R * sin(φ).
# This is the equation of a sphere in spherical coordinates with radius R, centered at the origin.
# However, the original equation has a division by (x^2 + y^2 + z^2), which means the origin (0, 0, 0) is excluded.
# Let's re-examine the original equation:
# x^2 + y^2 + z^2 = R^2 - R^2 * z^2 / (x^2 + y^2 + z^2)

# If x^2 + y^2 + z^2 = R^2, then R^2 = R^2 - R^2 * z^2 / R^2, which simplifies to R^2 = R^2 - z^2. This implies z^2 = 0, so z = 0.
# This gives a circle in the xy-plane with radius R.

# If z = 0, the equation becomes x^2 + y^2 = R^2 - 0 = R^2. This is a circle in the xy-plane with radius R.

# Let's consider the case where x^2 + y^2 + z^2 is not R^2 and not 0.
# (x^2 + y^2 + z^2)^2 = R^2 * (x^2 + y^2 + z^2) - R^2 * z^2
# x^4 + y^4 + z^4 + 2x^2y^2 + 2x^2z^2 + 2y^2z^2 = R^2x^2 + R^2y^2 + R^2z^2 - R^2z^2
# x^4 + y^4 + z^4 + 2x^2y^2 + 2x^2z^2 + 2y^2z^2 = R^2x^2 + R^2y^2

# This equation is quite complex to plot directly in Cartesian coordinates.
# Let's go back to the spherical coordinate form: ρ = R * sin(φ).
# This represents a cardioid of revolution around the z-axis.

# Let's generate points based on ρ = R * sin(φ) and then convert to Cartesian coordinates.
R = 1  # You can change the value of R
theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, np.pi, 100)
theta, phi = np.meshgrid(theta, phi)

rho = R * np.sin(phi)

# Convert spherical coordinates to Cartesian coordinates
x = rho * np.sin(phi) * np.cos(theta)
y = rho * np.sin(phi) * np.sin(theta)
z = rho * np.cos(phi)

# Create the 3D plot
fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, surfacecolor=np.sin(theta), colorscale='Pinkyl')]) # Add color based on phi for visualization and use Viridis colormap

# Customize the layout
fig.update_layout(
    title='Surface plot of the set A',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

# Show the plot
fig.show()