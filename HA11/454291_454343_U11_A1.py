import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting

# Define the datasets for each fruit class based on the image_b16a90.jpg
# Each row represents a fruit instance, and columns are [Color, Length-to-width ratio, Weight]

aepfel_data = np.array([
    [0.61, 1.01, 170],
    [0.59, 1.06, 180],
    [0.57, 1.03, 160],
    [0.64, 1.04, 190],
    [0.67, 1.08, 210],
    [0.62, 1.02, 165],
    [0.62, 1.07, 175],
    [0.55, 1.01, 150],
    [0.59, 1.03, 170],
    [0.66, 1.03, 185]
])

birnen_data = np.array([
    [0.54, 1.11, 130],
    [0.56, 1.06, 145],
    [0.51, 1.13, 120],
    [0.59, 1.13, 150],
    [0.59, 1.08, 135],
    [0.55, 1.12, 140],
    [0.53, 1.08, 130],
    [0.49, 1.10, 125],
    [0.56, 1.11, 140],
    [0.57, 1.07, 135]
])

orangen_data = np.array([
    [0.64, 0.98, 165],
    [0.66, 1.01, 180],
    [0.65, 0.97, 150],
    [0.69, 1.02, 190],
    [0.70, 1.05, 200],
    [0.62, 0.96, 145],
    [0.68, 1.03, 210],
    [0.69, 0.99, 155],
    [0.65, 0.95, 160],
    [0.67, 1.00, 175]
])

# a) Erstellen Sie aus den gegebenen Daten einen Datensatz X, der alle drei Klassen (Äpfel, Birnen, Orangen) enthält
X = np.vstack((aepfel_data, birnen_data, orangen_data))

# Erstellen Sie einen Label-Array Y
# 0 for Apples, 1 for Pears, 2 for Oranges
y = np.array([0]*len(aepfel_data) + [1]*len(birnen_data) + [2]*len(orangen_data))

print("Datensatz X (erste 5 Zeilen):\n", X[:5])
print("\nLabel-Array y:\n", y)

# Visualisieren Sie diesen Datensatz in einem 3D-Plot.
# Verwenden Sie dabei die Merkmale Farbe, Längenverhältnis und Gewicht als Achsen.
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d') # Create a 3D subplot

# Scatter plot for each class
ax.scatter(aepfel_data[:, 0], aepfel_data[:, 1], aepfel_data[:, 2], label='Äpfel', c='red', marker='o')
ax.scatter(birnen_data[:, 0], birnen_data[:, 1], birnen_data[:, 2], label='Birnen', c='green', marker='^')
ax.scatter(orangen_data[:, 0], orangen_data[:, 1], orangen_data[:, 2], label='Orangen', c='blue', marker='s')

# Set labels for the axes
ax.set_xlabel('Farbe F (in µm)')
ax.set_ylabel('Längenverhältnis L (in mm)')
ax.set_zlabel('Gewicht G (in g)')
ax.set_title('3D-Visualisierung des Frucht-Datensatzes')
ax.legend()
plt.grid(True)
plt.show()