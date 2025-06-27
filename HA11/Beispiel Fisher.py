import numpy as np
import matplotlib.pyplot as plt

# --- Daten aus der Übungsangabe ---
# Trainingsdaten
aepfel = np.array([
    [0.61, 1.01, 170], [0.59, 1.06, 180], [0.57, 1.03, 160],
    [0.64, 1.04, 190], [0.67, 1.08, 210], [0.62, 1.02, 165],
    [0.62, 1.07, 175], [0.55, 1.01, 155], [0.59, 1.03, 170],
    [0.66, 1.03, 185]
])

birnen = np.array([
    [0.54, 1.11, 130], [0.56, 1.06, 145], [0.51, 1.13, 120],
    [0.59, 1.13, 150], [0.59, 1.08, 135], [0.55, 1.12, 140],
    [0.53, 1.08, 130], [0.49, 1.10, 125], [0.56, 1.11, 140],
    [0.57, 1.07, 135]
])

orangen = np.array([
    [0.64, 0.98, 165], [0.66, 1.01, 180], [0.65, 0.97, 150],
    [0.68, 1.02, 190], [0.70, 1.05, 200], [0.63, 0.96, 145],
    [0.68, 1.03, 210], [0.69, 0.99, 195], [0.62, 0.95, 160],
    [0.67, 1.00, 175]
])

# Testdaten
testdaten = np.array([
    [0.60, 1.00, 175],
    [0.58, 1.05, 145],
    [0.66, 1.02, 185],
    [0.52, 1.10, 130]
])


# --- Teil b: LDA Implementierung und Projektion ---

# 1. Daten kombinieren und Mittelwerte berechnen
X = np.concatenate((aepfel, birnen, orangen), axis=0)
d = X.shape[1] # Anzahl der Merkmale

class_means = {
    'Apfel': np.mean(aepfel, axis=0),
    'Birne': np.mean(birnen, axis=0),
    'Orange': np.mean(orangen, axis=0)
}
overall_mean = np.mean(X, axis=0)

# 2. Within-Class und Between-Class Streumatrizen berechnen
# Formeln aus dem Skript, Seite 18
S_W = np.zeros((d, d))
for i, data_class in enumerate([aepfel, birnen, orangen]):
    mean_vec = list(class_means.values())[i].reshape(d, 1)
    for row in data_class:
        row_vec = row.reshape(d, 1)
        S_W += (row_vec - mean_vec).dot((row_vec - mean_vec).T)

S_B = np.zeros((d, d))
for name, mean_vec in class_means.items():
    n = len({'Apfel': aepfel, 'Birne': birnen, 'Orange': orangen}[name])
    mean_vec = mean_vec.reshape(d, 1)
    overall_mean_vec = overall_mean.reshape(d, 1)
    S_B += n * (mean_vec - overall_mean_vec).dot((mean_vec - overall_mean_vec).T)

# 3. Eigenwertproblem lösen: S_W^-1 * S_B
S_W_inv = np.linalg.inv(S_W)
eigvals, eigvecs = np.linalg.eig(S_W_inv.dot(S_B))

# Eigenvektoren nach Eigenwerten sortieren
idx = eigvals.argsort()[::-1]
eigvecs = eigvecs[:, idx].real

# 4. Projektionsmatrix W aus den ersten K-1=2 Eigenvektoren erstellen
W = eigvecs[:, :2]

# 5. Trainingsdaten projizieren
X_lda = X.dot(W)

# Visualisierung der projizierten Trainingsdaten (Aufgabe 1b)
print("--- Ergebnis Aufgabe 1b: 2D-Projektion der Trainingsdaten ---")
plt.figure(figsize=(10, 7))
plt.scatter(X_lda[:10, 0], X_lda[:10, 1], c='red', label='Äpfel')
plt.scatter(X_lda[10:20, 0], X_lda[10:20, 1], c='blue', label='Birnen')
plt.scatter(X_lda[20:30, 0], X_lda[20:30, 1], c='orange', label='Orangen')
plt.title('LDA-Projektion der Trainingsdaten')
plt.xlabel('LDA Komponente 1')
plt.ylabel('LDA Komponente 2')
plt.legend()
plt.grid(True)
plt.show()


# --- Teil c: Klassifizierung der Testdaten ---

# 1. Testdaten projizieren
testdaten_lda = testdaten.dot(W)

# 2. Klassenmittelwerte projizieren, um Zentren für die Klassifizierung zu erhalten
projected_means = {name: mean.dot(W) for name, mean in class_means.items()}

# 3. Testpunkte klassifizieren (basierend auf der Distanz zum nächsten Klassen-Zentrum)
classified_as = []
class_names = list(class_means.keys())
for point in testdaten_lda:
    distances = [np.linalg.norm(point - p_mean) for p_mean in projected_means.values()]
    closest_class_idx = np.argmin(distances)
    classified_as.append(class_names[closest_class_idx])

# Klassifizierungsergebnisse ausgeben
print("\n--- Ergebnis Aufgabe 1c: Klassifizierung der Testdaten ---")
for i, p_class in enumerate(classified_as):
    print(f"Testdatenpunkt {i+1} {testdaten[i]} wird als '{p_class}' klassifiziert.")

# Visualisierung der Trainings- und Testdaten (Aufgabe 1c)
plt.figure(figsize=(12, 8))
# Trainingsdaten
plt.scatter(X_lda[:10, 0], X_lda[:10, 1], c='red', label='Äpfel (Training)', alpha=0.6)
plt.scatter(X_lda[10:20, 0], X_lda[10:20, 1], c='blue', label='Birnen (Training)', alpha=0.6)
plt.scatter(X_lda[20:30, 0], X_lda[20:30, 1], c='orange', label='Orangen (Training)', alpha=0.6)

# Testdaten
colors = {'Apfel': 'red', 'Birne': 'blue', 'Orange': 'orange'}
for i, point_lda in enumerate(testdaten_lda):
    class_label = classified_as[i]
    plt.scatter(point_lda[0], point_lda[1], c=colors[class_label], marker='X', s=200,
                edgecolor='black', linewidth=1.5, label=f'Testpunkt {i+1} -> {class_label}')

plt.title('Klassifizierung von Testdaten mittels LDA')
plt.xlabel('LDA Komponente 1')
plt.ylabel('LDA Komponente 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()