import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Konfiguration
# =============================================================================
N_FEATURES = 4  # Anzahl der zu verwendenden Merkmale (1, 2, 3 oder 4)

CONFIG = {
    'fruits': {
        'aepfel': {
            'data': np.array([
                [0.61, 1.01, 170, 0.5], [0.59, 1.06, 180, 0.6], [0.57, 1.03, 160, 0.55],
                [0.64, 1.04, 190, 0.65], [0.67, 1.08, 210, 0.7], [0.62, 1.02, 165, 0.58],
                [0.62, 1.07, 175, 0.62], [0.55, 1.01, 155, 0.51], [0.59, 1.03, 170, 0.59],
                [0.66, 1.03, 185, 0.68]
            ]),
            'color': 'r',
            'label': 'Äpfel'
        },
        'birnen': {
            'data': np.array([
                [0.54, 1.11, 130, 0.8], [0.56, 1.06, 145, 0.85], [0.51, 1.13, 120, 0.78],
                [0.59, 1.13, 150, 0.88], [0.59, 1.08, 135, 0.82], [0.55, 1.12, 140, 0.86],
                [0.53, 1.08, 130, 0.81], [0.49, 1.10, 125, 0.75], [0.56, 1.11, 140, 0.84],
                [0.57, 1.07, 135, 0.83]
            ]),
            'color': 'g',
            'label': 'Birnen'
        },
        'orangen': {
            'data': np.array([
                [0.64, 0.98, 165, 0.3], [0.66, 1.01, 180, 0.35], [0.65, 0.97, 150, 0.28],
                [0.69, 1.02, 190, 0.38], [0.70, 1.05, 200, 0.4], [0.63, 0.96, 145, 0.25],
                [0.68, 1.03, 210, 0.42], [0.69, 0.99, 155, 0.32], [0.65, 0.95, 160, 0.29],
                [0.67, 1.00, 175, 0.33]
            ]),
            'color': 'b',
            'label': 'Orangen'
        }
    },
    'testdata': np.array([
        [0.60, 1.00, 175, 0.5], [0.58, 1.05, 145, 0.8],
        [0.66, 1.02, 185, 0.4], [0.52, 1.10, 130, 0.7]
    ])
}
# =============================================================================
# Datenaufbereitung
# =============================================================================
fruit_data = [fruit['data'][:, :N_FEATURES] for fruit in CONFIG['fruits'].values()]
X = np.concatenate(fruit_data)
y = np.array([i for i, fruit in enumerate(CONFIG['fruits'].values()) for _ in range(len(fruit['data']))])
testdaten = CONFIG['testdata'][:, :N_FEATURES]

# Plot der Daten
fig = plt.figure(figsize=(10, 8))
if N_FEATURES == 3:
    ax = fig.add_subplot(111, projection='3d')
    colors = [fruit['color'] for fruit in CONFIG['fruits'].values()]
    labels = [fruit['label'] for fruit in CONFIG['fruits'].values()]

    for i in range(len(CONFIG['fruits'])):
        ax.scatter(X[y == i, 0], X[y == i, 1], X[y == i, 2], c=colors[i], label=labels[i])

    ax.set_xlabel('Farbe F (in µm)')
    ax.set_ylabel('Längenverhältnis L (in mm)')
    ax.set_zlabel('Gewicht G (in g)')
    ax.set_title('3D-Darstellung der Obst-Daten')
    ax.legend()
    plt.show()
elif N_FEATURES == 2:
    ax = fig.add_subplot(111)
    colors = [fruit['color'] for fruit in CONFIG['fruits'].values()]
    labels = [fruit['label'] for fruit in CONFIG['fruits'].values()]

    for i in range(len(CONFIG['fruits'])):
        ax.scatter(X[y == i, 0], X[y == i, 1], c=colors[i], label=labels[i])

    ax.set_xlabel('Farbe F (in µm)')
    ax.set_ylabel('Längenverhältnis L (in mm)')
    ax.set_title('2D-Darstellung der Obst-Daten')
    ax.legend()
    plt.show()
elif N_FEATURES == 1:
    ax = fig.add_subplot(111)
    colors = [fruit['color'] for fruit in CONFIG['fruits'].values()]
    labels = [fruit['label'] for fruit in CONFIG['fruits'].values()]

    for i in range(len(CONFIG['fruits'])):
        ax.hist(X[y == i, 0], color=colors[i], label=labels[i], alpha=0.7)

    ax.set_xlabel('Farbe F (in µm)')
    ax.set_ylabel('Anzahl')
    ax.set_title('1D-Darstellung der Obst-Daten')
    ax.legend()
    plt.show()
else:
    print(f"{N_FEATURES}D-Darstellung nicht möglich.")


# =============================================================================
# Lineare Diskriminantenanalyse (LDA)
# =============================================================================
mean_vectors = []
for i in range(len(CONFIG['fruits'])):
    mean_vectors.append(np.mean(X[y == i], axis=0))

# Within-class scatter matrix S_W
S_W = np.zeros((N_FEATURES, N_FEATURES))
for cl, mv in zip(range(len(CONFIG['fruits'])), mean_vectors):
    class_sc_mat = np.zeros((N_FEATURES, N_FEATURES))
    for row in X[y == cl]:
        row, mv = row.reshape(N_FEATURES, 1), mv.reshape(N_FEATURES, 1)
        class_sc_mat += (row - mv).dot((row - mv).T)
    S_W += class_sc_mat

# Between-class scatter matrix S_B
overall_mean = np.mean(X, axis=0)
S_B = np.zeros((N_FEATURES, N_FEATURES))
for i, mean_vec in enumerate(mean_vectors):
    n = X[y == i, :].shape[0]
    mean_vec = mean_vec.reshape(N_FEATURES, 1)
    overall_mean = overall_mean.reshape(N_FEATURES, 1)
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

# Eigenwerte und Eigenvektoren berechnen
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# Eigenvektoren nach Eigenwerten sortieren
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

# Transformationsmatrix W
W = np.hstack([eig_pairs[i][1].reshape(N_FEATURES, 1) for i in range(min(N_FEATURES, 2))])

# Normierung der Eigenvektoren
for i in range(W.shape[1]):
    W[:, i] /= np.linalg.norm(W[:, i])

# Daten transformieren
X_lda = X.dot(W)

# 2D-Plot der transformierten Daten
plt.figure(figsize=(10, 8))
colors = [fruit['color'] for fruit in CONFIG['fruits'].values()]
labels = [fruit['label'] for fruit in CONFIG['fruits'].values()]
for i in range(len(CONFIG['fruits'])):
    if X_lda.shape[1] == 2:
        plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], c=colors[i], label=labels[i])
    else:
        plt.scatter(X_lda[y == i, 0], np.zeros_like(X_lda[y == i, 0]), c=colors[i], label=labels[i])


plt.xlabel('LDA Koordinate 1')
if X_lda.shape[1] == 2:
    plt.ylabel('LDA Koordinate 2')
plt.title('Darstellung der Daten nach LDA')
plt.legend()

# =============================================================================
# Klassifizierung
# =============================================================================
testdaten_lda = testdaten.dot(W)

print("Testdaten in der LDA-Projektion:")
print(testdaten_lda)

# Klassifizierung durch Abstand zu Klassenmittelpunkten im LDA-Raum
X_lda_means = np.array([np.mean(X_lda[y==i], axis=0) for i in range(len(CONFIG['fruits']))])
classified_labels = []
for point in testdaten_lda:
    distances = [np.linalg.norm(point - mean) for mean in X_lda_means]
    classified_labels.append(np.argmin(distances))

test_colors = [colors[l] for l in classified_labels]
test_labels_text = [labels[l] for l in classified_labels]

print("\nKlassifizierung der Testdaten:")
for i, label in enumerate(test_labels_text):
    print(f"Testpunkt {i+1}: {label}")

if testdaten_lda.shape[1] == 2:
    plt.scatter(testdaten_lda[:, 0], testdaten_lda[:, 1], c=test_colors, marker='x', s=100, label='Testdaten')
else:
    plt.scatter(testdaten_lda[:, 0], np.zeros_like(testdaten_lda[:, 0]), c=test_colors, marker='x', s=100, label='Testdaten')
plt.show()
