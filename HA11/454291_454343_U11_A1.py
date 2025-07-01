import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Konfiguration
# =============================================================================

# original dataset from the task
CONFIG = {
    'fruits': {
        'aepfel': {
            'data': np.array([
                [0.61, 1.01, 170], [0.59, 1.06, 180], [0.57, 1.03, 160],
                [0.64, 1.04, 190], [0.67, 1.08, 210], [0.62, 1.02, 165],
                [0.62, 1.07, 175], [0.55, 1.01, 155], [0.59, 1.03, 170],
                [0.66, 1.03, 185]
            ]),
            'color': 'r',
            'label': 'Äpfel'
        },
        'birnen': {
            'data': np.array([
                [0.54, 1.11, 130], [0.56, 1.06, 145], [0.51, 1.13, 120],
                [0.59, 1.13, 150], [0.59, 1.08, 135], [0.55, 1.12, 140],
                [0.53, 1.08, 130], [0.49, 1.10, 125], [0.56, 1.11, 140],
                [0.57, 1.07, 135]
            ]),
            'color': 'g',
            'label': 'Birnen'
        },
        'orangen': {
            'data': np.array([
                [0.64, 0.98, 165], [0.66, 1.01, 180], [0.65, 0.97, 150],
                [0.69, 1.02, 190], [0.70, 1.05, 200], [0.63, 0.96, 145],
                [0.68, 1.03, 210], [0.69, 0.99, 155], [0.65, 0.95, 160],
                [0.67, 1.00, 175]
            ]),
            'color': 'b',
            'label': 'Orangen'
        }
    },
    'feature_names': ['Farbe F (in µm)', 'Längenverhältnis L (in mm)', 'Gewicht G (in g)'],
    # Hinzugefügte Testdatensätze
    'testdata_sets': {
        'Test Set 1 (Original)': np.array([
            [0.60, 1.00, 175], [0.58, 1.05, 145],
            [0.66, 1.02, 185], [0.52, 1.10, 130]
        ]),
        # 'Test Set 2': np.array([
        #     [0.68, 1.04, 195], [0.56, 1.10, 138], [0.60, 1.04, 177]
        # ]),
        # 'Test Set 3': np.array([
        #     [0.50, 1.09, 122], [0.65, 0.99, 170]
        # ]),
        # 'Test Set 4': np.array([
        #     [0.63, 1.03, 180], [0.58, 1.09, 148], [0.71, 1.06, 205]
        # ])
    }
}

# new test dataset for 6 dimensions
# CONFIG = {
#     'fruits': {
#         'aepfel': {
#             'data': np.array([
#                 # Farbe, Längenverh., Gewicht, Festigkeit, Zuckergehalt, Säuregehalt
#                 [0.61, 1.01, 170, 18.5, 12.5, 3.50],
#                 [0.59, 1.06, 180, 19.2, 13.1, 3.45],
#                 [0.57, 1.03, 160, 17.8, 11.9, 3.55],
#                 [0.64, 1.04, 190, 20.1, 14.0, 3.30],
#                 [0.67, 1.08, 210, 21.0, 15.2, 3.25],
#                 [0.62, 1.02, 165, 18.0, 12.8, 3.48],
#                 [0.62, 1.07, 175, 19.5, 13.5, 3.40],
#                 [0.55, 1.01, 155, 17.2, 11.5, 3.60],
#                 [0.59, 1.03, 170, 18.8, 13.0, 3.52],
#                 [0.66, 1.03, 185, 20.5, 14.5, 3.35]
#             ]),
#             'color': 'r',
#             'label': 'Äpfel'
#         },
#         'birnen': {
#             'data': np.array([
#                 # Farbe, Längenverh., Gewicht, Festigkeit, Zuckergehalt, Säuregehalt
#                 [0.54, 1.11, 130, 14.0, 15.0, 3.90],
#                 [0.56, 1.06, 145, 15.5, 16.2, 3.85],
#                 [0.51, 1.13, 120, 12.8, 14.5, 4.10],
#                 [0.59, 1.13, 150, 16.0, 17.0, 3.75],
#                 [0.59, 1.08, 135, 14.5, 15.8, 3.95],
#                 [0.55, 1.12, 140, 15.0, 16.5, 3.80],
#                 [0.53, 1.08, 130, 13.5, 14.8, 4.05],
#                 [0.49, 1.10, 125, 12.0, 14.0, 4.15],
#                 [0.56, 1.11, 140, 15.2, 16.0, 3.88],
#                 [0.57, 1.07, 135, 14.8, 15.5, 3.92]
#             ]),
#             'color': 'g',
#             'label': 'Birnen'
#         },
#         'orangen': {
#             'data': np.array([
#                 # Farbe, Längenverh., Gewicht, Festigkeit, Zuckergehalt, Säuregehalt
#                 [0.64, 0.98, 165, 11.0, 10.5, 3.40],
#                 [0.66, 1.01, 180, 12.2, 11.8, 3.20],
#                 [0.65, 0.97, 150, 10.5, 9.8, 3.50],
#                 [0.69, 1.02, 190, 13.0, 12.5, 3.10],
#                 [0.70, 1.05, 200, 14.0, 13.0, 3.05],
#                 [0.63, 0.96, 145, 10.0, 9.5, 3.55],
#                 [0.68, 1.03, 210, 13.5, 12.8, 3.15],
#                 [0.69, 0.99, 155, 11.5, 10.0, 3.45],
#                 [0.65, 0.95, 160, 10.8, 10.2, 3.60],
#                 [0.67, 1.00, 175, 12.0, 11.5, 3.25]
#             ]),
#             'color': 'b',
#             'label': 'Orangen'
#         }
#     },
#     'feature_names': [
#         'Farbe F (in µm)', 
#         'Längenverhältnis L', 
#         'Gewicht G (in g)',
#         'Festigkeit (N)',
#         'Zuckergehalt (°Bx)',
#         'Säuregehalt (pH)'
#     ],
#     'testdata_sets': {
#         'Test Set 1': np.array([
#             [0.62, 1.02, 175, 19.0, 13.0, 3.6], # Eher Apfel
#             [0.55, 1.10, 140, 15.0, 16.0, 4.0], # Eher Birne
#             [0.68, 0.99, 180, 10.0, 11.5, 3.5]  # Eher Orange
#         ]),
#         'Test Set 2': np.array([
#             [0.60, 1.05, 160, 17.0, 14.0, 3.8], # Apfel/Birne?
#             [0.65, 1.00, 170, 12.0, 12.0, 3.4]  # Apfel/Orange?
#         ])
#     }
# }


# =============================================================================
# Datenaufbereitung
# =============================================================================
# a) Erstellen Sie aus den gegebenen Daten einen Datensatz X, der alle drei Klassen
#    enthält, erstellen Sie einen Label-Array y und visualisieren
#    Sie diesen Datensatz.

# Automatische Ermittlung der Anzahl der Merkmale und Klassen
N_FEATURES = next(iter(CONFIG['fruits'].values()))['data'].shape[1]
N_CLASSES = len(CONFIG['fruits'])

# Erstellen des Datensatzes X und des Label-Arrays y
fruit_data = [fruit['data'][:, :N_FEATURES] for fruit in CONFIG['fruits'].values()]
X = np.concatenate(fruit_data)
y = np.array([i for i, fruit in enumerate(CONFIG['fruits'].values()) for _ in range(len(fruit['data']))])

# Plot der Rohdaten (angepasst für variable Dimensionen)
fig = plt.figure(figsize=(10, 8))
feature_names = CONFIG.get('feature_names', [f'Merkmal {i+1}' for i in range(N_FEATURES)])
colors = [fruit['color'] for fruit in CONFIG['fruits'].values()]
labels = [fruit['label'] for fruit in CONFIG['fruits'].values()]

if N_FEATURES == 3:
    ax = fig.add_subplot(111, projection='3d')
    for i in range(N_CLASSES):
        ax.scatter(X[y == i, 0], X[y == i, 1], X[y == i, 2], c=colors[i], label=labels[i])
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel(feature_names[2])
    ax.set_title(f'{N_FEATURES}D-Darstellung der Obst-Rohdaten')
    ax.legend()
    plt.show()
elif N_FEATURES == 2:
    ax = fig.add_subplot(111)
    for i in range(N_CLASSES):
        ax.scatter(X[y == i, 0], X[y == i, 1], c=colors[i], label=labels[i])
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title('2D-Darstellung der Obst-Rohdaten')
    ax.legend()
    plt.show()
elif N_FEATURES == 1:
    ax = fig.add_subplot(111)
    for i in range(N_CLASSES):
        ax.hist(X[y == i, 0], color=colors[i], label=labels[i], alpha=0.7, bins=10)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel('Anzahl')
    ax.set_title('1D-Darstellung der Obst-Rohdaten')
    ax.legend()
    plt.show()
else:
    print(f"\nVisualisierung der Rohdaten für {N_FEATURES} Dimensionen wird nicht direkt unterstützt. Führe LDA aus, um die Dimensionen zu reduzieren.")
    plt.close(fig) # Schließt die leere Figur

# =============================================================================
# Lineare Diskriminantenanalyse (LDA)
# =============================================================================
# b) Führen Sie eine lineare Diskriminantenanalyse (LDA) durch und stellen Sie
#    die berechnete Projektion dar.

mean_vectors = []
for i in range(N_CLASSES):
    mean_vectors.append(np.mean(X[y == i], axis=0))

# Within-class scatter matrix S_W
S_W = np.zeros((N_FEATURES, N_FEATURES))
for cl, mv in zip(range(N_CLASSES), mean_vectors):
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
# Hinweis: np.linalg.inv(S_W) kann numerisch instabil sein. Für robuste Implementierungen
# wird oft die Pseudoinverse (np.linalg.pinv) verwendet.
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eig_vals = eig_vals.real
eig_vecs = eig_vecs.real

# Eigenvektoren nach Eigenwerten sortieren
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

# Transformationsmatrix W (Projektion auf maximal 2 Dimensionen für die Visualisierung)
# Die Anzahl der Diskriminanten ist min(N_FEATURES, N_CLASSES - 1)
n_discriminants = min(N_FEATURES, N_CLASSES - 1)
W = np.hstack([eig_pairs[i][1].reshape(N_FEATURES, 1) for i in range(n_discriminants)])

# Normierung der Eigenvektoren
for i in range(W.shape[1]):
    W[:, i] /= np.linalg.norm(W[:, i])

print("Die Transformations-matrix ist:\n", W)

# Daten transformieren
X_lda = X.dot(W)

# 2D-Plot der transformierten Daten
plt.figure(figsize=(10, 8))
for i in range(N_CLASSES):
    if X_lda.shape[1] == 2:
        plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], c=colors[i], label=labels[i])
    else: # Falls nur 1D-Projektion möglich ist
        plt.scatter(X_lda[y == i, 0], np.zeros_like(X_lda[y == i, 0]) + i*0.1, c=colors[i], label=labels[i])

plt.xlabel('LDA Koordinate 1')
if X_lda.shape[1] == 2:
    plt.ylabel('LDA Koordinate 2')
plt.title('Darstellung der Daten nach LDA')
plt.legend()

# =============================================================================
# Klassifizierung
# =============================================================================
# c) Klassifizieren Sie die neuen Testdatenpunkte.

# Mittelpunkte der Klassen im LDA-Raum berechnen
X_lda_means = np.array([np.mean(X_lda[y==i], axis=0) for i in range(N_CLASSES)])

classified_results = {}
all_testdata_lda = []
all_test_colors = []

# Iteration durch alle Testdatensätze
for name, test_data_raw in CONFIG['testdata_sets'].items():
    test_data = test_data_raw[:, :N_FEATURES]
    testdata_lda = test_data.dot(W)
    all_testdata_lda.append(testdata_lda)

    # Klassifizierung durch Abstand zu Klassenmittelpunkten im LDA-Raum
    classified_labels_indices = []
    for point in testdata_lda:
        distances = [np.linalg.norm(point - mean) for mean in X_lda_means]
        classified_labels_indices.append(np.argmin(distances))
    
    # Speichern der Ergebnisse
    test_colors = [colors[l] for l in classified_labels_indices]
    all_test_colors.extend(test_colors)
    classified_labels_text = [labels[l] for l in classified_labels_indices]
    classified_results[name] = classified_labels_text
    
print("Testendaten nach Tranformation:\n", all_testdata_lda)

# Ausgabe der Klassifizierungsergebnisse
print("\nKlassifizierung der Testdaten:")
for name, result_labels in classified_results.items():
    print(f"\n{name}:")
    for i, label in enumerate(result_labels):
        print(f"  Testpunkt {i+1}: {label}")

# Plotten der Testdaten auf dem LDA-Graphen
if all_testdata_lda:
    all_testdata_lda_np = np.concatenate(all_testdata_lda)
    if all_testdata_lda_np.shape[1] == 2:
        plt.scatter(all_testdata_lda_np[:, 0], all_testdata_lda_np[:, 1], c=all_test_colors, marker='x', s=150, linewidth=3, label='Testdaten')
    else: # Falls 1D
        plt.scatter(all_testdata_lda_np[:, 0], np.zeros_like(all_testdata_lda_np[:, 0]), c=all_test_colors, marker='x', s=150, linewidth=3, label='Testdaten')

# Legende aktualisieren, um Testdaten einzuschließen
handles, current_labels = plt.gca().get_legend_handles_labels()
if 'Testdaten' not in current_labels:
     # Finde den Handle für Testdaten und füge ihn hinzu
    from matplotlib.lines import Line2D
    test_handle = Line2D([0], [0], marker='x', color='k', label='Testdaten', markersize=10, linewidth=0, markeredgewidth=3)
    handles.append(test_handle)
plt.legend(handles=handles)
plt.show()