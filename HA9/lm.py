#! /usr/bin/env python3
# -*- coding: utf-8 -*-

### Vorlagendatei für die Übungen zur Computergestützten Physik ###

# Bitte ergänzen Sie hier die Daten der Abgebenden. Ersetzen Sie nur
# die Punkte ('...'), aber lassen Sie den Rest der Zeilen und ihre Reihenfolge
# ansonsten unverändert, da Ihre Abgabe sonst nicht elektronisch verarbeitet
# werden kann.
#
# 1)
# Matrikelnummer: 458471
# Name: Maximilian Kieser
# Email: maximilian.l.kieser@gmail.com
#
# 2)
# Matrikelnummer: 454505
# Name: Louisa Steffens
# Email: Louisa.sonne@web.de
# #

# Häufig benötigte Module (auskommentieren, wenn notwendig):
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy
from scipy import constants
import Library_simple_finite_elements as fem
from Library_simple_finite_elements import (
    lokale_rotationsmatrix,
    lokale_steifheitsmatrix_unrotiert,
    plottePositionen,
    plotMatrix
)

#------------- 1a) --------------

L = 0.7 #m
N = 7
N_knoten = N + 1
l = L/N
d = 10 # mm
g_to_N = constants.g # Erdbeschleunigung
m_lantern = 0.5 # kg
phi_deg = 35 # °
phi_rad = np.radians(phi_deg)
E = 1800  # N/mm^2
A = np.pi* (d/2)**2 # m^2
I = (np.pi / 64) * d**2 * (0.001 * d)**2 # m^4

x_knoten = [0]
y_knoten =[0]

for _ in range(N):
    x_knoten.append(x_knoten[-1] + l * np.cos(phi_rad))
    y_knoten.append(y_knoten[-1] + l * np.sin(phi_rad))

for i, (x, y) in enumerate(zip(x_knoten, y_knoten), start=1):
    print(f"Knoten {i}: ({x:.4f}, {y:.4f})")

phi_elements_deg = np.full(N_knoten, phi_deg)

stangen_zu_knoten = {
    i+1: (i+1, i+2) for i in range(N)
}


full_x = np.zeros(3 * N_knoten)
for i in range(N_knoten):
    full_x[3*i]     = x_knoten[i]
    full_x[3*i + 1] = y_knoten[i]
    full_x[3*i + 2] = 0.0  # theta = 0, da keine Last

fem.plottePositionen(full_x, stangen_zu_knoten)

# ------------ b) -------------

rotated_knoten = []
for i in range(N_knoten):
    K_local_unrotated = lokale_steifheitsmatrix_unrotiert(l, A=A, E=E, I=I)
    K_rotated = lokale_rotationsmatrix(phi_elements_deg[i]) @ K_local_unrotated @ lokale_rotationsmatrix(phi_elements_deg[i]).T
    rotated_knoten.append(K_rotated)

DOF = 3 # Freiheitsgrad pro Knoten ix, iy, iphi
N_DOF = DOF * N_knoten
K_global = np.zeros((N_DOF, N_DOF))

for element_id, (node1_id, node2_id) in stangen_zu_knoten.items():
    K_e = rotated_knoten[element_id - 1]  # -1, da Liste 0-indiziert

    # Globale Indizes für die Freiheitsgrade des linken Knotens
    # GDL-Schema: [ux1, uy1, theta1, ux2, uy2, theta2, ...]
    idx1_start = (node1_id - 1) * DOF
    idx1_end = idx1_start + DOF

    # Globale Indizes für die Freiheitsgrade des rechten Knotens
    idx2_start = (node2_id - 1) * DOF
    idx2_end = idx2_start + DOF

    # Assemblieren der Untermatrizen
    # K_e ist 6x6, entspricht den GDL [ux1, uy1, theta1, ux2, uy2, theta2] des Elements
    # Die lokalen Indizes 0,1,2 entsprechen Knoten 1, 3,4,5 entsprechen Knoten 2

    # Block 1,1 (Knoten 1 zu Knoten 1)
    K_global[idx1_start: idx1_end, idx1_start: idx1_end] += K_e[0:DOF, 0:DOF]
    # Block 1,2 (Knoten 1 zu Knoten 2)
    K_global[idx1_start: idx1_end, idx2_start: idx2_end] += K_e[0:DOF, DOF:6]
    # Block 2,1 (Knoten 2 zu Knoten 1)
    K_global[idx2_start: idx2_end, idx1_start: idx1_end] += K_e[DOF:6, 0:DOF]
    # Block 2,2 (Knoten 2 zu Knoten 2)
    K_global[idx2_start: idx2_end, idx2_start: idx2_end] += K_e[DOF:6, DOF:6]

print("\nGlobale Steifigkeitsmatrix (erste 9x9):")
print(K_global[:9, :9])
# Zur Kontrolle plotMatrix verwenden
print("\nPlot der globalen Steifigkeitsmatrix:")
plotMatrix(K_global, filename="Globale_Steifigkeitsmatrix.png")

# ------------- c) ------------

# Freiheitsgrade für Knoten 1 sind 0, 1, 2 (ux1, uy1, theta1).
fixed_dofs = [0, 1, 2] # Indizes der fixierten Freiheitsgrade im globalen System

# Löschung der Zeile und Spalte die zum ersten Knoten gehört: x1, y1, theta1
K_reduced = np.delete(K_global, fixed_dofs, axis=0)
K_reduced = np.delete(K_reduced, fixed_dofs, axis=1)

print("\nReduzierte Steifigkeitsmatrix (erste 9x9):")
print(K_reduced[:9, :9])

# Definieren des reduzierten, globalen Kraftvektors
# Die Gewichtskraft wirkt am letzten Knoten (Knoten 8, Index 7) in y-Richtung.
# Freiheitsgrade für den letzten Knoten: ux, uy, theta
# Index für uy des letzten Knotens im VOLLSTÄNDIGEN System: (N_knoten - 1) * DOF+ 1
F_global = np.zeros(N_knoten * DOF)
F_gravity = -m_lantern * g_to_N # Gewichtskraft nach unten
last_node_uy_dof = (N_knoten - 1) * DOF + 1
F_global[last_node_uy_dof] = F_gravity # Nur die y-Komponente am letzten Knoten

# Erzeugen des reduzierten Kraftvektors
F_reduced = np.delete(F_global, fixed_dofs, axis=0)

print(f"\nKraft am letzten Knoten (Y-Richtung): {F_gravity:.4f} N")
print(f"Reduzierter Kraftvektor (Auszug, z.B. die ersten 5 Elemente): {F_reduced[:5]}")

# Berechnen der reduzierten Koordinatenänderungen
try:
    delta_u_reduced = np.linalg.solve(K_reduced, F_reduced)
    print("\nReduzierter Vektor der Koordinatenänderungen (Auszug, z.B. die ersten 5 Elemente):")
    print(delta_u_reduced[:5])
except np.linalg.LinAlgError as e:
    print(f"\nFehler beim Lösen des Gleichungssystems: {e}")
    print("Möglicherweise ist die Steifigkeitsmatrix singulär oder schlecht konditioniert.")
    exit()

# Erzeugen des vollständigen Vektors der Koordinatenänderungen
delta_u_full = np.zeros(N_knoten * DOF)
# Füllen der Nicht-Null-Einträge
j = 0
for i in range(N_knoten * DOF):
    if i not in fixed_dofs:
        delta_u_full[i] = delta_u_reduced[j]
        j += 1

print("\nVollständiger Vektor der Koordinatenänderungen (Auszug, z.B. die ersten 10 Elemente):")
print(delta_u_full[:10])

# Berechnen des vollständigen Kraftvektors (Reaktionskräfte an den Lagern)
F_full_recalculated = K_global @ delta_u_full
print("\nVollständiger Kraftvektor (recalculated) (Auszug, z.B. die ersten 10 Elemente):")
print(F_full_recalculated[:10])


# Berechnen des vollständigen Koordinatenvektors bei Last
# initial_full_x hat die Form [x1, y1, theta1, x2, y2, theta2, ...]
final_full_x = full_x + delta_u_full

print("\nFinaler Koordinatenvektor bei Last :")
print(final_full_x)

# -------------- d) ---------------

print("\nPlot der Laternenstange unter Last:")
plottePositionen(final_full_x, stangen_zu_knoten, allphi=phi_elements_deg, filename="Laternenstange_unter_Last.png")

# -------------- e) -----------------
#Erhöhen Sie die Anzahl der Elemente N. Wie verhält sich die Kontur des Stabs unter Las? Konvergiert sie?

print("\n--- Analyse der Konvergenz (e) ---")
print("Um die Konvergenz zu analysieren, könnten Sie das Skript mit verschiedenen Werten für 'N_elements' aufrufen (z.B. N=14, N=28 etc.).")
print("Sie sollten beobachten, dass sich die Kontur der Stange mit zunehmender Elementanzahl glättet und einem")
print("physikalisch 'richtigeren' Ergebnis nähert, was auf Konvergenz hindeutet.")
print("Eine höhere Elementanzahl führt zu einer genaueren Diskretisierung der Geometrie")
print("und der Verformung, was zu einer besseren Approximation der kontinuierlichen Lösung führt.")
