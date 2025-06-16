import numpy as np
from Library_simple_finite_elements import lokale_rotationsmatrix, lokale_steifheitsmatrix_unrotiert, plottePositionen, check_Lengths, plotMatrix

# --- Aufgabe 1: St. Martins Laterne ---

# a) Geometrie definieren
L_gesamt = 0.70  # m (70 cm)
phi_laterne_deg = 35  # degrees (Winkel, in dem Katja den Stab hält)
N = 7  # Anzahl der Elemente

# Material- und Querschnittseigenschaften
d_mm = 10  # Durchmesser in mm
d_m = d_mm * 1e-3 # Durchmesser in m

E_Npmmsq = 1800  # N/mm^2
E_Npm2 = E_Npmmsq * (1e3)**2 # Umrechnung N/mm^2 zu N/m^2 (1 N/mm^2 = 1e6 N/m^2)

A = np.pi * (d_m / 2)**2  # Querschnittsfläche in m^2

# Trägheitsmoment I, gemäß Aufgabenstellung:
I_val_problem_units = (np.pi / 64) * (d_mm)**2 * (0.001 * d_mm)**2
I = I_val_problem_units * 1e-6  # Trägheitsmoment in m⁴

# Gravitationslast
m_load = 0.5  # kg (Masse am Ende)
g = 9.81  # m/s^2
F_gravity = -m_load * g # Kraft in N, negativ, da nach unten wirkend

# Länge jedes Elements
L_element = L_gesamt / N

num_nodes = N + 1
initial_nodal_coords = np.zeros((num_nodes, 2))
initial_element_angles_deg = np.zeros(N)
phi_laterne_rad = np.deg2rad(phi_laterne_deg)

for i in range(num_nodes):
    initial_nodal_coords[i, 0] = i * L_element * np.cos(phi_laterne_rad)
    initial_nodal_coords[i, 1] = i * L_element * np.sin(phi_laterne_rad)
for i in range(N):
    initial_element_angles_deg[i] = phi_laterne_deg

full_x_initial = np.zeros(num_nodes * 3)
for i in range(num_nodes):
    full_x_initial[i*3]     = initial_nodal_coords[i, 0]
    full_x_initial[i*3 + 1] = initial_nodal_coords[i, 1]
    full_x_initial[i*3 + 2] = 0.0

stangen_zu_knoten = {}
for i in range(1, N + 1):
    stangen_zu_knoten[i] = (i, i + 1)

print("--- Initiales Setup ---")
print(f"L_gesamt: {L_gesamt} m, N_elemente: {N}")
print(f"d: {d_m} m, A: {A:.4e} m^2, I: {I:.4e} m^4, E: {E_Npm2:.2e} N/m^2")

plottePositionen(full_x_initial, stangen_zu_knoten, allphi=initial_element_angles_deg, filename='initial_geometry.png')
check_Lengths(full_x_initial, stangen_zu_knoten)


# b) Gleichungssystem aufstellen
local_stiffness_matrices_rotated = []
K_unrotiert = lokale_steifheitsmatrix_unrotiert(L_element, A, E_Npm2, I)

for i in range(N):
    R = lokale_rotationsmatrix(initial_element_angles_deg[i])
    K_rotiert = R @ K_unrotiert @ R.T
    local_stiffness_matrices_rotated.append(K_rotiert)

global_stiffness_matrix_size = num_nodes * 3
K_global = np.zeros((global_stiffness_matrix_size, global_stiffness_matrix_size))

for element_id, (node_L_id, node_R_id) in stangen_zu_knoten.items():
    K_local_rot = local_stiffness_matrices_rotated[element_id - 1]
    idx_L_0 = node_L_id - 1
    idx_R_0 = node_R_id - 1
    dofs_L = [idx_L_0 * 3, idx_L_0 * 3 + 1, idx_L_0 * 3 + 2]
    dofs_R = [idx_R_0 * 3, idx_R_0 * 3 + 1, idx_R_0 * 3 + 2]
    global_dof_indices = np.array(dofs_L + dofs_R)
    for r_local in range(6):
        for c_local in range(6):
            r_global = global_dof_indices[r_local]
            c_global = global_dof_indices[c_local]
            K_global[r_global, c_global] += K_local_rot[r_local, c_local]

plotMatrix(K_global, filename='global_stiffness_matrix.png')


# c) Gleichungssystem lösen
fixed_dofs = [0, 1, 2]
K_reduced = np.delete(K_global, fixed_dofs, axis=0)
K_reduced = np.delete(K_reduced, fixed_dofs, axis=1)

F_global = np.zeros(global_stiffness_matrix_size)
last_node_idx_0 = N
y_dof_last_node = last_node_idx_0 * 3 + 1
F_global[y_dof_last_node] = F_gravity

F_reduced = np.delete(F_global, fixed_dofs, axis=0)
delta_reduced = np.linalg.solve(K_reduced, F_reduced)

delta_full = np.zeros(global_stiffness_matrix_size)
free_dofs_mask = np.ones(global_stiffness_matrix_size, dtype=bool)
free_dofs_mask[fixed_dofs] = False
delta_full[free_dofs_mask] = delta_reduced

full_x_load = full_x_initial + delta_full

print("\nHinweis: Zur Selbstkontrolle. Der globale Koordinatenvektor sei gegeben durch")
print("['x1', 'y1', 'phi1', 'x2', 'y2', 'phi2',")
print(" 'x3', 'y3', 'phi3', 'x4', 'y4', 'phi4',")
print(" 'x5', 'y5', 'phi5', 'x6', 'y6', 'phi6',")
print(" 'x7', 'y7', 'phi7', 'x8', 'y8', 'phi8']")

print("\nDie Koordinaten bei Last sind gegeben durch")
print(np.round(full_x_load, 8))

solution_target = np.array([0., 0., 0., 0.09060484, 0.04494409, -0.29547868,
                            0.19685395, 0.06754585, -0.5454991,  0.31613995, 0.071529,   -0.75006127,
                            0.44585546, 0.06061725, -0.90916517, 0.58339311, 0.03853434, -1.02281082,
                            0.72614552, 0.00900399, -1.09099821, 0.87150531, -0.02425009, -1.11372734])

print("\n--- Vergleich mit der Kontrolllösung ---")
abs_diff = np.abs(full_x_load - solution_target)
rel_diff = np.zeros_like(abs_diff)
non_zero_mask = np.abs(solution_target) > 1e-9 # Kleine Toleranz für Nicht-Null
rel_diff[non_zero_mask] = abs_diff[non_zero_mask] / np.abs(solution_target[non_zero_mask])

if np.allclose(full_x_load, solution_target, atol=1e-8, rtol=1e-6):
    print("Die berechneten Werte stimmen exakt (innerhalb der Toleranz) mit der Selbstkontrolle überein.")
else:
    print("WARNUNG: Die berechneten Werte weichen von der Selbstkontrolle ab!")
    print("Berechnet:", np.round(full_x_load,8))
    print("Erwartet: ", solution_target)
    max_abs_diff_idx = np.argmax(abs_diff)
    max_rel_diff_idx = np.argmax(rel_diff) #argmax auf rel_diff ist ok
    print(f"Maximale absolute Abweichung: {abs_diff[max_abs_diff_idx]:.2e} bei Index {max_abs_diff_idx}")
    print(f"Maximale relative Abweichung: {rel_diff[max_rel_diff_idx]:.2e} bei Index {max_rel_diff_idx} (für Nicht-Null-Zielwerte)")


# d) Plotten des Laternenstabs unter Last
phi_elements_loaded_deg = np.zeros(N)
for i in range(N):
    node_L_idx_0 = i
    node_R_idx_0 = i + 1
    x_L = full_x_load[node_L_idx_0 * 3]
    y_L = full_x_load[node_L_idx_0 * 3 + 1]
    x_R = full_x_load[node_R_idx_0 * 3]
    y_R = full_x_load[node_R_idx_0 * 3 + 1]
    delta_x = x_R - x_L
    delta_y = y_R - y_L
    phi_elements_loaded_deg[i] = np.rad2deg(np.arctan2(delta_y, delta_x))

plottePositionen(full_x_load, stangen_zu_knoten, allphi=phi_elements_loaded_deg, filename='loaded_geometry.png')
check_Lengths(full_x_load, stangen_zu_knoten)

print("\n--- Aufgabe e) Konvergenz ---")
print("Um die Konvergenz zu analysieren, führen Sie das Skript mit steigendem N erneut aus (z.B. 7, 14, 28).")
# ... (rest of the e) part explanation)