import numpy as np
from Library_simple_finite_elements import lokale_rotationsmatrix, lokale_steifheitsmatrix_unrotiert, plottePositionen, check_Lengths, plotMatrix
import matplotlib.pyplot as plt # Für den Plot der Konvergenz

# --- Aufgabe 1: St. Martins Laterne ---

# a) Geometrie definieren (Parameter für N=7)
L_gesamt_param = 0.70  # m (70 cm)
phi_laterne_deg_param = 35  # degrees (Winkel, in dem Katja den Stab hält)
N_param_aufg1 = 7  # Anzahl der Elemente für Aufgabe 1a-d
m_load_param = 0.5  # kg (Masse am Ende)
d_mm_param = 10  # Durchmesser in mm
E_Npmmsq_param = 1800  # N/mm^2

# Globale Parameter für die Schleife (werden in der Schleife mit N variiert)
L_gesamt = L_gesamt_param
phi_laterne_deg = phi_laterne_deg_param
m_load = m_load_param
d_mm = d_mm_param
E_Npmmsq = E_Npmmsq_param

# --- Berechnung für Aufgabe 1a-d mit N_param_aufg1 ---
print(f"--- Berechnung für Aufgabe 1a-d mit N = {N_param_aufg1} ---")
N = N_param_aufg1 # Setze N für diesen Teil

# Material- und Querschnittseigenschaften
d_m = d_mm * 1e-3
E_Npm2 = E_Npmmsq * (1e3)**2
A = np.pi * (d_m / 2)**2
I_val_problem_units = (np.pi / 64) * (d_mm)**2 * (0.001 * d_mm)**2
I = I_val_problem_units * 1e-6
g = 9.81
F_gravity = -m_load * g
L_element = L_gesamt / N
num_nodes = N + 1

# Initialgeometrie
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

plottePositionen(full_x_initial, stangen_zu_knoten, allphi=initial_element_angles_deg, filename=f'initial_geometry_N{N}.png')
check_Lengths(full_x_initial, stangen_zu_knoten)

# b) Gleichungssystem aufstellen
local_stiffness_matrices_rotated = []
K_unrotiert = lokale_steifheitsmatrix_unrotiert(L_element, A, E_Npm2, I)
for i in range(N):
    R = lokale_rotationsmatrix(initial_element_angles_deg[i])
    K_rotiert = R @ K_unrotiert @ R.T
    local_stiffness_matrices_rotated.append(K_rotiert)

K_global = np.zeros((num_nodes * 3, num_nodes * 3))
for element_id, (node_L_id, node_R_id) in stangen_zu_knoten.items():
    K_local_rot = local_stiffness_matrices_rotated[element_id - 1]
    idx_L_0=node_L_id-1; idx_R_0=node_R_id-1
    dofs_L=[idx_L_0*3,idx_L_0*3+1,idx_L_0*3+2]; dofs_R=[idx_R_0*3,idx_R_0*3+1,idx_R_0*3+2]
    global_dof_indices=np.array(dofs_L+dofs_R)
    for r_local in range(6):
        for c_local in range(6):
            K_global[global_dof_indices[r_local],global_dof_indices[c_local]]+=K_local_rot[r_local,c_local]
plotMatrix(K_global, filename=f'global_stiffness_matrix_N{N}.png')

# c) Gleichungssystem lösen
fixed_dofs = [0, 1, 2]
K_reduced = np.delete(K_global, fixed_dofs, axis=0)
K_reduced = np.delete(K_reduced, fixed_dofs, axis=1)
F_global_vec = np.zeros(num_nodes * 3)
F_global_vec[(num_nodes-1)*3 + 1] = F_gravity
F_reduced = np.delete(F_global_vec, fixed_dofs, axis=0)
try:
    delta_reduced = np.linalg.solve(K_reduced, F_reduced)
    delta_full = np.zeros(num_nodes*3); free_dofs_mask = np.ones(num_nodes*3,dtype=bool); free_dofs_mask[fixed_dofs]=False
    delta_full[free_dofs_mask] = delta_reduced
    full_x_load_aufg1 = full_x_initial + delta_full

    print("\nDie Koordinaten bei Last sind gegeben durch (für N=7):")
    print(np.round(full_x_load_aufg1, 8))

    solution_target = np.array([0.,0.,0.,0.09060484,0.04494409,-0.29547868,0.19685395,0.06754585,-0.5454991,0.31613995,0.071529,-0.75006127,0.44585546,0.06061725,-0.90916517,0.58339311,0.03853434,-1.02281082,0.72614552,0.00900399,-1.09099821,0.87150531,-0.02425009,-1.11372734])
    print("\n--- Vergleich mit der Kontrolllösung (N=7) ---")
    abs_diff = np.abs(full_x_load_aufg1 - solution_target)
    rel_diff = np.zeros_like(abs_diff); non_zero_mask = np.abs(solution_target) > 1e-9
    rel_diff[non_zero_mask] = abs_diff[non_zero_mask] / np.abs(solution_target[non_zero_mask])
    if np.allclose(full_x_load_aufg1, solution_target, atol=1e-7, rtol=1e-5): # Toleranz angepasst
        print("Die berechneten Werte stimmen sehr gut (innerhalb der Toleranz) mit der Selbstkontrolle überein.")
    else:
        print("WARNUNG: Die berechneten Werte weichen von der Selbstkontrolle ab!")
        print("Berechnet:", np.round(full_x_load_aufg1,8))
        print("Erwartet: ", solution_target)
        max_abs_diff_idx = np.argmax(abs_diff)
        max_rel_diff_idx = np.argmax(rel_diff)
        print(f"Maximale absolute Abweichung: {abs_diff[max_abs_diff_idx]:.2e} bei Index {max_abs_diff_idx}")
        print(f"Maximale relative Abweichung: {rel_diff[max_rel_diff_idx]:.2e} bei Index {max_rel_diff_idx}")

    # d) Plotten des Laternenstabs unter Last
    phi_elements_loaded_deg = np.zeros(N)
    for i in range(N):
        node_L_idx_0=i; node_R_idx_0=i+1
        x_L=full_x_load_aufg1[node_L_idx_0*3]; y_L=full_x_load_aufg1[node_L_idx_0*3+1]
        x_R=full_x_load_aufg1[node_R_idx_0*3]; y_R=full_x_load_aufg1[node_R_idx_0*3+1]
        phi_elements_loaded_deg[i] = np.rad2deg(np.arctan2(y_R-y_L, x_R-x_L))
    plottePositionen(full_x_load_aufg1, stangen_zu_knoten, allphi=phi_elements_loaded_deg, filename=f'loaded_geometry_N{N}.png')
    check_Lengths(full_x_load_aufg1, stangen_zu_knoten)

except np.linalg.LinAlgError:
    print(f"FEHLER: Singuläre Matrix bei der Berechnung für N={N_param_aufg1}. Berechnung abgebrochen.")
    full_x_load_aufg1 = None # Signalisiert Fehler

# --- Aufgabe e) Konvergenzstudie ---
print("\n\n--- Aufgabe e) Konvergenzstudie ---")
N_values_konvergenz = [4, 7, 10, 15, 20, 30, 40, 50, 70, 100] # Feinere Schritte und mehr Werte
results_last_node_x = []
results_last_node_y = []
results_last_node_phi = []
valid_N_values = [] # Für den Plot, nur Ns, bei denen die Berechnung erfolgreich war

for N_test in N_values_konvergenz:
    print(f"-- Berechne für N = {N_test} --")
    N = N_test # Wichtig: N für die aktuelle Iteration setzen

    # Neuberechnung der geometrieabhängigen Parameter für aktuelles N
    L_element = L_gesamt / N
    num_nodes = N + 1

    # Initialgeometrie für aktuelles N
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

    # Gleichungssystem aufstellen für aktuelles N
    local_stiffness_matrices_rotated = []
    # A, E_Npm2, I bleiben gleich, nur L_element ändert sich
    K_unrotiert = lokale_steifheitsmatrix_unrotiert(L_element, A, E_Npm2, I)
    for i in range(N):
        R = lokale_rotationsmatrix(initial_element_angles_deg[i])
        K_rotiert = R @ K_unrotiert @ R.T
        local_stiffness_matrices_rotated.append(K_rotiert)

    K_global = np.zeros((num_nodes * 3, num_nodes * 3))
    for element_id, (node_L_id, node_R_id) in stangen_zu_knoten.items():
        K_local_rot = local_stiffness_matrices_rotated[element_id - 1]
        idx_L_0=node_L_id-1; idx_R_0=node_R_id-1
        dofs_L=[idx_L_0*3,idx_L_0*3+1,idx_L_0*3+2]; dofs_R=[idx_R_0*3,idx_R_0*3+1,idx_R_0*3+2]
        global_dof_indices=np.array(dofs_L+dofs_R)
        for r_local in range(6):
            for c_local in range(6):
                K_global[global_dof_indices[r_local],global_dof_indices[c_local]]+=K_local_rot[r_local,c_local]

    # Gleichungssystem lösen für aktuelles N
    fixed_dofs = [0, 1, 2]
    K_reduced = np.delete(K_global, fixed_dofs, axis=0)
    K_reduced = np.delete(K_reduced, fixed_dofs, axis=1)
    F_global_vec = np.zeros(num_nodes * 3)
    F_global_vec[(num_nodes-1)*3 + 1] = F_gravity # Kraft am Y-DOF des LETZTEN Knotens
    F_reduced = np.delete(F_global_vec, fixed_dofs, axis=0)

    try:
        delta_reduced = np.linalg.solve(K_reduced, F_reduced)
        delta_full = np.zeros(num_nodes*3)
        free_dofs_mask = np.ones(num_nodes*3,dtype=bool); free_dofs_mask[fixed_dofs]=False
        delta_full[free_dofs_mask] = delta_reduced
        full_x_load_konvergenz = full_x_initial + delta_full

        results_last_node_x.append(full_x_load_konvergenz[-3])
        results_last_node_y.append(full_x_load_konvergenz[-2])
        results_last_node_phi.append(full_x_load_konvergenz[-1])
        valid_N_values.append(N_test)
        print(f"N={N_test}: x_tip={full_x_load_konvergenz[-3]:.8f}, y_tip={full_x_load_konvergenz[-2]:.8f}, phi_tip={full_x_load_konvergenz[-1]:.8f}")

        # Optional: Plotten der deformierten Struktur für ausgewählte N-Werte
        # if N_test in [7, 20, 50]:
        #     phi_elements_loaded_deg_conv = np.zeros(N_test)
        #     for i in range(N_test):
        #         node_L_idx_0=i; node_R_idx_0=i+1
        #         x_L=full_x_load_konvergenz[node_L_idx_0*3]; y_L=full_x_load_konvergenz[node_L_idx_0*3+1]
        #         x_R=full_x_load_konvergenz[node_R_idx_0*3]; y_R=full_x_load_konvergenz[node_R_idx_0*3+1]
        #         phi_elements_loaded_deg_conv[i] = np.rad2deg(np.arctan2(y_R-y_L, x_R-x_L))
        #     plottePositionen(full_x_load_konvergenz, stangen_zu_knoten, allphi=phi_elements_loaded_deg_conv, filename=f'loaded_geometry_N{N_test}_conv.png')

    except np.linalg.LinAlgError:
        print(f"WARNUNG: Singuläre Matrix für N={N_test} in Konvergenzstudie. Überspringe diesen Wert.")
        # Füge NaNs hinzu, um die Array-Längen konsistent zu halten, wenn ein Plot gewünscht wird
        results_last_node_x.append(np.nan)
        results_last_node_y.append(np.nan)
        results_last_node_phi.append(np.nan)
        # valid_N_values wird hier nicht erweitert


print("\n--- Ergebnisse der Konvergenzstudie für den letzten Knoten ---")
print("N \t x_tip (m) \t y_tip (m) \t phi_tip (rad)")
for i, n_val in enumerate(valid_N_values): # Nur über gültige N iterieren
    print(f"{n_val} \t {results_last_node_x[i]:.8f} \t {results_last_node_y[i]:.8f} \t {results_last_node_phi[i]:.8f}")

# Plotten der Konvergenz
if valid_N_values: # Nur plotten, wenn es gültige Ergebnisse gibt
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(valid_N_values, results_last_node_x, 'o-', label='x_tip')
    plt.xlabel('Anzahl der Elemente (N)')
    plt.ylabel('x-Koordinate Spitze (m)')
    plt.title('Konvergenz der x-Koordinate der Spitze')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(valid_N_values, results_last_node_y, 'o-', label='y_tip')
    plt.xlabel('Anzahl der Elemente (N)')
    plt.ylabel('y-Koordinate Spitze (m)')
    plt.title('Konvergenz der y-Koordinate der Spitze')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(valid_N_values, results_last_node_phi, 'o-', label='phi_tip')
    plt.xlabel('Anzahl der Elemente (N)')
    plt.ylabel('phi-Koordinate Spitze (rad)')
    plt.title('Konvergenz der phi-Koordinate der Spitze')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('konvergenz_studie_spitze.png')
    print("\nKonvergenzplot wurde als 'konvergenz_studie_spitze.png' gespeichert.")
    plt.show() # Uncomment to display plot interactively
else:
    print("\nKeine gültigen Ergebnisse für die Konvergenzstudie vorhanden, um zu plotten.")


print("\n--- Interpretation von Aufgabe e) ---")
print("Beobachten Sie, ob sich die Werte für x_tip, y_tip und phi_tip mit steigender Anzahl an Elementen N stabilisieren.")
print("Wenn die Kurven abflachen und sich einem Grenzwert annähern, konvergiert Ihre numerische Lösung.")
print("Vergleichen Sie diese konvergierten Werte (für großes N) mit der Kontrolllösung für N=7.")
print("Eine verbleibende Diskrepanz deutet auf unterschiedliche Modellannahmen zwischen Ihrem Code und der Kontrolllösung hin.")