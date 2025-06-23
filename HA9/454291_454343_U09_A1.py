import numpy as np
from Library_simple_finite_elements import lokale_rotationsmatrix, lokale_steifheitsmatrix_unrotiert, plottePositionen, check_Lengths, plotMatrix
import matplotlib.pyplot as plt # Für den Plot der Konvergenz
from scipy.constants import g as g_scipy

# a) Übertragen Sie die Geometrie in Ihren Code:
# • Definieren Sie dabei die Positionen xi, yi mit [xi, yi] = m der einzelnen Knoten
# sowie die Winkel ϕi in Grad der einzelnen Stangen in Abhängigkeit der oben
# beschriebenen Eingabegeometrie.
# • Definieren Sie ein Mapping von Stangen-Id zu den linken und rechten Knoten-Ids,
# mit dem Sie u.a. später die globale Steifheitsmatrix bestimmen. Dieses Mapping
# soll vom Python-Typ dictionary sein. Der Aufbau sei wie folgt:
# 1 stangen_zu_knoten = {
# 2 StangenID : ( linke_KnotenID ,
#                 rechte_KnotenID ),
# 3 ...
# 4 }
# Hierbei seien alle IDs ganze Zahlen größer 0 so wie in Abbildung 1 dargestellt.
# Verwenden Sie die Plot-Funktion in Library_simple_finite_elements.py um
# Ihre Geometrie zu kontrollieren.

# --- Globale Parameter (werden in der Konvergenzschleife für N konstant gehalten, außer N selbst) ---
L_gesamt_param = 0.75  # m (70 cm)
phi_laterne_deg_param = 37  # degrees (Winkel, in dem Katja den Stab hält)
m_load_param = 0.48  # kg (Masse am Ende)
d_mm_param = 10  # Durchmesser in mm
E_Npmmsq_param = 1800  # N/mm^2

# --- Berechnung für Aufgabe 1a-d ---
print("--- Berechnung für Aufgabe 1a-d ---")
N_aufg1 = 10  # N spezifisch für diesen Teil
N = N_aufg1 # Setze globales N für diesen Teil

# Geometrie und Material für N=7
L_gesamt = L_gesamt_param
phi_laterne_deg = phi_laterne_deg_param
m_load = m_load_param
d_mm = d_mm_param
E_Npmmsq = E_Npmmsq_param

d_m = d_mm * 1e-3
E_Npm2 = E_Npmmsq * (1e3)**2
A = np.pi * (d_m / 2)**2
I_val_problem_units = (np.pi / 64) * (d_mm)**2 * (0.001 * d_mm)**2
I = I_val_problem_units * 1e-6
g = g_scipy
F_gravity = -m_load * g
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

print("--- Initiales Setup (N=7) ---")
print(f"L_gesamt: {L_gesamt} m, N_elemente: {N}")
print(f"d: {d_m} m, A: {A:.4e} m^2, I: {I:.4e} m^4, E: {E_Npm2:.2e} N/m^2")

plottePositionen(full_x_initial, stangen_zu_knoten, allphi=initial_element_angles_deg, filename='initial_geometry.png')
lengths = check_Lengths(full_x_initial, stangen_zu_knoten)
assert np.allclose(lengths, np.repeat(L_element, N)), "Lengths and coordinates aren't consistent"

# b) Stellen Sie das Gleichungssystem auf:
# • Verwenden Sie die beiden entsprechenden Funktionen in
#                                           Library_simple_finite_elements.py,
# um eine Liste der korrekt um ϕi rotierten lokalen Steifheitsmatrix zu erzeugen.
# Hinweis: Die ϕi sind hier die Winkel, um die die jeweilige Stange von
# ihrer horizontalen Lage rotiert wurden. Diese sind zu unterscheiden von
# den Winkeln θi, welche an jedem Knoten angeben, um wie viel dieser im
# Vergleich zur Stange rotiert wurde. Ohne Last sind also alle θi = 0.
# • Erzeugen Sie aus dieser Liste und Ihrem Mapping die globale Steifheitsmatrix.
# Achten Sie darauf, die Untermatrizen der lokalen Steifheitsmatrizen an die richti-
#                                                                            gen Stellen der globalen Steifheitsmatrix zu setzen. Verwenden Sie hierfür Ihr
# Mapping stangen_zu_knoten, sodass dieser Code-Teil eine beliebige Geometrie
# korrekt in eine globale Steifheitsmatrix umsetzen kann.
# Hinweis: Sie können zur Kontrolle gerne die Funktion plotMatrix verwen-
#                                                                 den, um Ihre Matrizen anzuzeigen.

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
    idx_L_0 = node_L_id - 1; idx_R_0 = node_R_id - 1
    dofs_L = [idx_L_0 * 3, idx_L_0*3+1, idx_L_0*3+2]; dofs_R = [idx_R_0*3, idx_R_0*3+1, idx_R_0*3+2]
    global_dof_indices = np.array(dofs_L + dofs_R)
    print(f'{element_id}: {global_dof_indices}')
    for r_local in range(6):
        for c_local in range(6):
            K_global[global_dof_indices[r_local], global_dof_indices[c_local]] += K_local_rot[r_local, c_local]
plotMatrix(K_global, filename='global_stiffness_matrix.png')

fixed_dofs = [0, 1, 2]
K_reduced = np.delete(K_global, fixed_dofs, axis=0)
K_reduced = np.delete(K_reduced, fixed_dofs, axis=1)
F_global_vec_aufg1 = np.zeros(global_stiffness_matrix_size)
last_node_idx_0_aufg1 = N
y_dof_last_node_aufg1 = last_node_idx_0_aufg1 * 3 + 1
F_global_vec_aufg1[y_dof_last_node_aufg1] = F_gravity
F_reduced = np.delete(F_global_vec_aufg1, fixed_dofs, axis=0)
# c) Lösen Sie nun das Gleichungssystem, um die Auswirkung des Gewichts auf den
# Laternenstab zu simulieren:
# • Erzeugen Sie die reduzierte, globale Steifheitsmatrix. Löschen Sie dazu die
# Zeilen und Spalten Ihrer Steifheitsmatrix, die zu den Randbedingungen des
# Knoten 1 gehören.
# • Definieren Sie den reduzierten, globalen Kraftvektor. Dieser soll nur die nach
# unten gerichtete Gewichtskraft der Masse m enthalten. Diese sei frei beweglich
# am letzten Knoten aufgehangen.
# • Berechnen Sie den reduzierten Vektor der Koordinatenänderungen.
# • Erzeugen Sie daraus den vollständigen Vektor der Koordinatenänderungen.
# • Berechnen Sie daraus den vollständigen Kraftvektor.
# • Berechnen Sie den vollständigen Koordinatenvektor bei Last.
try:
    delta_reduced = np.linalg.solve(K_reduced, F_reduced)
    delta_full = np.zeros(global_stiffness_matrix_size)
    free_dofs_mask = np.ones(global_stiffness_matrix_size, dtype=bool)
    free_dofs_mask[fixed_dofs] = False
    delta_full[free_dofs_mask] = delta_reduced
    full_x_load_aufg1 = full_x_initial + delta_full

    print("\nDie Koordinaten bei Last sind gegeben durch (N=7):")
    print(np.round(full_x_load_aufg1, 3))

    Ms = full_x_load_aufg1[2:-1:3]
    sigmaBmax = (Ms*d_mm*10^(-3))/(2*I)
    for i in sigmaBmax:
        res = np.max(sigmaBmax)

    print(f"----- RES: {res}")
    # solution_target = np.array([0.,0.,0.,0.09060484,0.04494409,-0.29547868,0.19685395,0.06754585,-0.5454991,0.31613995,0.071529,-0.75006127,0.44585546,0.06061725,-0.90916517,0.58339311,0.03853434,-1.02281082,0.72614552,0.00900399,-1.09099821,0.87150531,-0.02425009,-1.11372734])
    # print("\n--- Vergleich mit der Kontrolllösung (N=7) ---")
    # print("Target:\n", solution_target)
    # abs_diff = np.abs(full_x_load_aufg1 - solution_target)
    # rel_diff = np.zeros_like(abs_diff); non_zero_mask = np.abs(solution_target) > 1e-9
    # rel_diff[non_zero_mask] = abs_diff[non_zero_mask] / np.abs(solution_target[non_zero_mask])
    # if np.allclose(full_x_load_aufg1, solution_target, atol=1e-7, rtol=1e-5):
    #     print("Die berechneten Werte stimmen sehr gut (innerhalb der Toleranz) mit der Selbstkontrolle überein.")
    # else:
    #     print("WARNUNG: Die berechneten Werte weichen von der Selbstkontrolle ab!")
    #     print("Berechnet:", np.round(full_x_load_aufg1,8)); print("Erwartet: ", solution_target)
    #     max_abs_diff_idx = np.argmax(abs_diff); max_rel_diff_idx = np.argmax(rel_diff)
    #     print(f"Maximale absolute Abweichung: {abs_diff[max_abs_diff_idx]:.2e} bei Index {max_abs_diff_idx}")
    #     print(f"Maximale relative Abweichung: {rel_diff[max_rel_diff_idx]:.2e} bei Index {max_rel_diff_idx}")



    # d) Plotten Sie den Laternenstab unter Last. Nutzen Sie dafür die gleiche Funktion wie
    # beim ersten Plot.
    phi_elements_loaded_deg = np.zeros(N_aufg1)
    for i in range(N_aufg1):
        node_L_idx_0=i; node_R_idx_0=i+1
        x_L=full_x_load_aufg1[node_L_idx_0*3]; y_L=full_x_load_aufg1[node_L_idx_0*3+1]
        x_R=full_x_load_aufg1[node_R_idx_0*3]; y_R=full_x_load_aufg1[node_R_idx_0*3+1]
        phi_elements_loaded_deg[i] = np.rad2deg(np.arctan2(y_R-y_L, x_R-x_L))
    plottePositionen(full_x_load_aufg1, stangen_zu_knoten, allphi=phi_elements_loaded_deg, filename='loaded_geometry.png')
    check_Lengths(full_x_load_aufg1, stangen_zu_knoten)
except np.linalg.LinAlgError:
    print(f"FEHLER: Singuläre Matrix bei der Berechnung für N={N_aufg1}. Berechnung für Teil a-d abgebrochen.")

print(f'--- Kontrolllösung aus der Aufgabenstellung ---')
print("\nHinweis: Zur Selbstkontrolle. Der globale Koordinatenvektor sei gegeben durch")
print("['x1', 'y1', 'phi1', 'x2', 'y2', 'phi2',")
print(" 'x3', 'y3', 'phi3', 'x4', 'y4', 'phi4',")
print(" 'x5', 'y5', 'phi5', 'x6', 'y6', 'phi6',")
print(" 'x7', 'y7', 'phi7', 'x8', 'y8', 'phi8']")

# --- Aufgabe e) Konvergenz ---
# Erhöhen Sie die Anzahl der Elemente N. Wie verhält sich die Kontur des Stabs unter
# Last? Konvergiert sie?
print("\n\n--- Aufgabe e) Konvergenzstudie ---")
N_values_konvergenz = [1, 4, 7, 10, 15, 20, 30, 40, 50, 70, 100]
results_last_node_x = []
results_last_node_y = []
results_last_node_phi = []
valid_N_values_konvergenz = []

L_gesamt = L_gesamt_param
phi_laterne_deg = phi_laterne_deg_param
m_load = m_load_param
d_mm = d_mm_param
E_Npmmsq = E_Npmmsq_param
g = g_scipy
F_gravity = -m_load * g

d_m_conv = d_mm * 1e-3
E_Npm2_conv = E_Npmmsq * (1e3)**2
A_conv = np.pi * (d_m_conv / 2)**2
I_val_problem_units_conv = (np.pi / 64) * (d_mm)**2 * (0.001 * d_mm)**2
I_conv = I_val_problem_units_conv * 1e-6

for N_iter in N_values_konvergenz:
    print(f"-- Berechne für N = {N_iter} --")
    N = N_iter

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

    local_stiffness_matrices_rotated = []
    K_unrotiert = lokale_steifheitsmatrix_unrotiert(L_element, A_conv, E_Npm2_conv, I_conv)
    for i in range(N):
        R = lokale_rotationsmatrix(initial_element_angles_deg[i])
        K_rotiert = R @ K_unrotiert @ R.T
        local_stiffness_matrices_rotated.append(K_rotiert)

    global_stiffness_matrix_size = num_nodes * 3
    K_global = np.zeros((global_stiffness_matrix_size, global_stiffness_matrix_size))
    for element_id, (node_L_id, node_R_id) in stangen_zu_knoten.items():
        K_local_rot = local_stiffness_matrices_rotated[element_id - 1]
        idx_L_0 = node_L_id - 1; idx_R_0 = node_R_id - 1
        dofs_L = [idx_L_0*3,idx_L_0*3+1,idx_L_0*3+2]; dofs_R = [idx_R_0*3,idx_R_0*3+1,idx_R_0*3+2]
        global_dof_indices = np.array(dofs_L + dofs_R)
        for r_local in range(6):
            for c_local in range(6):
                K_global[global_dof_indices[r_local],global_dof_indices[c_local]] += K_local_rot[r_local,c_local]

    fixed_dofs = [0, 1, 2]
    K_reduced = np.delete(K_global, fixed_dofs, axis=0)
    K_reduced = np.delete(K_reduced, fixed_dofs, axis=1)

    F_global_vec_konv = np.zeros(global_stiffness_matrix_size)
    last_node_idx_0_konv = N
    y_dof_last_node_konv = last_node_idx_0_konv * 3 + 1
    F_global_vec_konv[y_dof_last_node_konv] = F_gravity

    F_reduced = np.delete(F_global_vec_konv, fixed_dofs, axis=0)

    try:
        delta_reduced = np.linalg.solve(K_reduced, F_reduced)
        delta_full = np.zeros(global_stiffness_matrix_size)
        free_dofs_mask = np.ones(global_stiffness_matrix_size, dtype=bool)
        free_dofs_mask[fixed_dofs] = False
        delta_full[free_dofs_mask] = delta_reduced
        full_x_load_konvergenz = full_x_initial + delta_full

        results_last_node_x.append(full_x_load_konvergenz[-3])
        results_last_node_y.append(full_x_load_konvergenz[-2])
        results_last_node_phi.append(full_x_load_konvergenz[-1])
        valid_N_values_konvergenz.append(N_iter)
        # print(f"N={N_iter}: x_tip={full_x_load_konvergenz[-3]:.8f}, y_tip={full_x_load_konvergenz[-2]:.8f}, phi_tip={full_x_load_konvergenz[-1]:.8f}") # Weniger Output während der Schleife
    except np.linalg.LinAlgError:
        print(f"WARNUNG: Singuläre Matrix für N={N_iter} in Konvergenzstudie. Überspringe diesen Wert.")

print("\n--- Ergebnisse der Konvergenzstudie für den letzten Knoten ---")
print("N \t x_tip (m) \t y_tip (m) \t phi_tip (rad)")
for i, n_val in enumerate(valid_N_values_konvergenz):
    print(f"{n_val} \t {results_last_node_x[i]:.8f} \t {results_last_node_y[i]:.8f} \t {results_last_node_phi[i]:.8f}")

if valid_N_values_konvergenz:
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(valid_N_values_konvergenz, results_last_node_x, 'o-', label='x_tip')
    plt.xlabel('Anzahl der Elemente (N)'); plt.ylabel('x-Koordinate Spitze (m)'); plt.title('Konvergenz der x-Koordinate der Spitze'); plt.grid(True); plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(valid_N_values_konvergenz, results_last_node_y, 'o-', label='y_tip')
    plt.xlabel('Anzahl der Elemente (N)'); plt.ylabel('y-Koordinate Spitze (m)'); plt.title('Konvergenz der y-Koordinate der Spitze'); plt.grid(True); plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(valid_N_values_konvergenz, results_last_node_phi, 'o-', label='phi_tip')
    plt.xlabel('Anzahl der Elemente (N)'); plt.ylabel('phi-Koordinate Spitze (rad)'); plt.title('Konvergenz der phi-Koordinate der Spitze'); plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig('konvergenz_studie_spitze.png')
    print("\nKonvergenzplot wurde als 'konvergenz_studie_spitze.png' gespeichert.")
else:
    print("\nKeine gültigen Ergebnisse für die Konvergenzstudie vorhanden, um zu plotten.")

# --- Antwort zu Aufgabe e) als print statement ---
print("\n--- Antwort zu Aufgabe e) ---")
print("Verhalten der Kontur des Stabs unter Last bei Erhöhung der Elementanzahl N:")
print("Mit zunehmender Anzahl an Elementen (N) nähert sich die berechnete Kontur des Stabs unter Last")
print("immer genauer der tatsächlichen, kontinuierlichen Verformungskurve an. Bei geringer Elementanzahl")
print("ist die Diskretisierung grob, und die berechnete Verformung zwischen den Knoten ist nur eine")
print("lineare (oder kubische, je nach Elementtyp) Interpolation. Dies kann zu einer sichtbaren,")
print("segmentierten Darstellung der Biegelinie führen.")
print("\nKonvergenz der Kontur:")
print("Ja, die Kontur konvergiert. Das bedeutet, dass sich die berechneten Knotenverschiebungen (und damit")
print("die Form des Stabes) mit steigendem N einem stabilen Grenzwert annähern. Die Unterschiede zwischen")
print("den Lösungen für N und N+1 (oder einem deutlich größeren N) werden immer kleiner.")
print("Dies ist im Konvergenzplot der x-, y- und phi-Koordinaten der Stabspitze ersichtlich, wo die")
print("Kurven für größere N-Werte abflachen und sich einem asymptotischen Wert nähern.")
print("Diese Konvergenz ist ein wichtiges Kriterium für die Zuverlässigkeit einer Finite-Elemente-Analyse.")
print("Sie zeigt, dass die numerische Lösung nicht mehr stark von der Feinheit der Diskretisierung abhängt,")
print("sobald eine ausreichende Anzahl von Elementen verwendet wird.")