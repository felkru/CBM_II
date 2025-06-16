import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

# --- Functions from Library_simple_finite_elements.py (as provided by user) ---

def lokale_rotationsmatrix(phi):
    phi = np.deg2rad(phi)
    R = np.eye(6)
    c = np.cos(phi)
    s = np.sin(phi)
    r_sub_rotation = np.array([[c, -s, 0],
                               [s,  c, 0],
                               [0,  0, 1]])
    R[0:3, 0:3] = r_sub_rotation
    R[3:6, 3:6] = r_sub_rotation
    return R

def lokale_steifheitsmatrix_unrotiert(L, A=1, E=1, I=1): # This is the function from user's block
    alpha = A * E / L
    beta = E * I / (L**3)

    K_unrot = np.zeros((6,6))
    K_unrot[0,0] = K_unrot[3,3] = alpha
    K_unrot[0,3] = K_unrot[3,0] = -alpha

    K_unrot[1,1] = K_unrot[4,4] = 12 * beta
    K_unrot[2,2] = K_unrot[5,5] = 4 * (L**2) * beta

    K_unrot[1,4] = K_unrot[4,1] = -12 * beta

    K_unrot[1,2] = K_unrot[2,1] = K_unrot[1,5] = K_unrot[5,1] = 6 * L * beta

    # Implementing the specific line from the user's provided snippet.
    # R[2,4] = R[4,2] = R[4,5] = R[5,4] = -6*L*beta
    K_unrot[2,4] = -6 * L * beta
    K_unrot[4,2] = -6 * L * beta
    K_unrot[4,5] = -6 * L * beta
    K_unrot[5,4] = -6 * L * beta

    K_unrot[2,5] = K_unrot[5,2] = 2 * (L**2) * beta
    return K_unrot

# --- Plotting functions (kept from previous solution, slightly enhanced) ---
def plottePositionen(full_x, stangen_zu_knoten,allphi=None,filename=None, show_labels = True, title_suffix=""):
    # (Implementation from previous response - assumed correct by user)
    plt.figure(figsize=(10, 8))
    points = [(full_x[i], full_x[i + 1]) for i in range(0, len(full_x), 3)]
    cmap_points = get_cmap('viridis')
    cmap_stangen = get_cmap('plasma')

    num_points = len(points)
    points_colors = [cmap_points(i / (num_points -1 if num_points > 1 else 1) ) for i in range(num_points)]

    num_stangen = np.max(list(stangen_zu_knoten.keys())) if stangen_zu_knoten else 0
    stangen_colors = [cmap_stangen(i / (num_stangen-1 if num_stangen > 1 else 1)) for i in range(num_stangen)] if num_stangen > 0 else []


    for i, (x_i, y_i) in enumerate(points):
        plt.plot(x_i, y_i, 'o', label=f'Knoten {i + 1}',color=points_colors[i])
        if show_labels:
            # Adjust text offset based on overall scale of coordinates
            offset_scale_x = 0.01 * (np.max(full_x[0::3]) - np.min(full_x[0::3])) if num_points > 0 else 0.01
            offset_scale_y = 0.025 * (np.max(full_x[1::3]) - np.min(full_x[1::3])) if num_points > 0 else 0.025
            if offset_scale_x == 0: offset_scale_x = 0.01 # handle case of single point or all x same
            if offset_scale_y == 0: offset_scale_y = 0.025

            plt.text(x_i + offset_scale_x, y_i + offset_scale_y, f'{i + 1}', fontsize=10, color=points_colors[i],ha='center')

    if num_stangen > 0:
        for stange_id_1based, (knoten_links_idx_1based, knoten_rechts_idx_1based) in stangen_zu_knoten.items():
            knoten_links_idx_0based = knoten_links_idx_1based -1
            knoten_rechts_idx_0based = knoten_rechts_idx_1based -1
            stange_id_0based = stange_id_1based -1

            x_values = [points[knoten_links_idx_0based][0], points[knoten_rechts_idx_0based][0]]
            y_values = [points[knoten_links_idx_0based][1], points[knoten_rechts_idx_0based][1]]

            plt.plot(x_values, y_values, color=stangen_colors[stange_id_0based])

            mid_point_x = (x_values[0] + x_values[1]) / 2
            mid_point_y = (y_values[0] + y_values[1]) / 2

            elem_len = np.sqrt((x_values[1]-x_values[0])**2 + (y_values[1]-y_values[0])**2)
            if elem_len == 0: elem_len = 1.0 # avoid division by zero if coincident points

            if show_labels:
                dx = x_values[1] - x_values[0]
                dy = y_values[1] - y_values[0]
                norm = np.sqrt(dx**2 + dy**2)
                if norm == 0: norm = 1.0 # avoid division by zero

                label_offset_x = -dy/norm * 0.05 * elem_len
                label_offset_y = dx/norm * 0.05 * elem_len
                plt.text(mid_point_x + label_offset_x, mid_point_y + label_offset_y,
                         fr'$S_{stange_id_1based}$', fontsize=10,
                         ha='center', color=stangen_colors[stange_id_0based])

            if allphi is not None and stange_id_0based < len(allphi):
                phi_value_deg = allphi[stange_id_0based]
                if True:
                    arc_radius = elem_len * 0.15
                    arc_center_x = points[knoten_links_idx_0based][0]
                    arc_center_y = points[knoten_links_idx_0based][1]

                    arc_angle_start_deg = 0
                    arc_angle_end_deg = phi_value_deg

                    arc = Arc((arc_center_x, arc_center_y),
                              arc_radius * 2,
                              arc_radius * 2,
                              angle=0,
                              theta1=arc_angle_start_deg,
                              theta2=arc_angle_end_deg,
                              color=stangen_colors[stange_id_0based],
                              linewidth=1.5)
                    plt.gca().add_patch(arc)

                    if show_labels:
                        label_angle_rad = np.deg2rad((arc_angle_start_deg + arc_angle_end_deg) / 2)
                        label_radius_scale = 1.3
                        label_x = arc_center_x + arc_radius * label_radius_scale * np.cos(label_angle_rad)
                        label_y = arc_center_y + arc_radius * label_radius_scale * np.sin(label_angle_rad)

                        plt.text(label_x, label_y,
                                 fr'$\phi_{stange_id_1based}$', fontsize=10,
                                 ha='center', va='center', color=stangen_colors[stange_id_0based])

    plt.xlabel('X-Koordinate (m)')
    plt.ylabel('Y-Koordinate (m)')
    plt.title(f'Plot der Knoten und Verbindungen{title_suffix}')
    plt.grid(True)
    plt.axis('equal')
    if show_labels and num_points > 0 :
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if filename is not None:
        plt.savefig(filename, dpi = 300)
    plt.show()

def plotMatrix(matrix,filename=None, title="Matrix Plot"):
    # (Implementation from previous response - assumed correct by user)
    matrix_plot = np.copy(matrix)
    masked_matrix = np.ma.masked_where(matrix_plot == 0, matrix_plot)
    valid_values = masked_matrix.data[~masked_matrix.mask] if np.any(masked_matrix.mask) else matrix_plot

    if valid_values.size == 0:
        max_abs_val = 1.0
    else:
        max_abs_val = np.max(np.abs(valid_values))
        if max_abs_val == 0 : max_abs_val = 1.0

    cmap = plt.get_cmap('RdBu_r', lut=256)
    norm = Normalize(vmin=-max_abs_val, vmax=max_abs_val)

    plt.figure(figsize=(10, 8))
    plt.imshow(masked_matrix, cmap=cmap, norm=norm, interpolation='nearest')
    plt.colorbar(label='Value')
    plt.xticks(ticks=np.arange(matrix_plot.shape[1]), labels=np.arange(1, matrix_plot.shape[1]+1))
    plt.yticks(ticks=np.arange(matrix_plot.shape[0]), labels=np.arange(1, matrix_plot.shape[0]+1))
    plt.xlabel('Spalten Index')
    plt.ylabel('Zeilen Index')
    plt.title(title)
    if filename is not None:
        plt.savefig(filename,dpi=300)
    plt.show()

# --- Main solver function ---
def solve_st_martin_lantern(N_elements, plot_results=True, verbose=True):
    # --- Constants and Parameters ---
    L_gesamt_m = 0.7  # m
    phi_overall_deg = 35.0  # degrees

    d_mm = 10.0  # mm
    E_N_mm2 = 1800.0  # N/mm^2
    mass_kg = 0.5  # kg
    g_m_s2 = 9.81  # m/s^2

    # --- Derived Parameters ---
    phi_overall_rad = np.deg2rad(phi_overall_deg)
    L_element_m = L_gesamt_m / N_elements
    num_nodes = N_elements + 1

    A_mm2 = np.pi * (d_mm / 2)**2
    # Using the specific I formula from the problem
    I_val = (np.pi / 64) * (d_mm**2) * ((0.001 * d_mm)**2)
    # This I_val has units mm^2 * m^2 as per problem statement.

    # --- a) Geometrie ---
    full_x_initial_m_rad = np.zeros(3 * num_nodes)
    for i in range(num_nodes):
        full_x_initial_m_rad[3 * i] = i * L_element_m * np.cos(phi_overall_rad)
        full_x_initial_m_rad[3 * i + 1] = i * L_element_m * np.sin(phi_overall_rad)
        full_x_initial_m_rad[3 * i + 2] = 0.0

    stangen_zu_knoten = {i: (i, i + 1) for i in range(1, N_elements + 1)}
    all_phi_elements_deg = [phi_overall_deg] * N_elements

    if plot_results:
        plottePositionen(full_x_initial_m_rad, stangen_zu_knoten, all_phi_elements_deg,
                         title_suffix=f" (Initial, N={N_elements})", filename=f"initial_N{N_elements}.png")

    # --- b) Gleichungssystem aufstellen ---
    K_global = np.zeros((3 * num_nodes, 3 * num_nodes))

    for stangen_id_1based, (n1_1based, n2_1based) in stangen_zu_knoten.items():
        # Explicitly use the provided lokale_steifheitsmatrix_unrotiert function
        K_local_unrot = lokale_steifheitsmatrix_unrotiert(L_element_m, A=A_mm2, E=E_N_mm2, I=I_val)

        element_phi_deg = phi_overall_deg
        R_mat = lokale_rotationsmatrix(element_phi_deg) # Use R_mat to avoid conflict with R in lokale_steifheitsmatrix

        K_local_rotated = R_mat.T @ K_local_unrot @ R_mat

        dof_indices_n1 = [3 * (n1_1based - 1) + k for k in range(3)]
        dof_indices_n2 = [3 * (n2_1based - 1) + k for k in range(3)]
        global_dof_indices = np.array(dof_indices_n1 + dof_indices_n2, dtype=int)

        for r_local, r_global in enumerate(global_dof_indices):
            for c_local, c_global in enumerate(global_dof_indices):
                K_global[r_global, c_global] += K_local_rotated[r_local, c_local]

    if plot_results and N_elements <= 10:
        plotMatrix(K_global, title=f"Globale Steifheitsmatrix (N={N_elements})", filename=f"K_global_N{N_elements}.png")

    # --- c) Gleichungssystem lösen ---
    fixed_dofs = [0, 1, 2]
    free_dofs = [i for i in range(3 * num_nodes) if i not in fixed_dofs]
    K_reduced = K_global[np.ix_(free_dofs, free_dofs)]

    F_full = np.zeros(3 * num_nodes)
    force_gravity_N = -mass_kg * g_m_s2
    idx_y_last_node = 3 * (num_nodes - 1) + 1
    F_full[idx_y_last_node] = force_gravity_N
    F_reduced = F_full[free_dofs]

    try:
        u_reduced = np.linalg.solve(K_reduced, F_reduced)
    except np.linalg.LinAlgError:
        if verbose: print(f"Singular matrix for N={N_elements}. Using pseudo-inverse.")
        u_reduced = np.linalg.pinv(K_reduced) @ F_reduced

    u_full = np.zeros(3 * num_nodes)
    u_full[free_dofs] = u_reduced
    full_x_final_m_rad = full_x_initial_m_rad + u_full

    if N_elements == 7 and verbose:
        print("\n--- N=7 Detailed Output ---")
        print("Initial coordinates (x, y in m; phi in rad):")
        print(np.array2string(full_x_initial_m_rad.reshape(-1,3), precision=8, separator=', ', suppress_small=False))
        print("\nDisplacement vector u_full (dx, dy in m; dphi in rad):")
        print(np.array2string(u_full.reshape(-1,3), precision=8, separator=', ', suppress_small=False))
        print("\nFinal coordinates (x, y in m; phi in rad) full_x_final_m_rad:")
        # Precise printout for comparison
        print("array([", end="")
        for i, val in enumerate(full_x_final_m_rad):
            if i > 0: print(",", end="")
            if i % 3 == 0 and i > 0: print("\n        ", end="") # Newline for each node
            print(f"{val:12.8f}", end="")
        print("])")


    # --- d) Plotten des Laternenstabs unter Last ---
    if plot_results:
        plottePositionen(full_x_final_m_rad, stangen_zu_knoten, all_phi_elements_deg,
                         title_suffix=f" (Unter Last, N={N_elements})", filename=f"deformed_N{N_elements}.png")

    tip_displacement_y = u_full[idx_y_last_node]
    tip_displacement_x = u_full[idx_y_last_node-1]

    return tip_displacement_x, tip_displacement_y, full_x_final_m_rad

# --- Aufgabe 1 ---
N_base = 7
print(f"--- Solving for N = {N_base} ---")
tip_dx_base, tip_dy_base, coords_final_base = solve_st_martin_lantern(N_elements=N_base, plot_results=True, verbose=True)
print(f"\nN={N_base}: Tip Dx = {tip_dx_base:.8f} m, Tip Dy = {tip_dy_base:.8f} m")
last_node_phi_displacement = coords_final_base[-1] # This is phi_final_node8
print(f"N={N_base}: Tip Dphi = {last_node_phi_displacement:.8f} rad")


# --- e) Erhöhen Sie die Anzahl der Elemente N ---
print("\n--- Aufgabe 1e: Konvergenzstudie ---")
N_values = [7, 10, 15, 20, 30, 50] # Reduced for brevity if needed
tip_displacements_y = []
tip_displacements_x = []
tip_displacements_phi = []


for N_val in N_values:
    # For convergence study, plotting individual steps can be turned off for speed
    plot_indiv = (N_val == N_values[0] or N_val == N_values[-1])
    if N_val > 20 and len(N_values)>3 : plot_indiv = False # Further reduce plotting for many Ns

    # print(f"--- Solving for N = {N_val} (verbose=False for convergence) ---")
    tip_dx, tip_dy, coords_final = solve_st_martin_lantern(N_elements=N_val, plot_results=plot_indiv, verbose=False if N_val!=7 else True)
    tip_displacements_x.append(tip_dx)
    tip_displacements_y.append(tip_dy)
    tip_displacements_phi.append(coords_final[-1]) # Final rotation of the last node
    if N_val != 7: # Avoid double printing for N=7
        print(f"N={N_val}: Tip Dx = {tip_dx:.6f} m, Tip Dy = {tip_dy:.6f} m, Tip Dphi = {coords_final[-1]:.6f} rad")


plt.figure(figsize=(12, 7))
plt.subplot(1,2,1)
plt.plot(N_values, tip_displacements_y, 'o-', label='Tip Vertical Displacement (Dy)')
plt.plot(N_values, tip_displacements_x, 's-', label='Tip Horizontal Displacement (Dx)')
plt.xlabel('Anzahl der Elemente (N)')
plt.ylabel('Verschiebung am Ende (m)')
plt.title('Konvergenz der Endverschiebung')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(N_values, np.rad2deg(tip_displacements_phi), '^-', label='Tip Rotation (Dphi_endknoten)')
plt.xlabel('Anzahl der Elemente (N)')
plt.ylabel('Rotation am Ende (Grad)')
plt.title('Konvergenz der Endrotation')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("convergence_tip_displacement_and_rotation.png")
plt.show()

print("\nVerhalten der Kontur des Stabs unter Last:")
print("Mit steigender Anzahl an Elementen N wird die berechnete Verformung des Stabs genauer.")
print("Die Form der Biegelinie konvergiert gegen eine glatte Kurve. Die Verschiebungen/Rotationen an der Spitze")
print("nähern sich asymptotisch einem Grenzwert an.")