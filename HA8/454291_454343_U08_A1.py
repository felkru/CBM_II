# 1)
# Matrikelnummer: 454291
# Name: Julia Els
# Email: julia.els@rwth-aachen.de
#
# 2)
# Matrikelnummer: 454343
# Name: Felix Krückel
# Email: felix.krueckel@rwth-aachen.de


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from collections import deque
from matplotlib.collections import LineCollection # Import ist hier korrekt

# Globale Konstanten für Massen (verwendet von TheFunction)
m1 = 1.0
m2 = 1.0
m3 = 1.0
G_const = 1.0 # Explizite Gravitationskonstante für Energieberechnung

# Anfangszustand aus Aufgabe 1b) als Referenz für den Testaufruf
INITIAL_STATE_FIG8 = np.array([
    -0.97000436,  0.24308753,    # x1 (x, y)
    0.97000436, -0.24308753,    # x2 (x, y)
    0.0,          0.0,          # x3 (x, y)
    0.4662036850, 0.4323657300, # p1 (px, py)
    0.4662036850, 0.4323657300, # p2 (px, py)
    -0.93240737,  -0.86473146    # p3 (px, py)
])

# --- Systemdefinition und Integratoren ---
def TheFunction(t, theState): # G=1 ist hier implizit
    x1, y1, x2, y2, x3, y3, px1, py1, px2, py2, px3, py3 = theState
    x1d = px1 / m1; y1d = py1 / m1
    x2d = px2 / m2; y2d = py2 / m2
    x3d = px3 / m3; y3d = py3 / m3
    epsilon_div = 1e-9 # Für Kraftberechnung
    r_sq_12 = (x1 - x2)**2 + (y1 - y2)**2
    r_sq_13 = (x1 - x3)**2 + (y1 - y3)**2
    r_sq_23 = (x2 - x3)**2 + (y2 - y3)**2
    common12 = m1 * m2 * (r_sq_12 + epsilon_div)**(-1.5)
    common13 = m1 * m3 * (r_sq_13 + epsilon_div)**(-1.5)
    common23 = m2 * m3 * (r_sq_23 + epsilon_div)**(-1.5)
    p1xd = (x2 - x1) * common12 + (x3 - x1) * common13
    p1yd = (y2 - y1) * common12 + (y3 - y1) * common13
    p2xd = (x1 - x2) * common12 + (x3 - x2) * common23
    p2yd = (y1 - y2) * common12 + (y3 - y2) * common23
    p3xd = (x1 - x3) * common13 + (x2 - x3) * common23
    p3yd = (y1 - y3) * common13 + (y2 - y3) * common23
    return np.array([x1d, y1d, x2d, y2d, x3d, y3d, p1xd, p1yd, p2xd, p2yd, p3xd, p3yd])

def SimpleEuler(theState, current_h_step):
    return theState + current_h_step * TheFunction(0, theState)

def BetterEuler(theState, current_h_step):
    g = theState + current_h_step / 2 * TheFunction(0, theState)
    return theState + current_h_step * TheFunction(0, g)

def RK45Step(theState, t, h_val_step):
    sol = solve_ivp(TheFunction, (t, t + h_val_step), theState, method='RK45', t_eval=[t + h_val_step], rtol=1e-8, atol=1e-10)
    return sol.y[:, -1]

# --- Aufgabe 1g: Energieberechnung ---
def calculate_total_energy(theState):
    x1, y1, x2, y2, x3, y3, px1, py1, px2, py2, px3, py3 = theState

    T1 = (px1**2 + py1**2) / (2 * m1)
    T2 = (px2**2 + py2**2) / (2 * m2)
    T3 = (px3**2 + py3**2) / (2 * m3)
    T_total = T1 + T2 + T3

    epsilon_dist = 1e-12
    r12 = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + epsilon_dist)
    r13 = np.sqrt((x1 - x3)**2 + (y1 - y3)**2 + epsilon_dist)
    r23 = np.sqrt((x2 - x3)**2 + (y2 - y3)**2 + epsilon_dist)

    U12 = -G_const * m1 * m2 / r12
    U13 = -G_const * m1 * m3 / r13
    U23 = -G_const * m2 * m3 / r23
    U_total = U12 + U13 + U23

    return T_total + U_total

# --- Animationsfunktion mit Energieplot ---
def make_animation_with_energy(filename, initial_state_vector, h_step, total_time_to_simulate, trail_length=100, x_lims=None, y_lims=None, figure_title_prefix=""):
    print(f"Starte Animation mit Energieplot: Ziel-Datei='{filename}', h={h_step}, Simulationszeit={total_time_to_simulate}s")

    state_euler = np.array(initial_state_vector)
    state_better_euler = np.array(initial_state_vector)
    state_rk45 = np.array(initial_state_vector)
    current_t = 0.0

    time_points = [0.0]
    energy_euler_hist = [calculate_total_energy(state_euler)]
    energy_better_euler_hist = [calculate_total_energy(state_better_euler)]
    energy_rk45_hist = [calculate_total_energy(state_rk45)]

    trails = [deque(maxlen=trail_length) for _ in range(9)]
    for i_particle in range(3):
        trails[i_particle].append((state_euler[2*i_particle], state_euler[2*i_particle+1]))
        trails[3+i_particle].append((state_better_euler[2*i_particle], state_better_euler[2*i_particle+1]))
        trails[6+i_particle].append((state_rk45[2*i_particle], state_rk45[2*i_particle+1]))

    fig = plt.figure(figsize=(18, 7))
    ax_anim = fig.add_subplot(1, 2, 1)
    ax_energy = fig.add_subplot(1, 2, 2)

    ax_anim.set_facecolor('white')
    colors_scatter_full = ['blue']*3 + ['red']*3 + ['green']*3
    initial_positions_full = []
    for i in range(3): initial_positions_full.append((state_euler[2*i], state_euler[2*i+1]))
    for i in range(3): initial_positions_full.append((state_better_euler[2*i], state_better_euler[2*i+1]))
    for i in range(3): initial_positions_full.append((state_rk45[2*i], state_rk45[2*i+1]))

    valid_initial_scatter_pos = [p for p in initial_positions_full if np.all(np.isfinite(p))]
    valid_initial_colors = [colors_scatter_full[i] for i, p in enumerate(initial_positions_full) if np.all(np.isfinite(p))]

    scatter_zorder = 10
    if valid_initial_scatter_pos:
        scat = ax_anim.scatter(np.array(valid_initial_scatter_pos)[:,0], np.array(valid_initial_scatter_pos)[:,1],
                               c=valid_initial_colors, s=35, zorder=scatter_zorder,
                               edgecolors='black', linewidths=0.5)
    else:
        scat = ax_anim.scatter([], [], s=35, zorder=scatter_zorder, edgecolors='black', linewidths=0.5)

    base_trail_colors_rgb = [
                                np.array([0,0,1])]*3 + [np.array([1,0,0])]*3 + [np.array([0,0.7,0])]*3
    bg_color_rgb = np.array([1.0,1.0,1.0])
    trail_linecollections = []
    zorders_trails_map = {'euler':3,'better_euler':2,'rk45':1}
    zorders_trails = [zorders_trails_map['euler']]*3 + [zorders_trails_map['better_euler']]*3 + [zorders_trails_map['rk45']]*3
    for i_trail_lc in range(9):
        lc = LineCollection([], lw=2.5, zorder=zorders_trails[i_trail_lc])
        ax_anim.add_collection(lc)
        trail_linecollections.append(lc)

    if x_lims is None: x_lims = [-1.5, 1.5]
    if y_lims is None: y_lims = [-1.2, 1.2]
    ax_anim.set(xlim=x_lims, ylim=y_lims); ax_anim.set_aspect('equal', adjustable='box'); ax_anim.grid(True, linestyle=':', alpha=0.6)
    ax_anim.set_title('Partikelbewegung')

    legend_handles_anim = [
        plt.Line2D([0],[0],marker='o',color='w',label='Euler',markerfacecolor='blue',markersize=8,linestyle='None'),
        plt.Line2D([0],[0],marker='o',color='w',label='Verb. Euler',markerfacecolor='red',markersize=8,linestyle='None'),
        plt.Line2D([0],[0],marker='o',color='w',label='RK45',markerfacecolor='green',markersize=8,linestyle='None')]
    ax_anim.legend(handles=legend_handles_anim,loc='upper right',fontsize='small',frameon=True,facecolor='white',framealpha=0.85,title="Algorithmen")

    line_energy_euler, = ax_energy.plot([], [], 'b-', label='Euler Energie')
    line_energy_better_euler, = ax_energy.plot([], [], 'r-', label='Verb. Euler Energie')
    line_energy_rk45, = ax_energy.plot([], [], 'g-', label='RK45 Energie')
    initial_total_energy = energy_euler_hist[0]
    ax_energy.axhline(initial_total_energy, color='k', linestyle='--', label=f'Initiale E = {initial_total_energy:.3f}', lw=1.0)
    ax_energy.set_xlabel('Zeit (s)'); ax_energy.set_ylabel('Gesamtenergie E'); ax_energy.set_title('Energieerhaltung'); ax_energy.grid(True)
    ax_energy.legend(loc='upper left', fontsize='small')

    fig.suptitle(f'{figure_title_prefix}3-Körper-Problem mit Energie (h={h_step:.4f})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    num_frames = int(total_time_to_simulate / h_step)
    non_finite_logged = {'euler': False, 'better_euler': False, 'rk45': False}

    def update_frame(frame_num):
        nonlocal current_t, state_euler, state_better_euler, state_rk45, non_finite_logged

        current_t_before_step = current_t
        state_euler = SimpleEuler(state_euler, h_step)
        state_better_euler = BetterEuler(state_better_euler, h_step)
        state_rk45 = RK45Step(state_rk45, current_t_before_step, h_step)
        current_t += h_step
        time_points.append(current_t)

        energy_euler_hist.append(calculate_total_energy(state_euler))
        energy_better_euler_hist.append(calculate_total_energy(state_better_euler))
        energy_rk45_hist.append(calculate_total_energy(state_rk45))

        if not np.all(np.isfinite(state_euler)) and not non_finite_logged['euler']: print(f"INFO: Euler non-finite at t={current_t-h_step:.4f}"); non_finite_logged['euler']=True
        if not np.all(np.isfinite(state_better_euler)) and not non_finite_logged['better_euler']: print(f"INFO: BetterEuler non-finite at t={current_t-h_step:.4f}"); non_finite_logged['better_euler']=True
        if not np.all(np.isfinite(state_rk45)) and not non_finite_logged['rk45']: print(f"INFO: RK45 non-finite at t={current_t-h_step:.4f}"); non_finite_logged['rk45']=True

        current_scatter_positions_all_methods=[]
        for i_particle in range(3): pos=(state_euler[2*i_particle],state_euler[2*i_particle+1]); current_scatter_positions_all_methods.append(pos); trails[i_particle].append(pos) if np.all(np.isfinite(pos)) else None
        for i_particle in range(3): pos=(state_better_euler[2*i_particle],state_better_euler[2*i_particle+1]); current_scatter_positions_all_methods.append(pos); trails[3+i_particle].append(pos) if np.all(np.isfinite(pos)) else None
        for i_particle in range(3): pos=(state_rk45[2*i_particle],state_rk45[2*i_particle+1]); current_scatter_positions_all_methods.append(pos); trails[6+i_particle].append(pos) if np.all(np.isfinite(pos)) else None

        valid_scatter_positions_display=[p for p in current_scatter_positions_all_methods if np.all(np.isfinite(p))]
        valid_scatter_colors_display=[colors_scatter_full[i] for i,p in enumerate(current_scatter_positions_all_methods) if np.all(np.isfinite(p))]
        if valid_scatter_positions_display: scat.set_offsets(np.array(valid_scatter_positions_display)); scat.set_facecolors(valid_scatter_colors_display); scat.set_edgecolors(['black']*len(valid_scatter_positions_display))
        else: scat.set_offsets(np.empty((0,2)))

        for i_trail,trail_deque in enumerate(trails):
            if len(trail_deque)<2: trail_linecollections[i_trail].set_segments([]); continue
            points=np.array(list(trail_deque)); segments=[]; segment_colors_rgba=[]; base_color_rgb_current_trail=base_trail_colors_rgb[i_trail]; num_segments_in_trail=len(points)-1
            for j in range(num_segments_in_trail):
                segments.append([points[j],points[j+1]]); age_of_segment=(num_segments_in_trail-1)-j; fade_ratio=age_of_segment/max(1,trail_length-1); fade_ratio=np.clip(fade_ratio,0,1)
                faded_r=base_color_rgb_current_trail[0]*(1-fade_ratio)+bg_color_rgb[0]*fade_ratio; faded_g=base_color_rgb_current_trail[1]*(1-fade_ratio)+bg_color_rgb[1]*fade_ratio; faded_b=base_color_rgb_current_trail[2]*(1-fade_ratio)+bg_color_rgb[2]*fade_ratio
                segment_colors_rgba.append((faded_r,faded_g,faded_b,1.0))
            trail_linecollections[i_trail].set_segments(segments)
            if segment_colors_rgba: trail_linecollections[i_trail].set_colors(segment_colors_rgba)
            else: trail_linecollections[i_trail].set_segments([])
        ax_anim.set_title(f'Partikelbewegung (t={current_t:.3f}s)')

        line_energy_euler.set_data(time_points, energy_euler_hist)
        line_energy_better_euler.set_data(time_points, energy_better_euler_hist)
        line_energy_rk45.set_data(time_points, energy_rk45_hist)
        ax_energy.relim(); ax_energy.autoscale_view()
        current_energies = [e for hist in [energy_euler_hist, energy_better_euler_hist, energy_rk45_hist] for e in hist if np.isfinite(e)] # Flache Liste aller gültigen Energien
        if current_energies:
            emin = min(current_energies + [initial_total_energy])
            emax = max(current_energies + [initial_total_energy])
            padding = (emax - emin) * 0.1 if (emax - emin) > 1e-9 else 0.1
            ax_energy.set_ylim(emin - padding, emax + padding)
        ax_energy.set_xlim(0, max(time_points) if time_points else 1)

        return [scat] + trail_linecollections + [line_energy_euler, line_energy_better_euler, line_energy_rk45]

    ani = animation.FuncAnimation(fig, update_frame, frames=num_frames,
                                  interval=33, blit=True, repeat=False)
    try:
        print(f"Speichere Animation als '{filename}'..."); progress_callback=lambda c,t: print(f'Video: Frame {c+1}/{t}') if (c+1)%max(1,t//20)==0 or c==t-1 else None
        ani.save(filename, writer='ffmpeg', fps=30, dpi=150, progress_callback=progress_callback)
        print(f"Animation erfolgreich als '{filename}' gespeichert.")
    except Exception as e: print(f"Fehler beim Speichern: {e}")
    finally:
        if non_finite_logged['euler']: print("INFO: Euler NaN.")
        if non_finite_logged['better_euler']: print("INFO: BetterEuler NaN.")
        if non_finite_logged['rk45']: print("INFO: RK45 NaN.")
    plt.close(fig)
# --- Ende der Animationsfunktion ---

# --- Aufgabe 1e) ---
def print_comparison_of_algorithms():
    print("\n--- Aufgabe 1e: Vergleich der Algorithmen ---")
    comparison_text = """
    Das Verhalten der drei Algorithmen (Einfacher Euler, Verbesserter Euler/Mittelpunktmethode, 
    Runge-Kutta 4/5) im Drei-Körper-Problem, insbesondere bei der Figure-8-Lösung, 
    zeigt deutliche Unterschiede in Genauigkeit und Stabilität:

    1. Runge-Kutta 4/5 (RK45 - Grün):
       - Verhalten: Dient als die genaueste Referenz. Diese Methode ist von höherer Ordnung 
         und verfügt oft über eine Fehlerabschätzung, die eine Anpassung der internen Schrittweite 
         ermöglicht (obwohl wir hier eine feste externe Schrittweite h verwenden).
       - Genauigkeit: RK45 bleibt der analytischen (oder zumindest der stabilen numerischen) 
         Lösung der Figure-8-Bahn am längsten treu.
       - Stabilität: Aufgrund der höheren Ordnung und Fehlerkontrolle ist RK45 am stabilsten 
         und widerstandsfähigsten gegen die Akkumulation von Fehlern.

    2. Verbesserter Euler / Mittelpunktmethode (Rot):
       - Verhalten: Als Methode zweiter Ordnung ist sie eine deutliche Verbesserung gegenüber 
         dem einfachen Euler. Die Trajektorie folgt der RK45-Lösung länger.
       - Genauigkeit: Der globale Fehler ist proportional zu h^2.
       - Stabilität: Stabiler als der einfache Euler, aber anfällig für Fehlerakkumulation 
         über längere Zeiträume, besonders bei nahen Begegnungen. Bei h=0.01 kann er 
         instabil werden und NaN-Werte produzieren. Eine Verringerung der Schrittweite 
         verbessert die Stabilität. Die Überdeckung durch RK45 (wenn h klein genug ist) zeigt, 
         dass er anfangs eine gute Annäherung liefert.

    3. Einfacher Euler (Blau):
       - Verhalten: Als Methode erster Ordnung ist dies der ungenaueste Algorithmus. 
         Die Trajektorie weicht sehr schnell und signifikant von der RK45-Lösung ab.
       - Genauigkeit: Der globale Fehler ist proportional zu h.
       - Stabilität: Oft nicht energieerhaltend für Hamiltonsche Systeme. Führt typischerweise 
         dazu, dass die Orbits künstlich Energie gewinnen oder verlieren. Für eine qualitative 
         Simulation dieses Problems bei h=0.01 ist er ungeeignet.

    Zusammenfassend: Die Komplexität und Genauigkeit der Algorithmen spiegelt sich direkt 
    in ihrer Fähigkeit wider, das chaotische Drei-Körper-Problem korrekt zu simulieren. 
    RK45 bietet die beste Balance. Der verbesserte Euler ist ein Kompromiss. Der einfache Euler 
    dient primär als Demonstration einer fehleranfälligen Methode.
    """
    print(comparison_text)
    print("--- Ende Aufgabe 1e ---\n")

# --- Aufgabe 1f) Definitionen ---
L_triangle_config = 2.0
d_triangle_config = L_triangle_config / np.sqrt(3.0)
v_triangle_config = np.sqrt(m1 / L_triangle_config)
INITIAL_STATE_TRIANGLE = np.array([
    d_triangle_config,0.0, d_triangle_config*np.cos(2*np.pi/3),d_triangle_config*np.sin(2*np.pi/3),
                           d_triangle_config*np.cos(4*np.pi/3),d_triangle_config*np.sin(4*np.pi/3), 0.0,v_triangle_config,
                           v_triangle_config*np.cos(2*np.pi/3+np.pi/2),v_triangle_config*np.sin(2*np.pi/3+np.pi/2),
                           v_triangle_config*np.cos(4*np.pi/3+np.pi/2),v_triangle_config*np.sin(4*np.pi/3+np.pi/2)
])

# --- Aufgabe 1g) Textuelle Antwort ---
def print_energy_observations():
    print("\n--- Aufgabe 1g: Beobachtungen zur Energieerhaltung ---")
    energy_text = """
    Physikalische Erwartung:
    In einem abgeschlossenen System, das nur durch konservative Kräfte (wie die Gravitation) 
    beeinflusst wird, sollte die Gesamtenergie E = T + U (kinetische + potentielle Energie) 
    erhalten bleiben, d.h. über die Zeit konstant sein.

    Beobachtungen bei der numerischen Lösung (typischerweise):
    - RK45 (Grün): Zeigt die beste Energieerhaltung. Die Gesamtenergie schwankt nur minimal 
      um den Anfangswert. Die kleinen Schwankungen sind auf numerische Rundungsfehler und 
      die endliche Schrittweite zurückzuführen. Dies ist ein Zeichen für seine hohe Genauigkeit.
    - Verbesserter Euler (Rot): Hält die Energie besser als der einfache Euler. Es ist jedoch 
      meist ein leichter, aber systematischer Drift (Zunahme oder Abnahme) der Energie über 
      die Zeit zu beobachten. Dieser Drift ist bei größerer Schrittweite h ausgeprägter.
    - Einfacher Euler (Blau): Zeigt die schlechteste Energieerhaltung. Die Energie driftet 
      oft stark und systematisch vom Anfangswert weg. Für Hamiltonsche Systeme wie dieses 
      neigt der einfache Euler dazu, die Energie künstlich zu erhöhen, was zu unrealistisch 
      expandierenden oder instabilen Orbits führt. Dies liegt daran, dass er nicht 
      symplektisch ist.

    Fazit: Der Energieplot bestätigt die Hierarchie der Genauigkeit und Stabilität der 
    Methoden. Methoden höherer Ordnung und solche mit speziellen Eigenschaften (wie Symplektizität, 
    die RK45 in der 'solve_ivp'-Implementierung nicht unbedingt hat, aber dennoch sehr gut ist) 
    sind für die Langzeitsimulation physikalischer Systeme, bei denen Erhaltungsgrößen wichtig sind, 
    deutlich überlegen.
    """
    print(energy_text)
    print("--- Ende Aufgabe 1g ---\n")


if __name__ == '__main__':
    # Aufgabe 1e ausgeben
    print_comparison_of_algorithms()

    # Aufgabe 1g Text ausgeben
    print_energy_observations()

    # --- Animation für Aufgabe 1d mit Energieplot (Aufgabe 1g) ---
    print("Erstelle Animation für Aufgabe 1d mit Energieplot (Aufgabe 1g)...")
    simulation_h_step_d_g = 0.01
    animation_total_time_d_g = 7.0
    animation_trail_length_d_g = 100
    output_video_filename_d_g = "454291_454343_U08_A1_mit_Energie.mp4" # Dateiname angepasst
    # make_animation_with_energy(filename=output_video_filename_d_g,
    #                            initial_state_vector=INITIAL_STATE_FIG8,
    #                            h_step=simulation_h_step_d_g,
    #                            total_time_to_simulate=animation_total_time_d_g,
    #                            trail_length=animation_trail_length_d_g,
    #                            x_lims=[-1.5, 1.5], y_lims=[-1.2, 1.2],
    #                            figure_title_prefix="Figure-8: ")


    # --- Teil f) Animation (Stabiles Dreieck) mit Energieplot (Aufgabe 1g hier auch angewendet) ---
    print("\nErstelle Animation für Aufgabe 1f) (Stabiles Dreieck) mit Energieplot...")
    simulation_h_step_f_g = 0.01
    animation_total_time_f_g = 15.0
    animation_trail_length_f_g = 150
    output_video_filename_f_g = "454291_454343_U08_A2_mit_Energie.mp4" # Dateiname angepasst
    triangle_anim_xlims = [-2.5, 2.5] # Etwas größere Limits für L=2 Dreieck
    triangle_anim_ylims = [-2.2, 2.2]
    # make_animation_with_energy(filename=output_video_filename_f_g,
    #                            initial_state_vector=INITIAL_STATE_TRIANGLE,
    #                            h_step=simulation_h_step_f_g,
    #                            total_time_to_simulate=animation_total_time_f_g,
    #                            trail_length=animation_trail_length_f_g,
    #                            x_lims=triangle_anim_xlims,
    #                            y_lims=triangle_anim_ylims,
    #                            figure_title_prefix="Stabiles Dreieck: ")
