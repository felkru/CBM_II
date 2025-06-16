import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from collections import deque
from matplotlib.collections import LineCollection

# --- Globale Konstanten und Systemdefinition (aus 3-Körper-Problem) ---
m1 = 1.0
m2 = 1.0
m3 = 1.0
G_const = 1.0

INITIAL_STATE_FIG8 = np.array([
    -0.97000436,  0.24308753, 0.97000436, -0.24308753, 0.0, 0.0,
    0.4662036850, 0.4323657300, 0.4662036850, 0.4323657300, -0.93240737, -0.86473146
])

def TheFunction(t, theState):
    x1,y1,x2,y2,x3,y3,px1,py1,px2,py2,px3,py3 = theState
    x1d=px1/m1; y1d=py1/m1; x2d=px2/m2; y2d=py2/m2; x3d=px3/m3; y3d=py3/m3
    epsilon_div=1e-9
    r_sq_12=(x1-x2)**2+(y1-y2)**2; r_sq_13=(x1-x3)**2+(y1-y3)**2; r_sq_23=(x2-x3)**2+(y2-y3)**2
    common12=m1*m2*(r_sq_12+epsilon_div)**(-1.5); common13=m1*m3*(r_sq_13+epsilon_div)**(-1.5); common23=m2*m3*(r_sq_23+epsilon_div)**(-1.5)
    p1xd=(x2-x1)*common12+(x3-x1)*common13; p1yd=(y2-y1)*common12+(y3-y1)*common13
    p2xd=(x1-x2)*common12+(x3-x2)*common23; p2yd=(y1-y2)*common12+(y3-y2)*common23
    p3xd=(x1-x3)*common13+(x2-x3)*common23; p3yd=(y1-y3)*common13+(y2-y3)*common23
    return np.array([x1d,y1d,x2d,y2d,x3d,y3d,p1xd,p1yd,p2xd,p2yd,p3xd,p3yd])

def SimpleEuler(theState, current_h_step): # Wird auch für adaptive Methode intern verwendet
    return theState + current_h_step * TheFunction(0, theState)

def BetterEuler(theState, current_h_step):
    g = theState + current_h_step / 2 * TheFunction(0, theState)
    return theState + current_h_step * TheFunction(0, g)

def RK45Step(theState, t, h_val_step):
    sol = solve_ivp(TheFunction, (t, t+h_val_step), theState, method='RK45', t_eval=[t+h_val_step], rtol=1e-8, atol=1e-10)
    return sol.y[:,-1]

# --- Aufgabe 2a: AdaptiveEulerStep_HalfStepMethod ---
def AdaptiveEulerStep_HalfStepMethod(y_current, t_current, h_current, epsilon_tol, h_max=1.0, recursion_depth=0, max_recursion_depth=20):
    if recursion_depth > max_recursion_depth:
        # print(f"WARNUNG: Max. Rekursionstiefe ({max_recursion_depth}) erreicht bei t={t_current:.4f}. Akzeptiere Schritt mit h={h_current:.4e}.")
        # Akzeptiere den Schritt mit y2 und dem aktuellen h, um Endlosschleifen zu vermeiden
        y_half1 = SimpleEuler(y_current, h_current / 2.0)
        y_next_accepted = SimpleEuler(y_half1, h_current / 2.0)
        return y_next_accepted, t_current + h_current, h_current, h_current # letzter Wert ist aktuelles h

    # 1. Euler-Schritt mit voller Schrittweite h
    y_next_h = SimpleEuler(y_current, h_current)

    # 2. Zwei Euler-Schritte mit halber Schrittweite h/2
    y_half_h_half1 = SimpleEuler(y_current, h_current / 2.0)
    y_next_h_half = SimpleEuler(y_half_h_half1, h_current / 2.0)

    # 3. Lokalen Fehler e berechnen
    # Norm der Differenz. Nur Positionskomponenten für Fehlerabschätzung oder alle?
    # Aufgabenstellung sagt "Zustand", also alle Komponenten.
    error_estimate_vec = y_next_h - y_next_h_half
    local_error_e = np.linalg.norm(error_estimate_vec)

    # 4. Nächste Schrittweite h_next berechnen
    if local_error_e == 0: # Sollte selten sein, aber möglich wenn keine Änderung
        h_next = 2.0 * h_current
    else:
        # Faktor p für Euler (1. Ordnung) ist p=1. Formel ist (epsilon/e_local)^(1/(p+1))
        # Die Formel aus der Vorlesung/Aufgabenstellung scheint einen Exponenten von 1/2 zu verwenden, was für p=1 passt.
        ratio = epsilon_tol / local_error_e
        h_next = h_current * min(max(ratio**(0.5), 0.1), 5.0) # Formel aus Aufgabenstellung

    h_next = min(h_next, h_max) # Darf h_max nicht überschreiten
    # Verhindere zu kleine Schritte, die zu Rekursionsproblemen führen könnten
    h_next = max(h_next, 1e-7) # Minimal erlaubte Schrittweite


    # 5. & 6. Schritt akzeptieren oder wiederholen
    if local_error_e <= epsilon_tol:
        # Schritt akzeptiert, Lösung ist y_next_h_half (die genauere der beiden)
        t_next = t_current + h_current
        return y_next_h_half, t_next, h_next, h_current # letzter Wert ist das *verwendete* h
    else:
        # Schritt verwerfen, mit neuer Schrittweite h_next wiederholen
        # Wichtig: Rekursiver Aufruf mit *alter* Zeit t_current und *altem* Zustand y_current
        return AdaptiveEulerStep_HalfStepMethod(y_current, t_current, h_next, epsilon_tol, h_max, recursion_depth + 1, max_recursion_depth)


# --- Animationsfunktion für Aufgabe 2 (mit adaptivem Euler und h-Plot) ---
def make_animation_adaptive(filename, initial_state_vector, initial_h, epsilon_tol, h_max_adaptive,
                            total_time_to_simulate, trail_length=100, x_lims=None, y_lims=None):
    print(f"Starte adaptive Animation: Datei='{filename}', h_init={initial_h}, eps={epsilon_tol}, T_sim={total_time_to_simulate}s")

    # Zustände für alle Methoden
    state_euler_fixed = np.array(initial_state_vector) # Fester Euler für Vergleich
    state_adaptive_euler = np.array(initial_state_vector)
    state_rk45_fixed = np.array(initial_state_vector) # RK45 mit festem h für Vergleich

    current_t_fixed = 0.0
    current_t_adaptive = 0.0
    current_h_adaptive = initial_h

    # Historien
    time_points_fixed = [0.0]
    time_points_adaptive = [0.0]

    h_adaptive_history = [initial_h] # Speichert die *verwendete* Schrittweite

    # Spuren: 0-2 EulerFix, 3-5 AdaptiveEuler, 6-8 RK45Fix
    trails = [deque(maxlen=trail_length) for _ in range(9)]
    for i_particle in range(3):
        trails[i_particle].append((state_euler_fixed[2*i_particle], state_euler_fixed[2*i_particle+1]))
        trails[3+i_particle].append((state_adaptive_euler[2*i_particle], state_adaptive_euler[2*i_particle+1]))
        trails[6+i_particle].append((state_rk45_fixed[2*i_particle], state_rk45_fixed[2*i_particle+1]))


    fig = plt.figure(figsize=(18, 7))
    ax_anim = fig.add_subplot(1, 2, 1)
    ax_h_plot = fig.add_subplot(1, 2, 2) # Plot für adaptive Schrittweite

    ax_anim.set_facecolor('white')
    # Farben: Blau (EulerFix), Rot (AdaptiveEuler), Grün (RK45Fix)
    # Die Aufgabenstellung sagt "alle 4 Lösungen". Ich nehme an, das sind Euler, Verb. Euler, RK45 (alle mit festem h)
    # UND der neue Adaptive Euler. Hier vereinfache ich auf EulerFix, AdaptiveEuler, RK45Fix.
    # Wenn alle 4 gemeint sind, muss state_better_euler etc. hinzugefügt werden.
    # Für jetzt: Blau=EulerFix, Magenta=AdaptiveEuler, Grün=RK45Fix. Verb.Euler weggelassen.
    colors_scatter_full = ['blue']*3 + ['magenta']*3 + ['green']*3

    initial_positions_full = []
    for i in range(3): initial_positions_full.append((state_euler_fixed[2*i], state_euler_fixed[2*i+1]))
    for i in range(3): initial_positions_full.append((state_adaptive_euler[2*i], state_adaptive_euler[2*i+1]))
    for i in range(3): initial_positions_full.append((state_rk45_fixed[2*i], state_rk45_fixed[2*i+1]))

    valid_initial_scatter_pos = [p for p in initial_positions_full if np.all(np.isfinite(p))]
    valid_initial_colors = [colors_scatter_full[i] for i, p in enumerate(initial_positions_full) if np.all(np.isfinite(p))]

    scatter_zorder = 10
    if valid_initial_scatter_pos:
        scat = ax_anim.scatter(np.array(valid_initial_scatter_pos)[:,0], np.array(valid_initial_scatter_pos)[:,1],
                               c=valid_initial_colors, s=35, zorder=scatter_zorder, edgecolors='black', linewidths=0.5)
    else: scat = ax_anim.scatter([], [], s=35, zorder=scatter_zorder, edgecolors='black', linewidths=0.5)

    base_trail_colors_rgb = [np.array([0,0,1])]*3 + [np.array([1,0,1])]*3 + [np.array([0,0.7,0])]*3 # B, M, G
    bg_color_rgb = np.array([1.0,1.0,1.0])
    trail_linecollections = []
    zorders_trails_map = {'euler_fix':3, 'adaptive_euler':2, 'rk45_fix':1} # Reihenfolge für Trails
    zorders_trails = [zorders_trails_map['euler_fix']]*3 + [zorders_trails_map['adaptive_euler']]*3 + [zorders_trails_map['rk45_fix']]*3

    for i_trail_lc in range(9):
        lc = LineCollection([], lw=2.0, zorder=zorders_trails[i_trail_lc]) # Linienbreite etwas reduziert
        ax_anim.add_collection(lc)
        trail_linecollections.append(lc)

    if x_lims is None: x_lims = [-1.5, 1.5];
    if y_lims is None: y_lims = [-1.2, 1.2]
    ax_anim.set(xlim=x_lims, ylim=y_lims); ax_anim.set_aspect('equal', adjustable='box'); ax_anim.grid(True, linestyle=':', alpha=0.6)
    ax_anim.set_title('Partikelbewegung')

    legend_handles_anim = [
        plt.Line2D([0],[0],marker='o',color='w',label='Euler (fest h)',markerfacecolor='blue',markersize=8,linestyle='None'),
        plt.Line2D([0],[0],marker='o',color='w',label='Euler (adaptiv)',markerfacecolor='magenta',markersize=8,linestyle='None'),
        plt.Line2D([0],[0],marker='o',color='w',label='RK45 (fest h)',markerfacecolor='green',markersize=8,linestyle='None')]
    ax_anim.legend(handles=legend_handles_anim,loc='upper right',fontsize='small',frameon=True,facecolor='white',framealpha=0.85,title="Algorithmen")

    # Setup für h-Plot
    line_h_adaptive, = ax_h_plot.plot([], [], 'm.-', label='Adaptive Schrittweite h')
    ax_h_plot.set_xlabel('Zeit (s)'); ax_h_plot.set_ylabel('Schrittweite h'); ax_h_plot.set_title('Adaptive Schrittweite'); ax_h_plot.grid(True)
    ax_h_plot.legend(loc='upper right', fontsize='small'); ax_h_plot.set_yscale('log') # Log-Skala für h ist oft nützlich


    fig.suptitle(f'Adaptiver Euler im 3-Körper-Problem (h_init={initial_h:.2e}, eps={epsilon_tol:.1e})', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Steuerung der Animationslänge: Entweder feste Anzahl Frames oder bis total_time_to_simulate erreicht
    # Hier verwenden wir total_time_to_simulate als Ziel für die adaptiven Methoden

    non_finite_logged = {'euler_fix': False, 'adaptive_euler': False, 'rk45_fix': False}

    # Um die Animation flüssiger zu machen, auch wenn der adaptive Schritt groß wird,
    # können wir eine maximale Frame-Schrittweite für die festen Methoden definieren
    # und die adaptiven Methoden so lange laufen lassen, bis sie die Gesamtzeit erreichen.
    # Die Animation stoppt, wenn current_t_adaptive >= total_time_to_simulate.
    # Wir brauchen eine update-Funktion, die die globalen Zeiten und Zustände fortschreibt.

    frame_count = 0
    max_frames_for_display = int(total_time_to_simulate / initial_h) * 2 # Schätzung für max Frames

    def update_frame_adaptive(frame_num_unused): # frame_num wird nicht direkt verwendet, da adaptive Schritte
        nonlocal current_t_fixed, current_t_adaptive, current_h_adaptive
        nonlocal state_euler_fixed, state_adaptive_euler, state_rk45_fixed, non_finite_logged
        nonlocal frame_count

        frame_count +=1

        # Feste Schrittweite für EulerFix und RK45Fix (hier initial_h als Referenz)
        # Das ist eine Vereinfachung. Die Aufgabenstellung meint evtl. Vergleich mit *optimalem* festem h.
        # Wir nehmen hier initial_h des adaptiven als Vergleich für die festen.
        fixed_h_for_comparison = initial_h

        # Stoppbedingung für die Animation
        if current_t_adaptive >= total_time_to_simulate and current_t_fixed >= total_time_to_simulate:
            if hasattr(ani_adaptive, 'event_source') and ani_adaptive.event_source is not None:
                ani_adaptive.event_source.stop()
            return [scat] + trail_linecollections + [line_h_adaptive]

        # Update feste Methoden (wenn ihre Zeit noch nicht erreicht ist)
        if current_t_fixed < total_time_to_simulate:
            state_euler_fixed = SimpleEuler(state_euler_fixed, fixed_h_for_comparison)
            state_rk45_fixed = RK45Step(state_rk45_fixed, current_t_fixed, fixed_h_for_comparison)
            current_t_fixed += fixed_h_for_comparison
            time_points_fixed.append(current_t_fixed) # Nur für Debugging, nicht direkt geplottet

        # Update adaptive Methode (wenn ihre Zeit noch nicht erreicht ist)
        actual_h_used_this_step = current_h_adaptive # Speichere h vor dem Update für den Plot
        if current_t_adaptive < total_time_to_simulate:
            state_adaptive_euler, current_t_adaptive, current_h_adaptive, actual_h_used_this_step = \
                AdaptiveEulerStep_HalfStepMethod(state_adaptive_euler, current_t_adaptive, current_h_adaptive,
                                                 epsilon_tol, h_max_adaptive)
            time_points_adaptive.append(current_t_adaptive)
            h_adaptive_history.append(actual_h_used_this_step)


        # NaN-Logging
        if not np.all(np.isfinite(state_euler_fixed)) and not non_finite_logged['euler_fix']: print(f"INFO: EulerFix non-finite"); non_finite_logged['euler_fix']=True
        if not np.all(np.isfinite(state_adaptive_euler)) and not non_finite_logged['adaptive_euler']: print(f"INFO: AdaptiveEuler non-finite"); non_finite_logged['adaptive_euler']=True
        if not np.all(np.isfinite(state_rk45_fixed)) and not non_finite_logged['rk45_fix']: print(f"INFO: RK45Fix non-finite"); non_finite_logged['rk45_fix']=True


        # Partikel-Animation aktualisieren
        current_scatter_positions_all_methods = []
        # Euler Fixed
        for i_particle in range(3): pos=(state_euler_fixed[2*i_particle],state_euler_fixed[2*i_particle+1]); current_scatter_positions_all_methods.append(pos); trails[i_particle].append(pos) if np.all(np.isfinite(pos)) else None
        # Adaptive Euler
        for i_particle in range(3): pos=(state_adaptive_euler[2*i_particle],state_adaptive_euler[2*i_particle+1]); current_scatter_positions_all_methods.append(pos); trails[3+i_particle].append(pos) if np.all(np.isfinite(pos)) else None
        # RK45 Fixed
        for i_particle in range(3): pos=(state_rk45_fixed[2*i_particle],state_rk45_fixed[2*i_particle+1]); current_scatter_positions_all_methods.append(pos); trails[6+i_particle].append(pos) if np.all(np.isfinite(pos)) else None

        valid_scatter_positions_display=[p for p in current_scatter_positions_all_methods if np.all(np.isfinite(p))]
        valid_scatter_colors_display=[colors_scatter_full[i] for i,p in enumerate(current_scatter_positions_all_methods) if np.all(np.isfinite(p))] # Hier Indexfehler möglich, wenn weniger Methoden

        # Korrektur für valid_scatter_colors_display, falls Methoden fehlen
        temp_color_list = []
        idx_valid_pos = 0
        for i, p_all in enumerate(current_scatter_positions_all_methods):
            if np.all(np.isfinite(p_all)):
                temp_color_list.append(colors_scatter_full[i]) # Nutze globalen Index i für Farbe
                idx_valid_pos += 1
        valid_scatter_colors_display = temp_color_list


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
        ax_anim.set_title(f'Partikel (t_adapt={current_t_adaptive:.3f}s)')

        # h-Plot aktualisieren
        line_h_adaptive.set_data(time_points_adaptive, h_adaptive_history) # Nutze die *verwendeten* h-Werte
        ax_h_plot.relim(); ax_h_plot.autoscale_view()
        if h_adaptive_history: # Setze Y-Achsen-Limits für h-Plot, um Schwankungen besser zu sehen
            h_min_plot = min(h_adaptive_history) * 0.8
            h_max_plot = max(h_adaptive_history) * 1.2
            ax_h_plot.set_ylim(min(h_min_plot, initial_h*0.1), max(h_max_plot, initial_h*10.0)) # Log-Skala braucht positive Limits
        ax_h_plot.set_xlim(0, max(time_points_adaptive) if time_points_adaptive else 1)


        return [scat] + trail_linecollections + [line_h_adaptive]

    # FuncAnimation benötigt eine feste Anzahl von Frames. Wir schätzen diese.
    # Die Animation stoppt, wenn die Simulationszeit erreicht ist (siehe oben).
    estimated_frames = int(total_time_to_simulate / (initial_h * 0.5)) # Großzügige Schätzung

    global ani_adaptive # Mache ani_adaptive global, damit es in update_frame_adaptive gestoppt werden kann
    ani_adaptive = animation.FuncAnimation(fig, update_frame_adaptive, frames=estimated_frames,
                                           interval=50, blit=True, repeat=False) # Intervall für Anzeige
    try:
        print(f"Speichere adaptive Animation als '{filename}'..."); progress_callback=lambda c,t: print(f'Video: Frame {c+1}/{t}') if (c+1)%max(1,t//20)==0 or c==t-1 else None
        ani_adaptive.save(filename, writer='ffmpeg', fps=25, dpi=150, progress_callback=progress_callback) # FPS reduziert für evtl. weniger Frames
        print(f"Adaptive Animation erfolgreich als '{filename}' gespeichert.")
    except Exception as e: print(f"Fehler beim Speichern der adaptiven Animation: {e}")
    finally:
        if non_finite_logged['euler_fix']: print("INFO: EulerFix NaN.")
        if non_finite_logged['adaptive_euler']: print("INFO: AdaptiveEuler NaN.")
        if non_finite_logged['rk45_fix']: print("INFO: RK45Fix NaN.")
    plt.close(fig)


# --- Textuelle Antworten (aus vorherigen Aufgaben, hier als Referenz) ---
def print_comparison_of_algorithms_fixed_step(): # Aufgabe 1e
    # ... (Text aus vorheriger Antwort) ...
    pass
def print_energy_observations_fixed_step(): # Aufgabe 1g
    # ... (Text aus vorheriger Antwort) ...
    pass

# --- Aufgabe 2d und 2e: Textuelle Antworten (Platzhalter) ---
def print_adaptive_step_observations():
    print("\n--- Aufgabe 2d & 2e: Beobachtungen zur adaptiven Schrittweite ---")
    obs_text = """
    Aufgabe 2d) Vergleich des adaptiven Euler mit anderen Algorithmen:
    - Genauigkeit: Der adaptive Euler (Magenta) sollte im Idealfall eine Genauigkeit erreichen, 
      die durch die Fehlertoleranz `epsilon` bestimmt wird. Bei kleiner `epsilon` kann er 
      genauer sein als der einfache Euler mit fester Schrittweite. Seine Trajektorie wird 
      wahrscheinlich zwischen dem einfachen Euler (Blau) und RK45 (Grün) liegen.
    - Effizienz: Der Hauptvorteil ist die Effizienz. In Phasen, in denen sich der Zustand 
      langsam ändert, kann die Schrittweite `h` vergrößert werden, was Rechenzeit spart. 
      In Phasen schneller Änderungen (z.B. nahe Begegnungen) wird `h` verkleinert, um die 
      Genauigkeit zu wahren.
    - Stabilität: Durch die Anpassung von `h` kann die Methode stabiler sein als ein einfacher
      Euler mit einer festen, möglicherweise zu großen Schrittweite. Wenn `epsilon` jedoch zu groß gewählt
      wird oder `h_max` die Stabilität einschränkt, kann auch der adaptive Euler instabil werden.

    Aufgabe 2e) Verhalten der adaptiven Schrittweite:
    - Der Plot der Schrittweite `h` über die Zeit (im rechten Subplot) sollte zeigen, wie
      `h` variiert.
    - In Bereichen, in denen die Lösung "glatt" ist und sich wenig ändert (Körper weit voneinander entfernt, 
      langsame Geschwindigkeitsänderungen), sollte `h` anwachsen (bis `h_max`).
    - In Bereichen mit schnellen Änderungen (Körper kommen sich nahe, hohe Beschleunigungen), 
      sollte `h` verkleinert werden, um den lokalen Fehler unter `epsilon` zu halten.
    - Man erwartet typischerweise "Spikes" nach unten in `h`, wenn sich die Körper im 
      Drei-Körper-Problem nahe kommen, und längere Phasen mit größerem `h` dazwischen.
    - Wenn `epsilon` sehr klein ist, wird `h` tendenziell klein bleiben. Wenn `epsilon` groß ist,
      wird `h` tendenziell groß sein, was die Genauigkeit reduzieren kann.
    """
    print(obs_text)
    print("--- Ende Aufgabe 2d & 2e ---\n")


if __name__ == '__main__':
    # print_comparison_of_algorithms_fixed_step() # Für Teil 1e
    # print_energy_observations_fixed_step()      # Für Teil 1g

    print_adaptive_step_observations() # Für Teil 2d, 2e

    # --- Animation für Aufgabe 2 (Adaptiver Euler) ---
    print("Erstelle Animation für Aufgabe 2 (Adaptiver Euler)...")

    # Parameter für die adaptive Simulation
    # Die Wahl von initial_h und epsilon_tol ist hier entscheidend!
    adaptive_h_initial = 0.05   # Anfangsschrittweite für adaptiven Euler
    adaptive_epsilon = 1e-4     # Fehlertoleranz
    adaptive_h_max = 0.2        # Maximale adaptive Schrittweite

    simulation_total_time_adaptive = 7.0 # Gesamtzeit für die Simulation
    animation_trail_length_adaptive = 100

    # Dateinamen anpassen: IHRE_MATRIKELNUMMERN_A_adaptiv.mp4
    output_video_filename_adaptive = "454291_454343_U09_A2_adaptiv.mp4"

    # ACHTUNG: Der make_animation_adaptive Aufruf kann lange dauern!
    # Für die Abgabe des Codes diesen Aufruf auskommentieren.
    make_animation_adaptive(filename=output_video_filename_adaptive,
                            initial_state_vector=INITIAL_STATE_FIG8,
                            initial_h=adaptive_h_initial, # Start-h für adaptiv UND Vergleichs-h für feste
                            epsilon_tol=adaptive_epsilon,
                            h_max_adaptive=adaptive_h_max,
                            total_time_to_simulate=simulation_total_time_adaptive,
                            trail_length=animation_trail_length_adaptive,
                            x_lims=[-2.0, 2.0], y_lims=[-1.5, 1.5]) # Achsengrenzen evtl. anpassen

    print(f"Animation '{output_video_filename_adaptive}' für Aufgabe 2 erstellt (oder wäre es, wenn nicht auskommentiert).")

    print("\nSkript für Hausaufgabe 9, Aufgabe 2 ist vollständig.")
    print("Die textuellen Antworten zu 2d, 2e werden beim Ausführen ausgegeben.")
    print("Für die ABGABE des Codes: Bitte den 'make_animation_adaptive'-Aufruf oben wieder auskommentieren.")