# main.py (der von dir bereitgestellte und von mir verbesserte Code)
import numpy as np
import matplotlib.pyplot as plt
import random
import boostfactor # Dies importiert die Funktionen aus der boostfactor.py Datei

# Setze Seeds für Reproduzierbarkeit
SEED_VALUE = 42
np.random.seed(SEED_VALUE)
random.seed(SEED_VALUE)

# --- Lösung für Aufgabe 2a ---
def objective_function(distances_vec_mm, frequencies_vec_ghz):
    """
    Args:
        distances_vec_mm (np.ndarray): Ein numpy-Array der Scheibenabstände [mm].
                                    Dies ist der Zustandsvektor, der optimiert wird.
        frequencies_vec_ghz (np.ndarray): Ein numpy-Array der Frequenzen [GHz],
                                      über die das Minimum berechnet werden soll.

    Returns:
        float: Das Negative des Minimums des Boost-Faktors über die gegebenen
               Frequenzen für die gegebenen Abstände. Ein kleinerer Wert
               entspricht einer "besseren" Lösung im Sinne der Maximierung
               des Minimums.
    """
    # Konvertiere Abstände von mm in Meter für die boostfactor Funktion
    distances_m = np.asarray(distances_vec_mm) * 1e-3
    # Konvertiere Frequenzen von GHz in Hz
    frequencies_hz = np.asarray(frequencies_vec_ghz) * 1e9

    # Rufe die boostfactor Funktion aus dem importierten Modul auf
    beta_squared_values = boostfactor.boostfactor(frequencies_hz, distances_m)

    # Finde das Minimum des Boost-Faktors in diesem Frequenzbereich
    min_beta_squared = np.min(beta_squared_values)

    # Gib das Negative des Minimums zurück (für Minimierungsalgorithmen)
    return -min_beta_squared


# --- Lösung für Aufgabe 2b ---
def find_neighbour(current_solution, step_size_r_mm):
    """
    Findet einen Nachbarzustand durch Hinzufügen eines skalierten Zufallsvektors.
    Verwendet die space_uniform_rand Funktion aus der boostfactor.py Datei.

    Args:
        current_solution (np.ndarray): Der aktuelle Zustandsvektor (Abstände in mm).
        step_size_r_mm (float): Die maximale Schrittweite [mm] zur Skalierung des Zufallsvektors.

    Returns:
        np.ndarray: Der neue Nachbarzustand (Abstände in mm).
    """
    n = len(current_solution)
    # Erzeuge einen raumgleichverteilten, normierten Zufallsvektor
    random_direction = boostfactor.space_uniform_rand(n)

    # Erzeuge 'a' gleichverteilt zufällig zwischen 0 und step_size_r_mm, wie in der Aufgabenstellung gefordert
    a = random.uniform(0, step_size_r_mm)

    neighbour_solution = current_solution + a * random_direction

    # Stelle sicher, dass die Abstände nicht negativ sind.
    # Ein kleiner positiver Wert verhindert numerische Probleme und berücksichtigt physikalische Realität.
    return np.maximum(0.001, neighbour_solution)

def thermal_probability(delta_E, temperature):
    """
    Berechnet die thermische Wahrscheinlichkeit für die Akzeptanz eines schlechteren Zustands.

    Args:
        delta_E (float): Die Differenz der Objective-Funktion-Werte (Nachbar - Aktuell).
                         Sollte positiv sein für schlechtere Zustände.
        temperature (float): Die aktuelle Temperatur des Simulated Annealing.

    Returns:
        float: Die Akzeptanzwahrscheinlichkeit.
    """
    if temperature <= 1e-9: # Eine sehr kleine Zahl statt 0, um Division durch Null zu vermeiden
        return 0.0 # Bei quasi-Temperatur 0 werden keine schlechteren Zustände akzeptiert
    return np.exp(-delta_E / temperature)


def simulated_annealing(objective_func, initial_solution, temperatures, step_size_r_mm_init, frequencies_for_obj):
    """
    Implementiert den Simulated Annealing Algorithmus.

    Args:
        objective_func (callable): Die zu minimierende Objective Function.
                                   Erwartet (distances_vec_mm, frequencies_vec_ghz).
        initial_solution (np.ndarray): Der Startzustand (Abstände in mm).
        temperatures (list or np.ndarray): Eine abfallende Folge von Temperaturen.
        step_size_r_mm_init (float or list/np.ndarray): Die Schrittweite(n) [mm]. Kann konstant
                                                         oder eine Folge sein (wird pro Iteration genutzt).
        frequencies_for_obj (np.ndarray): Die Frequenzen [GHz] zur Bewertung der Objective Function.

    Returns:
        tuple: (best_solution, best_objective_value, history)
               best_solution (np.ndarray): Die beste gefundene Lösung (Abstände in mm).
               best_objective_value (float): Der Wert der Objective Function für die beste Lösung.
               history (dict): Verlauf der Optimierung.
    """
    # 1. Initialisierung
    current_solution = np.copy(initial_solution)
    best_solution = np.copy(initial_solution)

    current_objective_value = objective_func(current_solution, frequencies_for_obj)
    best_objective_value = current_objective_value

    # Optional: Speichern des Verlaufs
    history = {
        'solutions': [np.copy(current_solution)],
        'objective_values': [current_objective_value], # Aktueller Wert in jeder Iteration
        'best_objective_values': [best_objective_value] # Bester Wert bis zu dieser Iteration
    }

    num_iterations = len(temperatures)

    print("Starte Simulated Annealing...")
    for t_idx in range(num_iterations):
        current_temperature = temperatures[t_idx]

        # Bestimme die aktuelle Schrittweite
        current_step_size_r = step_size_r_mm_init if isinstance(step_size_r_mm_init, (int, float)) else step_size_r_mm_init[t_idx]

        # 2. Zufällige Wahl eines Nachbarwertes
        neighbour_solution = find_neighbour(current_solution, current_step_size_r)
        neighbour_objective_value = objective_func(neighbour_solution, frequencies_for_obj)

        # 3. Selektion (Akzeptanzkriterium)
        delta_E = neighbour_objective_value - current_objective_value

        if delta_E <= 0: # Verbesserung oder gleichwertig
            accept = True
        else: # Verschlechterung
            acceptance_prob = thermal_probability(delta_E, current_temperature)
            if random.random() < acceptance_prob:
                accept = True
            else:
                accept = False

        if accept:
            current_solution = np.copy(neighbour_solution)
            current_objective_value = neighbour_objective_value

        # 4. Bisher beste Lösung setzen
        # Wir suchen das MINIMUM der objective_function
        if current_objective_value < best_objective_value:
            best_solution = np.copy(current_solution)
            best_objective_value = current_objective_value

        # Optional: Verlauf speichern
        history['solutions'].append(np.copy(current_solution))
        history['objective_values'].append(current_objective_value)
        history['best_objective_values'].append(best_objective_value)

        # Optional: Fortschritt anzeigen
        if (t_idx + 1) % 500 == 0 or t_idx == 0 or t_idx == num_iterations - 1:
            print(f"Iteration {t_idx+1}/{num_iterations}, Temp: {current_temperature:.2e}, Current Obj: {current_objective_value:.4f}, Best Obj: {best_objective_value:.4f}")

    print("Simulated Annealing beendet.")
    return best_solution, best_objective_value, history


if __name__ == '__main__':
    # Setze Seeds zu Beginn des Hauptteils
    np.random.seed(SEED_VALUE)
    random.seed(SEED_VALUE)

    # --- Aufgabe 2c: Baseline Konfiguration ---
    print("\n--- Aufgabe 2c: Baseline Konfiguration ---")
    num_disks_c = 20
    distance_c_mm = 7.21
    initial_distances_c_mm = np.full(num_disks_c, distance_c_mm)

    f_min_plot_ghz = 21.9
    f_max_plot_ghz = 22.2
    num_freq_points_plot = 500 # Mehr Punkte für eine glatte Kurve

    frequencies_plot_ghz = np.linspace(f_min_plot_ghz, f_max_plot_ghz, num_freq_points_plot)

    print(f"Berechne Boost-Faktor für {num_disks_c} Scheibenabstände von {distance_c_mm} mm über Frequenzbereich {f_min_plot_ghz}-{f_max_plot_ghz} GHz...")
    beta_squared_c = boostfactor.boostfactor(frequencies_plot_ghz * 1e9, initial_distances_c_mm * 1e-3)

    plt.figure(figsize=(10, 6))
    plt.plot(frequencies_plot_ghz, beta_squared_c, label=f'Initial (7.21mm feste Abstände)')
    plt.xlabel('Frequenz f [GHz]')
    plt.ylabel('Boost-Faktor $\\beta^2$')
    plt.title('Boost-Faktor Kurve für Baseline Konfiguration')
    plt.grid(True)
    plt.axvline(f_min_plot_ghz, color='k', linestyle='--', linewidth=0.8)
    plt.axvline(f_max_plot_ghz, color='k', linestyle='--', linewidth=0.8)

    # Der relevante Optimierungsbereich für Aufgabe d ist 22.0 - 22.05 GHz
    f_min_opt_ghz_d = 22.0
    f_max_opt_ghz_d = 22.05

    plt.axvline(f_min_opt_ghz_d, color='r', linestyle=':', linewidth=0.8, label='Optimierungsbereich (Aufgabe 2d)')
    plt.axvline(f_max_opt_ghz_d, color='r', linestyle=':', linewidth=0.8)
    plt.legend()
    plt.show() # Zeige den Plot für Aufgabe 2c direkt an

    # Berechne den Objective Wert für die Baseline im Optimierungsbereich der Aufgabe d
    frequencies_opt_for_baseline_ghz = np.linspace(f_min_opt_ghz_d, f_max_opt_ghz_d, 10) # 10 Punkte wie in Aufgabe d
    baseline_objective_value = objective_function(initial_distances_c_mm, frequencies_opt_for_baseline_ghz)
    print(f"Objective Wert (Neg. Minimum in {f_min_opt_ghz_d:.2f}-{f_max_opt_ghz_d:.2f} GHz) für Baseline (Startwert): {baseline_objective_value:.4f}")

    # --- Aufgabe 2d: Simulated Annealing Optimierung ---
    print("\n--- Aufgabe 2d: Simulated Annealing Optimierung ---")

    initial_distances_d_mm = np.copy(initial_distances_c_mm)
    frequencies_for_optimization_ghz = np.linspace(f_min_opt_ghz_d, f_max_opt_ghz_d, 10)
    print(f"Verwende {len(frequencies_for_optimization_ghz)} Frequenzpunkte ({frequencies_for_optimization_ghz[0]:.2f}-{frequencies_for_optimization_ghz[-1]:.2f} GHz) für Objective Function während der Optimierung.")

    # Simulated Annealing Parameter
    num_iterations = 10000 # Erhöhe die Iterationen, um bessere Ergebnisse zu erzielen

    # Lauf 1: Parameter basierend auf dem Hinweis (lineare Temperaturabnahme)
    print("\nLauf 1: Parameter aus Hinweis (Linear fallende Temperatur)")
    sa_step_size_r_mm_1 = 0.1 # mm, wie im Hinweis empfohlen
    sa_initial_temperature_1 = 100.0 # Startwert wie im Hinweis
    sa_final_temperature_1 = 0.0 # Endwert wie im Hinweis
    sa_temperatures_1 = np.linspace(sa_initial_temperature_1, sa_final_temperature_1, num_iterations)

    best_distances_d1_mm, best_objective_value_d1, history1 = simulated_annealing(
        objective_function,
        initial_distances_d_mm,
        sa_temperatures_1,
        sa_step_size_r_mm_1,
        frequencies_for_optimization_ghz
    )

    print(f"\nOptimierung Lauf 1 abgeschlossen.")
    print(f"Bestes Objective Value gefunden: {best_objective_value_d1:.4f}")
    print(f"Entspricht einem maximalen Minimum Boost-Faktor von: {-best_objective_value_d1:.4f}")

    if -best_objective_value_d1 > 14000:
        print("Ziel: Boost-Faktor > 14000 ERREICHT! :-)")
    else:
        print("Ziel: Boost-Faktor > 14000 NOCH NICHT ERREICHT.")

    beta_squared_optimized_d1 = boostfactor.boostfactor(frequencies_plot_ghz * 1e9, best_distances_d1_mm * 1e-3)

    plt.figure(figsize=(10, 6))
    plt.plot(frequencies_plot_ghz, beta_squared_c, label='Initial (7.21mm feste Abstände)')
    plt.plot(frequencies_plot_ghz, beta_squared_optimized_d1, label='Optimiert mit SA (Lauf 1)', color='red')
    plt.xlabel('Frequenz f [GHz]')
    plt.ylabel('Boost-Faktor $\\beta^2$')
    plt.title('Boost-Faktor Kurven: Initial vs. SA Optimiert (Lauf 1)')
    plt.grid(True)
    plt.axvline(f_min_plot_ghz, color='k', linestyle='--', linewidth=0.8)
    plt.axvline(f_max_plot_ghz, color='k', linestyle='--', linewidth=0.8)
    plt.axvline(f_min_opt_ghz_d, color='r', linestyle=':', linewidth=0.8, label='Optimierungsbereich (Aufgabe 2d)')
    plt.axvline(f_max_opt_ghz_d, color='r', linestyle=':', linewidth=0.8)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(history1['objective_values'], label='Current Objective Value')
    plt.plot(history1['best_objective_values'], label='Best Objective Value Found', linestyle='--')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value (Neg. Min Boost-Faktor)')
    plt.title('Simulated Annealing Verlauf (Lauf 1)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Lauf 2: Experimentieren mit aggressiveren oder feiner abgestimmten Parametern (exponentiell)
    print("\nLauf 2: Experimentelle Parameter (Exponentielle Temperatur, Feinabstimmung)")
    sa_step_size_r_mm_2 = 0.08 # Eine etwas kleinere Schrittweite, aber nicht zu klein
    sa_initial_temperature_2 = 200.0 # Höhere Starttemp für mehr Exploration am Anfang
    sa_cooling_rate_2 = 0.9997 # Sehr langsames Abkühlen
    sa_temperatures_2 = sa_initial_temperature_2 * (sa_cooling_rate_2**np.arange(num_iterations))
    sa_temperatures_2[sa_temperatures_2 < 1e-9] = 1e-9 # Sicherstellen, dass Temp nicht Null wird

    best_distances_d2_mm, best_objective_value_d2, history2 = simulated_annealing(
        objective_function,
        initial_distances_d_mm,
        sa_temperatures_2,
        sa_step_size_r_mm_2,
        frequencies_for_optimization_ghz
    )
    print(f"\nOptimierung Lauf 2 abgeschlossen.")
    print(f"Bestes Objective Value gefunden: {best_objective_value_d2:.4f}")
    print(f"Entspricht einem maximalen Minimum Boost-Faktor von: {-best_objective_value_d2:.4f}")

    if -best_objective_value_d2 > 14000:
        print("Ziel: Boost-Faktor > 14000 ERREICHT! :-)")
    else:
        print("Ziel: Boost-Faktor > 14000 NOCH NICHT ERREICHT.")

    beta_squared_optimized_d2 = boostfactor.boostfactor(frequencies_plot_ghz * 1e9, best_distances_d2_mm * 1e-3)

    plt.figure(figsize=(10, 6))
    plt.plot(frequencies_plot_ghz, beta_squared_c, label='Initial (7.21mm feste Abstände)')
    plt.plot(frequencies_plot_ghz, beta_squared_optimized_d2, label='Optimiert mit SA (Lauf 2, exp. Temp)', color='green')
    plt.xlabel('Frequenz f [GHz]')
    plt.ylabel('Boost-Faktor $\\beta^2$')
    plt.title('Boost-Faktor Kurven: Initial vs. SA Optimiert (Lauf 2)')
    plt.grid(True)
    plt.axvline(f_min_plot_ghz, color='k', linestyle='--', linewidth=0.8)
    plt.axvline(f_max_plot_ghz, color='k', linestyle='--', linewidth=0.8)
    plt.axvline(f_min_opt_ghz_d, color='r', linestyle=':', linewidth=0.8, label='Optimierungsbereich (Aufgabe 2d)')
    plt.axvline(f_max_opt_ghz_d, color='r', linestyle=':', linewidth=0.8)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(history2['objective_values'], label='Current Objective Value (Lauf 2)')
    plt.plot(history2['best_objective_values'], label='Best Objective Value Found (Lauf 2)', linestyle='--')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value (Neg. Min Boost-Faktor)')
    plt.title('Simulated Annealing Verlauf (Lauf 2, exp. Temp)')
    plt.grid(True)
    plt.legend()
    plt.show()