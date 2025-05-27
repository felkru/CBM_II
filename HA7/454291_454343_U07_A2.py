import numpy as np
import matplotlib.pyplot as plt
import random
import boostfactor

# Setze Seeds für Reproduzierbarkeit
# Beachte: boostfactor.space_uniform_rand nutzt wahrscheinlich numpy.random
# und random.random() nutzt das eingebaute random Modul
SEED_VALUE = 42
np.random.seed(SEED_VALUE)
random.seed(SEED_VALUE)

# --- Lösung für Aufgabe 2a ---
def objective_function(distances_vec_mm, frequencies_vec_ghz):
    distances_m = np.asarray(distances_vec_mm) * 1e-3
    frequencies_hz = np.asarray(frequencies_vec_ghz) * 1e9
    beta_squared_values = boostfactor.boostfactor(frequencies_hz, distances_m)
    min_beta_squared = np.min(beta_squared_values)
    return -min_beta_squared

# --- Lösung für Aufgabe 2b ---
def find_neighbour(current_solution, step_size_r_mm):
    n = len(current_solution)
    random_direction = boostfactor.space_uniform_rand(n)

    # Erzeuge 'a' gleichverteilt zufällig zwischen 0 und step_size_r_mm
    # Der Hinweis war: 'a∈[0,r] gleichverteilt zufällig ist.'
    # Dein Originalcode verwendet step_size_r_mm * random_direction, was einen konstanten Schritt der Größe r erzeugt,
    # multipliziert mit einem normierten Vektor. Das war so nicht explizit in der Aufgabe gemeint.
    # Es soll a * delta_x sein, wobei a zufällig ist und delta_x ein normierter Vektor.
    a = random.uniform(0, step_size_r_mm) # oder np.random.uniform(0, step_size_r_mm)

    neighbour_solution = current_solution + a * random_direction
    # Stelle sicher, dass die Abstände nicht negativ sind
    # Dies ist sehr wichtig für die Physikalität
    return np.maximum(0.001, neighbour_solution) # Kleiner positiver Wert, um 0 zu vermeiden

def thermal_probability(delta_E, temperature):
    if temperature <= 1e-9: # Eine sehr kleine Zahl statt 0, um Division durch Null zu vermeiden
        return 0.0
    return np.exp(-delta_E / temperature)

def simulated_annealing(objective_func, initial_solution, temperatures, step_size_r_mm_init, frequencies_for_obj):
    current_solution = np.copy(initial_solution)
    best_solution = np.copy(initial_solution)

    current_objective_value = objective_func(current_solution, frequencies_for_obj)
    best_objective_value = current_objective_value

    history = {
        'solutions': [np.copy(current_solution)],
        'objective_values': [current_objective_value],
        'best_objective_values': [best_objective_value]
    }

    num_iterations = len(temperatures)

    print("Starte Simulated Annealing...")
    for t_idx in range(num_iterations): # Um Konflikte mit der Temperaturvariablen t zu vermeiden
        current_temperature = temperatures[t_idx]

        # Anpassung der Schrittweite, wie in der Aufgabe suggeriert (kann konstant bleiben oder sich ändern)
        # Wenn step_size_r_mm_init eine Liste/Array ist, dann wird die Schrittweite je nach Iteration gewählt
        # Ansonsten bleibt sie konstant.
        current_step_size_r = step_size_r_mm_init if isinstance(step_size_r_mm_init, (int, float)) else step_size_r_mm_init[t_idx]

        # Erzeuge einen Nachbarn
        neighbour_solution = find_neighbour(current_solution, current_step_size_r)
        neighbour_objective_value = objective_func(neighbour_solution, frequencies_for_obj)

        delta_E = neighbour_objective_value - current_objective_value

        # Akzeptanzkriterium
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

        # Bisher beste Lösung setzen (wir minimieren die Objective Function)
        if current_objective_value < best_objective_value:
            best_solution = np.copy(current_solution)
            best_objective_value = current_objective_value

        # Verlauf speichern
        history['solutions'].append(np.copy(current_solution))
        history['objective_values'].append(current_objective_value)
        history['best_objective_values'].append(best_objective_value)

        # Fortschrittsanzeige
        if (t_idx + 1) % 100 == 0 or t_idx == 0 or t_idx == num_iterations - 1:
            print(f"Iteration {t_idx+1}/{num_iterations}, Temp: {current_temperature:.2e}, Current Obj: {current_objective_value:.4f}, Best Obj: {best_objective_value:.4f}")

    print("Simulated Annealing beendet.")
    return best_solution, best_objective_value, history

if __name__ == '__main__':
    # Setze Seeds zu Beginn des Hauptteils
    np.random.seed(SEED_VALUE)
    random.seed(SEED_VALUE)

    # --- Aufgabe 2c ---
    print("\n--- Aufgabe 2c: Baseline Konfiguration ---")
    num_disks_c = 20
    distance_c_mm = 7.21
    initial_distances_c_mm = np.full(num_disks_c, distance_c_mm)

    f_min_plot_ghz = 21.9
    f_max_plot_ghz = 22.2
    num_freq_points_plot = 500

    frequencies_plot_ghz = np.linspace(f_min_plot_ghz, f_max_plot_ghz, num_freq_points_plot)

    print(f"Berechne Boost-Faktor für {num_disks_c} Scheibenabstände von {distance_c_mm} mm über Frequenzbereich {f_min_plot_ghz}-{f_max_plot_ghz} GHz...")
    beta_squared_c = boostfactor.boostfactor(frequencies_plot_ghz * 1e9, initial_distances_c_mm * 1e-3)

    plt.figure(figsize=(10, 6))
    plt.plot(frequencies_plot_ghz, beta_squared_c, label=f'{distance_c_mm} mm feste Abstände')
    plt.xlabel('Frequenz f [GHz]')
    plt.ylabel('Boost-Faktor $\\beta^2$')
    plt.title('Boost-Faktor Kurve für Baseline Konfiguration')
    plt.grid(True)
    plt.axvline(f_min_plot_ghz, color='k', linestyle='--', linewidth=0.8)
    plt.axvline(f_max_plot_ghz, color='k', linestyle='--', linewidth=0.8)

    # Anpassen des Optimierungsbereichs an die Aufgabe (22.0 - 22.05 GHz)
    f_min_opt_ghz_d = 22.0
    f_max_opt_ghz_d = 22.05 # Der Hinweis sagt 10 Frequenzpunkte von 22.0 - 22.05 GHz.

    plt.axvline(f_min_opt_ghz_d, color='r', linestyle=':', linewidth=0.8, label='Optimierungsbereich (Aufgabe 2d)')
    plt.axvline(f_max_opt_ghz_d, color='r', linestyle=':', linewidth=0.8)
    plt.legend()
    plt.show() # Zeige den Plot für Aufgabe 2c direkt an

    # Berechne den Objective Wert für die Baseline im Optimierungsbereich
    # Dies ist der Startwert für die Optimierung
    frequencies_opt_for_baseline_ghz = np.linspace(f_min_opt_ghz_d, f_max_opt_ghz_d, 10) # 10 Punkte wie in Aufgabe d
    baseline_objective_value = objective_function(initial_distances_c_mm, frequencies_opt_for_baseline_ghz)
    print(f"Objective Wert (Neg. Minimum in {f_min_opt_ghz_d:.2f}-{f_max_opt_ghz_d:.2f} GHz) für Baseline (Startwert): {baseline_objective_value:.4f}")

    # --- Aufgabe 2d ---
    print("\n--- Aufgabe 2d: Simulated Annealing Optimierung ---")

    initial_distances_d_mm = np.copy(initial_distances_c_mm)
    frequencies_for_optimization_ghz = np.linspace(f_min_opt_ghz_d, f_max_opt_ghz_d, 10)
    print(f"Verwende {len(frequencies_for_optimization_ghz)} Frequenzpunkte ({frequencies_for_optimization_ghz[0]:.2f}-{frequencies_for_optimization_ghz[-1]:.2f} GHz) für Objective Function während der Optimierung.")

    # Simulated Annealing Parameter - Verschiedene Läufe ausprobieren
    num_iterations = 5000 # Erhöhe die Iterationen, um bessere Ergebnisse zu erzielen

    # Lauf 1: Empfohlene Parameter aus dem Hinweis
    print("\nLauf 1: Empfohlene Parameter (Linear fallende Temperatur)")
    sa_step_size_r_mm_1 = 0.1 # mm
    sa_initial_temperature_1 = 100.0 # Startwert
    sa_final_temperature_1 = 0.0 # Endwert
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

    # Überprüfe, ob der Zielwert erreicht wurde
    if -best_objective_value_d1 > 14000:
        print("Ziel: Boost-Faktor > 14000 ERREICHT! :-)")
    else:
        print("Ziel: Boost-Faktor > 14000 NOCH NICHT ERREICHT. Versuchen Sie andere Parameter!")

    beta_squared_optimized_d1 = boostfactor.boostfactor(frequencies_plot_ghz * 1e9, best_distances_d1_mm * 1e-3)

    plt.figure(figsize=(10, 6))
    plt.plot(frequencies_plot_ghz, beta_squared_c, label='Initial (7.21mm feste Abstände)')
    plt.plot(frequencies_plot_ghz, beta_squared_optimized_d1, label='Optimiert mit SA', color='red')
    plt.xlabel('Frequenz f [GHz]')
    plt.ylabel('Boost-Faktor $\\beta^2$')
    plt.title('Boost-Faktor Kurven: Initial vs. SA Optimiert')
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
    plt.title('Simulated Annealing Verlauf')
    plt.grid(True)
    plt.legend()
    plt.show()