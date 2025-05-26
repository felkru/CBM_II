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
import random
# Importiere das boostfactor Modul. Dadurch sind die Funktionen boostfactor und space_uniform_rand
# unter boostfactor.boostfactor und boostfactor.space_uniform_rand verfügbar.
import boostfactor

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
        step_size_r_mm (float): Die Schrittweite oder "Temperatur"-Parameter
                                 zur Skalierung des Zufallsvektors [mm].

    Returns:
        np.ndarray: Der neue Nachbarzustand (Abstände in mm).
    """
    n = len(current_solution)
    # Erzeuge einen raumgleichverteilten, normierten Zufallsvektor
    # Rufe die space_uniform_rand Funktion aus dem importierten Modul auf
    random_direction = boostfactor.space_uniform_rand(n)
    # Skaliere den Zufallsvektor mit der Schrittweite r und addiere ihn zur aktuellen Lösung
    neighbour_solution = current_solution + step_size_r_mm * random_direction
    # Stelle sicher, dass die Abstände nicht negativ sind
    return np.maximum(0, neighbour_solution)

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
    if temperature <= 0:
        return 0.0 # Bei Temperatur 0 werden keine schlechteren Zustände akzeptiert
    # Wenn delta_E <= 0, ist es eine Verbesserung oder gleichwertig, Wahrscheinlichkeit ist 1 (wird aber anders behandelt)
    # Wenn delta_E > 0, ist es eine Verschlechterung, Wahrscheinlichkeit ist exp(-delta_E / temperature)
    # Die Formel exp(-(f(y)-f(x))/T_t) aus der Folie
    return np.exp(-delta_E / temperature)


def simulated_annealing(objective_func, initial_solution, temperatures, step_size_r_mm, frequencies_for_obj):
    """
    Implementiert den Simulated Annealing Algorithmus.

    Args:
        objective_func (callable): Die zu minimierende Objective Function.
                                   Erwartet (distances_vec_mm, frequencies_vec_ghz).
        initial_solution (np.ndarray): Der Startzustand (Abstände in mm).
        temperatures (list or np.ndarray): Eine abfallende Folge von Temperaturen.
        step_size_r_mm (float or list/np.ndarray): Die Schrittweite(n) [mm]. Kann konstant
                                                   oder eine Folge sein.
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
        'objective_values': [current_objective_value],
        'best_objective_values': [best_objective_value]
    }

    num_iterations = len(temperatures)

    print("Starte Simulated Annealing...")
    for t in range(num_iterations):
        current_temperature = temperatures[t]
        # Schrittweite kann sich auch mit der Zeit ändern, falls step_size_r_mm eine Liste ist
        current_step_size_r = step_size_r_mm if isinstance(step_size_r_mm, (int, float)) else step_size_r_mm[t]

        # 2. Zufällige Wahl eines Nachbarwertes
        neighbour_solution = find_neighbour(current_solution, current_step_size_r)
        neighbour_objective_value = objective_func(neighbour_solution, frequencies_for_obj)

        # 3. Selektion
        delta_E = neighbour_objective_value - current_objective_value

        # Wenn die neue Lösung besser ist ODER mit einer bestimmten Wahrscheinlichkeit akzeptieren
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
        # Wir suchen das MINIMUM der objective_function, also vergleichen wir mit <
        if current_objective_value < best_objective_value:
            best_solution = np.copy(current_solution)
            best_objective_value = current_objective_value

        # Optional: Verlauf speichern
        history['solutions'].append(np.copy(current_solution))
        history['objective_values'].append(current_objective_value)
        history['best_objective_values'].append(best_objective_value)


        # Optional: Fortschritt anzeigen
        if (t + 1) % 100 == 0 or t == 0 or t == num_iterations - 1:
            print(f"Iteration {t+1}/{num_iterations}, Temperatur: {current_temperature:.2e}, Current Obj: {current_objective_value:.4f}, Best Obj: {best_objective_value:.4f}")


    print("Simulated Annealing beendet.")
    return best_solution, best_objective_value, history


# --- Hauptteil zur Ausführung der Aufgaben c und d ---

if __name__ == '__main__':
    # --- Aufgabe 2c ---
    print("--- Aufgabe 2c: Baseline Konfiguration ---")
    num_disks_c = 20
    distance_c_mm = 7.21
    initial_distances_c_mm = np.full(num_disks_c, distance_c_mm)

    f_min_plot_ghz = 21.9
    f_max_plot_ghz = 22.2
    num_freq_points_plot = 500 # Mehr Punkte für eine glatte Kurve

    frequencies_plot_ghz = np.linspace(f_min_plot_ghz, f_max_plot_ghz, num_freq_points_plot)

    print(f"Berechne Boost-Faktor für {num_disks_c} Scheibenabstände von {distance_c_mm} mm über Frequenzbereich {f_min_plot_ghz}-{f_max_plot_ghz} GHz...")
    # Rufe die boostfactor Funktion aus dem importierten Modul auf
    beta_squared_c = boostfactor.boostfactor(frequencies_plot_ghz * 1e9, initial_distances_c_mm * 1e-3)

    plt.figure(figsize=(10, 6))
    plt.plot(frequencies_plot_ghz, beta_squared_c, label=f'{distance_c_mm} mm feste Abstände')
    plt.xlabel('Frequenz f [GHz]')
    plt.ylabel('Boost-Faktor $\\beta^2$')
    plt.title('Boost-Faktor Kurve für Baseline Konfiguration')
    plt.grid(True)
    plt.axvline(f_min_plot_ghz, color='k', linestyle='--', linewidth=0.8)
    plt.axvline(f_max_plot_ghz, color='k', linestyle='--', linewidth=0.8)
    # Den relevanten Bereich für Aufgabe d markieren (22.05 bis 22.25 GHz)
    f_min_opt_ghz = 22.05
    f_max_opt_ghz = 22.25
    plt.axvline(f_min_opt_ghz, color='r', linestyle=':', linewidth=0.8, label='Optimierungsbereich')
    plt.axvline(f_max_opt_ghz, color='r', linestyle=':', linewidth=0.8)
    plt.legend()


    # Berechne den Objective Wert für die Baseline im Optimierungsbereich
    frequencies_opt_for_baseline_ghz = np.linspace(f_min_opt_ghz, f_max_opt_ghz, 100) # Mehr Punkte für genaueres Minimum
    baseline_objective_value = objective_function(initial_distances_c_mm, frequencies_opt_for_baseline_ghz)
    print(f"Objective Wert (Neg. Minimum in {f_min_opt_ghz}-{f_max_opt_ghz} GHz) für Baseline: {baseline_objective_value:.4f}")


    # --- Aufgabe 2d ---
    print("\n--- Aufgabe 2d: Simulated Annealing Optimierung ---")

    # Initialer Zustand (Abstände von Aufgabe c)
    initial_distances_d_mm = np.copy(initial_distances_c_mm) # Starten Sie mit den gleichen Abständen

    # Frequenzen für die Bewertung der Objective Function (10 Punkte im Bereich 22.05-22.25 GHz)
    # Weniger Punkte für die Optimierung sind üblich, um die Berechnung pro Iteration zu beschleunigen
    frequencies_for_optimization_ghz = np.linspace(f_min_opt_ghz, f_max_opt_ghz, 10)
    print(f"Verwende {len(frequencies_for_optimization_ghz)} Frequenzpunkte ({frequencies_for_optimization_ghz[0]:.2f}-{frequencies_for_optimization_ghz[-1]:.2f} GHz) für Objective Function während der Optimierung.")

    # Simulated Annealing Parameter
    num_iterations = 1000 # Anzahl der Schritte
    # Verschiedene Parameter ausprobieren:
    # Schrittweite r: Kann konstant sein oder sich mit der Zeit ändern.
    # Temperaturabfall: T_t = T0 * alpha**t (exponentiell) oder T_t = T0 - alpha * t (linear)
    # Hier: Konstanter r und exponentieller Temperaturabfall

    # Beispiel 1: Moderate Parameter
    print("\nLauf 1: Moderate Parameter")
    sa_step_size_r_mm = 0.1 # Anfangsschrittweite in mm
    sa_initial_temperature = 1000 # Anfangstemperatur (Skaliert mit der Größenordnung des Objektivwerts)
    sa_cooling_rate = 0.995 # Abklingrate (nahe 1 für langsames Abkühlen)
    sa_temperatures = sa_initial_temperature * sa_cooling_rate**np.arange(num_iterations)

    best_distances_d1_mm, best_objective_value_d1, history1 = simulated_annealing(
        objective_function,
        initial_distances_d_mm,
        sa_temperatures,
        sa_step_size_r_mm,
        frequencies_for_optimization_ghz # Frequenzen für die Optimierung
    )

    print(f"\nOptimierung Lauf 1 abgeschlossen.")
    print(f"Bestes Objective Value gefunden: {best_objective_value_d1:.4f}")
    print(f"Entspricht einem maximalen Minimum Boost-Faktor von: {-best_objective_value_d1:.4f}")
    print(f"Initiales Objective Value: {baseline_objective_value:.4f} (Max Min Boost: {-baseline_objective_value:.4f})")

    # Berechne die finale Boost-Faktor Kurve für die besten gefundenen Abstände (über den vollen Plot-Bereich)
    print(f"Berechne Boost-Faktor Kurve für die optimierten Abstände (Lauf 1) über den vollen Plot-Bereich...")
    # Rufe die boostfactor Funktion aus dem importierten Modul auf
    beta_squared_optimized_d1 = boostfactor.boostfactor(frequencies_plot_ghz * 1e9, best_distances_d1_mm * 1e-3)

    # Plotten der initialen und optimierten Kurve
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies_plot_ghz, beta_squared_c, label='Initial (7.21mm feste Abstände)')
    plt.plot(frequencies_plot_ghz, beta_squared_optimized_d1, label='Optimiert mit SA (Lauf 1)', color='red')
    plt.xlabel('Frequenz f [GHz]')
    plt.ylabel('Boost-Faktor $\\beta^2$')
    plt.title('Boost-Faktor Kurven: Initial vs. SA Optimiert')
    plt.grid(True)
    plt.axvline(f_min_plot_ghz, color='k', linestyle='--', linewidth=0.8)
    plt.axvline(f_max_plot_ghz, color='k', linestyle='--', linewidth=0.8)
    plt.axvline(f_min_opt_ghz, color='r', linestyle=':', linewidth=0.8, label='Optimierungsbereich')
    plt.axvline(f_max_opt_ghz, color='r', linestyle=':', linewidth=0.8)
    plt.legend()

    # Verlauf der Objective Value während der Optimierung plotten
    plt.figure(figsize=(10, 6))
    plt.plot(history1['objective_values'], label='Current Objective Value')
    plt.plot(history1['best_objective_values'], label='Best Objective Value Found', linestyle='--')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value (Neg. Min Boost-Faktor)')
    plt.title('Simulated Annealing Verlauf (Lauf 1)')
    plt.grid(True)
    plt.legend()


    # Beispiel 2: Aggressivere Parameter
    print("\nLauf 2: Aggressivere Parameter")
    sa_step_size_r_mm_2 = 0.5
    sa_initial_temperature_2 = 500
    sa_cooling_rate_2 = 0.99
    sa_temperatures_2 = sa_initial_temperature_2 * sa_cooling_rate_2**np.arange(num_iterations)

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

    # Finale Boost-Faktor Kurve für die besten Abstände (Lauf 2)
    print(f"Berechne Boost-Faktor Kurve für die optimierten Abstände (Lauf 2) über den vollen Plot-Bereich...")
    # Rufe die boostfactor Funktion aus dem importierten Modul auf
    beta_squared_optimized_d2 = boostfactor.boostfactor(frequencies_plot_ghz * 1e9, best_distances_d2_mm * 1e-3)

    plt.figure(figsize=(10, 6))
    plt.plot(frequencies_plot_ghz, beta_squared_c, label='Initial (7.21mm feste Abstände)')
    plt.plot(frequencies_plot_ghz, beta_squared_optimized_d2, label='Optimiert mit SA (Lauf 2, aggressiv)', color='green')
    plt.xlabel('Frequenz f [GHz]')
    plt.ylabel('Boost-Faktor $\\beta^2$')
    plt.title('Boost-Faktor Kurven: Initial vs. SA Optimiert (Lauf 2)')
    plt.grid(True)
    plt.axvline(f_min_plot_ghz, color='k', linestyle='--', linewidth=0.8)
    plt.axvline(f_max_plot_ghz, color='k', linestyle='--', linewidth=0.8)
    plt.axvline(f_min_opt_ghz, color='r', linestyle=':', linewidth=0.8, label='Optimierungsbereich')
    plt.axvline(f_max_opt_ghz, color='r', linestyle=':', linewidth=0.8)
    plt.legend()

    # Verlauf der Objective Value plotten
    plt.figure(figsize=(10, 6))
    plt.plot(history2['objective_values'], label='Current Objective Value (Lauf 2)')
    plt.plot(history2['best_objective_values'], label='Best Objective Value Found (Lauf 2)', linestyle='--')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value (Neg. Min Boost-Faktor)')
    plt.title('Simulated Annealing Verlauf (Lauf 2, aggressiv)')
    plt.grid(True)
    plt.legend()


    plt.show()