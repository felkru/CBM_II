# 1)
# Matrikelnummer: 454291
# Name: Julia Els
# Email: julia.els@rwth-aachen.de
#
# 2)
# Matrikelnummer: 454343
# Name: Felix Krückel
# Email: felix.krueckel@rwth-aachen.de


import sympy as sp
import numpy as np
from scipy.optimize import minimize
from torch.autograd import variable
import matplotlib.pyplot as plt

def define_symbolically():
    x, y, a, b = sp.symbols('x y a b')
    a_val, b_val = 1, 100

    f = (a - x)**2 + b * (y - x**2)**2 + 0.1 * sp.sin(3 * sp.pi * x) * sp.sin(3 * sp.pi * y)
    f_specific = f.subs({a: a_val, b: b_val})

    grad_f_sym = [sp.diff(f_specific, x), sp.diff(f_specific, y)]

    hess_f_sym = sp.hessian(f_specific, (x, y))

    #Matrix([
        #[sp.diff(f_specific, x, x), sp.diff(f_specific, x, y)],
        #[sp.diff(f_specific, y, x), sp.diff(f_specific, y, y)]
    #]))

    return f_specific, grad_f_sym, hess_f_sym, (x, y)

# --- Teil a: Symbolische Berechnung ---
f_sym, grad_f_sym, hess_f_sym, (x, y) = define_symbolically()

print("--- Ergebnisse Teil a) ---")
print("Symbolische Funktion f(x, y):")
print(f_sym)
print("\nSymbolischer Gradient ∇f(x, y):")
grad_vector = sp.Matrix(grad_f_sym)
sp.pprint(grad_vector)
print("\nSymbolische Hesse-Matrix Hf(x, y):")
sp.pprint(hess_f_sym)
print("-------------------------\n")

# --- Vorbereitung für numerische Verfahren ---
# Lambdifizierung der symbolischen Ausdrücke für numpy
f_numerical = sp.lambdify(((x, y),), f_sym, 'numpy') #änderung: Input-Signatur zu ((x, y),) geändert, um Tupel als Input zu erlauben
grad_f_numerical = sp.lambdify(((x, y),), grad_f_sym, 'numpy') #änderung: Input-Signatur zu ((x, y),) geändert
hess_f_numerical = sp.lambdify(((x, y),), hess_f_sym, 'numpy') #änderung: Input-Signatur zu ((x, y),) geändert

# Wrapper-Funktionen für scipy.optimize und eigene Newton-Methode
def f_wrapper(params):
    return f_numerical(params)

def grad_f_wrapper(params):
    return np.array(grad_f_numerical(params))

def hess_f_wrapper(params):
    return np.array(hess_f_numerical(params))

x0_initial = np.array([0, 5])

# --- Teil b: Vergleich mit scipy.optimize ---
print("--- Ergebnisse Teil b) ---")
result = minimize(f_wrapper, x0_initial, method='trust-ncg', jac=grad_f_wrapper, hess=hess_f_wrapper)
x_optimized = result.x # = x~
print(f"Ergebnis von scipy.optimize:")
print(result) #ändrung: Das komplette Ergebnis ausgeben für mehr Details
print(f"\nGefundener Extremalpunkt (näherungsweise): x̃ = {x_optimized}")
print(f"Funktionswert an diesem Punkt: f(x̃) = {f_wrapper(x_optimized):.6f}")
print("-------------------------\n")


# --- Teil c: Implementierung Newton-Schritt ---
def newton_schritt(xk: np.ndarray, grad_f_xk: np.ndarray, hess_f_xk: np.ndarray) -> np.ndarray:
    try:
        hess_inv = np.linalg.inv(hess_f_xk)
    except np.linalg.LinAlgError:
        print("Hesse-Matrix ist singulär. Newton-Schritt nicht möglich.")
        return xk

    delta_x = hess_inv @ grad_f_xk
    xk_plus_1 = xk - (0.85 * delta_x) # gedämpft mit alpha=0.85
    # xk_plus_1 = xk - delta_x
    return xk_plus_1

# --- Teil d und e: Implementierung und Ausführung Newton-Verfahren ---
#änderung: Rückgabe ist Tupel (xM, erster_konvergierter_schritt, x_history).
def newton_verfahren(x0: np.ndarray, f_func: callable, grad_f_func: callable, hess_f_func: callable, M: int, x_optimized: np.ndarray = None, epsilon: float = None) -> tuple[np.ndarray, int | None, list[np.ndarray]]:
    xk = np.array(x0, dtype=float)
    print(f"Startpunkt x0 = {xk}")
    print(f"Funktionswert f(x0) = {f_func(xk):.6f}")

    first_converged_step = None

    # Speichern der Zwischenergebnisse für Selbstkontrolle (optional, basierend auf Aufgabenhinweis)
    x_history = [np.copy(xk)] #änderung: Array zum Speichern der Iterationspunkte. np.copy, um spätere Modifikationen von xk nicht zu beeinflussen.

    for k in range(M):
        grad_f_xk_num = np.array(grad_f_func(xk))
        hess_f_xk_num = np.array(hess_f_func(xk))

        xk_plus_1 = newton_schritt(xk, grad_f_xk_num, hess_f_xk_num)

        # Überprüfen der Konvergenzbedingung für Teil e)
        if epsilon is not None and x_optimized is not None and first_converged_step is None:
            norm_diff = np.linalg.norm(xk_plus_1 - x_optimized)
            if norm_diff <= epsilon:
                first_converged_step = k + 1 #Speichert den Schrittindex (beginnend bei 1)
                print(f"Konvergenz ||x_{k+1} - x̃|| <= {epsilon} erreicht bei Schritt {k+1}")

        xk = xk_plus_1

        x_history.append(np.copy(xk)) #änderung: Aktuellen Punkt zur Historie hinzufügen.

        print(f"\nNach Schritt {k+1}/{M}:")
        print(f"Aktueller Punkt x_{k+1} = {xk}")
        print(f"Funktionswert f(x_{k+1}) = {f_func(xk):.6f}")

    xM = xk # Das Ergebnis nach M Schritten
    return xM, first_converged_step, x_history #änderung: Rückgabe des Endpunkts, des Schritts bei Konvergenz und der Historie.

# --- Ausführung für Teil d: M=2 Schritte ---
print("\n--- Ausführung für Teil d: M=2 Newton-Schritte ---")
M_part_d = 5

xM_result_d, _, _ = newton_verfahren(x0_initial, f_wrapper, grad_f_wrapper, hess_f_wrapper, M_part_d) #änderung: Aufruf von newton_verfahren angepasst an neue Signatur.

print(f"\n--- Endergebnis Teil d) ---")
print(f"Ergebnis nach {M_part_d} Newton-Schritten, Startpunkt {x0_initial}:")
print(f"x_{M_part_d} = {xM_result_d}")
print(f"Funktionswert f(x_{M_part_d}) = {f_wrapper(xM_result_d):.6f}")
print("-------------------------\n")


# --- Ausführung für Teil e: Finden von M für gegebene Genauigkeit ---
print("\n--- Ausführung für Teil e: Finden der Schrittanzahl für Genauigkeit ---")
epsilon_part_e = 10**-2 # 10^-2
M_max_part_e = 20 # Maximale Anzahl Schritte, um Konvergenz zu finden

xM_result_e, first_converged_step, x_history_e = newton_verfahren(x0_initial, f_wrapper, grad_f_wrapper, hess_f_wrapper, M_max_part_e, x_optimized=x_optimized, epsilon=epsilon_part_e)

print(f"\n--- Ergebnis Teil e) ---")
if first_converged_step is not None:
    print(f"Die Genauigkeit ||x_k - x̃|| <= {epsilon_part_e} wird erstmalig nach M = {first_converged_step} Schritten unterschritten.")
else:
    print(f"Die Genauigkeit ||x_k - x̃|| <= {epsilon_part_e} wurde innerhalb von {M_max_part_e} Schritten nicht erreicht.")

print(f"Punkt nach {M_max_part_e} Schritten: x_{M_max_part_e} = {xM_result_e}")
print("-------------------------\n")

print("\n--- Ergebnisse Teil f) ---")
print("Erstelle Plot der Funktion und des Newton-Pfades...")

# Bereich für den Plot definieren
# Basierend auf der Form der Funktion und dem Startpunkt/Optimum
x_min, x_max = 0.8, 1.2 # Bereich um das erwartete Minimum [1,1]
y_min, y_max = 0.8, 1.2

# Gitter für die Konturplot erstellen
x_plot = np.linspace(x_min, x_max, 200) #änderung: Mehr Punkte für glattere Konturen
y_plot = np.linspace(y_min, y_max, 200) #änderung: Mehr Punkte für glattere Konturen
X, Y = np.meshgrid(x_plot, y_plot)

# Funktionswerte auf dem Gitter berechnen
# Die lambdify Funktion f_numerical erwartet ein Tupel (X, Y) wenn sie mit numpy meshgrid verwendet wird
Z = f_numerical((X, Y)) #änderung: Übergabe als Tupel

plt.figure(figsize=(10, 8))

# Konturplot der Funktion f(x, y)
# Verwenden einer logarithmischen Skala für die Levels, da die Funktion steil ansteigt
levels = np.logspace(np.log10(Z.min() + 1e-6), np.log10(Z.max()), 50) # Logarithmische Skala für Levels
plt.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.8)
plt.colorbar(label='f(x, y)')
plt.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.5, alpha=0.5) # Konturlinien zusätzlich zeichnen

# Newton-Punkte aus der Historie extrahieren
# Die Historie wurde in newton_verfahren (Teil e) gesammelt
# x_history_e enthält die Punkte für die M_max_part_e Schritte
x_coords = [p[0] for p in x_history_e]
y_coords = [p[1] for p in x_history_e]

# Newton-Pfad zeichnen
plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='red', markersize=5, label='Newton Path') #änderung: linestyle='-' für durchgezogene Linie

# Start- und Endpunkt markieren
plt.plot(x_coords[0], y_coords[0], 'go', markersize=8, label='Start $x_0$') # Grüner Kreis für Start
plt.plot(x_coords[-1], y_coords[-1], 'bo', markersize=8, label=f'End $x_{M_max_part_e}$') # Blauer Kreis für Ende

# Optimum (von scipy.optimize) markieren
plt.plot(x_optimized[0], x_optimized[1], '*', markersize=10, color='cyan', label='Optimum $x̃$ (SciPy)') # Stern für Optimum

plt.title('Newton-Verfahren auf Rosenbrock-ähnlicher Funktion')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.gca().set_aspect('equal', adjustable='box') # Gleiches Seitenverhältnis für x und y
plt.show()

print("Plot erstellt.")
print("-------------------------\n")

# --- Ende Teil f ---