# 1)
# Matrikelnummer: 454291
# Name: Julia Els
# Email: julia.els@rwth-aachen.de
#
# 2)
# Matrikelnummer: 454343
# Name: Felix Krückel
# Email: felix.krueckel@rwth-aachen.de

import scipy.stats as stats
import sympy as sp
import numpy as np
from scipy.optimize import minimize

def define_symbolically():
    x, y, a, b = sp.symbols('x y a b')
    a_val, b_val = 1, 100

    f = (a - x)**2 + b * (y - x**2)**2 + 0.1 * sp.sin(3 * sp.pi * x) * sp.sin(3 * sp.pi * y)
    f_specific = f.subs({a: a_val, b: b_val})

    grad_f_sym = [sp.diff(f_specific, x), sp.diff(f_specific, y)]

    hess_f_sym = sp.Matrix([
        [sp.diff(f_specific, x, x), sp.diff(f_specific, x, y)],
        [sp.diff(f_specific, y, x), sp.diff(f_specific, y, y)]
    ])

    return f_specific, grad_f_sym, hess_f_sym, (x, y)

f, grad_f_sym, hess_f_sym, (x, y) = define_symbolically()

print("--- Ergebnisse Teil a) ---")
print("Symbolische Funktion f(x, y):")
print(f)
print("\nSymbolischer Gradient ∇f(x, y):")
grad_vector = sp.Matrix(grad_f_sym)
sp.pprint(grad_vector)#die 2 ist ein Quadrat
print("\nSymbolische Hesse-Matrix Hf(x, y):")
sp.pprint(hess_f_sym)
print("-------------------------\n")

f_numerical = sp.lambdify((x, y), f, 'numpy')
grad_f_numerical = sp.lambdify((x, y), grad_f_sym, 'numpy')
hess_f_numerical = sp.lambdify((x, y), hess_f_sym, 'numpy')

def f_wrapper(params):
    x_val, y_val = params
    return f_numerical(x_val, y_val)

def grad_f_wrapper(params):
    x_val, y_val = params
    return np.array(grad_f_numerical(x_val, y_val))

def hess_f_wrapper(params):
    x_val, y_val = params
    return hess_f_numerical(x_val, y_val)


x0_initial = np.array([1.075, 1.0625])
genauigkeit = 0.01
abstand = 0
M_steps = 7
schritt = np.zeros(M_steps)

result = minimize(f_wrapper, x0_initial, method='trust-ncg', jac=grad_f_wrapper, hess=hess_f_wrapper)
x_optimized = result.x
print(f"\nGefundener Extremalpunkt (näherungsweise): x̃ = {x_optimized}")
print(f"Funktionswert an diesem Punkt: f(x̃) = {f_wrapper(x_optimized)}")
print("-------------------------")


def newton_schritt(xk: np.ndarray, grad_f_xk: np.ndarray, hess_f_xk: np.ndarray) -> np.ndarray:
    hess_inv = np.linalg.inv(hess_f_xk)
    delta_x = hess_inv @ grad_f_xk
    xk_plus_1 = xk - delta_x
    return xk_plus_1


def newton_verfahren(x0: np.ndarray, f_func: callable, grad_f_func: callable, hess_f_func: callable, M: int) -> np.ndarray:
    xk = np.array(x0, dtype=float)
    print(f"Startpunkt x0 = {xk}")
    print(f"Funktionswert f(x0) = {f_func(xk[0], xk[1]):.6f}")

    for k in range(M):
        grad_f_xk_num = np.array(grad_f_func(xk[0], xk[1]))
        hess_f_xk_num = np.array(hess_f_func(xk[0], xk[1]))

        xk_plus_1 = newton_schritt(xk, grad_f_xk_num, hess_f_xk_num)
        xk = xk_plus_1
        abstand = np.abs(xk-x_optimized)
        print(f"\nNach Schritt {k+1}/{M}:")
        print(f"Aktueller Punkt x_{k+1} = {xk}")
        print(f"Funktionswert f(x_{k+1}) = {f_func(xk[0], xk[1]):.6f}")

        if abstand < genauigkeit:
            schritt[k]=xk
    xM = xk
    return xM

#xM_result = newton_verfahren(x0_initial, f_numerical, grad_f_numerical, hess_f_numerical, M_steps)


non_zero_indices = np.where(schritt != 0)
first_non_zero = None
if non_zero_indices[0].size > 0:
    first_index = non_zero_indices[0][0]
    first_non_zero = schritt[first_index]
    print(f": Nach {first_index} schritten unterschreitet die Genauigkeit das gegebene ϵ mit einer normierten Differenz von {first_non_zero}")
else:
    print("Das gegebene ϵ wurde nicht unterschritten.")

print(f"\n--- Endergebnis Teil c) ---")
print(f"Ergebnis nach {M_steps} Newton-Schritten, Startpunkt {x0_initial}:")
print(f"x_{M_steps} = {xM_result}")
print(f"Funktionswert f(x_{M_steps}) = {f_numerical(xM_result[0], xM_result[1]):.6f}")
print("-------------------------")