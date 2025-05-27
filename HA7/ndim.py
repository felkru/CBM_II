#! /usr/bin/env python3
# -*- coding: utf-8 -*-

### Vorlagendatei für die Übungen zur Computergestützten Physik ###

# Bitte ergänzen Sie hier die Daten der Abgebenden. Ersetzen Sie nur
# die Punkte ('...'), aber lassen Sie den Rest der Zeilen und ihre Reihenfolge
# ansonsten unverändert, da Ihre Abgabe sonst nicht elektronisch verarbeitet
# werden kann.
#
# 1)
# Matrikelnummer: 458471
# Name: Maximilian Kieser
# Email: maximilian.l.kieser@gmail.com
#
# 2)
# Matrikelnummer: 454505
# Name: Louisa Steffens
# Email: Louisa.sonne@web.de
# #

#%%

# Häufig benötigte Module (auskommentieren, wenn notwendig):
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy
from scipy.optimize import minimize

# 1a)
def funk():
    x, y , z = sp.symbols('x,y,z')
    variables =(x,y,z)
    a = 1
    b = 100
    f = z*(a-x)**2+b*(y-x**2)**2+0.1*sp.sin(3*sp.pi*x)*sp.sin(3*sp.pi*y)
    grad_f = [sp.diff(f, var) for var in variables]
    Hesse_f = sp.hessian(f,variables)

    f_lamb = sp.lambdify(variables, f, "numpy")
    grad_f_lamb = sp.lambdify(variables, grad_f, "numpy")
    Hesse_f_lamb = sp.lambdify(variables, Hesse_f, "numpy")
    return f, grad_f, Hesse_f, f_lamb, grad_f_lamb, Hesse_f_lamb

# ------------- 1b) ------------------

def loesung_berechnen(f_lamb):
    def f_wrapped(xy):
        return f_lamb(*xy)
    x_opt = minimize(f_wrapped, x0=[1.075, 1.0625, 1]).x
    return x_opt

# --------------------- 1c) ------------------

def newton_schritt(x_k, grad_f, Hesse_f):
    return x_k - np.linalg.solve(Hesse_f, grad_f)

def newton_verfahren(x0, grad_f, Hesse_f, M):
    xk = np.array(x0, dtype=float)
    x_vals = [xk.copy()]

    for k in range(M):
        grad_k = np.array(grad_f(*xk), dtype=float)
        Hesse_k = np.array(Hesse_f(*xk), dtype=float)
        xk = newton_schritt(xk, grad_k, Hesse_k)
        x_vals.append(xk.copy())
        print(f"x{k + 1} =", xk)
    return np.array(x_vals)


# -------------------- 1e) -------------------
def newton_verfahren_opt(x0, grad_f, Hesse_f,x_opt, eps=1e-2, max_item=20):
    xk = np.array(x0, dtype=float)
    x_vals = [xk.copy()]

    for k in range(1, max_item+1):
        grad_k = np.array(grad_f(*xk,), dtype=float)
        hess_k = np.array(Hesse_f(*xk), dtype=float)
        xk = newton_schritt(xk, grad_k, hess_k)
        dist = np.linalg.norm(xk - x_opt)
        print(f"Schritt {k}: xk = {xk}, ||xk - x_opt|| = {dist:.5e}")
        if dist <= eps:
            return k, xk
    return None, xk

# ----------------- 1d) --------------------

if __name__ == "__main__":
    x0 = [1.075, 1.0625, -9]
    f, grad_f, Hesse_f, f_lamb, grad_f_lamb, Hesse_f_lamb = funk()
    print("Newton-Verfahren (M = 2 Schritte):")
    x_end = newton_verfahren(x0, grad_f_lamb, Hesse_f_lamb, M=2)

    # ----------------- 1e) --------------------

    x0 = [1.075, 1.0625, -9]
    f, grad_f, Hesse_f, f_lamb, grad_f_lamb, Hesse_f_lamb = funk()

    print("Teil b: Minimierung mit scipy")
    x_opt = loesung_berechnen(f_lamb)
    print("Optimale Lösung x̃:", x_opt)

    print("\nTeil d: Newton-Verfahren für M=2 Schritte")
    x_iter = newton_verfahren(x0, grad_f_lamb, Hesse_f_lamb, M=2)

    print("\nTeil e: Genauigkeit ϵ = 1e-2")
    k, xk = newton_verfahren_opt(x0, grad_f_lamb, Hesse_f_lamb, x_opt, eps=1e-2)
    if k:
        print(f"Erreicht nach {k} Schritten mit xk =", xk)
    else:
        print("Genauigkeit nicht erreicht.")


    def plot_newton_verlauf(f_lamb, x_vals, xrange=(-0.5, 1.5), yrange=(-0.5, 1.5), steps=200):
        # Raster generieren
        x = np.linspace(*xrange, steps)
        y = np.linspace(*yrange, steps)
        X, Y = np.meshgrid(x, y)
        Z = f_lamb(X, Y)

        # Plot vorbereiten
        fig, ax = plt.subplots(figsize=(8, 6))
        cs = ax.contour(X, Y, Z, levels=50, cmap='viridis')
        ax.clabel(cs, inline=True, fontsize=8)
        ax.set_title("Newton-Verfahren auf f(x, y)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Iterationspunkte
        x_vals = np.array(x_vals)
        ax.plot(x_vals[:, 0], x_vals[:, 1], 'ro-', label="Newton-Pfade")
        for i, (xk, yk) in enumerate(x_vals):
            ax.text(xk, yk, f"$x_{{{i}}}$", fontsize=9, color="red", ha='right')

        ax.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



    # Newton-Verlauf mit Speicherung aller Punkte
    x0 = [1.075, 1.0625, 1]
    _, _, _, f_lamb, grad_f_lamb, Hesse_f_lamb = funk()
    x_vals = [np.array(x0)]
    xk = np.array(x0)

    for _ in range(5):
        grad_k = np.array(grad_f_lamb(*xk), dtype=float)
        Hesse_k = np.array(Hesse_f_lamb(*xk), dtype=float)
        xk = newton_schritt(xk, grad_k, Hesse_k)
        x_vals.append(xk.copy())

    # Visualisierung
    #plot_newton_verlauf(f_lamb, x_vals)





# %%
