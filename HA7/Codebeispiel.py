#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 15:28:57 2025

@author: alschmid
"""

import numpy as np
import matplotlib.pyplot as plt

# Beispielhafte Funktionen
def f(x):
    return (x - 3)**2 + 1  # Minimum bei x = 2

# Beispielhafte unimodale Funktion
def f2(x):
    return (x - 2)**2 + 1  # Minimum bei x = 2


# Phase 1: Intervall finden:
def finde_intervall_phase1(f, x0, x1, alpha=1.5, max_iter=50):
    fx0 = f(x0)
    fx1 = f(x1)
    if fx0 < fx1:
        x0, x1 = x1, x0  # tauschen, damit f(x0) > f(x1)
    for _ in range(max_iter):
        x2 = x1 + alpha * (x1 - x0)
        fx2 = f(x2)
        if fx2 > f(x1):
            return x0, x2
        x0, x1 = x1, x2
    raise Exception("Kein geeignetes Intervall gefunden.")

# Phase 2: Minimumsuche mit goldenem Schnitt
def goldener_schnitt_phase2(f, a, b, tol=1e-5):
    tau = (np.sqrt(5) - 1) / 2  # ca. 0.618
    x1 = b - tau * (b - a)
    x2 = a + tau * (b - a)
    f1, f2 = f(x1), f(x2)
    while abs(b - a) > tol:
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - tau * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + tau * (b - a)
            f2 = f(x2)
    return (a + b) / 2

# Ausführung des Verfahrens
x_start, x_next = 0.0, 1.0
a, b = finde_intervall_phase1(f, x_start, x_next)
xmin = goldener_schnitt_phase2(f, a, b)

print(f"Geschätztes Minimum bei x ≈ {xmin:.5f}, f(x) ≈ {f(xmin):.5f}")

# Optional: Plot der Funktion und des gefundenen Minimums
x_vals = np.linspace(a, b, 500)
plt.plot(x_vals, f(x_vals), label="f(x)")
plt.axvline(xmin, color='r', linestyle='--', label=f"Minimum ≈ {xmin:.4f}")
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Einfaches Suchverfahren mit Goldenem Schnitt")
plt.grid(True)
plt.show()