#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import count

### Vorlagendatei für die Übungen zur Computergestützten Physik ###

#
# 1)
# Matrikelnummer: 454291
# Name: Julia Els
# Email: julia.els@rwth-aachen.de
#
# 2)
# Matrikelnummer: 454343
# Name: Felix Krückel
# Email: felix.krueckel@rwth-aachen.de
#

# Häufig benötigte Module (auskommentieren, wenn notwendig):
import numpy as np
import scipy.stats as stats
#import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#3a)
data = np.loadtxt('linfit_data_2.csv', delimiter=',', dtype=float, skiprows=1)
x = data[:, 0]
y = data[:, 1]
fig, ax = plt.subplots()
ax.scatter(x, y)
ax.set_title('Messwerte')
ax.set_xlabel('x-Werte')
ax.set_ylabel('y-Werte')
plt.tight_layout()
plt.show()

#3b) mit curve_fit und polyfit
def linear_model(xwerte, a, b):
    return a * xwerte + b

(a,b), cov = curve_fit(linear_model, x, y)
stderr_a, stderr_b = np.sqrt(cov[0,0]), np.sqrt(cov[1,1])

print(f"Fit-Parameter (curve_fit):")
print(f"  Steigung a = {a:.5f} ± {stderr_a:.5f}")
print(f"  Achsenabschnitt b = {b:.3f} ± {stderr_b:.3f}")

#3c)
corrcoef = cov[0,1] / np.sqrt(cov[0,0] * cov[1,1])
print(f'Ja, die Werte sind korreliert, da*param die Kovarianz nicht 0 ist sondern {cov[0,1]}. Der Korrelationskoeffizient ist {corrcoef}')

#3d)
plt.scatter(x, y, label='Daten')
plt.plot(x, linear_model(x, a, b), 'r-', label=f'Fit: a={a}, b={b}')
plt.title('Chi-Quadrat fit mit Plot der Werte')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

#3e)
x_values = np.linspace(900, 910, 100)

fit_stderror = np.sqrt(cov[0,0] * x_values ** 2 + cov[1,1] + 2 * x_values * cov[0,1])
plt.fill_between(x_values, linear_model(x_values, a, b) - fit_stderror, linear_model(x_values, a, b) + fit_stderror, color='red', alpha=0.3, label='Unsicherheit (±1σ)')
plt.show()

#3f)
residuals = y - linear_model(x, a, b)

plt.axhline(0, color='grey', linestyle='--', linewidth=1)
plt.scatter(x, residuals)
plt.show()

#3g)
print(f'--- Aufgabe 3g) ---')
ndof = len(residuals) - 2
print(f"  Freiheitsgrade (DoF) = {ndof}")

# Test 1
std1 = 0.5
chi2_value = np.sum((residuals / std1) ** 2)
alpha = 1 - stats.chi2.cdf(chi2_value, ndof)

print(f"\nChi-Quadrat-Test mit angenommener Unsicherheit σ_y = {std1}:")
print(f"  χ² = {chi2_value:.2f}")
print(f"  Reduziertes χ² (χ²/DoF) = {chi2_value / ndof:.2f}")
print(f"  p-Wert = {alpha:.3g}")

# Test 2
std1 = 0.005
chi2_value = np.sum((residuals / std1) ** 2)
alpha = 1 - stats.chi2.cdf(chi2_value, ndof)

print(f"\nChi-Quadrat-Test mit angenommener Unsicherheit σ_y = {std1}:")
print(f"  χ² = {chi2_value:.2f}")
print(f"  Reduziertes χ² (χ²/DoF) = {chi2_value / ndof:.2f}")
print(f"  p-Wert = {alpha:.3g}")

print(f'\n Zusammenfassend lässt sich sagen, dass unser lineares Modell für ein Sigma von 0.5 gut passen würde während für s=0.05 keine gute Übereinstimmung besteht.')
