#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import count  # Not used, can be removed

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

# Häufig benötigte Module:
import numpy as np
# import scipy.stats
# import scipy.stats as stats # Redundant alias
# import sympy as sp
import matplotlib.pyplot as plt
import scipy as sci
from scipy.stats import chi2 as chi2_dist  # For p-value calculation

# --- Task 3a: Load and Plot Data ---
print("--- Aufgabe 3a ---")
try:
    # Load data, skip header row
    data = np.loadtxt('linfit_data_2.csv', delimiter=',', dtype=float, skiprows=1)
    x_data = data[:, 0]  # Corrected indexing
    y_data = data[:, 1]  # Corrected indexing
except FileNotFoundError:
    print("Error: 'linfit_data_2.csv' not found. Make sure it's in the same directory.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.scatter(x_data, y_data, label='Messwerte', marker='o', s=15, zorder=5)  # s=size, zorder=draw on top
ax1.set_title('Messwerte')
ax1.set_xlabel('x-Werte')
ax1.set_ylabel('y-Werte')
ax1.grid(True, linestyle=':')

# --- Task 3b: Linear Fit using curve_fit ---
print("\n--- Aufgabe 3b ---")

# Define the linear model function (using a=slope, b=intercept)
def linear_model(x, a, b):
    """Linear model y = a*x + b"""
    return a * x + b

# Perform the fit using curve_fit (minimizes chi-squared)
# By default, curve_fit assumes sigma=1 for all data points if not provided.
params, cov = sci.optimize.curve_fit(linear_model, x_data, y_data)

# Extract parameters
a_fit = params[0]
b_fit = params[1]

# Extract standard errors from the covariance matrix diagonal
stderr_a = np.sqrt(cov[0, 0])
stderr_b = np.sqrt(cov[1, 1])

print(f"Fit-Parameter (curve_fit):")
print(f"  Steigung a = {a_fit:.5f} ± {stderr_a:.5f}")
print(f"  Achsenabschnitt b = {b_fit:.3f} ± {stderr_b:.3f}")

# Compare with polyfit (optional, just for verification)
# result_pars_poly, result_cov_poly = np.polyfit(x_data, y_data, 1, cov=True)
# print("\nFit-Parameter (polyfit):")
# print(f"  Steigung a = {result_pars_poly[0]:.5f} ± {np.sqrt(result_cov_poly[0,0]):.5f}")
# print(f"  Achsenabschnitt b = {result_pars_poly[1]:.3f} ± {np.sqrt(result_cov_poly[1,1]):.3f}")


# --- Task 3c: Check for Correlation ---
print("\n--- Aufgabe 3c ---")
# Get the covariance between a and b
cov_ab = cov[0, 1]
# Calculate the correlation coefficient
correlation_ab = cov_ab / (stderr_a * stderr_b)

print(f"Kovarianz zwischen a und b: {cov_ab:.5f}")
print(f"Korrelationskoeffizient zwischen a und b: {correlation_ab:.5f}")
if abs(correlation_ab) > 1e-6:
    print("Die Parameter a und b sind korreliert (Korrelationskoeffizient != 0).")
else:
    print("Die Parameter a und b sind (nahezu) unkorreliert.")

# --- Task 3d: Plot Fitted Line ---
print("\n--- Aufgabe 3d ---")
# Generate points for a smooth line plot
x_fit_line = np.linspace(x_data.min(), x_data.max(), 200)
y_fit_line = linear_model(x_fit_line, a_fit, b_fit)

ax1.plot(x_fit_line, y_fit_line, 'r-', linewidth=2,
         label=f'Fit: y = {a_fit:.3f}x + {b_fit:.3f}')
print("Angepasste Gerade wurde zum Plot hinzugefügt.")

# --- Task 3e: Plot Uncertainty Band ---
print("\n--- Aufgabe 3e ---")
# Calculate the standard deviation of the predicted y_fit values
var_y_fit = cov[0, 0] * x_fit_line ** 2 + cov[1, 1] + 2 * x_fit_line * cov[0, 1] # Var(y_fit) = Var(a*x + b) = x^2*Var(a) + Var(b) + 2*x*Cov(a, b)
sigma_y_fit = np.sqrt(var_y_fit)

# Plot the 1-sigma uncertainty band
ax1.fill_between(x_fit_line, y_fit_line - sigma_y_fit, y_fit_line + sigma_y_fit, color='red', alpha=0.3, label='Unsicherheitsband (±1σ)')

ax1.legend()
print("Unsicherheitsband wurde zum Plot hinzugefügt.")
# Display the first plot (data, fit, uncertainty)
plt.tight_layout()
# plt.show() # Show plot 1 now, or combine later


# --- Task 3f: Residual Plot ---
print("\n--- Aufgabe 3f ---")
# Calculate residuals
residuals = y_data - linear_model(x_data, a_fit, b_fit)

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.scatter(x_data, residuals, marker='o', s=15, label='Residuen')
ax2.axhline(0, color='grey', linestyle='--', linewidth=1)  # Line at y=0
ax2.set_title('Residuenplot')
ax2.set_xlabel('x-Werte')
ax2.set_ylabel('Residuen (y_data - y_fit)')
ax2.grid(True, linestyle=':')
ax2.legend()
plt.tight_layout()
# plt.show() # Show plot 2 now, or combine later


# --- Task 3g: Chi-squared Test ---
print("\n--- Aufgabe 3g ---")


def perform_chi2_test(x, y, model_func, params, sigma_y):
    """Performs a chi-squared test for goodness of fit."""
    y_model = model_func(x, *params)
    residuals = y - y_model
    chi2_value = np.sum((residuals / sigma_y) ** 2)

    n_points = len(y)
    n_params = len(params)
    dof = n_points - n_params  # Degrees of freedom

    if dof <= 0:
        print("  Fehler: Freiheitsgrade <= 0. Kann p-Wert nicht berechnen.")
        return chi2_value, dof, np.nan

    p_value = 1.0 - chi2_dist.cdf(chi2_value, dof)  # CDF = Cumulative Distribution Function

    return chi2_value, dof, p_value


# Test case 1: Assume sigma_y = 0.5 for all points
sigma_y_1 = 0.5
chi2_1, dof_1, p_value_1 = perform_chi2_test(x_data, y_data, linear_model, params, sigma_y_1)

print(f"\nChi-Quadrat-Test mit angenommener Unsicherheit σ_y = {sigma_y_1}:")
print(f"  χ² = {chi2_1:.2f}")
print(f"  Freiheitsgrade (DoF) = {dof_1}")
print(f"  Reduziertes χ² (χ²/DoF) = {chi2_1 / dof_1:.2f}")
print(f"  p-Wert = {p_value_1:.3g}")

if p_value_1 > 0.05:  # Common threshold, can be adjusted
    print(
        f"  Beurteilung (σ={sigma_y_1}): Guter Fit (p > 0.05). Das Modell beschreibt die Daten gut, wenn die Unsicherheiten bei {sigma_y_1} liegen.")
else:
    print(
        f"  Beurteilung (σ={sigma_y_1}): Schlechter Fit (p <= 0.05). Das Modell beschreibt die Daten schlecht oder die angenommene Unsicherheit ist zu klein.")
    if (chi2_1 / dof_1) > 1:
        print(
            "     (Reduziertes Chi² > 1 deutet darauf hin, dass das Modell unzureichend ist oder die Unsicherheiten unterschätzt wurden).")
    elif (chi2_1 / dof_1) < 1:
        print(
            "     (Reduziertes Chi² < 1 deutet darauf hin, dass das Modell zu gut passt oder die Unsicherheiten überschätzt wurden).")

# Test case 2: Assume sigma_y = 0.005 for all points
sigma_y_2 = 0.005
chi2_2, dof_2, p_value_2 = perform_chi2_test(x_data, y_data, linear_model, params, sigma_y_2)

print(f"\nChi-Quadrat-Test mit angenommener Unsicherheit σ_y = {sigma_y_2}:")
print(f"  χ² = {chi2_2:.2f}")
print(f"  Freiheitsgrade (DoF) = {dof_2}")
print(f"  Reduziertes χ² (χ²/DoF) = {chi2_2 / dof_2:.2f}")
print(f"  p-Wert = {p_value_2:.3g}")

if p_value_2 > 0.05:
    print(
        f"  Beurteilung (σ={sigma_y_2}): Guter Fit (p > 0.05). Das Modell beschreibt die Daten gut, wenn die Unsicherheiten bei {sigma_y_2} liegen.")
else:
    print(
        f"  Beurteilung (σ={sigma_y_2}): Schlechter Fit (p <= 0.05). Das Modell beschreibt die Daten schlecht oder die angenommene Unsicherheit ist zu klein.")
    if (chi2_2 / dof_2) > 1:
        print(
            "     (Reduziertes Chi² > 1 deutet darauf hin, dass das Modell unzureichend ist oder die Unsicherheiten unterschätzt wurden).")
    elif (chi2_2 / dof_2) < 1:
        print(
            "     (Reduziertes Chi² < 1 deutet darauf hin, dass das Modell zu gut passt oder die Unsicherheiten überschätzt wurden).")

# Show all plots at the end
plt.show()