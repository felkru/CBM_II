#! /usr/bin/env python3
# -*- coding: utf-8 -*-
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings

print("--- Aufgabe 2 ---")
df_blood = pd.read_csv('blutwerte.csv')
    # Handle potential NaNs
df_blood_clean = df_blood.dropna()
print(f"Daten aus 'blutwerte.csv' geladen. {len(df_blood) - len(df_blood_clean)} Zeilen mit NaN entfernt.")

    # Separate groups
group_a_data = df_blood_clean['Medikament'].to_numpy()
group_b_data = df_blood_clean['Placebo'].to_numpy()

n_a = len(group_a_data)
n_b = len(group_b_data)

print(f"Anzahl Patienten Gruppe A (Medikament): {n_a}")
print(f"Anzahl Patienten Gruppe B (Placebo): {n_b}")


# --- Teil a) Annahmen prüfen & Plot ---

# Mittelwert, Standardabweichung und Varianz berechnen
mean_a, std_a = np.mean(group_a_data), np.std(group_a_data, ddof=1) # Use sample std dev
mean_b, std_b = np.mean(group_b_data), np.std(group_b_data, ddof=1) # Use sample std dev
var_a, var_b = std_a**2, std_b**2

print(f"Gruppe A - Mean: {mean_a:.2f}, StdDev: {std_a:.2f}, Variance: {var_a:.2f}")
print(f"Gruppe B - Mean: {mean_b:.2f}, StdDev: {std_b:.2f}, Variance: {var_b:.2f}")

# Visually check normality and compare distributions with fitted Gaussians
plt.figure(figsize=(12, 7))

# Determine shared bins for better comparison
all_data = np.concatenate([group_a_data, group_b_data])#beide Arrays zusammenführen
min_val, max_val = np.min(all_data), np.max(all_data)#globalen minimal- und maximalwert beider Gruppen finden
bin_edges = np.linspace(min_val, max_val, 15) # Bins zwischen den minimal- und Maximalwerten erzeugen

# Histogram for Group A
counts_a, _, _ = plt.hist(group_a_data, bins=bin_edges, alpha=0.6, label=f'Gruppe A (Medikament, N={n_a})', density=True)
# Normalverteilung für Gruppe A
x_a = np.linspace(min_val, max_val, 300)
pdf_a = stats.norm.pdf(x_a, mean_a, std_a)
plt.plot(x_a, pdf_a, 'b-', lw=2, label=f'Fit Gruppe A (µ={mean_a:.2f}, σ={std_a:.2f})')

# Histogram for Group B
counts_b, _, _ = plt.hist(group_b_data, bins=bin_edges, alpha=0.6, label=f'Gruppe B (Placebo, N={n_b})', density=True)
# Normalverteilung für Gruppe B
x_b = np.linspace(min_val, max_val, 300)
pdf_b = stats.norm.pdf(x_b, mean_b, std_b)
plt.plot(x_b, pdf_b, 'r-', lw=2, label=f'Fit Gruppe B (µ={mean_b:.2f}, σ={std_b:.2f})')

plt.title('Verteilung der Entzündungswerte nach Behandlung')
plt.xlabel('Entzündungswert (mg/L)')
plt.ylabel('Dichte')
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()

print("Die Histogramme beider Gruppen scheinen grob glockenförmig zu sein, was eine Normalverteilung nahelegt.")
print("Gruppe A (blau) sieht der angepassten Normalverteilung recht ähnlich.")
print("Gruppe B (orange) scheint ebenfalls durch die Normalverteilung beschrieben zu werden, auch wenn der eine Bin zwischen 18 und 20 keine Einträge hat und der unterhab von 18 dafür mehr als bei einer Normalverteilung.")

# --- Teil c) Qualitative Erklärung ---
print("Der unabhängige t-Test testet hier die Nullhypothese (H0), dass die durchschnittlichen Entzündungswerte in der Population, aus der die Medikamentengruppe (A) stammt, gleich den durchschnittlichen Entzündungswerten in der Population sind, aus der die Placebogruppe (B) stammt (d.h. µ_A = µ_B).")
print("Die Alternativhypothese (H1), die wir untersuchen wollen, ist, dass das Medikament die Entzündungswerte signifikant *senkt*, also dass der durchschnittliche Entzündungswert in der Medikamentengruppe *niedriger* ist als in der Placebogruppe (d.h. µ_A < µ_B).")
print("Der Test prüft, ob der beobachtete Unterschied der Mittelwerte (x̄_A - x̄_B) groß genug ist (unter Berücksichtigung der Streuung und Stichprobengröße), um als statistisch signifikant zu gelten und nicht nur auf Zufall zu beruhen.")

# --- Teil d) Durchführung t-Test ---
ndof = len(group_a_data)+len(group_b_data)-2
t_value = np.mean(group_a_data) - np.mean(group_b_data) / np.sqrt((var_a+var_b)/ndof)
t = np.absolute(t_value)
alpha = stats.t.sf(t, df=ndof)

print(f"t-Statistik: {t}")
print(f"Beobachtetes Signifikanzniveau (p-Wert, einseitig): {alpha}")

# --- Teil e) Vergleich und Schlussfolgerung ---
print("\n--- Teil e) Vergleich mit Schwellniveau und Schlussfolgerung ---")
alpha_C = 0.05
print(f"Vergleich des p-Wertes ({alpha:.4f}) mit dem Schwellniveau αC = {alpha_C}")

if alpha < alpha_C:
    print("Da der p-Wert kleiner als αC ist, verwerfen wir die Nullhypothese (H0).")
    print("Schlussfolgerung: Es gibt statistisch signifikante Evidenz (auf dem 5%-Niveau), dass das Medikament zu niedrigeren Entzündungswerten führt als das Placebo.")
else:
    print("Da der p-Wert nicht kleiner als αC ist, können wir die Nullhypothese (H0) nicht verwerfen.")
    print("Schlussfolgerung: Es gibt keine ausreichende statistisch signifikante Evidenz (auf dem 5%-Niveau), um zu behaupten, dass das Medikament zu niedrigeren Entzündungswerten führt als das Placebo. Der beobachtete Unterschied könnte zufällig sein.")

# --- Teil f) Wahrscheinlichkeit für Zufall ---
print("\n--- Teil f) Wahrscheinlichkeit für zufällige Unterschiede ---")
print(f"Der p-Wert ({alpha:.4f}) gibt die Wahrscheinlichkeit an, einen Unterschied in den Mittelwerten zu beobachten, der mindestens so groß ist wie der gefundene (oder noch extremer), unter der Annahme, dass die Nullhypothese (kein echter Unterschied zwischen Medikament und Placebo, µ_A = µ_B) wahr ist.")
print("Man kann also sagen: Wenn das Medikament tatsächlich keine Wirkung hätte, beträgt die Wahrscheinlichkeit, rein zufällig einen solchen oder stärkeren Unterschied zugunsten des Medikaments in den Stichproben zu finden, ca. {:.2f}%. ".format(alpha * 100))


print("\n--- Ende Aufgabe 2 ---")