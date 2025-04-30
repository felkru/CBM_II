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

# Load data and preprocess
try:
    df_blood = pd.read_csv('blutwerte.csv')
    # Handle potential NaNs
    df_blood_clean = df_blood.dropna(subset=['Entzündungswert'])
    print(f"Daten aus 'blutwerte.csv' geladen. {len(df_blood) - len(df_blood_clean)} Zeilen mit NaN entfernt.")

    # Separate groups
    group_a_data = df_blood_clean[df_blood_clean['Gruppe'] == 'A']['Entzündungswert'].to_numpy()
    group_b_data = df_blood_clean[df_blood_clean['Gruppe'] == 'B']['Entzündungswert'].to_numpy()

    n_a = len(group_a_data)
    n_b = len(group_b_data)

    if n_a < 2 or n_b < 2:
        raise ValueError("Nicht genügend Daten in einer oder beiden Gruppen nach NaN-Entfernung.")

    print(f"Anzahl Patienten Gruppe A (Medikament): {n_a}")
    print(f"Anzahl Patienten Gruppe B (Placebo): {n_b}")

except FileNotFoundError:
    print("FEHLER: Datei 'blutwerte.csv' nicht gefunden. Bitte sicherstellen, dass sie im selben Verzeichnis liegt.")
    # Exit or set dummy data to prevent further errors
    exit()
except ValueError as e:
    print(f"FEHLER: {e}")
    exit()
except Exception as e:
    print(f"Ein unerwarteter Fehler beim Laden der Daten ist aufgetreten: {e}")
    exit()


# --- Teil a) Annahmen prüfen & Plot ---
print("\n--- Teil a) Überprüfung der Voraussetzungen (Normalität, Varianz) ---")

# Calculate statistics for fitting
mean_a, std_a = np.mean(group_a_data), np.std(group_a_data, ddof=1) # Use sample std dev
mean_b, std_b = np.mean(group_b_data), np.std(group_b_data, ddof=1) # Use sample std dev
var_a, var_b = std_a**2, std_b**2

print(f"Gruppe A - Mean: {mean_a:.2f}, StdDev: {std_a:.2f}, Variance: {var_a:.2f}")
print(f"Gruppe B - Mean: {mean_b:.2f}, StdDev: {std_b:.2f}, Variance: {var_b:.2f}")

# Visually check normality and compare distributions with fitted Gaussians
plt.figure(figsize=(12, 7))

# Determine shared bins for better comparison
all_data = np.concatenate([group_a_data, group_b_data])
min_val, max_val = np.min(all_data), np.max(all_data)
bin_edges = np.linspace(min_val, max_val, 15) # Adjust bin number as needed

# Histogram for Group A
counts_a, _, _ = plt.hist(group_a_data, bins=bin_edges, alpha=0.6, label=f'Gruppe A (Medikament, N={n_a})', density=True)
# Fitted Normal Distribution for Group A
x_a = np.linspace(min_val, max_val, 300)
pdf_a = stats.norm.pdf(x_a, mean_a, std_a)
plt.plot(x_a, pdf_a, 'b-', lw=2, label=f'Fit Gruppe A (µ={mean_a:.2f}, σ={std_a:.2f})')

# Histogram for Group B
counts_b, _, _ = plt.hist(group_b_data, bins=bin_edges, alpha=0.6, label=f'Gruppe B (Placebo, N={n_b})', density=True)
# Fitted Normal Distribution for Group B
x_b = np.linspace(min_val, max_val, 300)
pdf_b = stats.norm.pdf(x_b, mean_b, std_b)
plt.plot(x_b, pdf_b, 'r-', lw=2, label=f'Fit Gruppe B (µ={mean_b:.2f}, σ={std_b:.2f})')

plt.title('Verteilung der Entzündungswerte nach Behandlung')
plt.xlabel('Entzündungswert (mg/L)')
plt.ylabel('Dichte')
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()
# plt.savefig('123456_789012_U04_A2a.png')
plt.show()

# Comment on normality based on plot
print("\nKommentar zur Normalität (visuell):")
print("Die Histogramme beider Gruppen scheinen grob glockenförmig zu sein, was eine Normalverteilung nahelegt.")
print("Gruppe A (blau) sieht der angepassten Normalverteilung recht ähnlich.")
print("Gruppe B (orange) scheint ebenfalls plausibel durch die Normalverteilung beschrieben zu werden, auch wenn die Übereinstimmung möglicherweise nicht perfekt ist.")
# Add formal test (optional, but good practice)
shapiro_a = stats.shapiro(group_a_data)
shapiro_b = stats.shapiro(group_b_data)
print(f"Shapiro-Wilk Test (Normalität) Gruppe A: W={shapiro_a.statistic:.3f}, p={shapiro_a.pvalue:.3f}")
print(f"Shapiro-Wilk Test (Normalität) Gruppe B: W={shapiro_b.statistic:.3f}, p={shapiro_b.pvalue:.3f}")
normality_ok = shapiro_a.pvalue > 0.05 and shapiro_b.pvalue > 0.05
if normality_ok:
    print("Beide Gruppen scheinen laut Shapiro-Wilk Test (p > 0.05) mit Normalverteilungen vereinbar zu sein.")
else:
    print("Mindestens eine Gruppe weicht laut Shapiro-Wilk Test (p <= 0.05) signifikant von einer Normalverteilung ab. Die t-Test-Annahme ist möglicherweise verletzt.")


# Comment on homogeneity of variances
print("\nKommentar zur Varianzhomogenität:")
# Ratio of variances - rule of thumb: should not exceed ~3 or 4
var_ratio = var_b / var_a if var_a > 0 else np.inf
var_ratio_inv = var_a / var_b if var_b > 0 else np.inf
print(f"Verhältnis der Varianzen (größer/kleiner): {max(var_ratio, var_ratio_inv):.2f}")
# Levene's test (more robust than Bartlett's)
levene_test = stats.levene(group_a_data, group_b_data)
print(f"Levene-Test (Varianzhomogenität): W={levene_test.statistic:.3f}, p={levene_test.pvalue:.3f}")
variance_ok = levene_test.pvalue > 0.05
if variance_ok:
    print("Der Levene-Test (p > 0.05) deutet nicht auf signifikant unterschiedliche Varianzen hin. Annahme der Varianzhomogenität ist wahrscheinlich erfüllt.")
    equal_var_assumption = True
else:
    print("Der Levene-Test (p <= 0.05) deutet auf signifikant unterschiedliche Varianzen hin. Annahme der Varianzhomogenität ist verletzt. Welch's t-Test (equal_var=False) sollte verwendet werden.")
    equal_var_assumption = False

# Overall conclusion for t-test applicability
print("\nFazit zur Anwendbarkeit des t-Tests:")
if normality_ok and variance_ok:
    print("Die Annahmen der Normalität und Varianzhomogenität scheinen erfüllt. Ein Standard t-Test (Student's t-Test) kann angewendet werden.")
elif normality_ok and not variance_ok:
    print("Die Normalitätsannahme scheint erfüllt, aber die Varianzen sind unterschiedlich. Welch's t-Test (berücksichtigt ungleiche Varianzen) ist angemessen.")
else:
    print("Die Normalitätsannahme ist verletzt. Der t-Test ist nicht robust gegenüber dieser Verletzung. Ein nicht-parametrischer Test (z.B. Mann-Whitney U) wäre möglicherweise besser geeignet. Wir führen den t-Test dennoch wie gefordert durch, aber interpretieren das Ergebnis mit Vorsicht.")
    # We proceed as requested, maybe defaulting to Welch if variances are unequal
    equal_var_assumption = not variance_ok # Use Welch if variances unequal, even if normality fails

# --- Teil b) Herleitung ---
print("\n--- Teil b) Herleitung ---")
print("Die Herleitung wurde in den separaten Dateien '123456_789012_U04_A2b.tex' und '.pdf' erstellt.")

# --- Teil c) Qualitative Erklärung ---
print("\n--- Teil c) Qualitative Erklärung des t-Tests ---")
print("Der unabhängige t-Test testet hier die Nullhypothese (H0), dass die durchschnittlichen Entzündungswerte in der Population, aus der die Medikamentengruppe (A) stammt, gleich den durchschnittlichen Entzündungswerten in der Population sind, aus der die Placebogruppe (B) stammt (d.h. µ_A = µ_B).")
print("Die Alternativhypothese (H1), die wir untersuchen wollen, ist, dass das Medikament die Entzündungswerte signifikant *senkt*, also dass der durchschnittliche Entzündungswert in der Medikamentengruppe *niedriger* ist als in der Placebogruppe (d.h. µ_A < µ_B).")
print("Der Test prüft, ob der beobachtete Unterschied der Mittelwerte (x̄_A - x̄_B) groß genug ist (unter Berücksichtigung der Streuung und Stichprobengröße), um als statistisch signifikant zu gelten und nicht nur auf Zufall zu beruhen.")

# --- Teil d) Durchführung t-Test ---
print("\n--- Teil d) Durchführung des t-Tests ---")

# Perform independent t-test using scipy.stats.ttest_ind
# We want to test if group A is *less* than group B (µ_A < µ_B) -> one-sided test
# Use 'alternative="less"'
# Use equal_var based on Levene test result from part a)
t_stat, p_value = stats.ttest_ind(group_a_data, group_b_data,
                                  equal_var=equal_var_assumption,
                                  alternative='less') # H1: mean(A) < mean(B)

print(f"Durchgeführter Test: {'Student t-Test (gleiche Varianzen angenommen)' if equal_var_assumption else 'Welch t-Test (ungleiche Varianzen angenommen)'}")
print(f"t-Statistik: {t_stat:.4f}")
print(f"Beobachtetes Signifikanzniveau (p-Wert, einseitig): {p_value:.4f}")

# --- Teil e) Vergleich und Schlussfolgerung ---
print("\n--- Teil e) Vergleich mit Schwellniveau und Schlussfolgerung ---")
alpha_C = 0.05
print(f"Vergleich des p-Wertes ({p_value:.4f}) mit dem Schwellniveau αC = {alpha_C}")

if p_value < alpha_C:
    print("Da der p-Wert kleiner als αC ist, verwerfen wir die Nullhypothese (H0).")
    print("Schlussfolgerung: Es gibt statistisch signifikante Evidenz (auf dem 5%-Niveau), dass das Medikament zu niedrigeren Entzündungswerten führt als das Placebo.")
else:
    print("Da der p-Wert nicht kleiner als αC ist, können wir die Nullhypothese (H0) nicht verwerfen.")
    print("Schlussfolgerung: Es gibt keine ausreichende statistisch signifikante Evidenz (auf dem 5%-Niveau), um zu behaupten, dass das Medikament zu niedrigeren Entzündungswerten führt als das Placebo. Der beobachtete Unterschied könnte zufällig sein.")

# --- Teil f) Wahrscheinlichkeit für Zufall ---
print("\n--- Teil f) Wahrscheinlichkeit für zufällige Unterschiede ---")
print(f"Der p-Wert ({p_value:.4f}) gibt die Wahrscheinlichkeit an, einen Unterschied in den Mittelwerten zu beobachten, der mindestens so groß ist wie der gefundene (oder noch extremer), unter der Annahme, dass die Nullhypothese (kein echter Unterschied zwischen Medikament und Placebo, µ_A = µ_B) wahr ist.")
print("Man kann also sagen: Wenn das Medikament tatsächlich keine Wirkung hätte, beträgt die Wahrscheinlichkeit, rein zufällig einen solchen oder stärkeren Unterschied zugunsten des Medikaments in den Stichproben zu finden, ca. {:.2f}%. ".format(p_value * 100))


print("\n--- Ende Aufgabe 2 ---")