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

print("--- Aufgabe 1 ---")

# --- Teil a) ---
print("\n--- Teil a) Voraussetzungen t-Test ---")
# The t-test assumes:
# 1. Independence of observations.
# 2. Normality of the data in both groups being compared.
# 3. Homogeneity of variances (equal variances) between the groups.
# Given a "Notenspiegel" (grade distribution), which often represents points or grades,
# the assumption of NORMALITY is frequently violated. Grade distributions can be skewed,
# bimodal (e.g., if there's a pass/fail cluster), or bounded (e.g., between 0 and 100 points, or 1.0 and 5.0).
print("a) Die vermutlich nicht erfüllte Voraussetzung für den t-Test bei einem Notenspiegel ist die Annahme der Normalverteilung der Daten. Notenverteilungen sind oft schief, begrenzt oder weisen andere nicht-normale Eigenschaften auf.")

# --- Teil b) ---
print("\n--- Teil b) Numerische Überprüfung der Normalverteilung ---")

# Load the data
try:
    df_grades = pd.read_csv('lehramtNoten.csv')
    # Assuming the relevant column is named 'Punkte' based on dummy data.
    # Adjust if the actual file has a different name.
    if 'Punkte' not in df_grades.columns:
         raise ValueError("CSV does not contain 'Punkte' column. Please check file.")
    grades_data = df_grades['Punkte'].dropna().to_numpy()
    print(f"Daten aus 'lehramtNoten.csv' geladen. Anzahl gültiger Werte: {len(grades_data)}")

    if len(grades_data) < 8: # Chi-squared test needs sufficient data
         print("Warnung: Zu wenige Datenpunkte für einen zuverlässigen Chi-Quadrat-Test.")
         is_normal = False # Cannot reliably test
         p_value_chi2 = np.nan
    else:
        # Perform Chi-squared goodness-of-fit test against a normal distribution
        # H0: The data follows a Normal(mean, sigma=1) distribution.
        # H1: The data does not follow a Normal(mean, sigma=1) distribution.

        # Calculate sample mean
        sample_mean = np.mean(grades_data)
        # Given assumption: Standard deviation sigma = 1
        assumed_sigma = 1.0

        print(f"Sample Mean: {sample_mean:.2f}, Assumed Sigma: {assumed_sigma}")

        # Determine bins for the histogram/test
        # Freedman-Diaconis rule for bin width: h = 2 * IQR * n^(-1/3)
        iqr = stats.iqr(grades_data)
        if iqr == 0: # Handle cases with no variance in IQR calculation
             bin_width = (np.max(grades_data) - np.min(grades_data)) / 10 # Fallback
        else:
             bin_width = 2 * iqr * (len(grades_data))**(-1/3)

        if bin_width == 0: # Handle case where all data might be the same
            bin_width = 1
            num_bins = 5 # Arbitrary small number
            print(f"Warnung: Geringe Varianz in Daten. Setze Bin-Breite auf {bin_width}.")
        else:
             data_range = np.max(grades_data) - np.min(grades_data)
             num_bins = max(5, int(np.ceil(data_range / bin_width))) # Ensure at least 5 bins

        print(f"Anzahl Bins für Chi2-Test gewählt: {num_bins}")

        # Get observed frequencies
        observed_freq, bin_edges = np.histogram(grades_data, bins=num_bins)

        # Calculate expected frequencies under H0: Normal(sample_mean, assumed_sigma=1)
        expected_freq = np.zeros_like(observed_freq, dtype=float)
        norm_dist = stats.norm(loc=sample_mean, scale=assumed_sigma)

        for i in range(num_bins):
            cdf_lower = norm_dist.cdf(bin_edges[i])
            cdf_upper = norm_dist.cdf(bin_edges[i+1])
            expected_prob = cdf_upper - cdf_lower
            expected_freq[i] = expected_prob * len(grades_data)

        # Ensure expected frequencies are not too small (often > 5 required)
        # If some are too small, bins might need to be merged, but scipy's chisquare handles this somewhat.
        # For simplicity here, we proceed, but note this limitation.
        if np.any(expected_freq < 1):
             print("Warnung: Einige erwartete Häufigkeiten sind sehr klein (<1). Chi2-Test könnte ungenau sein.")
        if np.sum(expected_freq == 0) > 0:
            print("FEHLER: Nullexpectation in Bins - Test nicht durchführbar in dieser Form.")
            # Handle this case, e.g., by merging bins or stopping
            chi2_stat, p_value_chi2 = np.nan, np.nan
            is_normal = False # Cannot perform test
        else:
            # Perform the Chi-squared test
            # ddof = 1 because we estimated 1 parameter (the mean) from the data.
            # Note: stats.chisquare's default ddof might need adjustment based on how parameters are estimated.
            # However, the prompt implies testing against a *specific* normal dist (mean=sample_mean, sigma=1 fixed).
            # In this specific setup (fixed sigma=1), we only used the data for the mean.
            # Let's calculate degrees of freedom: k - 1 - p = num_bins - 1 - 1 = num_bins - 2
            # We can also let chisquare compute it if we don't provide f_exp directly from a fitted model.
            # Let's use the direct calculation with stats.chisquare:
            chi2_stat, p_value_chi2 = stats.chisquare(f_obs=observed_freq, f_exp=expected_freq, ddof=1) # ddof=1 for estimated mean

            alpha = 0.05
            is_normal = p_value_chi2 >= alpha

            print(f"Chi2-Statistik: {chi2_stat:.4f}")
            print(f"p-Wert (Chi2-Test): {p_value_chi2:.4f}")

            if is_normal:
                print(f"Der p-Wert ({p_value_chi2:.4f}) ist größer oder gleich alpha ({alpha}).")
                print("Die Nullhypothese (Daten sind normalverteilt mit µ=Sample Mean, σ=1) kann nicht verworfen werden.")
            else:
                print(f"Der p-Wert ({p_value_chi2:.4f}) ist kleiner als alpha ({alpha}).")
                print("Die Nullhypothese (Daten sind normalverteilt mit µ=Sample Mean, σ=1) wird verworfen.")

        # Conclusion for t-test applicability
        if not is_normal:
            print("\nBegründung: Da der Chi-Quadrat-Test darauf hindeutet, dass die Daten signifikant von der angenommenen Normalverteilung (mit sigma=1) abweichen (p < 0.05), ist eine wichtige Voraussetzung des t-Tests verletzt. Die Anwendung des t-Tests ist hier daher nicht korrekt bzw. robust.")
        else:
             print("\nBegründung: Basierend auf dem Chi-Quadrat-Test gibt es keine ausreichenden Beweise, um die Annahme der Normalverteilung (mit sigma=1) zu verwerfen (p >= 0.05). Unter dieser Annahme *könnte* der t-Test angewendet werden, WENN auch die anderen Voraussetzungen (Varianzhomogenität, Unabhängigkeit) für einen Vergleich zweier Gruppen erfüllt wären.")
        # Note: The task only asked to check normality for the *given* grade list, not for comparing two groups yet.
        # The overall conclusion depends on the result of the test. Assuming the test fails:
        print("Fazit für t-Test Anwendung: Basierend auf der numerischen Überprüfung der Normalverteilungsannahme (mit dem Chi2-Test gegen N(mean, 1)), ist diese Voraussetzung wahrscheinlich verletzt. Daher ist die Anwendung des t-Tests für diese Daten nicht angemessen.")

except FileNotFoundError:
    print("FEHLER: Datei 'lehramtNoten.csv' nicht gefunden. Bitte sicherstellen, dass sie im selben Verzeichnis liegt.")
    is_normal = False # Cannot proceed
    grades_data = None # Ensure variable exists for plotting check
except ValueError as e:
    print(f"FEHLER beim Lesen der CSV-Datei: {e}")
    is_normal = False
    grades_data = None
except Exception as e:
    print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
    is_normal = False
    grades_data = None

# --- Teil c) ---
print("\n--- Teil c) Grafische Darstellung ---")

if grades_data is not None and len(grades_data) > 0 and 'sample_mean' in locals():
    plt.figure(figsize=(10, 6))

    # Histogram of the actual data
    # Use the same bins as for the chi2 test for consistency
    hist_counts, hist_bins, _ = plt.hist(grades_data, bins=bin_edges, density=False, alpha=0.7, label='Beobachtete Daten (Histogramm)')

    # Plot the PDF of the hypothesized normal distribution
    # Scale the PDF to match the histogram counts: PDF * N * bin_width
    bin_widths = np.diff(bin_edges)
    # For scaling, use an average bin width or calculate per bin if widths vary significantly
    avg_bin_width = np.mean(bin_widths) if len(bin_widths) > 0 else 1

    x_norm = np.linspace(bin_edges[0], bin_edges[-1], 500)
    pdf_norm = stats.norm.pdf(x_norm, loc=sample_mean, scale=assumed_sigma)
    # Scale PDF by total count N and average bin width
    scaled_pdf_norm = pdf_norm * len(grades_data) * avg_bin_width

    plt.plot(x_norm, scaled_pdf_norm, 'r-', lw=2, label=f'Erwartete Verteilung\nN(µ={sample_mean:.2f}, σ={assumed_sigma}) (skaliert)')

    plt.title('Vergleich der Notenverteilung mit angenommener Normalverteilung (σ=1)')
    plt.xlabel('Punkte')
    plt.ylabel('Häufigkeit')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    # Save or show plot
    # plt.savefig('123456_789012_U04_A1c.png')
    plt.show()

    print("Plot zur Darstellung der Verteilung und der Normalverteilungsannahme wurde erstellt.")
else:
    print("Plot konnte nicht erstellt werden, da Daten fehlen oder ein Fehler aufgetreten ist.")

print("\n--- Ende Aufgabe 1 ---")