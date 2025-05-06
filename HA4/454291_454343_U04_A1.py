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
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#%%
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
print("a) Die vermutlich nicht erfüllte Voraussetzung für den t-Test bei einem Notenspiegel ist die Annahme der Normalverteilung der Daten. Die Verteilungen haben beide sowohl bei der 1 als auch bei der 5 zu hohe Häufigkeiten.")

# --- Teil b) ---
print("\n--- Teil b) Numerische Überprüfung der Normalverteilung ---")
data = pd.read_csv('lehramtNoten.csv')
lehramt = data['Lehramt_Noten'].dropna().to_numpy()

n = len(lehramt)
mean = np.mean(lehramt)
std = np.std(lehramt, ddof=1)

# Boundaries: start below 1.0, end above 5.0, use midpoints in between
grades = np.array([1.0, 1.3, 1.7, 2.0, 2.3, 2.7, 3.0, 3.3, 3.7, 4.0, 5.0])
boundaries = np.concatenate(([-np.inf], (grades[:-1] + grades[1:]) / 2, [np.inf]))

observed_grades = pd.Series(lehramt).value_counts().reindex(grades, fill_value=0)

expected_probs = []
for i in range(len(grades)):
    lower_bound = boundaries[i]
    upper_bound = boundaries[i+1]
    prob = stats.norm.cdf(upper_bound, loc=mean, scale=1) - \
           stats.norm.cdf(lower_bound, loc=mean, scale=1)
    expected_probs.append(prob)

chi2, p_value = stats.chisquare(f_obs=observed_grades, f_exp=np.array(expected_probs) * n, ddof=1)

print(f'P-Value Lehramt: {p_value:.2f}')
print(f'Die Notenverteilung ist nicht normalverteilt.')
print(f'Entsprechend wäre ein t-Test nicht aussagekräftig, weil er annimmt, dass beide Verteilungen normalverteilt sind.')

print('--- 1 c) ---')
# print('Siehe Plot')
plt.figure(figsize=(8, 6))

lehramt_mean = np.mean(lehramt)
hist_bins = np.sort(np.unique(lehramt))
if len(hist_bins) > 1:
    bin_edges = np.concatenate(( [hist_bins[0]-0.1] , (hist_bins[:-1] + hist_bins[1:])/2 , [hist_bins[-1]+0.1] ))
else:
    bin_edges = [hist_bins[0]-0.1, hist_bins[0]+0.1]

plt.bar(grades, observed_grades/n, width=0.1, label=f'Beobachtete Anteile (N={n})', align='center')
x_norm = np.linspace(0.5, 5.5, 200)
pdf_lehramt = stats.norm.pdf(x_norm, loc=lehramt_mean, scale=1)
plt.plot(x_norm, pdf_lehramt, 'r--', label=f'Normalverteilung (mean={lehramt_mean:.2f}, =1)')
plt.title('Lehramt Noten vs. Normalverteilung')
plt.xlabel('Note')
plt.ylabel('Dichte')
plt.xlim(0.95, 5.05)
plt.legend()
plt.grid(axis='y', linestyle=':')

plt.tight_layout()
plt.show()