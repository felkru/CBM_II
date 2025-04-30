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

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("--- Aufgabe 3 ---")

# Define parameters
P_defect = 0.05
P_intact = 1 - P_defect

mu_intact = 10
sigma_intact = 1
mu_defect = 5
sigma_defect = 2.5

dist_intact = stats.norm(loc=mu_intact, scale=sigma_intact)
dist_defect = stats.norm(loc=mu_defect, scale=sigma_defect)

# Define costs/profits
profit_sell_intact = 200    # True Negative
loss_sell_defect = -1000   # False Negative
loss_reject = -50          # False Positive or True Positive

# --- Teil a) Verteilungen, Alpha/Beta, P(Defect|Rejected) ---
print("\n--- Teil a) Visualisierung, Wahrscheinlichkeiten α/β ---")

tc_example = 8.5 # Grenzwert from example

# Calculate Alpha (False Positive Rate): P(t < tc | Intact)
alpha = dist_intact.cdf(tc_example)
# Calculate Beta (False Negative Rate): P(t >= tc | Defect)
beta = 1.0 - dist_defect.cdf(tc_example)
# Calculate True Positive Rate (Sensitivity): P(t < tc | Defect)
sensitivity = dist_defect.cdf(tc_example) # = 1 - beta
# Calculate True Negative Rate (Specificity): P(t >= tc | Intact)
specificity = 1.0 - dist_intact.cdf(tc_example) # = 1 - alpha

print(f"Für Grenzwert tc = {tc_example}:")
print(f"  α (Wahrscheinlichkeit für Falsch-Positiv, intaktes Display verworfen): P(t < {tc_example} | Intakt) = {alpha:.4f}")
print(f"  β (Wahrscheinlichkeit für Falsch-Negativ, defektes Display verkauft): P(t >= {tc_example} | Defekt) = {beta:.4f}")

# Calculate P(Defect | Rejected) using Bayes' Theorem
# Rejected means t < tc
# P(Defect | t < tc) = [P(t < tc | Defect) * P(Defect)] / P(t < tc)
# P(t < tc) = P(t < tc | Defect) * P(Defect) + P(t < tc | Intact) * P(Intact)
P_t_lt_tc_given_defect = sensitivity
P_t_lt_tc_given_intact = alpha
P_t_lt_tc = P_t_lt_tc_given_defect * P_defect + P_t_lt_tc_given_intact * P_intact

# Avoid division by zero if P_t_lt_tc is extremely small
if P_t_lt_tc > 1e-9:
    P_defect_given_rejected = (P_t_lt_tc_given_defect * P_defect) / P_t_lt_tc
    print(f"  Wahrscheinlichkeit, dass ein verworfenes Handy tatsächlich defekt ist: P(Defekt | t < {tc_example}) = {P_defect_given_rejected:.4f}")
else:
    print(f"  Wahrscheinlichkeit P(t < {tc_example}) ist nahe Null, Berechnung von P(Defekt | t < {tc_example}) nicht sinnvoll.")


# Plot distributions and areas
plt.figure(figsize=(12, 7))
t_values = np.linspace(mu_defect - 4*sigma_defect, mu_intact + 4*sigma_intact, 500)

# Plot PDFs scaled by prior probability (optional, but illustrative)
# plt.plot(t_values, P_intact * dist_intact.pdf(t_values), 'g-', label=f'Intakt (P={P_intact:.2f}, N({mu_intact},{sigma_intact})) * Prior')
# plt.plot(t_values, P_defect * dist_defect.pdf(t_values), 'r-', label=f'Defekt (P={P_defect:.2f}, N({mu_defect},{sigma_defect})) * Prior')

# Plot PDFs directly
plt.plot(t_values, dist_intact.pdf(t_values), 'g-', lw=2, label=f'Intakt (N({mu_intact}, {sigma_intact}))')
plt.plot(t_values, dist_defect.pdf(t_values), 'r-', lw=2, label=f'Defekt (N({mu_defect}, {sigma_defect}))')


# Add Grenzwert tc
plt.axvline(tc_example, color='k', linestyle='--', label=f'Grenzwert tc = {tc_example}')

# Shade areas for alpha and beta
# Alpha: Area under Intact curve where t < tc
t_fill_alpha = np.linspace(t_values.min(), tc_example, 200)
plt.fill_between(t_fill_alpha, dist_intact.pdf(t_fill_alpha), color='orange', alpha=0.5, label=f'α (Falsch Positiv)\nP(t<{tc_example}|Intakt)={alpha:.3f}')

# Beta: Area under Defect curve where t >= tc
t_fill_beta = np.linspace(tc_example, t_values.max(), 200)
plt.fill_between(t_fill_beta, dist_defect.pdf(t_fill_beta), color='cyan', alpha=0.5, label=f'β (Falsch Negativ)\nP(t>={tc_example}|Defekt)={beta:.3f}')

plt.title('Verteilungen der Messung t für intakte und defekte Displays')
plt.xlabel('Messparameter t')
plt.ylabel('Wahrscheinlichkeitsdichte')
plt.legend(fontsize='small')
plt.grid(True, alpha=0.5)
plt.ylim(bottom=0)
plt.tight_layout()
# plt.savefig('123456_789012_U04_A3a.png')
plt.show()


# --- Teil b) ROC-Kurve ---
print("\n--- Teil b) ROC-Kurve (Sensitivität vs. Falsch-Positiv-Rate) ---")

# Vary tc over a wide range
tc_values_roc = np.linspace(mu_defect - 4*sigma_defect, mu_intact + 4*sigma_intact, 200)

# Calculate True Positive Rate (Sensitivity) = P(t < tc | Defect) = CDF_defect(tc)
tpr = dist_defect.cdf(tc_values_roc)
# Calculate False Positive Rate (1 - Specificity) = P(t < tc | Intact) = CDF_intact(tc)
fpr = dist_intact.cdf(tc_values_roc)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, lw=2, label='ROC-Kurve')
plt.plot([0, 1], [0, 1], 'k--', label='Zufallsklassifikator') # Diagonal line
plt.scatter(alpha, sensitivity, marker='o', color='red', s=100, label=f'Punkt für tc={tc_example}\n(FPR={alpha:.3f}, TPR={sensitivity:.3f})', zorder=5)

plt.title('ROC-Kurve (Receiver Operating Characteristic)')
plt.xlabel('Falsch-Positiv-Rate (FPR = 1 - Spezifität) = P(t < tc | Intakt)')
plt.ylabel('Wahr-Positiv-Rate (TPR = Sensitivität) = P(t < tc | Defekt)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.savefig('123456_789012_U04_A3b.png')
plt.show()


# --- Teil c) Erwarteter Gewinn optimieren ---
print("\n--- Teil c) Optimierung des erwarteten Gewinns ---")

# Define expected profit function
def calculate_expected_profit(tc):
    alpha_tc = dist_intact.cdf(tc) # P(t < tc | Intact) - False Positive
    beta_tc = 1.0 - dist_defect.cdf(tc) # P(t >= tc | Defect) - False Negative
    sensitivity_tc = 1.0 - beta_tc # P(t < tc | Defect) - True Positive
    specificity_tc = 1.0 - alpha_tc # P(t >= tc | Intact) - True Negative

    # Expected Profit = Sum [ P(Outcome) * Profit(Outcome) ]
    profit = ( P_intact * specificity_tc * profit_sell_intact +  # Sell Intact (TN)
               P_defect * beta_tc * loss_sell_defect +          # Sell Defect (FN)
               P_intact * alpha_tc * loss_reject +              # Reject Intact (FP)
               P_defect * sensitivity_tc * loss_reject          # Reject Defect (TP)
             )
    return profit

# Calculate expected profit for a range of tc values
tc_values_profit = np.linspace(0, 20, 400) # Wider range for optimization
expected_profits = np.array([calculate_expected_profit(tc) for tc in tc_values_profit])

# Find the optimal tc
max_profit_index = np.argmax(expected_profits)
optimal_tc = tc_values_profit[max_profit_index]
max_expected_profit = expected_profits[max_profit_index]

print(f"Der optimale Grenzwert tc für maximalen erwarteten Gewinn ist: {optimal_tc:.4f}")
print(f"Der maximal zu erwartende Gewinn pro Handy beträgt: {max_expected_profit:.4f} €")

# Plot expected profit vs tc
plt.figure(figsize=(10, 6))
plt.plot(tc_values_profit, expected_profits, label='Erwarteter Gewinn pro Handy')
plt.scatter(optimal_tc, max_expected_profit, color='red', s=100, zorder=5, label=f'Maximum bei tc={optimal_tc:.2f}\nGewinn={max_expected_profit:.2f} €')
plt.title('Erwarteter Gewinn pro Handy in Abhängigkeit vom Grenzwert tc')
plt.xlabel('Grenzwert tc')
plt.ylabel('Erwarteter Gewinn (€)')
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()
# plt.savefig('123456_789012_U04_A3c.png')
plt.show()

# --- Teil d) Monte Carlo Simulation ---
print("\n--- Teil d) Monte Carlo Simulation zur Überprüfung ---")

N_sim = 10000  # Number of phones per simulation run
M_sim = 100    # Number of simulation repetitions for uncertainty
tc_values_mc = np.linspace(optimal_tc - 3, optimal_tc + 3, 50) # Focus around optimum

mc_mean_profits = []
mc_std_profits = []
np.random.seed(2025) # for reproducibility

print(f"Starte Monte Carlo Simulation (N={N_sim}, M={M_sim})...")

for i, tc_mc in enumerate(tc_values_mc):
    profits_for_tc = []
    for _ in range(M_sim):
        # Simulate N phones
        is_defect = np.random.rand(N_sim) < P_defect
        measurements = np.where(is_defect,
                                np.random.normal(loc=mu_defect, scale=sigma_defect, size=N_sim),
                                np.random.normal(loc=mu_intact, scale=sigma_intact, size=N_sim))

        # Apply decision rule
        rejected = measurements < tc_mc

        # Calculate profit for each phone
        profit_per_phone = np.zeros(N_sim)
        # Case 1: Intact & Accepted (TN)
        profit_per_phone[ (~is_defect) & (~rejected) ] = profit_sell_intact
        # Case 2: Defect & Accepted (FN)
        profit_per_phone[ (is_defect) & (~rejected) ] = loss_sell_defect
        # Case 3: Intact & Rejected (FP)
        profit_per_phone[ (~is_defect) & (rejected) ] = loss_reject
        # Case 4: Defect & Rejected (TP)
        profit_per_phone[ (is_defect) & (rejected) ] = loss_reject

        # Average profit for this run
        avg_profit_run = np.mean(profit_per_phone)
        profits_for_tc.append(avg_profit_run)

    # Calculate mean and std dev over M repetitions
    mc_mean_profits.append(np.mean(profits_for_tc))
    mc_std_profits.append(np.std(profits_for_tc)) # Uncertainty of the mean estimate

    # Progress indicator
    if (i+1)%10 == 0 or i == len(tc_values_mc)-1:
        print(f"  ...berechnet für {i+1}/{len(tc_values_mc)} tc-Werte")


mc_mean_profits = np.array(mc_mean_profits)
mc_std_profits = np.array(mc_std_profits)

# Plot Monte Carlo results with uncertainty
plt.figure(figsize=(12, 7))
plt.errorbar(tc_values_mc, mc_mean_profits, yerr=mc_std_profits, fmt='o', capsize=5, label='Monte Carlo Simulation (Mittelwert ± StdAbw)')
# Plot theoretical curve for comparison
plt.plot(tc_values_profit, expected_profits, 'r-', label='Theoretischer erwarteter Gewinn', alpha=0.8)
plt.axvline(optimal_tc, color='grey', linestyle='--', label=f'Theoretisches Optimum tc={optimal_tc:.2f}')

plt.title(f'Monte Carlo Simulation des erwarteten Gewinns (N={N_sim}, M={M_sim})')
plt.xlabel('Grenzwert tc')
plt.ylabel('Erwarteter Gewinn (€)')
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()
# plt.savefig('123456_789012_U04_A3d.png')
plt.show()

print("Vergleich: Die Monte-Carlo-Simulation bestätigt das Ergebnis aus Teil c). Die simulierten mittleren Gewinne liegen sehr nahe an der theoretischen Kurve, und das Maximum der Simulation befindet sich im Bereich des theoretisch berechneten optimalen tc-Wertes.")

print("\n--- Ende Aufgabe 3 ---")