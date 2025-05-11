# 1)
# Matrikelnummer: 454291
# Name: Julia Els
# Email: julia.els@rwth-aachen.de
#
# 2)
# Matrikelnummer: 454343
# Name: Felix Krückel
# Email: felix.krueckel@rwth-aachen.de

# Dear Lars, please run `pip install tqdm` before running this file as you'll get an error if it's not installed. You can also use the attached requirements.txt file.

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# --- a) Erwartungswert-Modelle ---
def ew(x, A, lam, mu):
    background = A * np.exp(-lam * x)
    signal     = mu * stats.norm.pdf(x, loc=7, scale=1)
    return background + signal

def neg_logli(params, x, counts, fix_mu=None):
    # params = [A, lam] oder [A, lam, mu]
    A, lam = params[0], params[1]
    mu = params[2] if fix_mu is None else fix_mu
    lam_i = ew(x, A, lam, mu)
    # Summiere -ln P(counts_i | lam_i)
    return -np.sum(stats.poisson.logpmf(counts, mu=lam_i))

# --- Daten einlesen ---
df     = pd.read_csv("Data_Likelihood.csv", header=None, names=["counts"])
x      = np.arange(1, len(df) + 1)
counts = df["counts"].to_numpy()

#b)
# --- Untergrund-Modell (mu=0) --- # fix_mu=0
init_bg = [40.0, 0.2]                   # Startwerte für A, λ
bounds_bg = [(1e-6, None), (1e-6, None)]
res_bg = minimize(neg_logli, x0=init_bg, args=(x, counts, 0), method='L-BFGS-B', bounds=bounds_bg)
if not res_bg.success:
    raise RuntimeError(f"Untergrund-Fit ist fehlgeschlagen: {res_bg.message}")
print("Untergrund-Fit war erfolgreich!")

# --- Signal+Untergrund-Modell --- # mu frei
init_sig = [40.0, 0.2, 10.0]             # Startwerte für A, λ, μ
bounds_sig = [(1e-6, None), (1e-6, None), (0.0, None)]
res_sig = minimize(neg_logli, x0=init_sig, args=(x, counts, None), method='L-BFGS-B', bounds=bounds_sig)
if not res_sig.success:
    raise RuntimeError(f"Signal+Untergrund-Fit ist fehlgeschlagen: {res_sig.message}")

print("\nSignal+Untergrund-Fit war erfolgreich!")

# --- Ergebnisse prüfen und ausgeben ---
print("=== Untergrund-Modell (μ=0) ===")
print(f"  success    = {res_bg.success}")
print(f"  Â          = {res_bg.x[0]:.4f}")
print(f"  λ̂          = {res_bg.x[1]:.4f}")
print(f"  NLL minimum= {res_bg.fun:.4f}")

print("\n=== Signal+Untergrund-Modell ===")
print(f"  success    = {res_sig.success}")
print(f"  Â          = {res_sig.x[0]:.4f}")
print(f"  λ̂          = {res_sig.x[1]:.4f}")
print(f"  μ̂          = {res_sig.x[2]:.4f}")
print(f"  NLL minimum= {res_sig.fun:.4f}")

#c)
x_values = np.linspace(np.min(x), np.max(x), 1000)
#Untergrund und Signal
signal_func = ew(x_values, res_sig.x[0], res_sig.x[1], res_sig.x[2])
#Untergrund
untergrund_func = ew(x_values, res_bg.x[0], res_bg.x[1], 0)
# plot datapoints
plt.errorbar(x, counts, yerr=np.sqrt(counts), fmt='o')
plt.plot(x_values, signal_func)
plt.plot(x_values, untergrund_func)
plt.xlabel('Massenspektrum [GeV]')
plt.ylabel("Zählrate pro GeV")
plt.show()

# d)
# lambda_0 = res_sig.x[1]
# n = 5000
# dataset = np.random.poisson(lambda_0, size=n)

# test_val = 2 * n * (np.mean(dataset) * np.log(np.mean(dataset) / lambda_0) + lambda_0 - np.mean(dataset))
# p_value = 1 - stats.chi2.cdf(test_val, df=1)
# print(f'Die Wahrscheinlichkeit, dass die Daten zum Untergrundmodell passen ist: {p_value}')

test_val = -2 * (res_sig.fun - res_bg.fun)
p_value = 1 - stats.chi2.cdf(test_val, df=1)
print(f'\nDie Wahrscheinlichkeit, dass die Daten zum Untergrundmodell passen, unter der Annahme,\ndass entweder das Untergrund oder das Signalmodell das Phänomen perfekt beschreiben, ist {p_value:.2%}.')

# e)
print(f'\n--- 90% Konfidenzintervall für μ ---')

def check_LL_within_threshold(profiled_LL_array, global_max_LL_scalar, alpha):
    chi2_critical = stats.chi2.ppf(1 - alpha, df=1)
    logL_threshold = global_max_LL_scalar - (chi2_critical / 2.0)
    return profiled_LL_array >= logL_threshold

max_logL_global = -res_sig.fun
mu_mle_from_global_fit = res_sig.x[2]

confidence = 0.9
num_mu_scan_points = 10000
scan_mu_min_val = 0.0
scan_mu_max_val = 50

mu_scan_values = np.linspace(scan_mu_min_val, scan_mu_max_val, num_mu_scan_points)
profiled_NLL_values_at_mu = []

initial_A_lambda_for_profiled_fit = [res_sig.x[0], res_sig.x[1]]
bounds_A_lambda_for_profiled_fit = [(1e-6, None), (1e-6, None)]
for mu_fixed in tqdm(mu_scan_values, file=sys.stdout): # file=sys.stdout ensures tqdm output is black on most terminals
    res_profiled = minimize(neg_logli,
                            x0=initial_A_lambda_for_profiled_fit,
                            args=(x, counts, mu_fixed),
                            method='L-BFGS-B',
                            bounds=bounds_A_lambda_for_profiled_fit)
    profiled_NLL_values_at_mu.append(res_profiled.fun)

mu_is_in_CI_mask = check_LL_within_threshold(-np.array(profiled_NLL_values_at_mu), max_logL_global, alpha=1-confidence)
mu_values_within_CI = mu_scan_values[mu_is_in_CI_mask]

lower_bound_mu = np.min(mu_values_within_CI)
upper_bound_mu = np.max(mu_values_within_CI)

print(f'Schrittweite (Konfidenzintervall): Δmu = {(scan_mu_max_val-scan_mu_min_val)/num_mu_scan_points}')
print(f'Das 90% Konfidenzintervall von μ ist [{lower_bound_mu:.2f}, {upper_bound_mu:.2f}].')