# 1)
# Matrikelnummer: 454291
# Name: Julia Els
# Email: julia.els@rwth-aachen.de
#
# 2)
# Matrikelnummer: 454343
# Name: Felix Krückel
# Email: felix.krueckel@rwth-aachen.de

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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
xvalues = np.linspace(np.min(x), np.max(x), 1000)
#Untergrund und Signal
signal_func = ew(xvalues, res_sig.x[0], res_sig.x[1], res_sig.x[2])
#Untergrund
untergrund_func = ew(xvalues, res_bg.x[0], res_bg.x[1], 0)
# plot datapoints
plt.errorbar(x, counts, yerr=np.sqrt(counts), fmt='o')
plt.plot(xvalues, signal_func)
plt.plot(xvalues, untergrund_func)
plt.xlabel('Massenspektrum [GeV]')
plt.ylabel("Zählrate pro GeV")
plt.show()
