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
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from numpy.random import chisquare
from scipy.optimize import curve_fit
import scipy.stats as stats
from scipy.stats import chi2

#3a) Daten darstellen
data = np.loadtxt('chi2.csv', delimiter=',', dtype=float)
xwerte = data[:,0]
ywerte = data[:,1]
plt.errorbar(xwerte, ywerte, yerr=0.01, fmt='o')
plt.title("Messwerte mit quadratischem Fit")
plt.xlabel("x")
plt.ylabel("y")

#3b) Modellanpassung
def func(x, a, b, c):
    return a*x**2+b*x+c

(a,b,c), cov = curve_fit(func, xwerte, ywerte)
plt.plot(xwerte, func(xwerte, a, b, c), label ='Fit')
plt.legend()
print(f'a = {a}, b = {b}, c = {c}')

#3c) chiquadrat berechnen
chiquadrat = np.sum(((ywerte-func(xwerte, a, b, c))**2)/(0.01**2))
print(f"Der Chiquadratwert ist {chiquadrat}")

#3d) Beurteilung Modell passend zu Daten?
ndof = len(xwerte)-3
mean = ndof
sigma = np.sqrt(2*ndof)
#xgauss = np.linspace(xwerte.min(), xwerte.max(), 100)
#ygauss = stats.norm.pdf(xgauss, loc=mean, scale=sigma)
#plt.plot(xgauss, ygauss)
plt.show()

#in cdf
alpha = 1 - stats.chi2.cdf(chiquadrat, ndof)
print(f'Alpha ist = {alpha}, das Modell passt also nicht gut zu den Daten')

# Kritischer χ²-Wert für 5%-Signifikanzniveau (rechtsseitig)
chi2_critical = chi2.ppf(0.95, ndof)
print(f"χ²-Wert bei 5%-Signifikanzniveau: {chi2_critical:.2f}, der tatsächliche Wert {chiquadrat:.2f} ist damit viel größer als der 5% Wert.")

# Interpretation
if chiquadrat > chi2_critical:
    print("→ Der Fit ist schlecht (χ² größer als kritischer Wert).")
else:
    print("→ Der Fit ist akzeptabel (χ² kleiner oder gleich kritischer Wert).")

#3e) Residuenplot
yres = ywerte - func(xwerte, a, b, c)
plt.scatter(xwerte, yres)
plt.title('Residuenplot')
plt.xlabel('Daten')
plt.ylabel('Residuen')
plt.axhline(0, color='red', linestyle='--')  # Nullinie
plt.show()
print(f'Da die Residuen einer systematischen Struktur folgen und nicht zufällig um die Nullinie verteilt sind, passt das Modell nicht gut zu den Daten')

#3f) besseres Modell finden (ausprobieren kubisches Modell)
print('Ab hier wird ein kubischer Fit gemacht:')
def kubisch (x, a, b, c, d):
    return a*x**3+b*x**2+c*x+d

(a,b,c,d), cov = curve_fit(kubisch, xwerte, ywerte)
plt.plot(xwerte, kubisch(xwerte, a, b, c, d), label ='Fit kubisch')
plt.errorbar(xwerte, ywerte, yerr=0.01, fmt='o')
plt.legend()
plt.title('Messwerte mit kubischem Fit')
plt.show()
print(f'a = {a}, b = {b}, c = {c}, d = {d}')

chiquadrat2 = np.sum(((ywerte-kubisch(xwerte, a, b, c, d))**2)/(0.01**2))
print(f"Der Chiquadratwert ist {chiquadrat}")
ndof = len(xwerte)-4
alpha = 1 - stats.chi2.cdf(chiquadrat2, ndof)
print(f'Alpha ist = {alpha}, das Modell passt also gut zu den Daten')

# Kritischer χ²-Wert für 5%-Signifikanzniveau (rechtsseitig)
chi2_critical = chi2.ppf(0.95, ndof)
print(f"χ²-Wert bei 5%-Signifikanzniveau: {chi2_critical:.2f}, der tatsächliche Wert {chiquadrat2:.2f} ist damit kleiner als der 5% Wert.")

# Interpretation
if chiquadrat2 > chi2_critical:
    print("→ Der Fit ist schlecht (χ² größer als kritischer Wert).")
else:
    print("→ Der Fit ist akzeptabel (χ² kleiner oder gleich kritischer Wert).")

# Residuenplot
yres = ywerte - kubisch(xwerte, a, b, c, d)
plt.scatter(xwerte, yres)
plt.title('Residuenplot für kubischen Fit')
plt.xlabel('Daten')
plt.ylabel('Residuen')
plt.axhline(0, color='red', linestyle='--')  # Nullinie
plt.show()
print(f'Da die Residuen keiner systematischen Struktur folgen, passt das Modell gut zu den Daten.')
