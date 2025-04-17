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
import scipy.stats
import scipy.stats as stats
#import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit


#3a)
a = np.loadtxt('linfit_data_2.csv', delimiter=',', dtype=str, skiprows=1)
xwerte =a[1:,0]
ywerte = a[1:,1]
xwerte = xwerte.astype(float)
ywerte = ywerte.astype(float)
fig, ax = plt.subplots()
ax.scatter(xwerte,ywerte)
ax.set_title('Messwerte')
ax.set_xlabel('x-Werte')
ax.set_ylabel('y-Werte')
ax.xaxis.set_major_locator(MultipleLocator(15))
ax.yaxis.set_major_locator(MultipleLocator(5))
plt.tight_layout()
plt.show()

def func(xwerte, b, c):
    return b * xwerte + c

#3b) mit curve_fit und polyfit
popt, pcov = curve_fit(func, xwerte, ywerte)
print(f'a ist {popt[0]}, b ist {popt[1]}')

result_pars, result_cov = np.polyfit(xwerte, ywerte, 1, cov=True)
print(f'Das Ergebnis für a ist {result_pars[0]} plus minus {np.sqrt(result_cov[0,0])} und für b {result_pars[1]} plus minus {result_cov[1,1]}')

#3c)
corrcoef = result_cov[0,1]/np.sqrt(result_cov[0,0]*result_cov[1,1])
print(f'Ja, die Werte sind korreliert, da die Kovarianz nicht 0 ist sondern {result_cov[0,1]}. Der Korrelationskoeffizient ist {corrcoef}')

#3d)
plt.plot(xwerte, ywerte, label='Daten')
plt.plot(xwerte, func(xwerte, *popt), 'r-',
         label='Fit: b=%5.3f, c=%5.3f' % tuple(popt))
plt.title('Chi-Quadrat fit mit Plot der Werte')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()


#3e)
xgerade = np.linspace(900, 910, 10)

def func2(xgerade, d, e):
    return d * xgerade +e

plt.plot(xgerade, func2(xgerade, popt[0]+np.sqrt(result_cov[0,0]),popt[1]))
plt.plot(xgerade, func2(xgerade, popt[0]-np.sqrt(result_cov[0,0]),popt[1]))
plt.show()

print(pcov)