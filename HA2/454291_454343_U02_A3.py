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
a = np.loadtxt('linfit_data_2.csv', delimiter=',', dtype=float, skiprows=1)
xwerte =a[:,0]
ywerte = a[:,1]
fig, ax = plt.subplots()
ax.scatter(xwerte,ywerte)
ax.set_title('Messwerte')
ax.set_xlabel('x-Werte')
ax.set_ylabel('y-Werte')
plt.tight_layout()
plt.show()

#3b) mit curve_fit und polyfit
def linear_model(xwerte, b, c):
    return b * xwerte + c

popt, pcov = curve_fit(linear_model, xwerte, ywerte)
#print(f'a ist {params[0]}, b ist {params[1]}')

pars, cov = np.polyfit(xwerte, ywerte, 1, cov=True)
print(f'Das Ergebnis für a ist {pars[0]} plus minus {np.sqrt(cov[0,0])} und für b {pars[1]} plus minus {cov[1,1]}')

#3c)
corrcoef = cov[0,1] / np.sqrt(cov[0,0] * cov[1,1])
print(f'Ja, die Werte sind korreliert, da die Kovarianz nicht 0 ist sondern {cov[0,1]}. Der Korrelationskoeffizient ist {corrcoef}')

#3d)
plt.plot(xwerte, ywerte, label='Daten')
plt.plot(xwerte, linear_model(xwerte, *popt), 'r-',
         label='Fit: b=%5.3f, c=%5.3f' % tuple(popt))
plt.title('Chi-Quadrat fit mit Plot der Werte')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()


#3e)
xgerade = np.linspace(900, 910, 10)

def func2(xgerade, d, e):
    return d * xgerade + e

# fit_stderror2= np.sqrt(xgerade**2*np.sqrt(cov[0,0])*np.sqrt(cov[1,1]))
fit_stderror = pcov[0,0] * xgerade**2 + pcov[1,1] * 1 + 2 * xgerade * pcov[0,1]
plt.plot(xgerade, func2(xgerade, popt[0]+fit_stderror,popt[1]))
plt.plot(xgerade, func2(xgerade, popt[0]-fit_stderror,popt[1]))
plt.show()

print(pcov)