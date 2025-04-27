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

data = np.loadtxt('chi2.csv', delimiter=',', dtype=float)
xwerte = data[:,0]
ywerte = data[:,1]
#plt.scatter(xwerte, ywerte, s=1)
plt.errorbar(xwerte, ywerte, yerr=0.01, fmt='o')

#3b)
def func(x, a, b, c):
    return a*x**2+b*x+c

(a,b,c), cov = curve_fit(func, xwerte, ywerte)
plt.plot(xwerte, func(xwerte, a, b, c))
#plt.show()

print(f'a = {a}, b = {b}, c = {c}')

#chi2= stats.chisquare(ywerte, func(xwerte, a, b, c))
#print(chi2)

#3c)
chiquadrat= np.sum(((ywerte-func(xwerte, a, b, c))**2)/(0.01**2))
print(f"Der Chiquadratwert ist {chiquadrat}")

#3d)
ndof = len(xwerte)-3
mean = ndof
xgauss = np.linspace(xwerte.min(), xwerte.max(), 100)
sigma = 2*ndof
ygauss = stats.norm.pdf(xgauss, loc=mean, scale=sigma)
plt.plot(xgauss, ygauss)
plt.show()
#in cdf

alpha = 1 - stats.chi2.cdf(chiquadrat, ndof)
print(alpha)


