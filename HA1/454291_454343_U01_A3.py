#! /usr/bin/env python3
# -*- coding: utf-8 -*-

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
import scipy.stats as stats
#import sympy as sp
import matplotlib.pyplot as plt

# Task 3
# 3 a)
n, m = 1, 1000
mu, sigma = 5, 0.8

ai = np.random.normal(mu, sigma, size=(m, n))
cs = np.sum(((ai - mu)/sigma)**2, axis=1)

# 3 b)
plt.title("Chi-Quadrat verteilte cj's mit N=1")
plt.xlabel('X')
plt.ylabel('Anzahl')
plt.hist(cs, density=True, bins=30)

# 3 c)
x = np.linspace(0, max(cs), 100)
y = stats.chi2.pdf(x, n)
plt.plot(x, y)
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.show()

# 3 d)
for n in [2,4,8]:
    ai = np.random.normal(mu, sigma, size=(m, n))
    cs = np.sum(((ai - mu) / sigma) ** 2, axis=1)

    # set up plot, plot hist of c's
    plt.title(f"Chi-Quadrat verteilte cj's mit N={n}")
    plt.xlabel('X')
    plt.ylabel('Anzahl')
    plt.hist(cs, density=True, bins=30)

    # theoretical chi2 distribution
    x = np.linspace(0, max(cs), 100)
    y = stats.chi2.pdf(x, n)
    plt.plot(x, y)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.show()