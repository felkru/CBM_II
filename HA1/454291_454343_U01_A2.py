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

# Task 2
tau = 100 # value in seconds

# Please find A2 a) in the attached .pdf and .tex files.

# 2 b)
n = 1000 # you can choose the size of the sample here
sample = np.random.exponential(tau, size=n)

# plot samples as histogram
plt.title('2b) Histogramm des simulierten Exponentialzerfalls')
plt.xlabel('t in Sekunden')
plt.ylabel('Counts')
plt.hist(sample, bins=30)
plt.show()

# print sample median and stat. error
mean = np.mean(sample)
stat_error = np.std(sample, ddof=1)/np.sqrt(len(sample))
print(f'Der Mittelwert der Stichprobe ist {mean}s ±{stat_error}s')

# 2 c)
x = list(range(2, 1000))
means = []
stds = []
errors = []

for k in x:
    sample = np.random.exponential(tau, size=k)
    means.append(np.mean(sample))
    stds.append(np.std(sample, ddof=1))
    errors.append(np.std(sample)/np.sqrt(len(sample)))

plt.title('2c) Stichprobenmittelwert in Abhängigkeit von Stichprobengröße')
plt.xlabel('Stichprobengröße')
plt.ylabel('Stichprobenmittelwert')
plt.plot(x, means)
plt.show()

plt.title('2c) Stichprobenstandardabweichung in Abhängigkeit von Stichprobengröße')
plt.xlabel('Stichprobengröße')
plt.ylabel('Stichprobenstandardabweichung')
plt.plot(x, stds)
plt.show()

plt.title('2c) Stat. Fehler d. Mittelwerts abhängig von Stichprobengröße')
plt.xlabel('Stichprobengröße')
plt.ylabel('Statistische Unsicherheit des Mittelwerts')
plt.plot(x, errors)
plt.show()