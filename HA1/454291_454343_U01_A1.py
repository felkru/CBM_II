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

a = np.loadtxt("Messwerte.csv")

# Task 1
# 1 a)
bins = 20
plt.title('1a) Histogramm der Messwerte')
plt.hist(a, bins=bins)
plt.xlabel("X")
plt.ylabel("Anzahl")
plt.show()

# 1 b)
# median
mu = np.mean(a)
print(f'Der Mittelwert der Stichprobe ist {mu}')
# statistical uncertainty of the median
sigma = np.std(a, ddof=1)
std_mean = sigma / np.sqrt(len(a))
print(f'Der Stichprobenmittelwert hat einen statistischen Fehler von {std_mean}')

# 1 c)
# the standard deviation was already calculated for the statistical uncertainty of the median
print(f'Man sollte als Parameter der Gaussfunktion das Stichproben-mu und -sigma wählen.')
print(f'Die Standardabweichung der Stichprobe ist {sigma}')

x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
gauss = stats.norm.pdf(x, mu, sigma)
# plot gaussian
plt.title('1c) Normalverteilung vs Histogramm')
plt.xlabel('X')
plt.ylabel('Wahrscheinlichkeit für x')
plt.hist(a, bins=bins, density=True)
plt.plot(x, gauss)
plt.show()



# 1 d)
cdfAt3p5 = stats.norm.pdf(3.5, mu, sigma)
bigger3p5freq = len([v for v in a if v > 3.5])/len(a)
print(f'Stimmt das Modell, dass die Messwerte mit mu und sigma normalverteilt sind, sollten {1-cdfAt3p5}% der Werte über 3,5 liegen.')
print(f'In unserere Stichprobe liegen {bigger3p5freq}% der Werte über 3,5.')