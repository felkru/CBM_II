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
#import sympy as sp
import matplotlib.pyplot as plt

a = np.loadtxt("Messwerte.csv")

# 1 a)
bins = 20
plt.title('1a) Histogramm der Messwerte')
plt.hist(a, bins=bins)
plt.xlabel("x")
plt.ylabel("Anzahl")

# 1 b)
# Mittelwert
mean = np.mean(a)
print(f'Der Mittelwert ist {mean}')
# Statistische Unsicherheit über den Mittelwert
std = np.std(a)
std_mean = std / np.sqrt(len(a))
print(f'Der Mittelwert hat eine statistische Unsicherheit von {std_mean}')

# 1 c)
# std wurde bereits für statistische Unsicherheit über den Mittelwert berechnet

# Histogramm mit Gaussverteilung vergleichen
gauss = np.random.normal(mean, std, len(a))
# Plot gaussian
# plt.hist(gauss, bins=bins, alpha=0.5, label='Gaussverteilung')
plt.show()

