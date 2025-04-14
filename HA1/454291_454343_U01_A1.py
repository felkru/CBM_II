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
plt.xlabel("x")
plt.ylabel("Anzahl")
plt.show()

# 1 b)
# median
mu = np.mean(a)
print(f'Der Mittelwert der Stichprobe ist {mu}')
# statistical uncertainty of the median
sigma = np.std(a)
std_mean = sigma / np.sqrt(len(a))
print(f'Der Stichprobenmittelwert hat einen statistischen Fehler von {std_mean}')

# 1 c)
# the standard deviation was already calculated for the statistical uncertainty of the median
print(f'Die Standardabweichung der Stichprobe ist {sigma}')

x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
gauss = stats.norm.pdf(x, mu, sigma)
# plot gaussian
plt.title('1c) Normalverteilung mit Stichproben-mu und -sigma')
plt.xlabel('X')
plt.ylabel('Wahrscheinlichkeit für x')
plt.plot(x, gauss)
plt.show()

# 1 d)
cdfAt3p5 = stats.norm.pdf(3.5, mu, sigma)
bigger3p5freq = len([v for v in a if v > 3.5])/len(a)
print(f'Stimmt das Modell, dass die Messwerte mit mu und sigma normalverteilt sind, sollten {1-cdfAt3p5}% der Werte über 3,5 liegen.')
print(f'In unserere Stichprobe liegen {bigger3p5freq}% der Werte über 3,5.')

# Task 2
tau = 100 # value in seconds

# 2 b)
n = 1000 # number of samples
sample = np.random.exponential(tau, size=n)

# plot samples as histogram
plt.title('2b) Histogramm des simulierten Exponentialzerfalls')
plt.xlabel('t in Sekunden')
plt.ylabel('Counts')
plt.hist(sample, bins=20)
plt.show()

# print sample median and stat. error
mean = np.mean(sample)
stat_error = np.std(sample)/np.sqrt(len(sample))
print(f'Der Mittelwert der Stichprobe ist {mean}±{stat_error}')

# 2 c)
x = list(range(2, 1000))
means = []
stds = []
errors = []

for k in x:
    sample = np.random.exponential(tau, size=k)
    means.append(np.mean(sample))
    stds.append(np.std(sample))
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

plt.title('2c) Statistische Unsicherheit des Mittelwerts in Abhängigkeit von Stichprobengröße')
plt.xlabel('Stichprobengröße')
plt.ylabel('Statistische Unsicherheit des Mittelwerts')
plt.plot(x, errors)
plt.show()

# Task 3
# 3 a)
n, m = 1, 1000
mu, sigma = 5, 0.8

cs = []
for _ in range(m):
    ai = np.random.normal(mu, sigma, n)
    cs.append(np.sum(((ai - mu)/sigma)**2))

print('Aufgabe 3a):', cs)

# 3 b)

plt.title('Chi-Quadrat-Verteilung')
plt.xlabel('X')
plt.ylabel('Anzahl')
plt.hist(cs, bins=20)
plt.show()