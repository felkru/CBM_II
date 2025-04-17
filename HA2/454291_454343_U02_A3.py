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



a = np.loadtxt('linfit_data_2.csv', delimiter=',', dtype=str)
for zeile in a:
    x, y = zeile
    #print(x, y)
#print(a)
fig, ax = plt.subplots()
ax.scatter(a[:,0],a[:,1])
ax.set_title('Messwerte')
ax.set_xlabel('x-Werte')
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_major_locator(MultipleLocator(5))
plt.tight_layout()
plt.show()

