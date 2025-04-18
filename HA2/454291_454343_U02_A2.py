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
from numpy.random import multivariate_normal

rho=0.6
mean=[1,2] # [mu_x, mu_y]
sigmax=1
sigmay=1
cov=[[1,0.6],[0.6,1]]
n = 1000
x = np.random.multivariate_normal(mean, cov, size=(n,))

plt.title('Scatterplot')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x[:, 0], x[:, 1], s=1)
plt.show()

#Stichprobenvarianzen von x und y
sigmasx=np.std(x[:,0], ddof=1)
sigmasy=np.std(x[:,1], ddof=1)
print(f'Die Stichprobenvarianzen von x und y sind {sigmasx} und {sigmasy}')

#Korrelationskoeffizient der erzeugten Datenpunkte
sigmasxy=1/(n-1)*np.sum((x[:,0]-x[:,0].mean())*(x[:,1]-x[:, 1].mean()))
corrs=sigmasxy/np.sqrt(sigmasx**2*sigmasy**2)
print(f'Der Korrelationskoeffizient der erzeugten Datenpunkte ist {sigmasxy}')

#Anteil der Ereignisse mit (x>2) und (y>3)
j=0
for i in range (0,n):
    if x[i,0]>2 and x[i,1]>3:
        j+=1
anteil=j/n
print(f'Der Anteil der Ereignisse mit (x>2) und (y>3) ist {anteil}')

#Wahrscheinlichkeit für (x>2) und (y>3) analytisch
m = scipy.stats.multivariate_normal(mean, cov)
lowlimit=np.array([2,3])
uplimit=np.array([np.inf, np.inf])
p = m.cdf(uplimit, lower_limit=lowlimit)
print(f'Wahrscheinlichkeit für (x>2) und (y>3) analytisch berechnet ist {p}')