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

import numpy as np
import matplotlib.pyplot as plt
import scipy as scp

def setup_plot(title, y_label='y'):
    plt.figure(figsize=(10, 6))
    plt.xlabel('x')
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)

def display_plot():
    plt.legend()
    plt.show()

data = np.genfromtxt('linfit_data_3.csv', delimiter=',' )

x, y, var = data[:, 0], data[:, 1], data[:, 2]
std = np.sqrt(var)

# %%
print('--- 2 a) ---')
setup_plot('Messpunkte mit Fehlerbalken (linfit_data_3.csv)')
plt.errorbar(x, y, yerr=std, fmt='o', label='Messpunkte mit Fehlerbalken', capsize=3)
print('Siehe Plot.')

print('--- 2 b) ---')

linear_model = lambda x, m, c: m * x + c

params_no_error, cov_no_error  = scp.optimize.curve_fit(linear_model, x, y, sigma=std, absolute_sigma=False)
params_w_error, cov_w_error = scp.optimize.curve_fit(linear_model, x, y, sigma=std, absolute_sigma=True)
plt.plot(x, linear_model(x, *params_no_error), label='Fitted linear model w/ and w/o error prop') # hier könnte man auch params_w_error verwenden, da die Werte identisch sind, da sie nur vom Datenset und den relativen Fehlern abhängen.
print(f'Ohne Fehlerpropagation (absolute_sigma=False) ergit sich für m={params_no_error[0]}±{np.sqrt(cov_no_error[0,0])} und c={params_no_error[1]}±{np.sqrt(cov_no_error[1,1])}')
print(f'Mit Fehlerpropagation (absolute_sigma=True) ergit sich für m={params_w_error[0]}±{np.sqrt(cov_w_error[0,0])} und c={params_w_error[1]}±{np.sqrt(cov_w_error[1,1])}')
print('Mit Fehlerpropagation ist der Fehler wesentlich kleiner und wir würden entsprechend unseren Fehler überschätzen, wenn absolute_error=False.')
display_plot()

print('--- 2 c) ---')
# Fehlerfortpflanzung ohne Korrelation
x_val = 3

y_at_3 = linear_model(x_val, *params_w_error)
y_at_3_error = np.sqrt(x_val**2 * cov_w_error[0,0] + cov_w_error[1,1])
print(f'f(3) = {y_at_3}±{y_at_3_error}, wenn man annimmt, dass m und c unkorreliert sind.')

print('--- 2 d) ---')

# 3d) Fehlerfortpflanzung mit Korrelation
std_y_corr = np.sqrt(x_val**2 * cov_w_error[0, 0] + cov_w_error[1,1] + 2 * x_val * cov_w_error[0,1])
print(f'f(3) = {y_at_3}±{std_y_corr}, wenn man annimmt, dass m und c korreliert sind.')