# 1)
# Matrikelnummer: 454291
# Name: Julia Els
# Email: julia.els@rwth-aachen.de
#
# 2)
# Matrikelnummer: 454343
# Name: Felix Krückel
# Email: felix.krueckel@rwth-aachen.de

import numpy as np
import scipy.stats as stats

confidence = 0.87
alpha = (1 - confidence)/2
observation = 9
mu = np.linspace(0.1,150, 1_000_000)
lower_limit = 0
upper_limit = 0

for mu_ in mu:
    if (1-stats.expon.cdf(observation, scale=mu_) >alpha):
        lower_limit = mu_
        break

for mu_ in mu:
    if (stats.expon.cdf(observation, scale=mu_) <alpha):
        upper_limit = mu_
        break

print("Unteres Limit =", lower_limit)
print("Oberes Limit = ", upper_limit)
print(f"Gerundet: {lower_limit:.2f}-{upper_limit:.2f}")

#stats.poisson.sf(observation-1, mu_)

# Unteres Limit = 3.292723292723293
# Oberes Limit =  133.9112169112169