import numpy as np
import scipy.stats as stats

confidence = 0.84
alpha = (1 - confidence)#/2
observation = 8
mu = np.linspace(0.1,100, 1000000)
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