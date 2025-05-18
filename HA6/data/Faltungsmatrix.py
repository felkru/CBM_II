import numpy as np

R = np.zeros((20, 20))

for i in range(2, 18):
    R[i, i]     = 0.3
    R[i-1, i]   = 0.25
    R[i+1, i]   = 0.25
    R[i-2, i]   = 0.1
    R[i+2, i]   = 0.1

# Edges
R[0, 0] = 0.5
R[1, 0] = 0.3
R[0, 1] = 0.3
R[1, 1] = 0.3
R[2, 1] = 0.3
R[2, 0] = 0.2

R[19, 19] = 0.5
R[18, 19] = 0.3
R[19, 18] = 0.3
R[18, 18] = 0.3
R[17, 18] = 0.3
R[17, 19] = 0.2

np.savetxt('convolution_matrix.csv', R,  delimiter=',', fmt='%f')