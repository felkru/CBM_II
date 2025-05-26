import numpy as np

# Lichtgeschwindigkeit im Vakuum
c0 = 299792458.0

# Helfer-Funktion für boostfactor
def cispi(x):
    return np.cos(x*np.pi)+1.0j*np.sin(x*np.pi)

# Berechnet den Boost-Faktor beta^2 für Frequenzen und Plattenabstände. Sie müssen diese
# Funktion NICHT verstehen, sie müssen sie nur anwenden wie in der Vorlesung gezeigt.
# frequencies und distances sollten numpy-Arrays sein.
# PS: Wir vergeben übrigens Bachelor-Arbeiten bei MADMAX
def boostfactor(frequencies,distances):
    eps=24.0
    thickness = 1e-3
    nd = np.sqrt(eps); nm = 1e15
    epsm = nm**2
    A = 1-1/eps; A0 = 1-1/epsm

    boost = np.empty(len(frequencies),dtype=complex)

    Gd = np.array([[(1+nd)/2, (1-nd)/2],
                   [(1-nd)/2, (1+nd)/2]],dtype=complex)
    Gv = np.array([[(nd+1)/(2*nd), (nd-1)/(2*nd)],
                   [(nd-1)/(2*nd), (nd+1)/(2*nd)]],dtype=complex)
    G0 = np.array([[(1+nm)/2, (1-nm)/2],
                   [(1-nm)/2, (1+nm)/2]],dtype=complex)

    S  = np.array([[A/2, 0.0],
                   [0.0,  A/2]],dtype=complex)
    S0 = np.array([[A0/2, 0.0],
                   [0.0, A0/2]],dtype=complex)

    M  = np.copy(S)
    T  = np.copy(Gd)

    for j in range(len(frequencies)):
        pd1 = cispi(-2*frequencies[j]*nd*thickness/c0)
        pd2 = cispi(+2*frequencies[j]*nd*thickness/c0)

        for i in reversed(range(len(distances))):
            T[:,0] *= pd1
            T[:,1] *= pd2

            M -= T@S
            T = T@Gv

            T[:,0] *= cispi(-2*frequencies[j]*distances[i]/c0)
            T[:,1] *= cispi(+2*frequencies[j]*distances[i]/c0)

            if i > 0:
                M += T@S
                T = T@Gd
            else:
                M += T@S0
                T = T@G0

        boost[j] = M[0,0]+M[0,1]-(M[1,0]+M[1,1])*T[0,1]/T[1,1]

        np.copyto(M,S)
        np.copyto(T,Gd)

    return np.abs(boost)**2

# Erzeugt einen zufälligen raumgleichverteilten, normierten Vektor mit n Einträgen
def space_uniform_rand(n):
    r = 2*np.random.rand(n)-1
    return r/np.linalg.norm(r)