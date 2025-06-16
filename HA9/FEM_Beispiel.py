#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

#Problem: Drei flexible Metallträger, als Stern in der Mitte verbunden (siehe Vorlesung)

#Definiere Konstanten und Randbedingungen

# Externe Kraft an Knoten 3
F3 = np.array([50000.0, -100000.0]) # Newton

# Realistische Werte für die Metallträger
L1 = 50.0 # m
L2 = 50.0
L3 = 20.0
A = 50E-4  # m^2
E = 200.0E9

# Definiere lokale Steifheitsmatrizen
k = np.array([ [1,0, -1,0],
               [0,0,0,0],
               [-1, 0, 1, 0],
               [0,0,0,0]] ,dtype=float)

kl1 = 1.0/L1* np.array(k)
kl2 = 1.0/L2*  np.array(k)
kl3 = 1.0/L3*  np.array(k)

# Rotationsmatrix
def rot(phi):
    return np.array([  [np.cos(phi), np.sin(phi), 0 ,0  ],
                       [-np.sin(phi), np.cos(phi), 0, 0],
                       [0,0, np.cos(phi), np.sin(phi)],
                       [0,0,-np.sin(phi), np.cos(phi) ] ])

R13 = rot(np.deg2rad(53))
R23 = rot(np.deg2rad(127))
R34 = rot(np.deg2rad(90))

#Rotiere die Steifheitsmatrizen
kg1 = R13.transpose() @ kl1 @ R13
kg2 = R23.transpose() @ kl2 @ R23
kg3 = R34.transpose() @ kl3 @ R34

# Schwierigster Teil: zusammenbauen der 8x8 Steifheitsmatrix aus den einzelnen
# drei lokalen Steifheitsmatrixzen
# Am besten auf Papier aufzeichnen, um das nachzuvollziehen

#starte mit leerer Matrix
kgg = np.zeros((8,8))

# Erste zwei Zeilen, nur Beiträge aus kg1
kgg[0:2,0:2 ] = kg1[0:2,0:2 ]
kgg[0:2, 4:6] = kg1[0:2,2:4 ]

#Zeilen3 und 4 erhalten nur Beiträge aus erster und zweiter Zeile der kg2
kgg[2:4, 2:6 ] = kg2[0:2, :]

#Zeilen 5 und 6 erhalten Beiträge aus allen Matrizen...
kgg[4:6, 0:2] = kg1[2:4, 0:2]
kgg[4:6, 2:4] = kg2[2:4, 0:2]
kgg[4:6, 4:6] = kg1[2:4, 2:4] + kg2[2:4, 2:4] + kg3[0:2, 0:2]
kgg[4:6, 6:8] = kg3[0:2, 2:4]

#Zeile 7 und 8 erhält nur Beiträge aus kg3
kgg[6:8, 4:8] = kg3[2:4,0:4 ]

# Im Folgenden wird nur die für das Problem spezifische Submatrix invertiert.
kggSub = kgg[4:6, 4:6]
kggSubInv = np.linalg.inv(kggSub)

# Berechnung der globalen Verschiebungen des Knotens 3
globVec3 =1.0/(E*A)* np.matmul(kggSubInv, F3)

print('Der Knoten 3 wird nach x verschoben um ', globVec3[0], ' Meter, ')
print('und er wird nach y verschoben um ', globVec3[1], 'Meter.')

# Restliche globale Kräfte an den anderen Knoten berechnen

# Zuerst den globalen Vektor der Verschiebungen definieren
# Die sind alle Null, ausser x3 und y3
globVec = np.zeros(8)
globVec[4:6] = globVec3

# Den Ortsvektor mit der globalen Steifheitsmatrix multiplizieren
Fglob =E*A* np.matmul(kgg, globVec)

print('Externe Kraft am Aufhängungspunkt 1 F1 = ', np.linalg.norm(Fglob[0:2]) )
print('Externe Kraft am Aufhängungspunkt 2 F2 = ', np.linalg.norm(Fglob[2:4]) )
print('Externe Kraft am Aufhängungspunkt 4 F4 = ', np.linalg.norm(Fglob[6:8]) )

# Als Beispiel Bestimmung der Kräfte in Element 1

#Verschiebungsvektor von Element 1 (knoten 1 und 3)
x1g  = np.array( [0.0,0.0,globVec3[0] , globVec3[1]])

#Multiplikation mit lokaler Steifheitsmatrix (in globalen Koordinaten)
f1 = E*A*np.matmul(kg1, x1g)

print('Die Kraftvektoren in Element 1 (Punkte 1 und 3', f1)
print('Betrag der Kraft in Element 1: ', np.linalg.norm(f1[0:2] ))