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

import numpy as np
import scipy as scp

def input_mode():
    # ask for user input as long they
    new_rolls = []
    while True:
        try:
            dice_res = int(input("Welche Augenzahl zeigt ihr Würfel?\n").strip())
        except Exception:
            print('Eingabe konnte nicht geparst werden. Geben sie eine Zahl ein.')
            continue
        new_rolls.append(dice_res)
        add_more = input('Do you want to add additional dice rolls? (y/n): ').lower().strip() == 'y'
        if (not add_more):
            np.savetxt("My_Dice_Rolls.csv", [*rolls, *new_rolls])
            return

# @Tutor - Program execution starts here
print('--- 1 a) ---')
try:
    rolls = np.loadtxt("My_Dice_Rolls.csv", dtype=int, converters=float)
    if (np.any(np.unique(rolls, return_counts=True)[1], rolls < 5)):
        print('Die Stichprobe ist noch nicht groß genug. Der Test ist entsprechend nicht aussagekräftig. Füge mehr Daten hinzu.')
    add_input = input('Do you want to add additional dice rolls? (y/n): ').lower().strip() == 'y'
    if (add_input):
        input_mode()
except FileNotFoundError:
    print('Würfelergebnisdatei nicht gefunden. Starte den interaktiven Modus.')
    rolls = []
    input_mode()
except Exception:
    print('Es ist beim laden der Datei ein Fehler aufgetreten, der sich nicht automatisch korrigieren lässt. Bitte überprüfen sie den Inhalt und Namen ihrer Würfelergebnisdatei.')
    print('Continuing with example dice_rolls.')
#    rolls = [1,2,5,6,1,3,5,4]

outcomes, counts = np.unique(rolls, return_counts=True)
print(f'Augenzahlen: {outcomes}')
print(f'Anzahl:      {counts}')


alpha = 0.05 # significance level
ndof = 6 - 1
std = np.std(rolls, ddof=1)

chi2 = np.sum((rolls - np.mean(rolls))**2 / std**2)
p_value = 1 - scp.stats.chi2.cdf(chi2, ndof)

if (alpha < p_value):
    print(f'Der Würfel ist laut Chi2-Test (alpha: {alpha}, tatsächliches alpha: {p_value}) nicht gezinkt.')
else:
    print(f'Der Würfel ist gezinkt. (alpha: {alpha}, tatsächliches alpha: {p_value})')