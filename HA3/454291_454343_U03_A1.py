#! /usr/bin/env python3
# -*- coding: utf-8 -*-
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
import scipy.stats
import os

SAVE_FILE = 'roll_counts.csv'
NUM_FACES = 6
ALPHA = 0.05

# --- Load Counts ---
counts = np.zeros(NUM_FACES, dtype=int)
if os.path.exists(SAVE_FILE):
    try:
        loaded_counts = np.loadtxt(SAVE_FILE, delimiter=',', dtype=int)
        if loaded_counts.ndim == 1 and loaded_counts.size == NUM_FACES and not np.any(loaded_counts < 0):
            print(f'Existing counts found: {loaded_counts}')
            counts = loaded_counts
        else:
            print(f"WARNUNG: Falsches Format in {SAVE_FILE} (erwartet {NUM_FACES} Zahlen). Starte bei Null.")
    except ValueError as e:
        print(f"WARNUNG: Fehler beim Laden/Verarbeiten von {SAVE_FILE} ({type(e).__name__}), starte bei Null.")

# --- Get New Rolls ---
print("Würfe eingeben ('q' zum Beenden):")
new_rolls = []
while True:
    inp = input("> ").strip().lower()
    if inp == 'q': break
    try:
        roll = int(inp)
        if 1 <= roll <= NUM_FACES:
            new_rolls.append(roll)
        else: print(f"Nur 1-{NUM_FACES} oder 'q'")
    except ValueError: print("Ungültiger Input")

# --- Update Counts ---
if new_rolls:
    session_counts = np.bincount(np.array(new_rolls) - 1, minlength=NUM_FACES)
    counts += session_counts

# --- Analysis ---
total_rolls = int(np.sum(counts))
print(f"\n--- Analyse ({total_rolls} Würfe gesamt) ---")
print(f"Zählungen: {counts.tolist()}")

if total_rolls > 0:
    expected_counts = total_rolls / NUM_FACES
    if expected_counts < 5:
        print("WARNUNG: Chi2-Test nicht aussagekräftig, weil Erwartungswert pro Bin < 5.")

    chi2_stat = np.sum((counts - expected_counts) ** 2 / expected_counts)
    ndof = NUM_FACES - 1
    p_value = 1 - scipy.stats.chi2.cdf(chi2_stat, ndof)

    print(f"Chi2: {chi2_stat:.4f}, p-Wert: {p_value:.4f}")
    result = "GEZINKT" if p_value < ALPHA else "NICHT GEZINKT"
    print(f"Ergebnis (alpha={ALPHA}): {result}")
else:
    print("Keine Würfe für Analyse.")

# --- Save Counts ---
try:
    np.savetxt(SAVE_FILE, counts, delimiter=',', fmt='%d')
    print(f"Gesamte Zählungen in {SAVE_FILE} gespeichert.")
except Exception as e:
    print(f"FEHLER beim Speichern mit NumPy: {e}")

print('--- Task 1b) ---')

# --- Load sicherGleichverteilt.csv ---
rolls = np.loadtxt('sicherGleichverteilt.csv', delimiter=',', dtype=int)
counts = np.bincount(rolls - 1, minlength=6)

# --- Analysis ---
total_rolls = int(np.sum(counts))
print(f"\n--- Analyse ({total_rolls} Würfe gesamt) ---")
print(f"Zählungen: {counts.tolist()}")

expected_counts = total_rolls / NUM_FACES
if expected_counts < 5:
    print("WARNUNG: Chi2-Test nicht aussagekräftig, weil Erwartungswert pro Bin < 5.")

chi2_stat = np.sum((counts - expected_counts) ** 2 / expected_counts)
p_value = 1 - scipy.stats.chi2.cdf(chi2_stat, ndof)

print(f"Chi2: {chi2_stat:.4f}, p-Wert: {p_value:.4f}")
result = "GEZINKT" if p_value < ALPHA else "NICHT GEZINKT"
print(f"Ergebnis (alpha={ALPHA}): {result}")
print('Dieser Würfel ist entsprechend ziemlich gleichverteilt.')
print('Ich gehe davon aus, dass er allerding synthetisch generiert wurde, weil die Counts so nah beieinander sind.')