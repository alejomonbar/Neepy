#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 11:05:55 2022

@author: alejomonbar
"""
# =============================================================================
# Neepy functions 
# =============================================================================
from neepy import QuantumSystem, SEAQT, vonNeumann
from neepy import StateFromBloch, ToBlochVector

# Other libraries t
import numpy as np # commonly used library to work with arrays
import matplotlib.pyplot as plt #commonly used library to plot 

po = StateFromBloch(*[0.95, 0, 0]) #Coordinates x, y, and z Bloch Sphere

p = QuantumSystem(po) # Python class to describe quantum systems
omega = 5 # Qubit frequency
H = lambda t: p.hbar * omega * p.sz # Hamiltonian 
time = np.linspace(0, 1, 100) # time evolution from 0 to 1 with 100 intermediate steps
# von Neumann evolution
p_von, dpdt_von = p.evolve(time, H, vonNeumann)
# SEAQT evolution
tauD = [0.5]
p_sea, dpdt_sea = p.evolve(time, H, SEAQT, tauD)

# =============================================================================
# Visualization
# =============================================================================

labels = ["x", "y", "z"]
titles = ["von Neumann", "SEAQT"]
fig, ax = plt.subplots(1,2, figsize=(12,5))
for _, p_eq in enumerate([p_von, p_sea]):
    for i in range(3):
        ax[_].plot(time, ToBlochVector(p_eq)[0,:,i])
    ax[_].set_title(titles[_])
    ax[_].set_xlabel("time")
    ax[_].set_ylabel("amplitude")
    ax[_].legend(labels)
fig.savefig("./Images/Single_qubit_evol.pdf")