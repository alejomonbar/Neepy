#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:01:19 2022

@author: alejomonbar
"""
from neepy import QuantumSystem, SEAQT_gen, vonNeumann, Lindblad
from neepy import StateFromBloch, ToBlochVector, entropy, entropy_production

# Other libraries t
import numpy as np # commonly used library to work with arrays
import matplotlib.pyplot as plt #commonly used library to plot 

po1 = StateFromBloch(*[0.95, 0, 0]) #Coordinates x, y, and z Bloch Sphere
po2 = StateFromBloch(*[0.95, 0, 0]) #Coordinates x, y, and z Bloch Sphere

po = np.kron(po1, po2) # Kronecker product between both qubits
p = QuantumSystem(po) # Python class to describe quantum systems
omega1 = 5 # Qubit 1 frequency
omega2 = 1 # Qubit 2 frequency
J = 2 # Coupling term for an interaction between both systems
H = lambda t: - 0.5 * p.hbar * (omega1 * np.kron(p.sz, p.si) +  omega2 * np.kron(p.si, p.sz) - J * np.kron(p.sx, p.sx)) # Hamiltonian 
time = np.linspace(0, 1, 100) # time evolution from 0 to 1 with 100 intermediate steps
# von Neumann evolution
p_von, dpdt_von = p.evolve(time, H, vonNeumann)
# SEAQT evolution
tauD = [0.5, 0.5]
p_sea, dpdt_sea = p.evolve(time, H, SEAQT_gen, tauD) # SEAQT for a general system
# Lindblad evolution
gammas_Q1 = [0, 1] # the first term "0" is associated with relaxation
                    #the second "1" with dephasing
gammas_Q2 = [0, 1] # gammas for the qubit 2
gammas = [gammas_Q1, gammas_Q2] 
p_Lind, dpdt_Lind = p.evolve(time, H, Lindblad, gammas) # Lidblad equation for a general system

# =============================================================================
# Visualization
# =============================================================================

labels = ["x", "y", "z"]
titles = ["von Neumann", "SEAQT", "Lindblad"]
fig, ax = plt.subplots(3, 2, figsize=(15, 15))
for qubit in range(2):
    for _, p_eq in enumerate([p_von, p_sea, p_Lind]):
        for i in range(3):
            ax[_][qubit].plot(time, ToBlochVector(p_eq)[qubit,:,i])
        ax[_][qubit].set_title(titles[_] + f" Q{qubit}")
        ax[_][qubit].set_xlabel("time")
        ax[_][qubit].set_ylabel("amplitude")
        ax[_][qubit].legend(labels)
fig.savefig("./Images/Two_qubit_evol.pdf")

# =============================================================================
# Entropy
# =============================================================================

fig, ax = plt.subplots()
for _, p_eq in enumerate([p_von, p_sea, p_Lind]):
    ax.plot(time, entropy(p_eq), label=titles[_])
ax.set_xlabel("time")
ax.set_ylabel(r"$S/k_B$")
ax.legend()
fig.savefig("./Images/Entropy.pdf")

# =============================================================================
# Entropy generation
# =============================================================================

fig, ax = plt.subplots()
for _, p_eq in enumerate([(p_von, dpdt_von), (p_sea, dpdt_sea), (p_Lind, dpdt_Lind)]):
    ax.plot(time, entropy_production(*p_eq), label=titles[_])
ax.set_xlabel("time")
ax.set_ylabel(r"$S/k_B$")
ax.legend()
fig.savefig("./Images/EntropyProduction.pdf")