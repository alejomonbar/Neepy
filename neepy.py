#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:54:16 2020
Neepy (Non-Equilibrium Evolution pyhton Based library)
This library 
@author: Alejandro Montañez
"""

import numpy as np
import scipy.linalg as matrix
import scipy.integrate as integrate
from functions_neepy import dag, commutator, pro
from functions_neepy import partial_trace, J_hats, map_to_composite, O_J, FG_J
from functions_neepy import anticom , kron_i, trA, trB
from scipy.constants import hbar

class QuantumSystem:
    
    def __init__(self, po, dim=[]):
        """
        Generate a Quantum system based on a density state initial condition

        Parameters
        ----------
        po : np.array
            density state initial condition.
        dim : list, optional
            If the dimensions are differents from those of a qubit, they must be
            specified. The default is [].

        Returns
        -------
        None.

        """
        self.po = po # initila condition
        
        # Pauli Matrices
        self.si = np.eye(2, dtype = complex)
        self.sx = np.array([[0, 1], [1, 0]], dtype = complex)
        self.sy = np.array([[0, -1j],[1j, 0]], dtype = complex)
        self.sz = np.array([[1, 0],[0, -1]], dtype = complex)
        self.hbar = hbar
        if len(dim) > 0:
            self.num_subsystems = len(dim)
            self.dim = dim
            if len(po) != np.prod(dim):
                raise Exception(f"The dimensions of the density state po: {len(po)} and dim: {np.prod(dim)} differ.")
        else:
            # By default the system is considered as a composition of qubits.
            self.num_subsystems = int(np.log2(len(po))) 
            self.dim = self.num_subsystems * [2] # Individual qubits with dimension 2
    def evolve(self, time, H, equation, dissipative_term = [], solver="dopri5", prin = False):
        """
        Return the integration of the equation of motion for a system in a non-equilibrium 
        state
        
        Arguments:
            po -- initial condition of the density state
            time -- vector with the sequence of steps that the system has to follow
            in order to get to the final state
            H -- Hamiltionian
            dissipative_term -- For the case of SEAQT the relaxation time and for Lindblad the gamma factor
            equation -- equation to be solved, e.g., von Neumann, SEAQT, Lindblad
            solver -- Method used to integrate the evolution.
        """
        nn = len(self.po)
        n = len(time) # number of the steps for the evolution 
        p = np.zeros((n, nn, nn),dtype = np.complex64)
        dpdt = np.zeros((n, nn,nn),dtype = np.complex64)
        p[0,:,:] = self.po
        
        if equation == vonNeumann:
            dpdt[0,:,:] = equation(time[0], p[0,:,:].reshape(nn**2,), H(time[0])).reshape(nn,nn)
            eq = lambda t, p: equation(t, p, H(t)) 
        else:
            dpdt[0,:,:] = equation(time[0],p[0,:,:].reshape(nn**2,),H(time[0]),dissipative_term).reshape(nn,nn)
            eq = lambda t, p: equation(t, p, H(t), dissipative_term)
        p_C = integrate.complex_ode(eq)
        po = np.reshape(self.po, (nn**2,))    
        p_C.set_initial_value(po,time[0])
        p_C.set_integrator(solver, method = 'bdf', rtol = 1e-4)
        for i, t in enumerate(time[1:]):
            dt = t - time[i]
            p[i+1,:,:] = p_C.integrate(p_C.t+dt).reshape((nn,nn))
            if equation != vonNeumann:
                dpdt[i+1,:,:] = equation(t, p[i+1,:,:].reshape(nn**2,), H(t), dissipative_term).reshape(nn,nn)
            else:
                dpdt[i+1,:,:] = equation(t, p[i+1,:,:].reshape(nn**2,), H(t)).reshape(nn,nn)
            if (i % np.ceil(n/10) == 0) and prin:
                print('--------------------',str(i))
        self.p = p
        self.dpdt = dpdt
        return p, dpdt
    
def ToBlochVector(pt):
    """
    Convert a density state operator in its representation on the Bloch sphere
    for a given time. This is only valid for qubit systems.
    Arguments:
        p -- density state operator in time steps with dimensions
        [t, n, n] where t is the time and n = 2 ** q where q is the number of qubits.
    Returns
    -------
    xyz_t : np.array
        Bloch vector for the different qubits as a function of time.
        the dimension is [qubits, time, xyz]

    """
    nt = pt.shape[0] # time steps
    qb = int(np.log2(len(pt[0])))
    xyz_t = np.zeros((qb, nt, 3)) # 3 because x, y, and z components
    for t in range(nt):
        for q in range(qb):
            pi = pt[t, :, :]
            p = partial_trace(pi, [q])
            xyz_t[q, t, :] = [2 * p[0,1].real, 2 * p[1,0].imag, p[0,0].real - p[1,1].real]
    return xyz_t

    
def StateFromBloch(u, v, w):
    """Based on the components u, v, and w of the Bloch vector retruns the density state.
        p array (2,2):density state
    """
    p = 0.5 * np.array([[1 + w , u - 1j * v],[u + 1j * v , 1 - w]], dtype=complex)
    return p

def SEAQT(t, p, H, tauD):
    """
    Return the evolution of a density state following a non-equilibrium trajectory
    through the steepest entropy ascente or entropy gradient. This is a proposal
    of Gian Paolo Beretta (1984) based on the postulates of Gyftopoulos and Hatsopoulos.
    
    Arguments:
        t -- Scalar, time
        p -- square matrix of 4 x 4, density state
        H -- Hamiltonian
        tauD -- disipative constant
    
    Return:
        dp/dt -- the change of the density state respect to the time
    """
    nn = int(np.sqrt(len(p)))
    p = np.reshape(p,(nn,nn))
    logp = matrix.logm(p,disp =False)[0]
    plogp = p @ logp
    Trplogp = plogp.trace()       
    TrpHlogp = (p @ H @ logp).trace()
    TrpH = (p @ H).trace()
    TrpH_2 = (p @ H @ H).trace()
    Gamma = TrpH_2 - TrpH ** 2.0
    beta = (TrpH * Trplogp - TrpHlogp) / Gamma
    alpha = (TrpHlogp * TrpH - Trplogp * TrpH_2)/Gamma
    ac = p @ H + H @ p
    D = plogp + alpha * p + 0.5 * beta * ac 
    D_T = 0.5 * (D + dag(D)) 
    term1 = (-1.0j / hbar) * commutator(H, p)
    term2 =  -(1.0 / tauD[0]) * D_T
    dpdt = term1 + term2
    dpdt = np.reshape(dpdt, (1, nn**2))
    return dpdt
    
def SEAQT_reservoir(self, t, p, H, tauD):
    """
    Return the evolution of a density state following a non-equilibrium trajectory
    through the steepest entropy ascente or entropy gradient. This is a proposal
    of Gian Paolo Beretta (1984) based on the postulates of Gyftopoulos and Hatsopoulos.
    The modification of the equation is based on the thesis of Tyler 2019.
    Arguments:
        t -- Scalar, time
        p -- square matrix of 4 x 4, density state
        H -- Hamiltonian
        tauD -- disipative constant
        beta_R -- beta = 1/kB*Tr where kB is the Boltzmann constant ant Tr is the reservoir temperature
    
    Return:
        dp/dt -- the change of the density state respect to the time
    """
    nn = int(np.sqrt(len(p)))
    I = np.eye(nn)
    p = np.reshape(p,(nn,nn))
    try:
        logp = matrix.logm(p,disp =False)[0]
    except:
        logp = np.zeros((nn,nn))
            
    plogp = p @ logp
    term1 = (-1.0j / hbar) * commutator(H, p)
# =============================================================================
#       Reservoir modification
# =============================================================================
    HT = H + self.H0
    e = (p @ HT).trace()
    D = self.beta_R * (p @ (HT - e*I))
# =============================================================================

    Trplogp = np.matrix.trace(plogp)         
    TrpHlogp = pro(p.dot(HT),logp)
    TrpH = pro(p, HT)
    TrpH_2 = pro(p, HT @ HT)
    Gamma = TrpH_2 - TrpH ** 2.0
    beta = (TrpH*Trplogp - TrpHlogp) / Gamma
    alpha = (TrpHlogp * TrpH - Trplogp * TrpH_2) / Gamma
    ac = p @ HT + HT @ p
    D1 = plogp + alpha * p + 0.5 * beta * ac 
    
    D_T = 0.5 * (D + dag(D)) 
    tauDR = tauD[0]*(1 + (p @ self.sz).trace())  
    term2 =  (1.0 / tauDR) * D_T + (1 / tauD[1]) * D1

    dpdt = term1 - term2
    dpdt = np.reshape(dpdt,(1,nn**2))
    return dpdt

    
def SEAQT_2Res(self, t, p, H, tauD):
    """
    Return the evolution of a density state following a non-equilibrium trajectory
    through the steepest entropy ascente or entropy gradient. This is for a 2 qubit 
    system subject to an interaction with the environment.
    
    Arguments:
        t -- Scalar, time
        p -- square matrix of 4 x 4, density state
        H -- Hamiltonian
        tauD -- Relaxation time
    Return:
        dp/dt -- the change of the density state respect to the time
    """
    p = np.reshape(p,(4,4))
    try:
        logp = matrix.logm(p, disp=False)[0]
    except:
        logp = np.zeros(p.shape, dtype = complex)
    pA = trB(p)
    IA = self.si
    pB = trA(p)
    IB = self.si
    lnpA = trB(np.kron(IA,pB).dot(logp))
    lnpB = trA(np.kron(pA,IB).dot(logp))
    HT = H + self.H0
        
    
    #       A 
    HA = trB(np.kron(IA,pB) @ HT)
    TrplnpA = 0.5*pA.dot(anticom(IA,lnpA)).trace()[0,0]
    TrpHA = 0.5*pA.dot(anticom(IA,HA)).trace()[0,0]
    TrpH_2A = 0.5*pA.dot(anticom(HA,HA)).trace()[0,0]
    acA = anticom(pA,HA)
    TrpHlnpA = 0.5*pA.dot(anticom(HA,lnpA)).trace()[0,0]
    rA = np.sqrt((pA @ self.sy).trace().real**2 +  (pA @ self.sx).trace().real**2)
    tauDA1 = float(tauD[0][0]*(1 - rA)) # Beretta's time constant
    tauDA2 = tauD[0][1]
    
    plnpA = pA @ lnpA
    GammaA = TrpH_2A - TrpHA**2.0
    betaA = (TrpHA*TrplnpA - TrpHlnpA)/GammaA
    alphaA = (TrpHlnpA*TrpHA - TrplnpA*TrpH_2A)/GammaA
    DA = plnpA + alphaA*pA + 0.5*betaA*acA
    dA = 0.5*(DA + dag(DA)) 
    eA = (pA @ HA).trace()[0,0]
    dAr = self.beta_R * (pA @ (HA - eA*IA))
    DAr =  0.5*(dAr + dag(dAr)) 

    
    #           B
    HB = trA(np.kron(pA,IB) @ HT)
    TrplnpB = 0.5*pB.dot(anticom(IB,lnpB)).trace()[0,0]
    TrpHB = 0.5*pB.dot(anticom(IB,HB)).trace()[0,0]
    TrpH_2B = 0.5*pB.dot(anticom(HB,HB)).trace()[0,0]
    acB = anticom(pB, HB)
    TrpHlnpB = 0.5*pB.dot(anticom(HB,lnpB)).trace()[0,0]
    
    rB = np.sqrt((pB @ self.sy).trace().real**2 +  (pB @ self.sx).trace().real**2)
    tauDB1 = float(tauD[1][0]*(1 - rB)) # Beretta's time constant
    tauDB2 = tauD[1][1]

    plnpB = pB @ lnpB
    GammaB = TrpH_2B - TrpHB**2.0
    betaB = (TrpHB*TrplnpB - TrpHlnpB)/GammaB
    alphaB = (TrpHlnpB*TrpHB - TrplnpB*TrpH_2B)/GammaB
    DB = plnpB + alphaB * pB + 0.5*betaB * acB
    dB = 0.5 * (DB + dag(DB)) 
    eB = (pB @ HB).trace()[0,0]
    dBr = self.beta_R * (pB @ (HB - eB*IB))
    DBr = 0.5*(dBr + dag(dBr)) 
    # Combining the two effects 
    term1 = -(1.0j/self.hbar)*commutator(H,p)
    term2 = - (1.0/tauDB1) * np.kron(pA, DBr) - (1.0/tauDA1) * np.kron(DAr, pB)
    term3 = - (1.0/tauDA2) * np.kron(dA,pB) - (1.0/tauDB2) * np.kron(pA,dB)
    
    dpdt = term1 + term2  + term3
    dpdt = np.reshape(dpdt,(1,16))
    return dpdt

def SEAQT_gen(t, p, H, tauD):
    """
    Return the GENERAL evolution of a density state following a non-equilibrium
    trajectory throughout the steepest entropy ascent or entropy gradient. This
    is a proposal of Gian Paolo Beretta (1985) in the paper Quantum Thermodynamics.
    A new equation of motion for a general quantum system.
    
    Arguments:
        t -- Scalar, time
        p -- square matrix of 4 x 4, density state
        H -- Hamiltonian
        tauD -- array with the number of subsistems for the disipative constant 
    
    Return:
        dp/dt -- the change of the density state respect to the time
    """
    nn = int(np.sqrt(len(p)))
    p = np.reshape(p, (nn, nn))
    nq = int(np.log2(nn))
    j_hats = J_hats(nq)
    I = np.eye(2)
    
    D = np.zeros((nn,nn),dtype = np.complex128)
    logp = matrix.logm(p, disp=False)[0]
    Dims = nq * [2]
    
    for j in range(nq):
        pj = partial_trace(p, [j])
        pj_bar = partial_trace(p, j_hats[j])
        p_perm = map_to_composite(pj_bar,Dims,j)# product np.kron(I_J, pj_bar) Eq 12 Beretta 1985 paper
        lnp_js = O_J(logp,p_perm, [j])
        pjlogpj = pj @ lnp_js
        H_js = O_J(H, p_perm, [j])
        Trplogp = FG_J(I,lnp_js, pj) 
        TrpH = FG_J(I,H_js, pj) 
        TrpH2 = FG_J(H_js, H_js, pj) 
        ac_s = anticom(pj, H_js)
        TrpHlogp = FG_J(H_js,lnp_js, pj) 
        Gamma = TrpH2 - TrpH ** 2.0
        alpha = (TrpHlogp * TrpH - Trplogp * TrpH2) / Gamma
        beta = (TrpH * Trplogp - TrpHlogp) / Gamma
        Dj = (pjlogpj + alpha * pj + 0.5 * beta * ac_s)
        Dj_T = 0.5 * (Dj + dag(Dj))
        D += -(1/tauD[j]) * kron_i(Dj_T, j, nq) @ p_perm
        
    term1 = (-1.0j / hbar) * commutator(H, p)
    term2 = D 
    dpdt = term1 + term2
    dpdt = np.reshape(dpdt, (1, nn ** 2))
    return dpdt

def entropy(p_v):
    """
    Return the von Neumann entropy for the density state p_v
    Arguments:
        p_v (array): array with n x nnx nn
    
    Return:
        s (array): array n x 1 with the values of the entropy per each density state 
                    in p_v
                    
    """
    n = len(p_v)
    s = np.zeros((n, 1))
    for i, p in enumerate(p_v):
        s[i] = - (p @ matrix.logm(p, disp=False)[0]).trace().real
    return s

def entropy_production(p_v, dpdt_v):
    """
    Return the von Neumann entropy for the density state p_v
    Arguments:
        p_v (array): array with n x nnx nn
    
    Return:
        s (array): array n x 1 with the values of the entropy per each density state 
                    in p_v
         """
    n = len(p_v)
    dS = np.zeros((n,1))
    for i, p in enumerate(p_v):
        dS[i] = np.real(-dpdt_v[i,:,:].dot(matrix.logm(p,disp =False)[0]) - (dpdt_v[i,:,:])).trace()
    return dS

def Lindblad(t, p, H, gamma):
    """
    Return the evolution of a density state following a non-equilibrium trajectory
    through the Lindblad equation.
    
    Arguments:
        t -- Scalar, time
        p -- square matrix of 4 x 4, density state
        H -- Hamiltonian
        gamma(list) -- Disipative term 0:amplitud damping 1: phase damping
    Return:
        dp/dt -- the change of the density state respect to the time
    """
    nn = int(np.sqrt(len(p)))
    p = np.reshape(p,(nn,nn))
    I = np.eye(2)
    qb = int(np.log2(len(p)))
    pos = np.eye(qb)
    D1 = np.zeros(p.shape, dtype=complex)
    D2 = np.zeros(p.shape, dtype=complex)
    for i in range(qb):
        if qb > 1:
            gamma1 = gamma[i][0]
            gamma2 = gamma[i][1]
        else:
            gamma1 = gamma[0]
            gamma2 = gamma[1]
        L_a = np.sqrt(gamma1) * np.array([[0,1],[0,0]])
        L_a_s = [I, L_a]
        L_p = np.sqrt(gamma2) * np.array([[1,0],[0,-1]])
        L_p_s = [I, L_p]
        L1 = L_a_s[int(pos[i,0])]
        L2 = L_p_s[int(pos[i,0])]
        for j in range(1,qb):
            L1 = np.kron(L1,L_a_s[int(pos[i,j])])
            L2 = np.kron(L2,L_p_s[int(pos[i,j])])
        D1 += 2 * L1 @ p @ dag(L1) - dag(L1) @ L1 @ p - p @ dag(L1) @ L1
        D2 += 2 * L2 @ p @ dag(L2) - dag(L2) @ L2 @ p - p @ dag(L2) @ L2
    dpdt = (-1.0j / hbar) * commutator(H, p) + D1 + D2
    dpdt = np.reshape(dpdt, (1, nn * nn))
    return dpdt

def vonNeumann(t, p, H):
    """
    Return the evolution of a density state following the von Neumann equation.
    This equation describes the ideal path of evolution.

    
    Arguments:
        t -- Scalar, time
        p -- square matrix of n x n, density state
        H -- Hamiltonian
    
    Return:
        dp/dt -- the change of the density state respect to the time
    """
    nn = int(np.sqrt(len(p)))
    p = np.reshape(p,(nn,nn))
    dpdt = (-1.0j / hbar)*commutator(H,p)
    dpdt = np.reshape(dpdt,(1, nn * nn))
    return dpdt
