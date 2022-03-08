#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:03:32 2020
Posprocessing functions
@author: jmon
"""
import numpy as np
from neepy import Neepy
import scipy.linalg as matrix
from functions_neepy import partial_trace, partial_trace_mul
from scipy.constants import k

sx = np.array([[0,1],[1,0]], dtype = complex)
sy = np.array([[0,-1j],[1j,0]], dtype = complex)
sz = np.array([[1,0],[0,-1]], dtype = complex)
sn = [sx, sy, sz]

def xyz(p_v):
    """
    Return the cartesian coordinates x,y,z of a vector decribed by a density state 
    
    Arguments:
        p -- square matrix of n x 2 x 2 
    
    Return:
        xyz_v -- 3 x n

    """
    n = len(p_v)
    xyz_v = np.zeros((3,n))
    for i,p in enumerate(p_v):
        xyz_v[:,i] = [2*p[0,1].real,2*p[1,0].imag,p[0,0].real - p[1,1].real]     
    return xyz_v

def xyz_mul(p_v):
    n = p_v.shape[0]
    qb = int(np.log2(p_v.shape[1]))
    xyz_v = np.zeros((3,n,qb))
    for i in range(n):
        for ii in range(qb):
            pt = p_v[i,:,:]
            p = partial_trace(pt,[ii])
            xyz_v[:,i,ii] = [2*p[0,1].real,2*p[1,0].imag,p[0,0].real - p[1,1].real]
    return xyz_v

def energy(pv,H):
    n = pv.shape[0]
    e = []
    for i in range(n):
        pt = pv[i,:,:]
        e.append((pt.dot(H[i,:,:])).trace())
    return np.array(e,dtype = complex)

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
    s = np.zeros((n,1))
    for i, p in enumerate(p_v):
        s[i] = - np.real(p.dot(matrix.logm(p,disp =False)[0])).trace()
    return s

def entropy_production(p_v,dpdt_v):
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

def dQ(dpdt_v, H):
    """
    Return the rate of heat transfer
    Arguments:
        dpdt_v (array): array with n x nnx nn density state derivative
    
    Return:
        s (array): array n x 1 with the values of the entropy per each density state 
                    in p_v
         """
    n = len(dpdt_v)
    Q = np.zeros((n,1))
    for i, dp in enumerate(dpdt_v):
        Q[i] = (H[i,:,:] @ dp).trace()
    return Q
def observable(p_v,O):
    """
    

    Parameters
    ----------
    p_v : numpy array
        density state evolution.
    O : numpy array matrix
        Operator from which we want to extract the observable.

    Returns
    -------
    np.array 
        observable evolution through time.

    """
    val = []
    for p in p_v:
        val.append((p @ O).trace())
    return np.array(val)

def concurrence(p_v):
    """
    Return the concurrence based on the paper of Shulman 2012
    "Demonstration of entanglement of electrostatically coupled singlet-triplet
    qubits"
    Arguments:
        p_v (array n x nn x nn): the evolution in time the density operator based
        in the evolution equation used
    Return:
        con(array n x 1): array with the values of concurrence for the timeline
        of the density state.
    """
    sy = np.array([[0,-1j],[1j,0]])
    n = len(p_v)
    con = np.zeros((n,1))
    for i,p in enumerate(p_v):
        pb = np.dot(np.dot(np.kron(sy,sy),np.conjugate(p)),np.kron(sy,sy))
        psqrt = matrix.sqrtm(p)
        R = matrix.sqrtm(np.dot(np.dot(psqrt,pb),psqrt))
        eig = sorted(np.linalg.eigh(R)[0])
        con[i] = eig[3] - eig[2] - eig[1] - eig[0]
    return con
def concurrence2(p_v):
    """
    Return the concurrence based on the paper of Shulman 2012
    "Demonstration of entanglement of electrostatically coupled singlet-triplet
    qubits"
    Arguments:
        p_v (array n x nn x nn): the evolution in time the density operator based
        in the evolution equation used
    Return:
        con(array n x 1): array with the values of concurrence for the timeline
        of the density state.
    """
    sy = np.array([[0,-1j],[1j,0]])
    n = len(p_v)
    con = np.zeros((n,1))
    for i,p in enumerate(p_v):
        eig = sorted(np.linalg.eigh(p)[0])
        con[i] = eig[3] - eig[2] - eig[1] - eig[0]
    return con

def fidelity(p_ideal,p_real):
    """
    

    Parameters
    ----------
    p_ideal : square matrix or array of square matrices
        The ideal density state
    p_real : square matrix or array of square matrices
        The experimental or simulated density state.

    Returns
    -------
    F : value or array
        Fidelity of the output signal.

    """
    
    if len(p_real.shape) == 3:
        F = []
        for i,p in enumerate(p_real):
            if len(p_ideal.shape) == 3:
                sqrt_p_ideal = matrix.sqrtm(p_ideal[i,:,:])
            else:
                sqrt_p_ideal = matrix.sqrtm(p_ideal)
            F.append(np.trace(matrix.sqrtm(sqrt_p_ideal.dot(p).dot(sqrt_p_ideal)))**2)
        F = np.array(F)
    else:
        sqrt_p_ideal = matrix.sqrtm(p_ideal)
        F = np.trace(matrix.sqrtm(sqrt_p_ideal.dot(p_real).dot(sqrt_p_ideal)))**2
    return F

def distanceBS(gamma1, gamma2):
    return np.arccos(0.5*(gamma1.T.conjugate() @ gamma2 + gamma2.T.conjugate() @ gamma1))

def mutualInf(p):
    pa = partial_trace_mul(p, [2,2], axis = 0)
    pb = partial_trace_mul(p, [2,2], axis = 1)
    return (pa @ matrix.logm(pa)).trace() + (pb @ matrix.logm(pb)).trace() + (p @ matrix.logm(p)).trace()

def CHSH(p):
    """Clauser-Horne-Shimony-Holt"""
    
    T = np.zeros((3,3), dtype = complex) 
    for i in range(3):
        for j in range(3):
            T[i,j] = (p @ np.kron(sn[i], sn[j])).trace()
    eig = sorted(matrix.eig(T)[0])
    t11 = eig[-1]
    t22 = eig[-2]
    return 2 * np.sqrt(t11**2 + t22**2)

def eigenvalues(p_v):
    n,l1,l2 = np.shape(p_v)
    eigen = np.zeros((n,l1))
    for i,p in enumerate(p_v):
        eigen[i,:] = matrix.eigh(p)[0]
    return eigen

def eigen_evol(p_v):
    n,l1,l2 = np.shape(p_v)
    evol = np.zeros((n,l1),dtype = complex)
    for i,p in enumerate(p_v):
        for nn in range(l1):
            evol[i,nn] = p[nn,nn]
    return evol

def trace_mul(p_v, partial):
    """

    Parameters
    ----------
    p_v : array 
        matrix with dimensions of the number of subsystems in the case of a 
        qubit coupled to a harmonic oscillator with 5 energy levelsit has shape
        2 X 5 = (10,10).
    partial : List 
    
        Information of the dimensions of the subsystems and the axis over which
        the partial trace is taken.
    Returns
    -------
    p_sub : array
        Matrix with dimension of the subsystem times n(the number of evolution
                                                       steps).

    """
    dim = partial[0]
    axis = partial[1]
    n, l1, l2 = np.shape(p_v)
    p_sub = np.zeros((n,l1//dim[axis],l1//dim[axis]),dtype = complex)
    for i in range(n):
        p_sub[i,:,:] = partial_trace_mul(p_v[i,:,:],dim,axis)
    return p_sub

def tauDf(p_v, x):
    """
    Supposition that the tauD = x[0] Tr(p(t) @ sz) + x[1]

    Parameters
    ----------
    p_v : array
        density state.
    x : array
        Based on the two-qubit paper linear relation of the dissipative constant.

    Returns
    -------
    tauD : array
        Dissipative time of the SEAQT equation of motion with the supossition that
        it depends on the energy variation

    """
    s3 = np.array([[1,0],[0,-1]])
    dims = int(np.log2(len(p_v[0])))
    tauD = {_:[] for _ in range(dims)}
    for p in p_v:
        for q in range(dims):
            tauD[q].append(x[q] * (np.trace(partial_trace(p, [q]) @ s3) + 1))
    return tauD

def inform(p_v,dpdt_v,properties,p_ideal=None, partial=None, H=None, x=None):
    data = {}
    for i in properties:
        if i == 's':
            data['s'] = entropy(p_v)
        elif i == 'ds':
            data['ds'] = entropy_production(p_v,dpdt_v)
        elif i == 'xyz':
            data['xyz'] = xyz(p_v)
        elif i == 'xyz_mul':
            data['xyz_mul'] = xyz_mul(p_v)
        elif i == 'con':
            data['con'] = concurrence(p_v)
        elif i == 'con2':
            data['con2'] = concurrence2(p_v)
        elif i == 'eigen':
            data['eigen'] = eigenvalues(p_v)
        elif i == 'F':
            data['F'] = fidelity(p_ideal,p_v)
        elif i == 'eigen_evol':
            data["eigen_evol"] = eigen_evol(p_v)
        elif i == "trace_mul":
            data["trace_mul"] = trace_mul(p_v, partial)
        elif i == 'energy':
            data['energy'] = energy(p_v, H)
        elif i == "dQ":
            data["dQ"] = dQ(dpdt_v, H)
        elif i == "temperature":
            data["temperature"] = dQ(dpdt_v, H) / (k*entropy_production(p_v, dpdt_v))
        elif i == "tauD":
            data["tauD"] = tauDf(p_v, x)
        else:
            raise Warning("This propertie is not included!")

    return data




            
                