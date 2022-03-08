#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:33:00 2020
This is a recopilation of those functions used in the evolution of the system 
but that are nor required in the main text
@author: jmon
"""
import numpy as np
import scipy.integrate as sol
import scipy.linalg as matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

# =============================================================================
# Pauli matrices
# =============================================================================
sx = np.array([[0,1],[1,0]], dtype=complex)
sy = np.array([[0,-1j],[1j,0]], dtype=complex)
sz = np.array([[1,0],[0,-1]],dtype=complex)
h = (1/np.sqrt(2))*np.array([[1,1],[1,-1]], dtype=complex)
si = np.eye(2)
IZ = np.kron(si,sz)
IX = np.kron(si,sx)
ZI = np.kron(sz,si)
ZZ = np.kron(sz,sz)
ZY = np.kron(sz,sy)
XI = np.kron(sx,si)

def pro(A,B):
        """
        Compute the trace of the matrix product between A and B.
        
        Arguments:
            A,B -- square matrices 
        
        Return:
            com -- Scalar
        """
        prod = (A.dot(B)).trace()
        return prod
def dag(v):
    """
    Compute the transpose-conjugate of a square matrix v.
    
    Arguments:
        v -- square matrix
    
    Return:
        daga -- square matrix of the Conjugate-transpose of v
    """
    daga = np.conjugate(v.T)
    return daga
def commutator(A,B):
    """
    Compute the commutator between matrices A and B.
    
    Arguments:
        A,B -- square matrices 
    
    Return:
        com -- square matrix of the commutator [A,B]
    """
    com = A.dot(B) - B.dot(A)
    return com

def anticom(A,B):
    """
    Compute the anticommutator between matrices A and B.
    
    Arguments:
        A,B -- square matrices 
    
    Return:
        antcom -- square matrix of the anticommutator {A,B}

    """
    antcom = A.dot(B) + B.dot(A)
    return antcom

def rot(theta,gate):
     """
     Unitary rotation for a qubit
     Args:
         theta(float): angle in radians for the rotation of qubit
         gate (matrix): Pauli matrix sx, sy, sz 
     """
     r = matrix.expm((-1.0j/2)*theta*gate)
     return r
         
def density_state(u,v,w):
    """Based on the components u, v, w from the Bloch vector,
    transfor to the density state.
    Return 
    p array (2,2):density state
    """
    p = 0.5*np.array([[1 + w , u - 1j*v],[u + 1j*v , 1 - w]])
    return p
    
# =============================================================================
# MULTIQUIBTIS SYSTEM
# =============================================================================
def partial_trace(rho, qubit_2_keep):
        """ Calculate the partial trace for qubit systems
        Parameters
        ----------
        rho: np.ndarray
            Density matrix
        qubit_2_keep: list
            Index of qubit to be kept after taking the trace
        Returns
        -------
        rho_res: np.ndarray
            Density matrix after taking partial trace
        """
        num_qubit = int(np.log2(rho.shape[0]))
        qubit_axis = [(i, num_qubit + i) for i in range(num_qubit)
                      if i not in qubit_2_keep]
        minus_factor = [(i, 2 * i) for i in range(len(qubit_axis))]
        minus_qubit_axis = [(q[0] - m[0], q[1] - m[1])
                            for q, m in zip(qubit_axis, minus_factor)]
        rho_res = np.reshape(rho, [2, 2] * num_qubit)
        qubit_left = num_qubit - len(qubit_axis)
        for i, j in minus_qubit_axis:
            rho_res = np.trace(rho_res, axis1=i, axis2=j)
        if qubit_left > 1:
            rho_res = np.reshape(rho_res, [2 ** qubit_left] * 2)
    
        return rho_res
    
def partial_trace_mul(rho, dims, axis=0):
    """
    Takes partial trace over the subsystem defined by 'axis'
    rho: a matrix
    dims: a list containing the dimension of each subsystem
    axis: the index of the subsytem to be traced out
    (We assume that each subsystem is square)
    """
    dims_ = np.array(dims)
    # Reshape the matrix into a tensor with the following shape:
    # [dim_0, dim_1, ..., dim_n, dim_0, dim_1, ..., dim_n]
    # Each subsystem gets one index for its row and another one for its column
    reshaped_rho = rho.reshape(np.concatenate((dims_, dims_), axis=None))

    # Move the subsystems to be traced towards the end
    reshaped_rho = np.moveaxis(reshaped_rho, axis, -1)
    reshaped_rho = np.moveaxis(reshaped_rho, len(dims)+axis-1, -1)

    # Trace over the very last row and column indices
    traced_out_rho = np.trace(reshaped_rho, axis1=-2, axis2=-1)

    # traced_out_rho is still in the shape of a tensor
    # Reshape back to a matrix
    dims_untraced = np.delete(dims_, axis)
    rho_dim = np.prod(dims_untraced)
    return traced_out_rho.reshape([rho_dim, rho_dim])
    
def J_hats(nq):
        j_hats = []
        for j in range(nq):
            j_hats.append([x for x in range(nq) if x != j])
            
        return j_hats
def map_to_composite(F_Sub, Dims, F_space):
    """
    This function will return the correct Eq.12 in the Beretta Equation 
    of 1985 paper, 'Quantum thermodynamics. A new equation of motion for a
    General Quantum system
    Args:
        F_sub (array 2**(qb-1) by 2**(qb-1)) -> subsytem J_hat in the equation
        Dims(list with qb elements)-> dimensions of each subsytem in the case of 
                        3 qubits [2,2,2]
        F_Space (int) position of J
                    
    """
    F_comp = 0
    #Number of subsystems
    nSS = len(Dims)
    #Subsystem index vector
    ss = [x for x in range(nSS)]
    rdim = Dims[::-1]
    #Subsystems on which F_sub operates
#        j = [x for x in ss if x != F_space]
    #Subsystems on which F_sub does not operate
    jbar = [x for x in ss if x == F_space]
    #Dimension on which F_sub operates
#        dimj = [Dims[x] for x in j]
    #Total dimensions on which F_sub operates
#        dimF_sub = dimj.prod()
    #WARNING
    dimJbar = [Dims[x] for x in jbar][0]  
    
    F_sub = np.zeros((F_Sub.shape[0],F_Sub.shape[1],int(dimJbar**2)),dtype = np.complex128)
    
    inds = np.arange(0,dimJbar**2,dimJbar+1)
    for i in inds:    
        F_sub[:,:,i] = F_Sub
    
    keep = np.arange(nSS)
    keep = keep[keep != jbar[0]]
    
    perm = nSS -1 - np.concatenate((keep[::-1],keep[::-1]-nSS,np.array(jbar),np.array(jbar)-nSS))
    
    rdim2 = rdim + rdim
    perm_rdim2 = np.array(rdim2)[perm]
#        F_Sub = np.zeros((64,64,4))
    F_sub = F_sub.reshape(perm_rdim2,order = 'F')
    F_sub = ipermute(F_sub,perm)
    
    F_comp = F_sub.reshape((np.prod(Dims),np.prod(Dims)),order = 'F')
    
    return F_comp

def ipermute(b,order):
        invers_order = np.argsort(order)
        b = np.transpose(b,invers_order)
        return b
    
def O_J(F,p_perm,j):
    """
    Function that describe equation 12 in the pape rBeretta (1985) in the 
    paper Quantum Thermodynamics. A new equation of motion for a general 
    quantum system.
    
    Args:
        F (array nq by nq): Operator which is wanted to operate (F)^J
        pjs (array (nq,2^nq,2^nq)): partial trace over system j
        j (int): qubit which is goning to operate over F
    
    """
    f_j = partial_trace(np.dot(p_perm,F),j)
    return f_j

def FG_J(F_J,G_J,pj):
    """
    Function that describe equation 13 in the pape rBeretta (1985) in the 
    paper Quantum Thermodynamics. A new equation of motion for a general 
    quantum system.
    
    Args:
        F_j (array nq by nq): Operator (F)^J
        G_J (array nq by nq): Operator (G)^J
        pj (array nq by nq): density operator for qubit j
    
    """
    return 0.5*pj.dot(anticom(F_J,G_J)).trace() 

def kron_i(matrix1,pos,nq):
    """
    This is for a specific position of a kronecker product 
    Args:
        matrix1(array simple qubit dimensions)
        pos(int): position where matrix1 is in the kronecker product
    
    For Example: 
         p =np.kron(pa,pb,pc)
         
         and I want kron(I pb I)
         matrix1 = pb
         matrix2 = I
         pos = 1
    """
    p_j = np.kron(np.eye(int(2**pos)),np.kron(matrix1,np.eye(int(2**(nq-pos-1))))) 
    return p_j

def c_gate(gate,qb,pos):
        """
        THIS FUNCTION IS NOT GENERIC REVIEW TO FIND THE MISTAKE!!!!!!!!
        WARNING 
        WARNING
        WARNING
        Function to produce a cnot gate between the the last two values of the
        gate_list.
        
       Args:
            g(str):gate to be applied 
            pos(list):[C,T] where C is the control and T is the target 
                Example
                g = z90 rotation about z of 90 degrees 
                pos = [0,1] qubit 0 controll and qubit 1 target
        """
        I = si
        s00 = np.array([[1,0],[0,0]])
        s11 = np.array([[0,0],[0,1]])
        gate1 = 1
        gate2 = 1
        for i in range(qb):
            if pos[0] == i:
                gate1 = np.kron(gate1,s00)
                gate2 = np.kron(gate2,s11)
            elif pos[1] == i:
                gate1 = np.kron(gate1,I)
                gate2 = np.kron(gate2,gate)
            else:
                gate1 = np.kron(gate1,I)
                gate2 = np.kron(gate2,I)
        gateT = gate1 + gate2
        return gateT
    
def cnot(pos,nq):
    """
    This function allows to make the CNOT gate between two qubits in 
    an array of n qubits.
    
    Args:
        pos(list): list with the numbber of the qubits of control and target
        in the respective order.
        
    Returns:
        gate(numpy array): qb X qb matrix with the CNOT gate between qubits in 
        pos
        
        Ex:
            cnot([1,2]) -> Control = 1, Target = 2. Because this is a clase 
            you need first to create a object with this clase and inside 
            it is implicit the number of qubits. Therefore, it's not 
            necessary to mention the number of qubits
    """
    gate = 0.5*(twoOp(pos,0,0,nq) + twoOp(pos,3,0,nq) + 
                twoOp(pos,0,1,nq) - twoOp(pos,3,1,nq))
    return gate

def twoOp(pos,op1,op2,nq):
    """
    This is a Function for the CNOT gate presented by Bruno Juliá-Díaz and
    Frank Tabakin
    Args:
        nq(int): number of qubits
    """
    zerosT = np.zeros((nq,))
    zerosT[pos] = [op1,op2]
    return sp(nq,zerosT)

def sp(n,Q):
    """
    This function is another step for the toffoli gate from the mathematica
    code of Bruno Juliá and Frank Tabakin
    """
    if n > 1:
        res = np.kron(sp(n-1,Q[:-1]),pauliMatrix(Q[n-1]))    
    else:
        res = pauliMatrix(Q[0])
    return res   
def pauliMatrix(n):
    """
    Return one of the Pauli matrix in this order i,x,y,z
    
    Args:
        n(int): 0 for i, 1 for x, and so on...
        
    Return
        PauliMatrix(Matrix): 2X2 matrix of the set of Pauli Matrices
        
    """
    if n == 0:
        s = si
    elif n == 1:
        s = sx
    elif n == 2:
        s = sy
    elif n == 3:
        s = sz
    else:
        print('This is not a valid gate, check your inputs')
   
    return s
# =============================================================================
# 2-Quibit system
# =============================================================================
    
def trA(p):
    """
    Compute the partial trace of A of the composed matrix of two subsystems A and B
    
    Arguments:
        p -- square matrix of 4 x 4 
    
    Return:
        pA -- square matrix of 2 x 2

    """
    pA = np.matrix([[p[0,0]+p[2,2],p[0,1]+p[2,3]],[p[1,0]+p[3,2],p[1,1]+p[3,3]]])
    return pA

def trB(p):
    """
    Compute the partial trace of B of the composed matrix of two subsystems A and B
    
    Arguments:
        p -- square matrix of 4 x 4 
    
    Return:
        pB -- square matrix of 2 x 2

    """
    pB = np.matrix([[p[0,0]+p[1,1],p[0,2]+p[1,3]],[p[2,0]+p[3,1],p[2,2]+p[3,3]]])
    return pB

# =============================================================================
# Gates
# =============================================================================
def thetaG(t,t1,t2):
    """
    Return a Gaussian pulse.
    
    Arguments:
        t -- time of the pulse
            t1 -- initial time
            t2 -- final time
    
    Return:
        theta -- Scalar or vector with the dimensions of t,
        
    
    """
    tau = (t2-t1)/5
    to = t1 + (t2-t1)/2
    theta = (np.sqrt(np.pi)/(2*tau))*np.exp(-((t-to)/tau)**2)
    return theta

def gaussian_Pulse(t,t1,t2,sigma):
    """
    Gaussian pulse.
    
    Arguments:
        t -- time of the pulse
            t1 -- initial time
            t2 -- final time
    
    Return:
        theta -- Scalar or vector with the dimensions of t,
        
    
    """
    to = t1 + (t2-t1)/2 #duration/2
    theta = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((t-to)/sigma)**2)
    return theta

def gaussian_qiskit(t,t1,t2,sigma):
    """
    Gaussian pulse.
    
    Arguments:
        t -- time of the pulse
            t1 -- initial time
            t2 -- final time
    
    Return:
        theta -- Scalar or vector with the dimensions of t,
        
    
    """
    to = t1 + (t2-t1)/2 #duration/2
    # theta0 = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((t1-to)/sigma)**2)
    theta0 = np.exp(-0.5*((t1-to)/sigma)**2)
    # theta = fstep(t,t1,t2)*((1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((t-to)/sigma)**2) - theta0)
    theta = fstep(t,t1,t2)*(np.exp(-0.5*((t-to)/sigma)**2) - theta0)
    return theta

def gaussianSquare(t,t1,t2,sigma,width):
    """
    Gaussian pulse.
    
    Arguments:
        t -- time of the pulse
            t1 -- initial time for the Gaussian
            t2 -- final time for the Gaussian pulse
    
    Return:
        theta -- Scalar or vector with the dimensions of t,
        
    
    """
    tau = t2 - t1
    to = t1 + tau/2 #duration/2
    # theta0 = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((t1-to)/sigma)**2)
    theta0 = np.exp(-0.5*((t1-to)/sigma)**2)
    # theta = fstep(t,t1,t2)*((1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((t-to)/sigma)**2) - theta0)
    rise = fstep(t,t1,t1+tau/2)*(np.exp(-0.5*((t-to)/sigma)**2) - theta0) / (1 - theta0) 
    middle = fstep(t,t1 + tau/2, t1 + tau/2 + width)
    to_f = t1 + width + tau/2 #duration/2
    fall = fstep(t,t1+tau/2 + width, t1 + tau + width)*(np.exp(-0.5*((t-to_f)/sigma)**2) - theta0)/(1 - theta0) 
    return rise + middle + fall

def gaussian_qiskitB(t,t1,t2,sigma):
    """
    Gaussian pulse.
    
    Arguments:
        t -- time of the pulse
            t1 -- initial time
            t2 -- final time
    
    Return:
        theta -- Scalar or vector with the dimensions of t,
        
    
    """
    to = t1 + (t2-t1)/2 #duration/2
    theta = fstep(t,t1,t2)*(np.exp(-0.5*((t-to)/sigma)**2))
    return theta

def drag_Pulse(t,t1,t2,sigma,function = gaussian_qiskit):
    """

    Parameters
    ----------
    t : float or array
        time for which is it required the pulse.
    t1 : float
        initial time of the pulse.
    t2 : float
        final time of the pulse.
    sigma : float
        pulse width of the pulse.

    Returns
    -------
    pulse : float or array
        pulse amplitude for the time t

    """
    to = t1 + (t2-t1)/2 #duration/2
    drag = - ((t - to)/sigma**2)*(function(t,t1,t2,sigma)) 
    return drag

def abs_drag(t,t1,t2,sigma,function = gaussian_qiskit):
    return abs(drag_Pulse(t,t1,t2,sigma,function))

def sdrag_Pulse(t,t1,t2,sigma,function = gaussian_qiskit):
    """

    Parameters
    ----------
    t : float or array
        time for which is it required the pulse.
    t1 : float
        initial time of the pulse.
    t2 : float
        final time of the pulse.
    sigma : float
        pulse width of the pulse.

    Returns
    -------
    pulse : float or array
        pulse amplitude for the time t

    """
    to = t1 + (t2-t1)/2 #duration/2
    drag = (((t-to)**2 - sigma**2)/sigma**3)*function(t,t1,t2,sigma)
    return drag
def abs_sdrag(t,t1,t2,sigma,function = gaussian_qiskit):
    return abs(sdrag_Pulse(t,t1,t2,sigma,function))
def tdrag_Pulse(t,t1,t2,sigma,function = gaussian_qiskit):
    """

    Parameters
    ----------
    t : float or array
        time for which is it required the pulse.
    t1 : float
        initial time of the pulse.
    t2 : float
        final time of the pulse.
    sigma : float
        pulse width of the pulse.
    Returns
    -------
    pulse : float or array
        pulse amplitude for the time t

    """
    to = t1 + (t2-t1)/2 #duration/2
    drag = ((-(t-to)**3 - 3*(t - to)*sigma**2)/sigma**5)*function(t,t1,t2,sigma)
    return drag
def abs_tdrag(t,t1,t2,sigma,function = gaussian_qiskit):
    return abs(tdrag_Pulse(t,t1,t2,sigma,function))
def cdrag_Pulse(t,t1,t2,sigma,function = gaussian_qiskit):
    """

    Parameters
    ----------
    t : float or array
        time for which is it required the pulse.
    t1 : float
        initial time of the pulse.
    t2 : float
        final time of the pulse.
    sigma : float
        pulse width of the pulse.
    Returns
    -------
    pulse : float or array
        pulse amplitude for the time t

    """
    to = t1 + (t2-t1)/2 #duration/2
    drag = (((t-to)**4 - 6*(t - to)**2*sigma**2 + 3*sigma**4)/sigma**7)*function(t,t1,t2,sigma)
    return drag

def abs_cdrag(t,t1,t2,sigma,function = gaussian_qiskit):
    return abs(cdrag_Pulse(t,t1,t2,sigma,function))

def fdrag_Pulse(t,t1,t2,sigma,function = gaussian_qiskit):
    """

    Parameters
    ----------
    t : float or array
        time for which is it required the pulse.
    t1 : float
        initial time of the pulse.
    t2 : float
        final time of the pulse.
    sigma : float
        pulse width of the pulse.
    Returns
    -------
    pulse : float or array
        pulse amplitude for the time t

    """
    to = t1 + (t2-t1)/2 #duration/2
    drag = ((-(t-to)**5 + 10*(t - to)**3*sigma**2 + 15*(-t + to)*sigma**4)/sigma**9)*function(t,t1,t2,sigma)
    return drag

def abs_fdrag(t,t1,t2,sigma,function = gaussian_qiskit):
    return abs(fdrag_Pulse(t,t1,t2,sigma,function))    

def nfun(t1,t2,function = thetaG):
    """
    Return the normalization of the soft pulse for the soft square pulse.
    
    Arguments:
        t -- time of the pulse
        to -- Central time
        tau -- width of the pulse
    
    Return:
        Nf -- Normalization of the soft square pulse
    """
    tau = (t2 - t1)
    func, error = sol.quad(lambda t : function(t,0,tau),-tau,tau)
    Nf = func
    return Nf

def normalized_gaussian(t1,t2,sigma,function = gaussian_qiskit):
    """
    Return the normalization of the soft pulse for the soft square pulse.
    
    Arguments:
        t -- time of the pulse
        to -- Central time
        tau -- width of the pulse
    
    Return:
        Nf -- Normalization of the soft square pulse
    """
    func, error = sol.quad(lambda t : function(t,t1,t2,sigma),t1,t2)
    Nf = func
    return Nf

def normalized_squared(t1,t2,sigma,width,function = gaussianSquare):
    """
    Return the normalization of the soft pulse for the soft square pulse.
    
    Arguments:
        t -- time of the pulse
        to -- Central time
        tau -- width of the pulse
    
    Return:
        Nf -- Normalization of the soft square pulse
    """
    func, error = sol.quad(lambda t : function(t,t1,t2,sigma,width),t1,t2+width)
    Nf = func
    return Nf

def fstep(t,t1,t2):
    """
    This is a step function to activate a signal strating at t1 and finishing at t1+tau

    Arguments:
        t1(float) = initial time
        tau(float) = pulse width
        
    Return:
        t1 < t < t1+tau -> f(t1,t2+tau) == 1
        else f == 0
        
    """
    f = np.heaviside(t-t1,0) - np.heaviside(t-t2,0)
    return f


# =============================================================================
# Armonk IBMQ quantum device gates
# =============================================================================
def x_pulse_VZ(t,t1,t2,sigma):
    """
    Virtual Rotation Pulse proposed in McKay2017
    Parameters
    ----------
    t : float or array
        time point for which is needed the x_pulse.
    t1 : float
        Initial time of the pulse.
    t2 : float
        Final time of the pulse usually t1 + 4*sigma.
    sigma : float
        Pulse width.

    Returns
    -------
    float or array
        Normalized two consecutive pulses with intermediated virtual Z rotations.

    """
    tg = t2 - t1
    norm = normalized_gaussian(t1,tg/2,sigma)
    pulsef = 0.5*gaussian_qiskit(t,t1,t1 + tg/2, sigma) - 0.5*gaussian_qiskit(t,t1 + tg/2, t2, sigma) 
    return pulsef/norm

def x_pulse_SP(t,t1,t2,sigma,function):
    norm = normalized_gaussian(t1,t2,sigma,function)
    pulsef = function(t,t1,t2, sigma)
    return pulsef/norm


# =============================================================================
# Different times gates generation
# =============================================================================
def pulse_samples(n,dt,sigma,beta,function,abs_func):
    t1 = 0; t2 = n*dt
    time = np.linspace(t1,t2,n)
    
    gauss_fun = lambda t : x_pulse_SP(t,t1,t2,sigma,gaussian_qiskit)
    ndrag_fun = lambda t : function(t,t1,t2,sigma,gaussian_qiskit)
    n_ndrag = normalized_gaussian(t1,t2,sigma,abs_func)
    gau_max = max(gauss_fun(time))
    samples = (gauss_fun(time) + 1j*beta*ndrag_fun(time)/n_ndrag)/gau_max
    return samples
    
# =============================================================================
# New Realization of derivative pulses gates: Here, I take into account the 
# Motzi2009 paper where the DRAG should be the first derivative of the pulse
# =============================================================================
def pulse_new(n,dt,sigma,beta,function,dfunction,abs_func,abs_dfunc):
    t1 = 0; t2 = n*dt
    time = np.linspace(t1,t2,n)
    
    gauss_fun = lambda t : function(t,t1,t2,sigma,gaussian_qiskit)
    ndrag_fun = lambda t : dfunction(t,t1,t2,sigma,gaussian_qiskit)
    n_dfun = normalized_gaussian(t1,t2,sigma,abs_dfunc)
    n_fun = normalized_gaussian(t1,t2,sigma,abs_func)
    gau_max = max(gauss_fun(time)/n_fun)
    samples = (gauss_fun(time)/n_fun + 1j*beta*ndrag_fun(time)/n_dfun)/gau_max
    return samples

# DELETE AFTER USE IT!!!!
def pulse_new1(n,dt,sigma,beta,function,dfunction,abs_func,delta):
    t1 = 0; t2 = n*dt
    time = np.linspace(t1,t2,n)
    
    gauss_fun = lambda t : function(t,t1,t2,sigma)
    ndrag_fun = lambda t : dfunction(t,t1,t2,sigma,gaussian_qiskit)
    n_fun = normalized_gaussian(t1,t2,sigma,abs_func)
    gau_max = max(gauss_fun(time)/n_fun)
    samples = (gauss_fun(time)/n_fun + 1j*(beta/delta)*ndrag_fun(time)/n_fun)/gau_max
    return samples


# =============================================================================
# What if instead of using Gaussian Pulse, I use a different function
# =============================================================================
def gauss_new(t,t1,t2,sigma):
    """
    Gaussian pulse.
    
    Arguments:
        t -- time of the pulse
            t1 -- initial time
            t2 -- final time
    
    Return:
        theta -- Scalar or vector with the dimensions of t,
        
    
    """
    to = t1 + (t2-t1)/2 #duration/2
    theta0 = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((t1-to)/sigma)**4)
    # theta0 = np.exp(-0.5*((t1-to)/sigma)**4)
    theta = fstep(t,t1,t2)*((1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((t-to)/sigma)**4) - theta0)
    # theta = fstep(t,t1,t2)*(np.exp(-0.5*((t-to)/sigma)**4) - theta0)
    return theta
def gauss_newB(t,t1,t2,sigma):
    """
    Gaussian pulse.
    
    Arguments:
        t -- time of the pulse
            t1 -- initial time
            t2 -- final time
    
    Return:
        theta -- Scalar or vector with the dimensions of t,
        
    
    """
    to = t1 + (t2-t1)/2 #duration/2
    # theta0 = np.exp(-0.5*((t1-to)/sigma)**4)
    theta = fstep(t,t1,t2)*((1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((t-to)/sigma)**4))
    # theta = fstep(t,t1,t2)*(np.exp(-0.5*((t-to)/sigma)**4) - theta0)
    return theta

def dgauss_new(t,t1,t2,sigma,function = gauss_new):
    """
    Gaussian pulse.
    
    Arguments:
        t -- time of the pulse
            t1 -- initial time
            t2 -- final time
    
    Return:
        theta -- Scalar or vector with the dimensions of t,
        
    
    """
    to = t1 + (t2-t1)/2 #duration/2
    theta = -(2*(t - to)**3/(sigma**4))*function(t,t1,t2,sigma)
    return theta

def area_pulse(t1,t2,sigma,fun, width= None):
    fun_abs = lambda t,t1,t2,sigma: abs(fun(t,t1,t2,sigma))
    area = normalized_gaussian(t1,t2,sigma,function = fun_abs)  
    return  area


# =============================================================================
# Classification states
# =============================================================================
def state_classifier(zero,one,two = None):
    
    dim = len(zero)
    if two is None:
        in_data = np.concatenate((zero,one))
        out_data = np.concatenate((np.zeros(dim,),np.ones(dim,)))
    else:
        in_data = np.concatenate((zero,one,two))
        out_data = np.concatenate((np.zeros(dim,),np.ones(dim,),2*np.ones(dim,)))
    
    train_in, test_in, train_out, test_out = train_test_split(in_data,out_data,test_size = 0.2)
    
    # Linear discriminant analysis
    LDA = LinearDiscriminantAnalysis()
    LDA.fit(train_in,train_out)
    return LDA

def classification(data,classifier,two_try = False):
    if len(data.shape) == 3:
        zero = []; one = []; two = []
        points_zero = []; points_one = []; points_two = []
        for i in data:
            dim = len(i)
            zero.append(np.sum(classifier.predict(i) == 0.0)/dim)
            points_zero.append(i[classifier.predict(i) == 0.0,:])
            one.append(np.sum(classifier.predict(i) == 1.0)/dim)
            points_one.append(i[classifier.predict(i) == 1.0,:])
            two.append(np.sum(classifier.predict(i) == 2.0)/dim)
            points_two.append(i[classifier.predict(i) == 2.0,:])
    elif len(data.shape) == 2:
        dim = len(data)
        zero = np.sum(classifier.predict(data) == 0.0)/dim
        points_zero = data[classifier.predict(data) == 0.0,:]
        one = np.sum(classifier.predict(data) == 1.0)/dim
        points_one = data[classifier.predict(data) == 1.0,:]
        two = np.sum(classifier.predict(data) == 2.0)/dim
        points_two = data[classifier.predict(data) == 2.0,:]
    else:
        print("This is not a valid array")
        return
    if two_try:
        return zero, one, two, points_zero, points_one, points_two
    else:
        return zero, one, points_zero, points_one
            
def gate_sequence(t, sigma, tT, width, circuit, beta, func, dfunc, func_CR,
                  n_qubits, CR_real= False, AC=False, delay=None, width_def=None):
    
    pulse = np.zeros((2**n_qubits,2**n_qubits), dtype = complex)
    t_c = 0;
    a_fun =  area_pulse(0,tT,sigma,func)
    a_dfun = area_pulse(0,tT,sigma,dfunc)
    if isinstance(width_def, float):
        a_CR = normalized_squared(0, tT, sigma, width_def)
    else:
        a_CR = normalized_squared(0, tT, sigma, width)
    for names, params in circuit:
        t1 = t_c
        t2 = t_c + tT
        if names[0] == 'CR':
            ZX = 1
            for i in range(n_qubits):
                if i == params[0][0]:
                    ZX = np.kron(ZX,sz)
                elif i == params[0][1]:
                    ZX = np.kron(ZX,sx)
                else:
                    ZX = np.kron(ZX, si)
            if CR_real:
                pulse += func_CR(t,t1,t2,sigma,width) * \
                    (params[1]/a_CR * ZX + params[2]['IZ']*IZ + params[2]['IX'] * 
                     IX + params[2]['ZI']*ZI + params[2]['ZZ']*ZZ + params[2]['ZY']*ZY +
                     params[2]["XI"]*XI)
            else:
                pulse += params[1]*func_CR(t,t1,t2,sigma,width) * ZX / a_CR
            if AC: # Active cancellation
                pulse += AC * func_CR(t, t1, t2, sigma, width) * IX / a_CR
            t_c += tT + width
        elif names[0] == "delay":
            t_c += params[1]
        else:
            t_prov = 0
            for n, name in enumerate(names):
                num = params[n]
                if name == 'rx':
                    num = params[0][n]
                X = kron_i(sx, num, n_qubits)
                Y = kron_i(sy, num, n_qubits)
                if name == 'x':
                    pulse += func(t,t1,t2,sigma)/a_fun * X + beta*dfunc(t,t1,t2,sigma)/a_dfun *  Y        
                    pulse += func(t,t2,t2 + tT,sigma)/a_fun * X + beta*dfunc(t,t2,t2 + tT,sigma)/a_dfun * Y
                    if 2*tT > t_prov:
                        t_prov = 2*tT
                elif name == 'y':
                    pulse += func(t,t1,t2,sigma)/a_fun * X + beta*dfunc(t,t1,t2,sigma)/a_dfun * Y          
                    pulse += func(t,t2,t2 + tT,sigma)/a_fun * X + beta*dfunc(t,t2,t2 + tT,sigma)/a_dfun * Y
                    if 2*tT > t_prov:
                        t_prov = 2*tT
                elif name == 'h':
                    pulse += func(t,t1,t2,sigma)/a_fun * X + beta*dfunc(t,t1,t2,sigma)/a_dfun * Y        
                    if tT > t_prov:
                        t_prov = tT
                elif name == 'rx':
                    pulse += params[1]*(func(t,t1,t2,sigma)/a_fun * X + beta*dfunc(t,t1,t2,sigma)/a_dfun * Y)  
                    if tT > t_prov:
                        t_prov = tT  
                elif name == 'ry':
                    pulse += params[1]*(func(t,t1,t2,sigma)/a_fun * Y + beta*dfunc(t,t1,t2,sigma)/a_dfun * X)  
                    if tT > t_prov:
                        t_prov = tT 
            t_c += t_prov
    return pulse


def vz_time(circuit,tG, width, n_qubits,n_gate, n_delay=None, delay=None):
    vz = {};t_v = np.array([0]);t_c = 0; gb = {};n_gates = 0
    for i in range(n_qubits):
        gb[i] = ""
        vz[i] = []
    for names, params in circuit:
        for ii, name in enumerate(names):
            num = params[ii]
            if name == 'x':
                if gb[num] == 'virtual':
                    vz[num][-1][1] -= np.pi/2
                else:
                    vz[num].append([t_c,-np.pi/2])
                vz[num].append([t_c+2*tG,-np.pi/2])
                if ii + 1 == len(names):
                    t_v = np.concatenate((t_v,np.linspace(t_c,t_c+tG,n_gate)[1:]))
                    t_v = np.concatenate((t_v,np.linspace(t_c+tG,t_c+2*tG,n_gate)[1:]))
                    t_c += 2*tG   
                    n_gates += 2
                gb[num] = 'virtual'
            elif name == 'y':
                if gb[num] == 'virtual':
                    vz[num][-1][1] -= np.pi
                else:
                    vz[num].append([t_c,-np.pi])
                if ii + 1 == len(names):
                    t_v = np.concatenate((t_v,np.linspace(t_c,t_c+tG,n_gate)[1:]))
                    t_v = np.concatenate((t_v,np.linspace(t_c+tG,t_c+2*tG,n_gate)[1:]))
                    t_c += 2*tG   
                    n_gates += 2
                    
                gb[num] = 'non-virtual'
            elif name == 'h':
                if gb[num] == 'virtual':
                    vz[num][-1][1] += np.pi/2 
                else:
                    vz[num].append([t_c, np.pi/2])
                vz[num].append([t_c + tG, np.pi/2])
                if ii + 1 == len(names):
                    t_v = np.concatenate((t_v,np.linspace(t_c,t_c+tG,n_gate)[1:]))
                    t_c += tG
                    n_gates += 1
                gb[num] = 'virtual'
            elif name == 'rx':
                t_v = np.concatenate((t_v,np.linspace(t_c,t_c+tG,n_gate)[1:]))
                t_c += tG
                n_gates += 1
                for num in params[0]:
                    gb[num] = 'non-virtual'
            elif name == 'ry':
                t_v = np.concatenate((t_v,np.linspace(t_c,t_c+tG,n_gate)[1:]))
                t_c += tG
                n_gates += 1
                for num in params[0]:
                    gb[num] = 'non-virtual'
            elif name == 's':
                if gb[num] == 'virtual':
                    vz[num][-1][1] +=  np.pi/2
                else:
                    vz[num].append([t_c,+np.pi/2])
                gb[num] = 'virtual'
            elif name == 'sdg':
                if gb[num] == 'virtual':
                    vz[num][-1][1] -= np.pi/2 
                else:
                    vz[num].append([t_c, -np.pi/2])
                gb[num] = 'virtual'
            elif name == 'z':
                if gb[num] == 'virtual':
                    vz[num][-1][1] += np.pi 
                else:
                    vz[num].append([t_c,np.pi])
                gb[num] = 'virtual'
            elif name == 'CR':
                t_v = np.concatenate((t_v,np.linspace(t_c,t_c+tG+width,n_gate)[1:]))
                t_c += tG + width   
                n_gates += 1
                for num in params[0]:
                    gb[num] = 'non-virtual'
            elif name == "delay":
                t_v = np.concatenate((t_v, np.linspace(t_c, t_c+params[1], n_delay)[1:]))
                t_c += params[1]
                for num in params[0]:
                    gb[num] = 'non-virtual'
    t_v = np.concatenate((t_v,np.array(t_v[-1]+tG/n_gate).reshape(1,)))
    return vz,t_v,n_gates

def mean_std(Exp, backend):
    data = []
    for day in Exp[backend].values():
        data.append(np.squeeze(day))
    data = 2 * np.array(data)/Exp["shots"] - 1
    return data.mean(axis=0), data.std(axis=0)

def width_def(backend):
    """
    Function to extract the width of a CNOT pulse

    Parameters
    ----------
    backend : qiskit backend
        DESCRIPTION.

    Returns
    -------
    width : int
        samples of the duration of the pulse, in terms of time it will be 
        multiplied by 0.22ns.

    """
    cx_def = backend.defaults().instruction_schedule_map.get("cx",[0,1])
    for inst in cx_def.instructions:
        if (inst[1].duration > 0) and ("CR" in inst[1].name):
            width = inst[1].pulse.width
            return width