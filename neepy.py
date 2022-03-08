#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:54:16 2020
Class for the SEAQT equation of motion but in a general form 
@author: jmon
"""
import numpy as np
import scipy.linalg as matrix
import scipy.integrate as integrate
from functions_neepy import dag, commutator, pro
from functions_neepy import partial_trace, partial_trace_mul, J_hats, ipermute, map_to_composite, O_J, FG_J
from functions_neepy import anticom , kron_i, trA, trB
from functions_neepy import sz
from scipy.constants import hbar

class Neepy:
    hbar = hbar
    si = np.eye(2, dtype = complex)
    sx = np.array([[0,1],[1,0]], dtype = complex)
    sy = np.array([[0,-1j],[1j,0]], dtype = complex)
    sz = np.array([[1,0],[0,-1]], dtype = complex)
    # sz = np.array([[1,0,0,0],[0,-1,0,0],[0,0,0,0],[0,0,0,0]])

    def __init__(self):
        self.a = 1
    
    def SEAQT(self,t,p,H,tauD):
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
        plogp = p.dot(logp)
        Trplogp = np.matrix.trace(plogp)         
        TrpHlogp = pro(p.dot(H),logp)
        TrpH = pro(p,H)
        TrpH_2 = pro(p,H.dot(H))
        Gamma = TrpH_2 - TrpH**2.0
        beta = (TrpH*Trplogp - TrpHlogp)/Gamma
        alpha = (TrpHlogp*TrpH - Trplogp*TrpH_2)/Gamma
        ac = p.dot(H) + H.dot(p)
        D = plogp + alpha*p + 0.5*beta*ac 
        D_T = 0.5*(D + dag(D)) 
        term1 = (-1.0j/self.hbar)*commutator(H,p)
        term2 =  -(1.0/tauD)*D_T
        dpdt = term1 + term2
        dpdt = np.reshape(dpdt,(1,nn**2))
        return dpdt
    
    def SEAQT_tab(self,t,p,H,tauD):
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
        logp = - matrix.logm(p,disp =False)[0]
        plogp = p @ logp
        HD = self.H0 + H
        Trplogp = plogp.trace()         
        TrpHlogp = pro(p, HD @ logp)
        TrpH = pro(p, HD)
        TrpH_2 = pro(p, HD @ HD)
        Trplogp_2 = pro(p, logp @ logp)
        Gamma = TrpH_2 - TrpH**2.0
        ac = p @ HD + HD @ p
        tau = (tauD[0]*tauD[1])/(tauD[0] + tauD[1])
        beta3 = self.beta_R2
        # beta3 = ((TrpHlogp - TrpH*Trplogp) - self.kT * (Trplogp_2 - Trplogp**2))/(Gamma - self.kT * (TrpHlogp - TrpH*Trplogp))
        beta2 = (TrpHlogp - TrpH*Trplogp)/Gamma
        beta_23 = tau * (beta2/tauD[0] + beta3/tauD[1])
        D = (1/tau) * (p @ (logp - Trplogp) + beta_23*(0.5*ac - TrpH*p))
        term1 = (-1.0j/self.hbar)*commutator(H,p)
        term2 = D
        dpdt = term1 + term2
        dpdt = np.reshape(dpdt,(1,nn**2))
        return dpdt
    
    def SEAQT_rot(self,t,p,H,tauD):
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
        plogp = p.dot(logp)
        HD = self.H0 + H
        Trplogp = np.matrix.trace(plogp)         
        TrpHlogp = pro(p.dot(HD),logp)
        TrpH = pro(p,HD)
        TrpH_2 = pro(p,HD.dot(HD))
        Gamma = TrpH_2 - TrpH**2.0
        beta = (TrpH*Trplogp - TrpHlogp)/Gamma
        alpha = (TrpHlogp*TrpH - Trplogp*TrpH_2)/Gamma
        ac = p @ HD + HD @ p
        D = plogp + alpha*p + 0.5*beta*ac 
        D_T = 0.5*(D + dag(D)) 
        term1 = (-1.0j/self.hbar)*commutator(H,p)
        if len(tauD) == 2:
            r = np.sqrt((p @ self.sy).trace().real**2 +  (p @ self.sx).trace().real**2)
            tau_D2 = tauD[0]*(1 - r) #+ tauD[1]
            # tau_D2 = tauD[0]/np.exp(-t/tauD[1])
            term2 =  -(1.0/tau_D2)*D_T
        else:
            term2 =  -(1.0/tauD)*D_T
        dpdt = term1 + term2
        dpdt = np.reshape(dpdt,(1,nn**2))
        return dpdt
    
    def SEAQT_reservoir(self,t,p,H,tauD):
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
        sz = np.array([[1,0],[0,-1]],dtype=complex)
        # sz = np.array([[1,0,0,0],[0,-1,0,0],[0,0,0,0],[0,0,0,0]], dtype=complex)

        sy = np.array([[0,-1j],[1j,0]],dtype=complex)
        sx = np.array([[0,1],[1,0]],dtype=complex)
        try:
            logp = matrix.logm(p,disp =False)[0]
        except:
            logp = np.zeros((nn,nn))
                
        plogp = p @ logp
        term1 = (-1.0j/self.hbar)*commutator(H,p)
# =============================================================================
#         Tyler modification
# =============================================================================
        HT = H + self.H0
        
        e = (p @ HT).trace()
        # s = (p @ logp).trace()
        
        Trplogp = np.matrix.trace(plogp)         
        TrpHlogp = pro(p.dot(HT),logp)
        TrpH = pro(p,HT)
        TrpH_2 = pro(p,HT.dot(HT))
        Gamma = TrpH_2 - TrpH**2.0
        beta = (TrpH*Trplogp - TrpHlogp)/Gamma
        alpha = (TrpHlogp*TrpH - Trplogp*TrpH_2)/Gamma
        ac = p @ HT + HT @ p
        D1 = plogp + alpha*p + 0.5*beta*ac 
        # D1 = plogp + alpha*p
# =============================================================================
        # D = (plogp - s*p) + self.beta_R * (p @ (HT - e*I))
        # D = 0.5*self.beta_R * (p @ (HT - e*I) + (HT - e*I) @ p)
        # pz = np.array([[0, p[0,1]],[p[1,0], 0]], dtype = complex)
        # D = self.beta_R * (pz @ (HT - e*I))
        D = self.beta_R * (p @ (HT - e*I))
        # D = np.array([[D[0,0],0],[0, D[1,1]]])
        D_T = 0.5*(D + dag(D)) 
        # tau_D = tauD[0]*(p @ sz).trace() + tauD[1]
        tau_D = tauD[0]*(1 + (p @ sz).trace())
        if tau_D < 0:
            print("Negative!!!")
        # tau_D = tauD[0]*np.exp(-t/tauD[1])
        # tau_D2 = tauD[0]*(p @ sz).trace() + tauD[1]
        # if (t < 240e-6) and (t > 206e-6):
        #     term2 = 0
        # else:   
        term2 =  (1.0/tau_D) * D_T #+ (1/tauD[2])*D1
        # if np.random.rand() < 0.001:
        #     print(f"for time t: {t} ---- D: {D_T}")
        if len(tauD) >= 2:
            r = np.sqrt((p @ self.sy).trace().real**2 +  (p @ self.sx).trace().real**2)
            tau_D2 = tauD[1]*(1 - r)
            # tau_D2 = tauD[2]/np.exp(-t/tauD[3])
            # tau_D2 = tauD[2]/np.sqrt((p @ sy).trace().real**2 +  (p @ sx).trace().real**2)+ tauD[3]
            # tau_D2 = tauD[2]*t + tauD[3]
            # tau_D2 = tauD[2]*(p @ sz).trace() + tauD[3]
            term2 += (1/tau_D2)*D1
        
        dpdt = term1 - term2
        dpdt = np.reshape(dpdt,(1,nn**2))
        return dpdt
    
    def SEAQT_reservoir2(self,t,p,H,tauD):
        """
        Return the evolution of a density state following a non-equilibrium trajectory
        through the steepest entropy ascente or entropy gradient. This is a proposal
        of Gian Paolo Beretta (1984) based on the postulates of Gyftopoulos and Hatsopoulos.
        The modification of the equation is based on the thesis of Tyler 2019 for a 
        non-approximation form .
        Arguments:
            t -- Scalar, time
            p -- square matrix of 4 x 4, density state
            H -- Hamiltonian
            tauD -- disipative constant
            beta_R -- beta = 1/kB*Tr where kB is the Boltzmann constant ant Tr is the reservoir temperature
        
        Return:
            dp/dt -- the change of the density state respect to the time
        """
        e_r, s_r, es_r, e2_r = self.reservoir_terms
        
        nn = int(np.sqrt(len(p)))
        I = np.eye(nn, dtype=complex)
        p = p.reshape((nn,nn))
        try:
            logp = matrix.logm(p, disp=False)[0]
        except:
            logp = np.zeros((nn, nn))
                
        plogp = p @ logp
        term1 = (-1.0j/self.hbar)*commutator(H,p)
# =============================================================================
#         Tyler modification
# =============================================================================
        HT = H + self.H0
        
        e_s = (p @ HT).trace()
        s_s = (p @ logp).trace()
        es_s = (p @ HT @ logp).trace()
        e2_s = (p @ HT @ HT).trace()
        
        B1 = s_s * (e2_s + e2_r - e_r ** 2) + e_s * (s_r * e_r - (es_s + es_r))
        B3 = - s_s * e_s + (es_s + es_r) - s_r * e_r
        gamma = e2_s + e2_r - e_r ** 2 - e_s ** 2 
        
        Dt = plogp - (B1 / gamma) * p + (B3 / gamma) * HT 
# =============================================================================
        D_T = 0.5 * (Dt + dag(Dt)) 
        term2 =  (1.0 / tauD(t)) * D_T
        dpdt = term1 - term2
        dpdt = np.reshape(dpdt, (1, nn ** 2))
        return dpdt
    
    def SEAQT_2(self,t,p,H,tauD):
        """
        Return the evolution of a density state following a non-equilibrium trajectory
        through the steepest entropy ascente or entropy gradient. This is a proposal
        of Gian Paolo Beretta (1984) based on the postulates of Gyftopoulos and Hatsopoulos.
        
        Arguments:
            t -- Scalar, time
            p -- square matrix of 4 x 4, density state
        
        Return:
            dp/dt -- the change of the density state respect to the time
        """
        p = np.reshape(p,(4,4))
        try:
            logp = matrix.logm(p,disp=False)[0]
        except:
            logp = np.zeros(p.shape)
        pA = trB(p)
        IA = self.si
        pB = trA(p)
        IB = self.si
        lnpA = trB(np.kron(IA,pB).dot(logp))
        lnpB = trA(np.kron(pA,IB).dot(logp))

            
        
        #       A 
        HA = trB(np.kron(IA,pB).dot(H))
        TrplnpA = 0.5*pA.dot(anticom(IA,lnpA)).trace()[0,0]
        TrpHA = 0.5*pA.dot(anticom(IA,HA)).trace()[0,0]
        TrpH_2A = 0.5*pA.dot(anticom(HA,HA)).trace()[0,0]
        acA = anticom(pA,HA)
        TrpHlnpA = 0.5*pA.dot(anticom(HA,lnpA)).trace()[0,0]
        tauDA = tauD[0] # Beretta's time constant
        plnpA = pA.dot(lnpA)
        GammaA = TrpH_2A - TrpHA**2.0
        betaA = (TrpHA*TrplnpA - TrpHlnpA)/GammaA
        alphaA = (TrpHlnpA*TrpHA - TrplnpA*TrpH_2A)/GammaA
        DA = plnpA+alphaA*pA + 0.5*betaA*acA
        dA = 0.5*(DA + dag(DA)) 
    
        
        #           B
        HB = trA(np.kron(pA,IB).dot(H))
        TrplnpB = 0.5*pB.dot(anticom(IB,lnpB)).trace()[0,0]
        TrpHB = 0.5*pB.dot(anticom(IB,HB)).trace()[0,0]
        TrpH_2B = 0.5*pB.dot(anticom(HB,HB)).trace()[0,0]
        acB = anticom(pB,HB)
        TrpHlnpB = 0.5*pB.dot(anticom(HB,lnpB)).trace()[0,0]
        
        tauDB = tauD[1]
        plnpB = pB.dot(lnpB)
        GammaB = TrpH_2B - TrpHB**2.0
        betaB = (TrpHB*TrplnpB - TrpHlnpB)/GammaB
        alphaB = (TrpHlnpB*TrpHB - TrplnpB*TrpH_2B)/GammaB
        DB = plnpB + alphaB*pB + 0.5*betaB*acB
        dB = 0.5*(DB + dag(DB)) 
        # Combining the two effects 
        term1 = (-1.0j/self.hbar)*commutator(H,p)
        term2 = - (1.0/tauDA)*np.kron(dA,pB) -(1.0/tauDB)*np.kron(pA,dB)
        dpdt = term1 + term2
        dpdt = np.reshape(dpdt,(1,16))
        return dpdt
    
    def SEAQT_2Res(self,t,p,H,tauD):
        """
        Return the evolution of a density state following a non-equilibrium trajectory
        through the steepest entropy ascente or entropy gradient. This is a proposal
        of Gian Paolo Beretta (1984) based on the postulates of Gyftopoulos and Hatsopoulos.
        
        Arguments:
            t -- Scalar, time
            p -- square matrix of 4 x 4, density state
        
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
        # tauDA1 = tauD[0][0]
        # tauDA2 = tauD[0][1]
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
        # tauDB1 = tauD[1][0]
        # tauDB2 = tauD[1][1]
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
    
    def SEAQT_2ResTay(self,t,p,H,tauD):
        """
        Return the evolution of a density state following a non-equilibrium trajectory
        through the steepest entropy ascente or entropy gradient. This is a proposal
        of Gian Paolo Beretta (1984) based on the postulates of Gyftopoulos and Hatsopoulos.
        
        Arguments:
            t -- Scalar, time
            p -- square matrix of 4 x 4, density state
        
        Return:
            dp/dt -- the change of the density state respect to the time
        """
        p = p.reshape(4,4)
        try:
            logp = matrix.logm(p, disp=False)[0]
        except:
            logp = np.zeros(p.shape, dtype = complex)
        pA = trB(p)
        IA = self.si
        pB = trA(p)
        IB = self.si
        lnpA = trB(np.kron(IA,pB) @ logp)
        lnpB = trA(np.kron(pA,IB) @ logp)
        HT = H + self.H0
            
        
        #       A 
        HA = trB(np.kron(IA,pB) @ HT)
        TrplnpA = 0.5*pA.dot(anticom(IA,lnpA)).trace()[0,0]
        TrpHA = 0.5*pA.dot(anticom(IA,HA)).trace()[0,0]
        TrpH_2A = 0.5*pA.dot(anticom(HA,HA)).trace()[0,0]
        acA = anticom(pA,HA)
        TrpHlnpA = 0.5*pA.dot(anticom(HA,lnpA)).trace()[0,0]
        # tauDA1 = tauD[0][0]*(pA @ self.sz).trace()[0,0] + tauD[0][1] # Beretta's time constant
        # tauDA2 = tauD[0][2]/np.exp(-t/tauD[0][3])
        tauDA1 = tauD[0]
        tauDA2 = tauD[0]
        plnpA = pA @ lnpA
        GammaA = TrpH_2A - TrpHA**2.0
        betaA = (TrpHA*TrplnpA - TrpHlnpA)/GammaA
        alphaA = (TrpHlnpA*TrpHA - TrplnpA*TrpH_2A)/GammaA
        DA = plnpA + alphaA*pA + 0.5*betaA*acA
        dA = 0.5*(DA + dag(DA)) 
        eA = (pA @ HA).trace()[0,0]
        sA = -(pA @ lnpA).trace()[0,0]
        dAr = (lnpA - sA*IA) + self.beta_R * ((HA - eA*IA))
        DAr =  0.5 * (pA @ dAr + pA @ dag(dAr)) 
    
        
        #           B
        HB = trA(np.kron(pA,IB) @ HT)
        TrplnpB = 0.5*pB.dot(anticom(IB,lnpB)).trace()[0,0]
        TrpHB = 0.5*pB.dot(anticom(IB,HB)).trace()[0,0]
        TrpH_2B = 0.5*pB.dot(anticom(HB,HB)).trace()[0,0]
        acB = anticom(pB,HB)
        TrpHlnpB = 0.5*pB.dot(anticom(HB,lnpB)).trace()[0,0]
        
        # tauDB1 = tauD[1][0]*(pB @ self.sz).trace()[0,0] +  tauD[1][1]
        # tauDB2 = tauD[1][2]/np.exp(-t/tauD[1][3]) # Beretta's time constant
        tauDB1 = tauD[1]
        tauDB2 = tauD[1]
        plnpB = pB @ lnpB
        GammaB = TrpH_2B - TrpHB**2.0
        betaB = (TrpHB*TrplnpB - TrpHlnpB)/GammaB
        alphaB = (TrpHlnpB*TrpHB - TrplnpB*TrpH_2B)/GammaB
        DB = plnpB + alphaB*pB + 0.5*betaB*acB
        dB = 0.5*(DB + dag(DB)) 
        eB = (pB @ HB).trace()[0,0]
        sB = -(pB @ lnpB).trace()[0,0]
        dBr = (lnpB - sB*IB) + self.beta_R * (HB - eB*IB)
        DBr = 0.5 * (pB @ dBr + pB @ dag( dBr))
        # Combining the two effects 
        term1 = -(1.0j/self.hbar)*commutator(H,p)
        term2 = - (1.0/tauDB1) * np.kron(pA, DBr) - (1/tauDA1) * np.kron(DAr, pB)
        term3 = - (1.0/tauDA2)*np.kron(dA,pB) - (1.0/tauDB2)*np.kron(pA,dB)
        
        dpdt = term1 + term2  + term3
        dpdt = np.reshape(dpdt,(1,16))
        return dpdt
    
    def SEAQT_gen(self,t,p,H,tauD):
        """
        Return the GENERAL evolution of a density state following a non-equilibrium trajectory
        through the steepest entropy ascente or entropy gradient. This is a proposal
        of Gian Paolo Beretta (1985) in the paper Quantum Thermodynamics. A new equation of 
        motion for a general quantum system.
        
        Arguments:
            t -- Scalar, time
            p -- square matrix of 4 x 4, density state
            H -- Hamiltonian
            tauD -- array with the number of subsistems for the disipative constant 
        
        Return:
            dp/dt -- the change of the density state respect to the time
        """
        nn = int(np.sqrt(len(p)))
        p = np.reshape(p,(nn,nn))
        nq = int(np.log2(nn))
        j_hats = J_hats(nq)
        I = np.eye(2)
        
        D = np.zeros((nn,nn),dtype = np.complex128)
        logp = matrix.logm(p,disp=False)[0]
        Dims = nq*[2]

        H_D = H + self.H0
        for j in range(nq):
            pj = partial_trace(p,[j])
            pj_bar = partial_trace(p,j_hats[j])
            p_perm = map_to_composite(pj_bar,Dims,j)#product np.kron(I_J,pj_bar) Eq 12 Beretta 1985 paper
            lnp_js = O_J(logp,p_perm,[j])
            pjlogpj = pj @ lnp_js
            H_js = O_J(H_D,p_perm,[j])
            Trplogp = FG_J(I,lnp_js,pj) 
            TrpH = FG_J(I,H_js,pj) 
            TrpH2 = FG_J(H_js,H_js,pj) 
            ac_s = anticom(pj,H_js)
            TrpHlogp = FG_J(H_js,lnp_js,pj) 
            Gamma = TrpH2 - TrpH**2.0
            alpha = (TrpHlogp*TrpH - Trplogp*TrpH2)/Gamma
            beta = (TrpH*Trplogp - TrpHlogp)/Gamma
            Dj = (pjlogpj + alpha*pj + 0.5*beta*ac_s)
            Dj_T = 0.5*(Dj + dag(Dj))
            D += -(1/tauD[j])*kron_i(Dj_T,j,nq) @ p_perm
            
        term1 = (-1.0j/self.hbar)*commutator(H,p)
        term2 = D 
        dpdt = term1 + term2
        dpdt = np.reshape(dpdt,(1,nn**2))
        return dpdt
    
    def SEAQT_mul(self,t,p,H,tauD):
        """
        Return the GENERAL evolution of a density state following a non-equilibrium trajectory
        through the steepest entropy ascente or entropy gradient. This is a proposal
        of Gian Paolo Beretta (1985) in the paper Quantum Thermodynamics. A new equation of 
        motion for a general quantum system.
        
        This differe from the others because it combines different size
        subsystems
        
        Arguments:
            t -- Scalar, time
            p -- square matrix of N x N, density state with the subsystems
            H -- Hamiltonian
            tauD -- array with the number of subsistems for the disipative constant 
        
        Return:
            dp/dt -- the change of the density state respect to the time
        """
        nn = int(np.sqrt(len(p)))
        p = np.reshape(p,(nn,nn))
        
        j_hats = [1,0]
        
        D = np.zeros((nn,nn),dtype = np.complex128)
        try:
            logp = matrix.logm(p,disp=False)[0]
        except:
            logp = np.zeros(p.shape)
        Dims = self.Dims
        ns = len(Dims)
        H_D = H + self.H0
        for j in range(ns):
            I = np.eye(Dims[j])
            pj = partial_trace_mul(p, Dims, j_hats[j])
            pj_bar = partial_trace_mul(p, Dims, j)
            p_perm = map_to_composite(pj_bar,Dims,j)#product np.kron(I_J,pj_bar) Eq 12 Beretta 1985 paper
            lnp_js = partial_trace_mul(np.dot(p_perm,logp),Dims,j_hats[j])
            pjlogpj = pj.dot(lnp_js)
            H_js = partial_trace_mul(np.dot(p_perm,H_D),Dims,j_hats[j])
            Trplogp = FG_J(I,lnp_js,pj) 
            TrpH = FG_J(I,H_js,pj) 
            TrpH2 = FG_J(H_js,H_js,pj) 
            ac_s = anticom(pj,H_js)
            TrpHlogp = FG_J(H_js,lnp_js,pj) 
            Gamma = TrpH2 - TrpH**2.0
            alpha = (TrpHlogp*TrpH - Trplogp*TrpH2)/Gamma
            beta = (TrpH*Trplogp - TrpHlogp)/Gamma
            Dj = (pjlogpj + alpha*pj + 0.5*beta*ac_s)
            Dj_T = 0.5*(Dj + dag(Dj))
            if j == 0:
                DD = np.kron(Dj_T,np.eye(Dims[1]))
            else:
                DD = np.kron(np.eye(Dims[0]), Dj_T)
            D += -(1/tauD[j])*DD.dot(p_perm)
            
        term1 = (-1.0j/self.hbar)*commutator(H,p)
        term2 = D 
        dpdt = term1 + term2
        dpdt = np.reshape(dpdt,(1,nn**2))
        return dpdt
    
    def Lindblad(self,t,p,H,gamma):
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
            L_a = np.sqrt(gamma1)*np.array([[0,1],[0,0]])
            L_a_s = [I,L_a]
            L_p = np.sqrt(gamma2)* np.array([[1,0],[0,-1]])
            L_p_s = [I,L_p]
            L1 = L_a_s[int(pos[i,0])]
            L2 = L_p_s[int(pos[i,0])]
            for j in range(1,qb):
                L1 = np.kron(L1,L_a_s[int(pos[i,j])])
                L2 = np.kron(L2,L_p_s[int(pos[i,j])])
            D1 += 2 * L1 @ p @ dag(L1) - dag(L1) @ L1 @ p - p @ dag(L1) @ L1
            D2 += 2 * L2 @ p @ dag(L2) - dag(L2) @ L2 @ p - p @ dag(L2) @ L2
        
        # D1 = 2*L1.dot(p.dot(dag(L1)))- dag(L1).dot(L1.dot(p))- p.dot(dag(L1).dot(L1))
        # D2 = 2*L2.dot(p.dot(dag(L2)))- dag(L2).dot(L2.dot(p))- p.dot(dag(L2).dot(L2))
        dpdt = (-1.0j/self.hbar) * commutator(H,p) + D1 + D2
        dpdt = np.reshape(dpdt,(1,nn*nn))
        return dpdt
    
    def vonNeumann(self,t,p,H,tauD):
        """
        Return the evolution of a density state following a non-equilibrium trajectory
        through the steepest entropy ascente or entropy gradient. This is a proposal
        of Gian Paolo Beretta (1984) based on the postulates of Gyftopoulos and Hatsopoulos.
        
        Arguments:
            t -- Scalar, time
            p -- square matrix of 4 x 4, density state
        
        Return:
            dp/dt -- the change of the density state respect to the time
        """
        nn = int(np.sqrt(len(p)))
        p = np.reshape(p,(nn,nn))
        dpdt = (-1.0j/self.hbar)*commutator(H,p)
        dpdt = np.reshape(dpdt,(1,nn*nn))
        return dpdt
    
    def solution(self,po,time,H,tauD,equation,prin = False):
        """
        Return the integration of the equation of motion for a system in a non-equilibrium 
        state
        
        Arguments:
            po -- initial condition of the density state
            time -- vector with the sequence of steps that the system has to follow
            in order to get to the final state
            H -- Hamiltionian
            tauD -- Dissipative time
            equation -- equation to be solved
        """
        nn = len(po)
        n = len(time) # number of the steps for the evolution 
        p = np.zeros((n,nn,nn),dtype = np.complex64)
        dpdt = np.zeros((n,nn,nn),dtype = np.complex64)
        p[0,:,:] = po
        dpdt[0,:,:] = equation(time[0],p[0,:,:].reshape(nn**2,),H(time[0]),tauD).reshape(nn,nn)
        seaqt = lambda t,p:equation(t,p,H(t),tauD)
        p_C = integrate.complex_ode(seaqt)
        po = np.reshape(po,(nn**2,))    
        p_C.set_initial_value(po,time[0])
        # p_C.set_integrator('Isoda',method = 'bdf',rtol = 1e-12)
        p_C.set_integrator('dopri5',method = 'bdf',rtol = 1e-4)
        # p_C.set_integrator('vode')
        for i,ii in enumerate(time[1:]):
            dt = ii - time[i]
            p[i+1,:,:] = p_C.integrate(p_C.t+dt).reshape((nn,nn))
            dpdt[i+1,:,:] = equation(ii,p[i+1,:,:].reshape(nn**2,),H(ii),tauD).reshape(nn,nn)
            if (i % np.ceil(n/10) == 0) and prin:
                print('--------------------',str(i))
        return p, dpdt
    
    def solution_qiskit(self,po,time,H,tauD,equation,vz):
        """
        Return the integration of the equation of motion for a system in a non-equilibrium 
        state
        
        Arguments:
            po -- initial condition of the density state
            time -- vector with the sequence of steps that the system has to follow
            in order to get to the final state
            H -- Hamiltionian
            tauD -- Dissipative time
            equation -- equation to be solved
            vz (list) -- virtual z rotation time and angle 
        """
        nn = len(po)
        n = len(time) # number of the steps for the evolution 
        p = np.zeros((n,nn,nn),dtype = np.complex64)
        dpdt = np.zeros((n,nn,nn),dtype = np.complex64)
        p[0,:,:] = po
        dpdt[0,:,:] = equation(time[0],p[0,:,:].reshape(nn**2,),H(time[0]),tauD).reshape(nn,nn)
        seaqt = lambda t,p:equation(t,p,H(t),tauD)
        p_C = integrate.complex_ode(seaqt)
        po = np.reshape(po,(nn**2,))    
        p_C.set_initial_value(po,time[0])
        # p_C.set_integrator('Isoda',method = 'bdf')
        # p_C.set_integrator('dopri5',method = 'bdf',rtol = 1e-12)
        num_vz = 0
        for i,ii in enumerate(time[1:]):
            dt = ii - time[i]
            if num_vz < len(vz):
                # print(abs(time[i] - vz[num_vz][0]))
                if (abs(time[i] - vz[num_vz][0]) < 1e-3*dt):
                    z_gate = matrix.expm(-0.5j*vz[num_vz][1]*sz)
                    pC = z_gate.dot(p_C.y.reshape((nn,nn)).dot(dag(z_gate))).reshape(1,nn**2)
                    p_C.set_initial_value(pC,time[i])
                    # p_C.temp = pC.reshape((nn**2,))
                    num_vz += 1
                
            p[i+1,:,:] = p_C.integrate(p_C.t+dt).reshape((nn,nn))
            dpdt[i+1,:,:] = equation(ii,p[i+1,:,:].reshape(nn**2,),H(ii),tauD).reshape(nn,nn)
            if (i % int(n/10) == 0):
                print('--------------------',str(i))
        return p, dpdt
    
    def solution_qiskit_mul_qb(self, po, time, H, tauD, equation, vz, n_qubits):
        """
        Return the integration of the equation of motion for a system in a non-equilibrium 
        state
        
        Arguments:
            po -- initial condition of the density state
            time -- vector with the sequence of steps that the system has to follow
            in order to get to the final state
            H -- Hamiltionian
            tauD -- Dissipative time
            equation -- equation to be solved
            vz (list) -- virtual z rotation time and angle 
        """
        nn = len(po)
        n = len(time) # number of the steps for the evolution 
        p = np.zeros((n,nn,nn),dtype = np.complex64)
        dpdt = np.zeros((n,nn,nn),dtype = np.complex64)
        p[0,:,:] = po
        dpdt[0,:,:] = equation(time[0],p[0,:,:].reshape(nn**2,), H(time[0]), tauD).reshape(nn,nn)
        seaqt = lambda t, p : equation(t, p, H(t), tauD)
        p_C = integrate.complex_ode(seaqt)
        # p_C = integrate.ode(seaqt)
        po = np.reshape(po, (nn**2,))    
        p_C.set_initial_value(po, time[0])
        p_C.set_integrator('Isoda',method = 'bdf', rtol = 1e-3)
        # p_C.set_integrator('dopri5',method = 'bdf', rtol = 1e-6)
        # p_C.set_integrator('dopri5',method = 'adams', rtol= 0.001)
        num_vz = {}
        for jj in range(n_qubits):
            num_vz[jj] = 0
        for i, ii in enumerate(time[1:]):
            dt = ii - time[i]
            # print(f"time:{ii}")
            # print(abs(time[i] - vz[num_vz][0]))
            for n_q in range(n_qubits):
                if num_vz[n_q] < len(vz[n_q]):
                    num_vz_q = num_vz[n_q]
                    if (abs(time[i] - vz[n_q][num_vz_q][0]) < 1e-3*dt):
                        z_gate = matrix.expm(-0.5j*vz[n_q][num_vz_q][1]*kron_i(sz,n_q,n_qubits)) #This is wrong but works for the CR gate
                        pC = (z_gate @ p_C.y.reshape((nn,nn))@ dag(z_gate)).reshape(1,nn**2)
                        p_C.set_initial_value(pC, time[i])
                        # p_C.temp = pC.reshape((nn**2,))
                        num_vz[n_q] += 1
                
            p[i+1,:,:] = p_C.integrate(p_C.t + dt).reshape((nn,nn))
            dpdt[i+1,:,:] = equation(ii,p[i+1,:,:].reshape(nn**2,),H(ii),tauD).reshape(nn,nn)
            # if (i % (n//10) == 0):
            #     print('--------------------',str(i))
        return p, dpdt
    
        
class Neepy2(Neepy):
    a = 5
    def __init__(self):
        self.b = 2
        
    def sum_a(self,a):
        return self.a + a