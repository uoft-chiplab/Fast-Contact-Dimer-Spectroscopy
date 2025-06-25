# -*- coding: utf-8 -*-
"""
2023-09-25
@author: Chip Lab

Functions to call in analysis scripts
"""
import os
current_dir = os.path.dirname(__file__)

from scipy.constants import pi, hbar, h, k as kB
from scipy.integrate import simpson, cumulative_trapezoid
from scipy.optimize import fsolve
import numpy as np

import pandas as pd

# TODO: add units and docstrings
uatom = 1.660538921E-27
a0 = 5.2917721092E-11
uB = 9.27400915E-24
gS = 2.0023193043622
gJ = gS
mK = 39.96399848 * uatom
ahf = -h * 285.7308E6 # For groundstate 
gI = 0.000176490 # total nuclear g-factor

def VVAtoVppInterpolation(file):
	"""Returns interpolation function based on VVA to Vpp file."""
	VVAs, Vpps = np.loadtxt(file, unpack=True)
	interp_func = lambda x: np.interp(x, VVAs, Vpps)
	return interp_func
		 
def quotient_propagation(f, A, B, sA, sB, sAB):
	return f* (sA**2/A**2 + sB**2/B**2 - 2*sAB/A/B)**(1/2)

def OmegaRcalibration():
	"""
	Returns function that interpolates the recent calibration from 
	VVAtoVpp.txt which should be in the root of the analysis folder.
	Input of function is VVA, output is OmegaR in kHz.
	"""
	try: 
		VVAtoVppfile = os.path.join("VVAtoVpp.txt") # calibration file
	except:
		FileNotFoundError("VVAtoVpp.txt not found. Check CWD or that file exists.")
	VVAs, Vpps = np.loadtxt(VVAtoVppfile, unpack=True)
	VpptoOmegaR = 27.5833 # kHz
	OmegaR_interp = lambda x: VpptoOmegaR*np.interp(x, VVAs, Vpps)
	
	return OmegaR_interp

def ChipBlackman(x, a_n=[0.42659, 0.49656, 0.076849]):
	"""The ChipLab Blackman that exists in the pulse generation 
	MatLab script. Coefficients slightly differ from conventional.
	Defined as a pulse with length 1 starting at 0."""
	zero_func = lambda y: 0
	pulse_func = lambda y: a_n[0] - a_n[1]*np.cos(2*np.pi*y) \
		+ a_n[2]*np.cos(4*np.pi*y)
	return np.piecewise(x, [x<0, x>1, (x>=0) & (x<=1)], 
					 [zero_func, zero_func, pulse_func])

def chi_sq(y, yfit, yerr, dof):
	return 1/dof * np.sum((np.array(y) - np.array(yfit))**2/(yerr**2))

def deBroglie(T):
	return h/np.sqrt(2*pi*mK*kB*T)

def EhfFieldInTesla(B, F, mF):
	term1 = -ahf/4 + gI * uB * mF * B
	term2 = (2*(gJ - gI)*uB *B /ahf/9)
	term3 = (-1)**(F-1/2) *9 *ahf/4 *np.sqrt(1+4*mF/9 * term2 + term2**2)
	return term1 + term3
	
def Ehf(B, F, mF):
	return EhfFieldInTesla(1E-4 *B, F, mF)

def FreqMHz(B, F1, mF1, F2, mF2):
  return 1E-6 *( Ehf(B, F1, mF1) - Ehf(B, F2, mF2))/h

def FermiEnergy(n, w):
	return hbar * w * (6 * n)**(1/3)

def GammaTilde(transfer, EF, OmegaR, trf):
	return EF/(hbar * pi * OmegaR**2 * trf) * transfer

def FirstMoment(data):
	"""
	integrated with simpsonons rule
	"""
	return [np.trapz(data[:,1]*data[:,0], x=data[:,0]), 
		 cumulative_trapezoid(data[:,1]*data[:,0], x=data[:,0])[-1],
		 simpson(data[:,1]*data[:,0], x=data[:,0])]

def BlackmanFourier2(omega):
	A = 1060.9629086785837
	B = -3.5209670498557566
	C = 0.002744323946881455
	D = 6234.181826176155
	E = -197.39208802178717
	return np.abs((2 *np.sin(omega/2) * (A+B*omega**2 + C*omega**4))/ \
		(D*omega + E*omega**3 + omega**5))**2

