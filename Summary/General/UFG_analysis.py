# compute trap averaged bulk viscosity of unitary Fermi gas
# given \mu/T, trap \bar\omega/T and the drive frequency \omega/T
# (all quantities E/h in units of Hz or lengths in units of the thermal length lambda_T)
#
# (c) LW Enss 2024
#

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import root_scalar
from library import pi, mK, hbar
from baryrat import BarycentricRational
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(current_dir)
theory_path = os.path.join(parentdir, 'theory')
#
# properties of homogeneous (bulk) gas
#

# from contact_tabulated.py
# Tabulated by eye from Vale paper and 
# T. Enss, R.Haussmann, W. Zwerger,
# Ann.of Phys. 326, 3,2011,770-796,
df = pd.read_csv(theory_path + "\\luttward-thermodyn.txt",skiprows=4,sep=' ')
# contact density c/(k_F n) = C/k_F^4 * (3 pi^2)
ContactInterpolation = lambda x: np.interp(x, df['T/T_F'], df['C/k_F^4']* 3*np.pi**2)


# from contact_interpolation.py
ToTFs = np.linspace(0.2, 1.2, 29)
Cs = np.array([2.1783244 , 2.02855915, 1.90539411, 1.78810748, 1.67342239,
       1.56085269, 1.45088512, 1.34443262, 1.24180172, 1.14444157,
       1.05270978, 0.96701767, 0.88750127, 0.81417138, 0.74688656,
       0.68537897, 0.62934843, 0.57842509, 0.53227651, 0.49047896,
       0.45264838, 0.41839089, 0.38739359, 0.35932629, 0.33388573,
       0.31080807, 0.28985089, 0.27077148, 0.25339308])
contact_interpolation = lambda x: np.interp(x, ToTFs, Cs)


eosfit = {'nodes': np.array([5.45981500e+01, 3.35462628e-04, 4.48168907e+00, 1.28402542e+00]), 
		  'values': np.array([2.66603452e+01, 3.35574145e-04, 5.63725236e+00, 1.91237718e+00]), 
		  'weights': np.array([ 0.52786226, -0.10489219, -0.69208542,  0.48101646])}
eosrat = BarycentricRational(eosfit['nodes'],eosfit['values'],eosfit['weights'])

eps = 1e-4

def eos_ufg(betamu):
	"""EOS of unitary gas: phase space density f_n(beta*mu) for both spin components (Zwierlein data)"""
	z = np.exp(betamu)
	f_n = 2*np.where(betamu<-8,z,eosrat(z)) # approximant is for a single spin component, so multiply by 2
	return f_n

def Theta(betamu):
	return (4*pi)/((3*pi**2)* eos_ufg(betamu))**(2/3)

def mutrap_est(ToTF):
	a = -50e3
	b = 21e3
	return a*ToTF + b

def weight_harmonic(v,betabaromega):
	"""area of equipotential surface of potential value V/T=v=0...inf"""
	return 2/(betabaromega**3)*np.sqrt(v/np.pi)

def number_per_spin(betamu,betabaromega,weight_func):
	"""compute number of particles per spin state for trapped unitary gas:
	   N_sigma = int_0^infty dv w(v) f_n_sigma*lambda^3(mu-v)"""
	N_sigma, __ = quad(lambda v: weight_func(v,betabaromega)*eos_ufg(betamu-v)/2,0,np.inf,epsrel=eps)
	return N_sigma

def Epot_trap(betamu,betabaromega,weight_func):
	"""compute trapping potential energy (in units of T):
	   E_trap = int_0^infty dv w(v) f_n*lambda^3(mu-v) v"""
	Epot, __ = quad(lambda v: weight_func(v,betabaromega)*eos_ufg(betamu-v)*v,0,np.inf,epsrel=eps)
	return Epot

def thermo_trap(T,betamu,betabaromega,weight_func):
	"""compute thermodynamics of trapped gas"""
	Ns = number_per_spin(betamu,betabaromega,weight_func)
	EF = T*betabaromega*(6*Ns)**(1/3) # in Hz, without 2pi
	Theta = T/EF
	Epot = T*Epot_trap(betamu,betabaromega,weight_func) # in Hz, without 2pi
	return Ns,EF,Theta,Epot

def find_betamu(T, ToTF, betabaromega, weight_func, guess=None):
	"""solves for betamu that matches T, EF and betabaromega of trap"""
	sol = root_scalar(lambda x: T/ToTF - T*betabaromega*(6*number_per_spin(x, 
				 betabaromega, weight_func))**(1/3), bracket=[20e3/T, -300e3/T], x0=guess)
	return sol.root, sol.iterations

def C_trap(betamu, betabaromega,weight_func):
	"""compute Contact Density averaged over the trap"""
	Ctrap, __ = quad(lambda v: weight_func(v,betabaromega)*eos_ufg(betamu-v)**(4/3)*ContactInterpolation(Theta(betamu-v)),0,np.inf,epsrel=eps)
	return Ctrap

def calc_contact(ToTF, EF, barnu, mutrap_guess=None):
	""" Calculates the harminic trap-averaged contact using ToTF, EF, barnu
		(geometric mean trap freq) and an optional guess mu. Returns the contact."""
	T = ToTF * EF
	betabaromega = barnu/T
	lambda_T = np.sqrt(hbar/(mK*T))
	kF = np.sqrt(4*pi*mK*EF/hbar) # global k_F, i.e. peak k_F
	
	# find mu
	if mutrap_guess:
		betamutrap_guess = mutrap_guess/T
	else:
		betamutrap_guess = mutrap_est(ToTF)
	betamutrap, __ = find_betamu(T, ToTF, betabaromega, weight_harmonic, 
								   guess=betamutrap_guess)

	# calculate thermodynamics
	Ns, EF, Theta, __ = thermo_trap(T,betamutrap,betabaromega,weight_harmonic)
	
	# calculate C
	Ctrap =  C_trap(betamutrap, betabaromega, weight_harmonic)/(kF*lambda_T)* \
				(3*pi**2)**(1/3)/Ns/2
				
	return Ctrap, Ns, EF, Theta