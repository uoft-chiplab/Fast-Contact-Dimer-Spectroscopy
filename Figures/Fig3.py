# -*- coding: utf-8 -*-
"""
Created by Chip lab circa 2024-2025

Analysis script for four shot scans: HFT, bg, dimer, bg.
Produces the current Fig. 3 in the clockshift manuscript.

"""
# paths
from scipy.constants import pi, hbar, h
from plot_settings import paper_settings, generate_plt_styles
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from warnings import filterwarnings	
filterwarnings('ignore')

# plotting options
colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']
styles = generate_plt_styles(colors, ts=0.6)
### plot settings
plt.rcParams.update(paper_settings) 
font_size = paper_settings['legend.fontsize']
fig_width = 3.4 # One-column PRL figure size in inches

uatom = 1.660538921E-27
a0 = 5.2917721092E-11
mK = 39.96399848 * uatom

def a13(B):
	''' ac scattering length '''
	abg = 167.6*a0
	DeltaB = 7.2
	B0 = 224.2
	return abg*(1 - DeltaB/(B-B0))

def min_avg_max(my_array):
	return (np.min(my_array), np.mean(my_array), np.max(my_array))

def xstar(B, EF):
	return Eb/EF # hbar**2/mK/a13(B)**2 * (1-re/a13(Bfield))**(-1)

re = 103 * a0 # ac dimer range estimate
Eb = 3.98 # MHz

### Script options
Talk = False
Plot_HFT_Data = True

### Analysis options
Filter_Low_Atom_Number_Shots = True
Final_State_Correction = True
Correct_ac_Loss = True

# select spin state simaging for analysis
state_selection = '97'
spins = ['c5', 'c9', 'ratio95']

### Calibrations
RabiperVpp_47MHz_2024 = 17.05/0.728 
e_RabiperVpp_47MHz_2024 = 0.15

RabiperVpp_43MHz_2024 = 14.44/0.656 
e_RabiperVpp_43MHz_2024 = 0.14

RabiperVpp_47MHz_2025 = 12.01/0.452 
e_RabiperVpp_47MHz_2025 = 0.28

RabiperVpp_43MHz_2025 = RabiperVpp_43MHz_2024 * RabiperVpp_47MHz_2025/ \
													RabiperVpp_47MHz_2024
e_RabiperVpp_43MHz_2025 = RabiperVpp_43MHz_2025 * np.sqrt(\
		  (e_RabiperVpp_43MHz_2024/RabiperVpp_43MHz_2024)**2 + \
			  (e_RabiperVpp_47MHz_2025/RabiperVpp_47MHz_2025)**2 +
			  (e_RabiperVpp_47MHz_2024/RabiperVpp_47MHz_2024)**2)
# this is about 0.35
dimer_x0 = 5211 
e_dimer_x0 = 216

HFT_x0 = 924
e_HFT_x0 = 93
HFT_x0_cold = HFT_x0
e_HFT_x0_cold = e_HFT_x0

def saturation_scale(x, x0):
	""" x is OmegaR^2 and x0 is fit 1/e Omega_R^2 """
	return x/x0*1/(1-np.exp(-x/x0))

### ac loss corrections
ToTFs = [0.26, 0.36, 0.6, 1.1]
corr_cs = [1.00, 1.15, 1.31, 1.31]
e_corr_cs = [0.05, 0.06, 0.08, 0.08]

corr_c_interp = lambda x: np.interp(x, np.array(ToTFs), np.array(corr_cs))
e_corr_c_interp = lambda x: np.interp(x, np.array(ToTFs), np.array(e_corr_cs))
	
### constants
re = 103 * a0 # ac dimer range estimate
Eb = 3.98 # MHz

### Summary plot lists
results_list = []
results_list = pd.read_csv("Figures/data/Fig3/4shot_results_testing.csv")

# convert results into dataframe
df_total = pd.DataFrame(results_list)

# 0 is min, avg is 1, max is 2
min_avg_max_choice = 1

### EF systematics 
# barnu systematics in thermometry
e_EF_from_barnu_sys = 0.02 # error in EF from barnu is 2%

# imaging systematics in thermometry
e_EF_from_light_saturation_fudge = 0.01

e_EF_from_therm_sys = np.sqrt(e_EF_from_light_saturation_fudge**2 + \
							  e_EF_from_barnu_sys**2)

	
### Transfer rate systematics

# this is where EF errors come in
e_kF_therm_sys = e_EF_from_therm_sys/2 # C propto 1/kF = 1/sqrt(EF), check my error propagation

dimer_systematic_labels = ["saturation_correction", "Omega_R", "EF_therm"]

dimer_systematic_factors = [
	 min_avg_max(df_total['e_sat_scale_dimer']/df_total['sat_scale_dimer'])[min_avg_max_choice],
	 min_avg_max(df_total['e_OmegaR_dimer_kHz2']/df_total['OmegaR_dimer_kHz2'])[min_avg_max_choice],
# 	 e_kF_therm_sys, # don't double count, because it's in HFT systematic below
	 ]

dimer_systematics = dict(zip(dimer_systematic_labels, dimer_systematic_factors))

HFT_systematic_labels = ["saturation_correction", "Omega_R", "ac_loss", 
						 "fudge_factor", "EF_therm"]

HFT_systematic_factors = [
	 min_avg_max(df_total['e_sat_scale_HFT']/df_total['sat_scale_HFT'])[min_avg_max_choice],
	 min_avg_max(df_total['e_OmegaR_HFT_kHz2']/df_total['OmegaR_HFT_kHz2'])[min_avg_max_choice],
	 min_avg_max(df_total['e_ff'])[min_avg_max_choice],
	 e_kF_therm_sys,
	 ]

HFT_systematics = dict(zip(HFT_systematic_labels, HFT_systematic_factors))

### Temperature dependent factors

# invert C_interp
ToTF_list = np.linspace(0.2, 1.0, 100)
C_list = np.array(C_interp(ToTF_list))
ToTF_interp = lambda x: np.interp(x, C_list, ToTF_list)

# now make ac_loss correction a function of C
e_corr_c_interp_fn_C = lambda x: e_corr_c_interp(ToTF_interp(x))
		
###############################
###### Summary Plotting #######
###############################	

# plotting options
dimertype2024 = 'c5'
dimertype2025 = 'c5'
plot_options = {
				"Loss Contact": False,
				"Binned": True,
				"not Binned": False,
				"plot_fits": False,
				"CS_pred": True,
				}

true_options = []
for key, val in plot_options.items():
	if val:
		true_options.append(key)
			
title_end = ' , '.join(true_options)
if title_end != '':
	title_end = " with " + title_end
plot_title = "Four shot analysis" + title_end

# choose contact
if plot_options['Loss Contact'] == False:
	df_total['C_data'] = df_total['C']
	df_total['e_C_data'] = df_total['e_C']
else:
	df_total['C_data'] = df_total['C_loss']
	df_total['e_C_data'] = df_total['e_C_loss']
	
sum_rule = 0.5
I_d_conv = 2
# correct data for sumrule

# make spectral weight a fraction out of 1
df_total['SW_c5'] = df_total['SW_c5'] * I_d_conv
df_total['SW_c9'] = df_total['SW_c9'] * I_d_conv
df_total['e_SW_c5'] = df_total['e_SW_c5'] * I_d_conv
df_total['e_SW_c9'] = df_total['e_SW_c9'] * I_d_conv

df_total['FM_c5'] = df_total['FM_c5']
df_total['FM_c9'] = df_total['FM_c9']
df_total['e_FM_c5'] = df_total['e_FM_c5']
df_total['e_FM_c9'] = df_total['e_FM_c9']

df_total['CS_c5'] = df_total['FM_c5'] / sum_rule
df_total['CS_c9'] = df_total['FM_c9'] / sum_rule
df_total['e_CS_c5'] = df_total['e_FM_c5'] / sum_rule
df_total['e_CS_c9'] = df_total['e_FM_c9'] / sum_rule


### THEORY CALCULATIONS ####
# calculate theoretical sum rule and first moment vs contact
Bfield = 202.14
open_channel_fraction = 0.93

C = np.linspace(0, max(df_total['C_data']), 50) 
kF = np.mean(df_total['kF'])
EF = np.mean(df_total['EF'])
xstar = xstar(Bfield, EF)
a13kF = kF * a13(202.14)
kappa = np.sqrt((Eb*h*10**6) *mK/hbar**2) # convert Eb back to kappa

### ZY single-channel square well w/ effective range
# divide I_d by a13 kF,
not_small_kappa_correction = 1.08
ell_d_SqW = 1/(kappa * (1 + re/a13(Bfield))) * open_channel_fraction  * not_small_kappa_correction / a0
ell_d_SqW = 160 # most up to date calculation
I_d_SqW = kF * C/pi * ell_d_SqW * a0 / a13kF
# I_d_SqW = C/a13kF * kF * 1/(pi*kappa) / (1 + re*kappa)
I_d_ZR = C/pi * open_channel_fraction


# compute clock shift
#CS_d = sum_rule*-2*kappa/(pi*kF) * (1/1+re/a13(Bfield)) * C 
CS_d_SqW = -I_d_SqW * a13kF**2 * 2 * (kappa/kF)**2 # convert I_d to CS_d to avoid rewriting sum_rule and o_c_f
# multiply FM (Eq. 7) by a13 kF
FM_d_SqW =  CS_d_SqW * sum_rule
	
### PJ CCC
# spectral weight, clockk shift, first moment
spin_me = 32/42 # spin matrix element
ell_d_CCC = spin_me * 42 * pi
I_d_CCC =  kF / a13kF / pi * ell_d_CCC * a0 * C
just_I_d = I_d_CCC * a13kF
CS_d_CCC = -I_d_CCC *a13kF**2 /kF**2 * kappa**2 * 2
FM_d_CCC = CS_d_CCC * sum_rule

### Other analytical models for bounding
I_d_max =  kF * 1/pi * a13(Bfield) * C / a13kF  # shallow bound state 
I_d_min =  kF * 1/(pi*kappa) * 1/(1+re*kappa) * C/a13kF  # another version of square well with eff range
CS_d_max = -I_d_max * a13kF**2 /kF**2 * kappa**2
CS_d_min = -I_d_min * a13kF**2 /kF**2 * kappa**2
FM_d_max = CS_d_max * sum_rule
FM_d_min = CS_d_min * sum_rule

### Choose Model for clock shift plot
FM_d = FM_d_CCC
CS_d = CS_d_CCC

### HFT clockshifts
FM_HFT = C /(2 * pi) * kappa/kF * a13kF
CS_HFT = FM_HFT / sum_rule

### final dataset manipulations 
# calculate kF a13 for each data point
df_total['a13kF'] = np.array(df_total['kF']) * a13(202.14)
	
# split df
dfs = []
labels = ['2024', '2025']
for label in labels:
	dfs.append(df_total.loc[df_total.year==label])
	
		
### Error bands 
# add all error sources in quadrature
HFT_error_const = np.sqrt(np.sum(np.array(HFT_systematic_factors)**2))
print(f"The constant HFT systematic error band is {HFT_error_const:.2f}")

# add ToTF/C dependent factors
HFT_error_fn_C = lambda x: np.sqrt(HFT_error_const**2+e_corr_c_interp_fn_C(x)**2)

HFT_error = lambda x: np.sqrt(HFT_error_const**2+e_corr_c_interp(x)**2)

# same for dimer, but no T depedent factors
dimer_error = np.sqrt(np.sum(np.array(dimer_systematic_factors)**2))
print(f"Dimer systematic error band is {dimer_error:.2f}")

# for the spectral weight vs. contact plots, SW propto C, so any error in C
# should be propagated
SWvC_error = lambda x: np.sqrt(HFT_error_fn_C(x)**2 + dimer_error**2)
# on second thought, maybe not, since that isn't really an error in the theory
SWvC_error = lambda x: dimer_error**2

alpha = 0.3
sty_i = 0

### intitialize plots
fig, axs = plt.subplots(2,1, figsize=[fig_width, fig_width*5/5], height_ratios=[0.8,0.9])
axes = axs.flatten()

contact_label = r"Contact,  $C/N k_F$"
spectral_weight_label = r"$I_d/k_Fa_{13}$"
clock_shift_label = r"Clock Shift,  $\tilde\Omega k_Fa_{13}$"
temperature_label = r"Temperature,  $T/T_F$"

#-- spectral weight vs. C
ax = axes[1]
ax.set(xlabel=contact_label, ylabel=spectral_weight_label, 
	   xlim=[0, C.max()+0.05], ylim=[0, 0.4])

i = 0

# theory curves
ax.plot(C, I_d_ZR, 'k:'
		, label='zero range'
		)
ax.plot(C, I_d_SqW, '--', color=colors[sty_i+2], label='SqW')
ax.plot(C, I_d_CCC, '-', color=colors[sty_i+3], label='CC')
ax2 = ax.twinx()
ax2.plot(C,just_I_d, marker='')
ax2.set_ylabel(r'Dimer weight, $I_d$')
ax2.set_yticks([0, 0.02, 0.04])
ax2.set_yticklabels(['0', '0.02', '0.04'])
ax2.set_ylim([0, 0.05])
ax.legend(frameon=False, loc='lower right')
