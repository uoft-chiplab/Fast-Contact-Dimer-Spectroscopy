# -*- coding: utf-8 -*-
"""
Created by Chip lab circa 2024-2025

Analysis script for four shot scans: HFT, bg, dimer, bg.
Produces the current Fig. 3 in the clockshift manuscript.

"""
from scipy.constants import pi, hbar, h
from plot_settings import paper_settings, generate_plt_styles, bin_data
from Summary.General.contact_interpolation import contact_interpolation as C_interp
from Summary.General.library import uatom, a0, mK
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
alpha = 0.3
nbins = 16

def a13(B):
	''' ac scattering length '''
	abg = 167.6*a0
	DeltaB = 7.2
	B0 = 224.2
	return abg*(1 - DeltaB/(B-B0))

re = 103 * a0 # ac dimer range estimate
Eb = 3.98 # MHz

# read data
data = pd.read_csv("Figures/data/Fig3/4shot_results_testing.csv")

### ac loss corrections
ToTFs = [0.26, 0.36, 0.6, 1.1]
corr_cs = [1.00, 1.15, 1.31, 1.31]
e_corr_cs = [0.05, 0.06, 0.08, 0.08]

corr_c_interp = lambda x: np.interp(x, np.array(ToTFs), np.array(corr_cs))
e_corr_c_interp = lambda x: np.interp(x, np.array(ToTFs), np.array(e_corr_cs))
	
### EF systematics 
# barnu systematics in thermometry
e_EF_from_barnu_sys = 0.02 # error in EF from barnu is 2%

# imaging systematics in thermometry
e_EF_from_light_saturation_fudge = 0.01

e_EF_from_therm_sys = np.sqrt(e_EF_from_light_saturation_fudge**2 + \
							  e_EF_from_barnu_sys**2)


### THEORY CALCULATIONS ####
# calculate theoretical sum rule and first moment vs contact
Bfield = 202.14
open_channel_fraction = 0.93
sum_rule=0.5

C = np.linspace(0, max(data['C']), 50) 
kF = np.mean(data['kF'])
EF = np.mean(data['EF'])
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
	
### PJ CCC
# spectral weight, clockk shift, first moment
spin_me = 32/42 # spin matrix element
ell_d_CCC = spin_me * 42 * pi
I_d_CCC =  kF / a13kF / pi * ell_d_CCC * a0 * C
just_I_d = I_d_CCC * a13kF
CS_d_CCC = -I_d_CCC *a13kF**2 /kF**2 * kappa**2 * 2
FM_d_CCC = CS_d_CCC * sum_rule

### final dataset manipulations 
# calculate kF a13 for each data point
data['a13kF'] = np.array(data['kF']) * a13(202.14)

### intitialize plots
fig, axs = plt.subplots(2,1, figsize=[fig_width, fig_width*5/5], height_ratios=[0.8,0.9])
axes = axs.flatten()

contact_label = r"Contact,  $C/N k_F$"
spectral_weight_label = r"$I_d/k_Fa_{13}$"
clock_shift_label = r"Clock Shift,  $\tilde\Omega k_Fa_{13}$"
temperature_label = r"Temperature,  $T/T_F$"

### =================================== 3 a) ===================================
### HFT Error bands 
# add all error sources in quadrature
e_kF_therm_sys = e_EF_from_therm_sys/2 # C propto 1/kF = 1/sqrt(EF), check my error propagation

HFT_systematic_factors = [
	 np.mean(data['e_sat_scale_HFT']/data['sat_scale_HFT']),
	 np.mean(data['e_OmegaR_HFT_kHz2']/data['OmegaR_HFT_kHz2']),
	 np.mean(data['e_ff']),
	 e_kF_therm_sys,
	 ]
HFT_error_const = np.sqrt(np.sum(np.array(HFT_systematic_factors)**2))

HFT_error = lambda x: np.sqrt(HFT_error_const**2+e_corr_c_interp(x)**2)


ax = axes[0]
ax.set(ylabel=contact_label, xlabel=temperature_label, xlim=[0.2, 0.85])

# Create inset
inset_colors = ['#1b1044', '#f3655c']
inset_styles = generate_plt_styles(inset_colors, ts=0.6)

inset_ax = fig.add_axes([0.595, 0.795, 0.235, 0.15]) # [left, bottom, width, height]

for i, T in enumerate([306, 616]):
	a_data = pd.read_csv(f"Figures/data/Fig3/saturation_data_0p{T}ToTF.csv")
	a_fit = pd.read_csv(f"Figures/data/Fig3/saturation_fit_0p{T}ToTF.csv")

	inset_ax.plot(a_fit['xs'], a_fit['Gammas_Sat'], '-', color=inset_colors[i])
	inset_ax.plot(a_fit['xs'], a_fit['Gammas_Lin'], ls="--", color=inset_colors[i])
	inset_ax.errorbar(a_data['x'], a_data['y'], yerr=a_data['yerr'], 
				   **inset_styles[i],  markersize=3.5
				   )

# format axes
inset_ax.set(
	ylabel = r'$\alpha$',
	xlabel = r'$\Omega_{23}^2/(2\pi)^2$ [kHz$^2]$',
	ylim = [0, 0.5],
	xlim = [0,1000])
inset_ax.tick_params(labelsize=5)
inset_ax.xaxis.label.set_size(6)
inset_ax.yaxis.label.set_size(7)
inset_ax.set_xticks([400,800])
inset_ax.set_yticks([0.2,0.4])

# plot trap-averaged contact (theory)
xs = np.linspace(min(data['ToTF'])*0.9, max(data['ToTF'])*1.1, 100)
ax.plot(xs, C_interp(xs), '--', color=colors[1], label='trap-averaged theory', zorder=10)
ax.fill_between(xs, C_interp(xs)*(1-HFT_error(xs)), C_interp(xs)*(1+HFT_error(xs)), 
				color=colors[1], alpha=alpha, zorder=10)

# binned contact data
binx, biny, binyerr, binxerr = bin_data(data['ToTF'], data['C'], 
											  data['e_C'], 16, xerr=data['e_ToTF'])
ax.errorbar(binx, biny, yerr=binyerr, xerr=binxerr, label='binned', 
			**styles[1], zorder=10)

### =================================== 3 b) ===================================
sty_i = 0

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

# binned dimer weight data
I_d_conv = 2 # what's this?
data[['SW_c5', 'e_SW_c5']] *= I_d_conv

x = data['C']
xerr = data['e_C']
y = data['SW_c5'] / data['a13kF']
yerr = np.abs(data['e_SW_c5'])/ data['a13kF']

binx, biny, binyerr, binxerr = bin_data(x, y, yerr, nbins, xerr=xerr)
ax.errorbar(binx, biny, yerr=binyerr, xerr=binxerr, label='binned', **styles[0])

# final plot settings
fig.tight_layout()  # note this is done before the labels on purpose
subplot_labels = ['(a)', '(b)']
for n, ax in enumerate(axs):
	label = subplot_labels[n]
	ax.text(-0.18, 1.08, label, transform=ax.transAxes, 
		 )
	
plt.subplots_adjust(top=0.95)
