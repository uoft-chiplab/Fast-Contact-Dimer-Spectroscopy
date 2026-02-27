# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 10:23:06 2025

@author: Chip Lab
"""
from scipy.optimize import curve_fit
from scipy.constants import pi

from plot_settings import paper_settings, generate_plt_styles

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

mycolors = ['#2827D8', '#D82827', '#3C9A34']
mystyles = generate_plt_styles(mycolors)
mpl.rcParams.update(paper_settings)

hbar = 1.0545718e-34  # J*s
h = 2*np.pi*hbar   

def Linear(x,m,b):
	return m*x + b

# x should be Rabi*t
def mySin2(x, Z):
	return Z*np.sin(x/2)**2

#"universal quadratic" form for transfer alpha = Z/4 Omega^2t^2
# x is Omega^2t^2
def UniversalQuad(x, Z):
	return Z/4*x
	
def RabiFreq(x, A, b, x0, C):
	return A*(np.sin(2*np.pi*b/2 * x - x0))**2 + C

labels = ['209 G pol b to c',
		  '209 G mix b to c',
		  '202p14 G mix b to c']

files = ['data/FigS3/2025-07-16_I_e.dat', # 209 G pol b's into c's
		 'data/FigS3/2025-07-18_I_e.dat', # 209 G ab spin mix into c's, only diff 07-16_J is date taken
		 'data/FigS3/2025-07-18_H_e.dat']# 202.14 G ab spin mix into C's but fr this time,
		

pulse_freqs = [48.369,
			   48.369,
			   47.2227,]

file_path = 'data/FigS3/phaseofreq_to_Vpp_VVA_2p3_square.txt'
cal = pd.read_csv(file_path, sep='\t', skiprows=2, names=[ 'Freq','Vpp'])
# normalize by 47 MHz value
cal['relVpp'] = cal['Vpp']/cal.loc[cal['Freq'] == 47]['Vpp'].values[0]
calInterpFreq = lambda x: np.interp(x, cal['Freq'], cal['relVpp'])

file_path = 'data/FigS3/VVAtoVpp_47MHz_squarePhaseO_4GSps_scope.txt'
cal_VVA_PO = pd.read_csv(file_path, sep='\t', skiprows=1, names=['VVA','Vpp'])
cal_x_PO, cal_y_PO = cal_VVA_PO['VVA'], cal_VVA_PO['Vpp']
calInterpVVA_PO = lambda x: np.interp(x, cal_x_PO, cal_y_PO)

file_path = 'data/FigS3/VVAtoVppMicro_43MHz.txt'
cal_VVA_MO = pd.read_csv(file_path, sep='\t', skiprows=1, names=['VVA','Vpp'])
cal_x_MO, cal_y_MO = cal_VVA_MO['VVA'], cal_VVA_MO['Vpp']
calInterpVVA_MO = lambda x: np.interp(x, cal_x_MO, cal_y_MO)

def Vpp_from_VVAfreq(VVA, freq, rfsource="phaseo"):
	''' Returns ... '''
	if rfsource == "phaseo":
		Vpp = calInterpVVA_PO(VVA)*calInterpFreq(freq)
		
	elif rfsource == "micro":
		Vpp = calInterpVVA_MO(VVA)
		
	return Vpp

AC_LOSS_CORR = False
PLOT_RABI_CAL = False
transfer_loss_strs = ['transfer','loss']
fit_func = RabiFreq
ff = 0.82
RabiperVpp_47MHz_July2025 = 12.13/0.452 # 2025-02-12 and slightly modified for July 2025 data
bg_cutoff = 1 # VVA
pulse_time_ms = 0.01
df_list = []
EF = 0.019 # GUESS, MHZ

for i, file in enumerate(files):
	if not PLOT_RABI_CAL and i==3:
		continue
	run = pd.read_csv(file) # , names=['VVA','c5','c9']	
	
	run['c9'] = run['c9'] * ff
	if AC_LOSS_CORR and i > 0 and i < 3: # only apply ac loss to datasets that had a's present
		ac_loss = 1.2 # based on most recent July 2025 ac loss correction for typical ToTF gas
	else:
		ac_loss = 1 # spin pol should have no loss

	bg_data = run[run['VVA'] < bg_cutoff]
	run = run[run['VVA'] > bg_cutoff]
	bg_c5 = bg_data['c5'].mean()
	bg_c9 = bg_data['c9'].mean()

	run['c5'] = run['c5'] * ac_loss
	run['N'] = run['c5'] + run['c9']
	run['alpha_transfer'] = (run['c5'] - bg_c5) / \
		((run['c5']-bg_c5) + run['c9'])
	run['alpha_loss'] = (bg_c9 - run['c9'])/bg_c9

	run['OmegaR'] = Vpp_from_VVAfreq(run['VVA'], pulse_freqs[i]) * \
							RabiperVpp_47MHz_July2025 * 2 * pi
	run['OmegaR2'] = run['OmegaR']**2
	run['time'] = pulse_time_ms
	run['OmegaRt'] = pulse_time_ms * run['OmegaR']
	run['OmegaR2t2'] = pulse_time_ms**2 * run['OmegaR']**2
	
	xname = 'OmegaRt'
	for string in transfer_loss_strs:
		yname = 'alpha_' + string
		guess = [1,0.15,0,0]
		popt, pcov = curve_fit(fit_func, run[xname], run[yname], p0=guess)
		run['A_' + string] = popt[0]

	run['scaled_time'] = run['time']/1e3 * h*EF*1e6/hbar
	run['scaled_alpha_transfer'] = run['alpha_transfer']*(h*EF*1e6/hbar/(run['OmegaR']*1e3))**2
	df_list.append(run)

#################################################
##################### let's make finalized plots
##################################################

# average data, plot and fit to linear response under a cutoff ~0.1
fig, ax = plt.subplots(figsize=(3,2))
err_type = 'std'
x_max = 30

#### SUBPLOT SETTINGS
# plot alpha_transfer vs Rabi^2time^2
ax.set(xlabel=r'$\Omega_{23}^2 t^2$', ylabel=r'Transfer $\alpha_\mathrm{res}$', ylim=[-0.02, 1],
		xlim=[-0.02, x_max]
		)

# inset axis
inset_ax = fig.add_axes([0.42, 0.4, 0.2, 0.2])
inset_ax.set(xlim = [4, 17.5],
			 ylim=[0.6, 1.0],
			 xlabel = r'$\Omega_{23}^2t^2$',
			 ylabel=r'Loss $\alpha_\mathrm{res}$')

# actual plotting and fitting loop
fit_func = mySin2
for i, df in enumerate(df_list):
	# TRANSFER
	xname = 'OmegaR2t2'
	yname = 'alpha_transfer'
	dfgrp = df.groupby(xname).agg(['mean', 'std','sem']).reset_index()
	ax.errorbar(dfgrp[xname], dfgrp[yname]['mean'], yerr=dfgrp[yname][err_type], **mystyles[i])
	cutoff_mask = (dfgrp[xname] < x_max) & (dfgrp[yname]['mean']>0.01)
	sub_df = dfgrp[cutoff_mask]
	guess = [1]
	popt, pcov = curve_fit(fit_func, np.sqrt(sub_df[xname]), sub_df[yname]['mean'],
						sigma=sub_df[yname][err_type])
	perr = np.sqrt(np.diag(pcov))
	print(f'i={i}, Z={popt[0]:.2f}+/-{perr[0]:.2f}')
	xs = np.linspace(0, x_max, 100)
	ax.plot(xs, fit_func(np.sqrt(xs), *popt), '-', color=mycolors[i], label=rf'$Z=${popt[0]:.2f}({round(perr[0]*100)})')
	ax.plot(xs, UniversalQuad(xs, *popt), '--', color=mycolors[i])

	# LOSS
	xname = 'OmegaR2t2'
	yname = 'alpha_loss'
	dfgrp = df.groupby(xname).agg(['mean', 'std','sem']).reset_index()

	cutoff_mask = (dfgrp[yname]['mean'] > 0.02) & (dfgrp[xname]<x_max)
	sub_df = dfgrp[cutoff_mask]
	guess = [1]
	popt, pcov = curve_fit(fit_func, np.sqrt(sub_df[xname]), sub_df[yname]['mean'],
						sigma=sub_df[yname][err_type])
	xs = np.linspace(0, sub_df[xname].max(), 100)
	perr = np.sqrt(np.diag(pcov))

	# repeat but zoomed in
	inset_ax.errorbar(dfgrp[xname], dfgrp[yname]['mean'], yerr=dfgrp[yname][err_type], label=labels[i]+', loss', **mystyles[i])
	inset_ax.plot(xs, fit_func(np.sqrt(xs), *popt), '-', color=mycolors[i])

ax.legend(frameon=False, loc='upper right')
fig.tight_layout()
# plt.savefig(r'\\UNOBTAINIUM\Carmen_Sandiego\Analysis Scripts\analysis\clockshift\manuscript\manuscript_figures\SM_Zfactors.pdf', dpi=300)
plt.show()