# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 08:33:01 2025

@author: Chip lab
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.cm

from plot_settings import paper_settings, generate_plt_styles

### Fit functions
def Linear(x,m,b):
	return m*x + b

def Saturation(x, A, x0):
	return A*(1-np.exp(-x/x0))

### =================================== S2 a) ===================================

# plotting options
colors = ['#1b1044', '#812581', '#c03a76', '#f3655c', '#fde0a2']
styles = generate_plt_styles(colors, ts=0.6)

linestyles = ['--', 
			  '--','--','--','--',
			  ':'
			  ]

### plot settings
plt.rcParams.update(paper_settings) 
font_size = paper_settings['legend.fontsize']
fig_width = 3.4 # One-column PRL figure size in inches
		
fig, axs = plt.subplots(1, 2, figsize=(fig_width*6/5, fig_width*3/5)
						)

axs[0].set(xlabel=r'$\Omega_{\mathrm{23}}^2/(2\pi)^2$ [kHz$^2$]', 
		   ylabel=r'Transfer $\alpha_\mathrm{d}$',
		   ylim=[-0.02, .22],
		   )

# loop over dimers

dimer_fits = pd.read_csv("Figures/data/FigS2/dimer_saturation_curves.csv")
dimer_data = ["Figures/data/FigS2/2025-02-13_J_e.dat",
			  "Figures/data/FigS2/2025-02-14_C_e.dat"]
# select data
dimer_fits = dimer_fits.iloc[[0,2]].reset_index()

for i, fit_params in dimer_fits.iterrows():
	
	# convert str to array, ignore [ ]
	popt = np.fromstring(fit_params['popt_b'][1:-1], dtype=float, sep=' ')
	
	# dimer
	df = pd.read_csv(dimer_data[i])
	x = df['OmegaR2']
	y = df['c5_transfer']
	yerr = df['em_c5_transfer']

	# plot
	ax = axs[0]
	xs = np.linspace(0, max(x), 1000)  # linspace of rf powers
	ax.errorbar(x, y, yerr=yerr, **styles[i])
	ax.plot(xs, Saturation(xs, *popt), '-', color=colors[i])
	ax.plot(xs, Linear(xs, popt[0]/popt[1], 0), linestyles[i], color=colors[i])
	
### =================================== S2 b) ===================================

### HFT various ToTF, same detuning ###
# save files 
fit_files = [
		 'Figures/data/FigS2/100kHz_saturation_curves.csv', 
		 'Figures/data/FigS2/near-res_saturation_curves.csv',
		 'Figures/data/FigS2/various_ToTF_saturation_curves.csv',
		 ]
# sorted to match sorted ToTF fit_data files
data_files = [
		"Figures/data/FigS2/2025-02-13_I_e.dat",
		"Figures/data/FigS2/2024-11-29_G_e.dat",
		"Figures/data/FigS2/2024-11-29_G_e_2.dat",
		"Figures/data/FigS2/2025-02-13_O_e.dat",
		"Figures/data/FigS2/2024-11-28_O_e.dat"
		]		

dfs = []
for f in fit_files:
	dfs.append(pd.read_csv(f))

# turn dictionary list into dataframe
fit_data = pd.concat(dfs) #.reset_index()
fit_data = fit_data.loc[(fit_data.detuning == 100)]
fit_data = fit_data.sort_values("ToTF").reset_index()

# chosen average saturation Rabi
OmegaRabi2 = np.mean([716.46, 962.81])

ax = axs[1]

for i, fit in fit_data.iterrows():
	xs = np.linspace(0, 4.05 * OmegaRabi2, 100)
	popt =  np.fromstring(fit.popt[1:-1], dtype=float, sep=' ')
	Gammas_Sat = Saturation(xs, *popt)
	Gammas_Lin = xs * popt[0] / popt[1]
	alpha0 = popt[0]

	# raw data plots
	ax.plot(xs , Gammas_Sat , '-', color=colors[i])
	ax.plot(xs , Gammas_Lin , linestyles[i], color=colors[i])
	
	data = pd.read_csv(data_files[i])
	x = data['OmegaR2']
	y = data['transfer'] 
	yerr = data['em_transfer']

	ax.errorbar(x , y, yerr=yerr, **styles[i])

ax.set(xlabel=r'$\Omega_{\mathrm{23}}^2/(2\pi)^2$ [kHz$^2$]', 
		ylabel=r'Transfer $\alpha_{\mathrm{HFT}}$',
		ylim=[0, .8],
		xlim = [0,2200]
)
cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', colors)
norm = mcolors.Normalize(min(fit_data.ToTF), max(fit_data.ToTF))
sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax)
cbar.set_label(r'$T/T_F$', fontsize=7)

fig.tight_layout()
subplot_labels = ['(a)', '(b)']
for n, ax in enumerate(axs):
	label = subplot_labels[n]
	ax.text(-0.33, 1.05, label, 
		 transform=ax.transAxes, 
		 )
	
plt.subplots_adjust(top=0.9)
