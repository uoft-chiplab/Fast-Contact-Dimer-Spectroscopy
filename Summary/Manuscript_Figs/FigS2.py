# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 08:33:01 2025

@author: Chip lab
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import matplotlib.colors as mcolors
import matplotlib.cm
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
analysis_dir = os.path.dirname(parent_dir)
data_path = os.path.join(parent_dir, 'Data')
general_files_path = os.path.join(parent_dir, 'General')
# Add the parent directory to sys.path
if general_files_path not in sys.path:
	sys.path.append(general_files_path)
from plot_settings import paper_settings, generate_plt_styles
import pickle as pkl
import pandas as pd

dimer_data_path = os.path.join(analysis_dir, 'clockshift\\rf_saturation_analysis\\saturation_data')

### Fit functions
def Linear(x,m,b):
	return m*x + b

def Saturation(x, A, x0):
	return A*(1-np.exp(-x/x0))

# plotting options
colors = ['#1b1044', '#812581', '#c03a76', '#f3655c', '#fde0a2']
colors = [
	colors[0],
	colors[1],
	colors[2],
	colors[3],
	colors[4]
]
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
dimer_file = 'dimer_saturation_curves.pkl'

with open(os.path.join(dimer_data_path, dimer_file), 'rb') as f:
	    dimer_data = pkl.load(f)
		
# select data
dimer_data = [dimer_data[0], dimer_data[2]]
		
# chosen average saturation Rabi
OmegaRabi2 = 3272
e_OmegaRabi2 = 136

for i, data in enumerate(dimer_data):
	df = data['df']
	
	popt = data['popt_b']
	pcovd = data['pcov_b']
	data['e_ToTF'] = data['ToTF']*0.03
	alpha0d = popt[0]
	# dimer
	sty = styles[i]
	color = colors[i]
	label = r'$T/T_F$ = {:.2f}({:.0f})'.format(data['ToTF'], 1e2*data['e_ToTF'])
	label = r'{:.2f}({:.0f}) $T_F$'.format(data['ToTF'], data['e_ToTF']*1e2)
	x = df['OmegaR2']
	y = df['c5_transfer']
	yerr = df['em_c5_transfer']
	xs = np.linspace(0, max(x), 1000)  # linspace of rf powers
	ax = axs[0]
	ax.errorbar(x, y, yerr=yerr, label=label, **sty)
	ax.plot(xs, Saturation(xs, *popt), '-', color=color)
	ax.plot(xs, Linear(xs, popt[0]/popt[1], 0), linestyles[i], 
	  color=color)
	

### HFT various ToTF, same detuning ###
# save files 
files = [
		 '100kHz_saturation_curves.pkl', 
		 'near-res_saturation_curves.pkl',
		 'various_ToTF_saturation_curves.pkl',
		 ]

loaded_data = []

# grab dictionary lists from pickle files
for i, file in enumerate(files):
	with open(os.path.join(data_path, file), "rb") as input_file:
		loaded_data = loaded_data + pkl.load(input_file)
		
# turn dictionary list into dataframe
df = pd.DataFrame(loaded_data)
sub_df = df.loc[(df.detuning == 100)]

ToTFs = sub_df.ToTF.unique()
ToTFs.sort()
ToTFs = [
	ToTFs[0],
		  ToTFs[1],
		 ToTFs[2], 
		 ToTFs[3],
		  ToTFs[4]
		  ]

# chosen average saturation Rabi
OmegaRabi2 = np.mean([716.46, 962.81])

ax = axs[1]

ax.set(xlabel=r'$\Omega_{\mathrm{23}}^2/(2\pi)^2$ [kHz$^2$]', 
		ylabel=r'Transfer $\alpha_{\mathrm{HFT}}$',
		ylim=[0, .8],
		xlim = [0,2200]
)
plot_data = []
for i, ToTF in enumerate(ToTFs):
	sty = styles[i]
	color = colors[i]
	e_ToTF = ToTF * 0.03
	label = r'{:.2f}({:.0f}) $T_F$'.format(ToTF, e_ToTF * 1e2)

	data = sub_df.loc[sub_df.ToTF == ToTF].squeeze()
	xs = np.linspace(0, 4.05 * OmegaRabi2, 100)
	popt = data.popt
	Gammas_Sat = Saturation(xs, *popt)
	Gammas_Lin = xs * popt[0] / popt[1]
	alpha0 = popt[0]

	# raw data plots
	ax.plot(xs , Gammas_Sat , '-', color=color)
	ax.plot(xs , Gammas_Lin , linestyles[i], color=color)
	
	x = data.df['OmegaR2']
	y = data.df['transfer'] 
	yerr = data.df['em_transfer']

	plot_data.append({
		'ToTF': ToTF,
		'label': label,
		'color': color,
		'style': sty,
		'linestyle': linestyles[i],
		'xs': xs.tolist(),
		'Gammas_Sat': Gammas_Sat.tolist(),
		'Gammas_Lin': Gammas_Lin.tolist(),
		'x': x.tolist(),
		'y': y.tolist(),
		'yerr': yerr.tolist()
	})
	ax.errorbar(x , y, yerr=yerr, **sty, label=label)


pklpath = os.path.join(data_path,"saturation_plot_data.pkl")

with open(pklpath, "wb") as f:
	pkl.dump(plot_data, f)	

colors_list = colors
# Save only what you need to rebuild sm
sm_config = {
    'colors': colors_list,  # list of hex or RGB colors you used
    'vmin': min(ToTFs),
    'vmax': max(ToTFs)
}

pklsmpath = os.path.join(data_path,"sm_config.pkl")

with open(pklsmpath, "wb") as f:
    pkl.dump(sm_config, f)
cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', colors)
norm = mcolors.Normalize(min(ToTFs), max(ToTFs))
sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, 
)
cbar.set_label(r'$T/T_F$', fontsize=7)

fig.tight_layout()
subplot_labels = ['(a)', '(b)']
for n, ax in enumerate(axs):
	label = subplot_labels[n]
	ax.text(-0.33, 1.05, label, 
		 transform=ax.transAxes, 
		 )
	
plt.subplots_adjust(top=0.9)
plt.show()
