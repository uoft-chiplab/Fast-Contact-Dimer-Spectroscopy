# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:34:41 2025

These are a collection of select figures for drafting the clock shift manuscript.
Typically it pulls data from manuscript_data/ and saves in manuscript_figures/

@author: coldatoms
"""

from scipy.constants import pi
from scipy.optimize import curve_fit
from scipy import interpolate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from plot_settings import paper_settings, bin_data, adjust_lightness
plt.rcParams.update(paper_settings)

fig = plt.figure(layout='constrained', figsize=(3.4,2.8))
gs = GridSpec(2, 2, figure=fig, hspace=0.1)

### =================================== 2 a) ===================================

# PLOT DIMER BINDING ENERGY VS FIELD
Eb_color =  '#48E0B6'
Eb_style= {'color':Eb_color,
				'mec':adjust_lightness(Eb_color, 0.3),
				'mfc':Eb_color,
				'mew':1,
				'marker':'o',
				'markersize':3}

colornaive = '#000000'
colorSqW = '#7570b3'
colorCC = '#e7298a'

# load theory 
Ebs = pd.read_csv("Figures/data/Fig2/Ebs.csv")
SqW = pd.read_csv("Figures/data/Fig2/sqw_theory_line.csv")
Tmat = pd.read_csv("Figures/data/Fig2/t_matrix_theory_line.csv")
CCC = pd.read_csv("Figures/data/Fig2/ac_s_Eb_vs_B_220-225G.dat", header=None, 
				  names=['B','E'], delimiter='\s')
# load data
ExpEbs = pd.read_csv("Figures/data/Fig2/Eb_results.csv")
ExpEbs = ExpEbs.sort_values(by='B')
binx, biny, binxerr, binyerr = bin_data(ExpEbs['B'], ExpEbs['Eb'], 
										xerr=np.ones(len(ExpEbs['B'])), 
										yerr= np.ones(len(ExpEbs['Eb'])), 
										nbins=25)

# plot
ax_a = fig.add_subplot(gs[0, 0])
ax_a.plot(Ebs['B'], Ebs['Ebs_naive'], ls='dotted', color=colornaive, marker='',  label=r'$1/a_{13}^2$')
ax_a.plot(SqW['Magnetic Field (G)'], SqW['Energy (MHz)'], color=colorSqW, marker='', ls = '--')
ax_a.plot(CCC['B'], CCC['E'], marker='', ls='-' , color = colorCC)
ax_a.plot(binx, biny, binyerr, **Eb_style)

xlabel=r'$B$ [G]'
ylabel = r'$\omega_d/2\pi$ (MHz)'
ax_a.set(xlabel=xlabel, ylabel=ylabel,
	xlim=[199, 210],
ylim = [-4.5, -1.5]
	)

### =================================== 2 b) ===================================

scaling = 1000 # y axis scale factor

EF_data_640 = 0.0133
EF_data_10 = 0.0182

# pulse time in us
color640 = '#4093ff'
style640 = {'color':color640,
				'mec':adjust_lightness(color640, 0.3),
				'mfc':color640,
				'mew':1,
				'marker':'o'}
color10 = '#ff5447'
style10 = {'color':color10,
			'mec':adjust_lightness(color10, 0.3),
			'mfc':color10,
			'mew':1,
			'marker':'s'}
msize = 3 # markersize

# 640 us pulse data and fit
dimer_data_640 = pd.read_csv("Figures/data/Fig2/2024-07-17_J_e.dat_sat_corr.csv")
custom_bins_640 = [-4.05,  -4.03, -4.02, -4.01, -4.005, -4, -3.995, -3.990, -3.985, -3.98, -3.97, -3.96, -3.94]
x_dimer_640, y_dimer_640, yerr_dimer_640 = bin_data(dimer_data_640['detuning'], 
										dimer_data_640['c5_scaledtransfer']*scaling, 
										dimer_data_640['em_c5_scaledtransfer']*scaling, 
										nbins=custom_bins_640)

dimer_fit_640 = pd.read_csv("Figures/data/Fig2/fit_2024-07-17_J_e.dat_sat_corr.csv")
dimer_fit_xs_640 = dimer_fit_640['xs']/1e6
dimer_fit_ys_640 = dimer_fit_640['ys']*scaling

# subtract background
y_dimer_640 -= dimer_fit_ys_640.min()
dimer_fit_ys_640 -= dimer_fit_ys_640.min()

# get convolved lineshape
lineshape_data = pd.read_json("Figures/data/Fig2/lineshape_2024-07-17_J_e_backup.json")
lineshape = interpolate.interp1d(*lineshape_data[['x', 'y']].values.T, 
								 'linear', bounds_error=False, fill_value='extrapolate')

def convls(x, A, x0):
	return A*lineshape((x-x0)/EF_data_640)

p0 = [0.00001, -3.98]
bounds = ([0, -600],[1*scaling, 0])

popt, __ = curve_fit(convls, x_dimer_640, y_dimer_640, 
					 sigma=yerr_dimer_640, p0=p0, bounds=bounds)
dimer_fit_xs_640 = np.linspace(-4.3, -3.7,1000)
dimer_fit_ys_640 = convls(dimer_fit_xs_640, *popt)

# load 10 us pulse data and fit
dimer_data_10 = pd.read_csv("Figures/data/Fig2/2024-09-27_B_e.dat_sat_corr.csv")
dimer_data_10 = dimer_data_10.sort_values(by='detuning')

custom_bins_10 = [-4.25,  -4.15, -4.1, -4.05, -4.025, -4, -3.975, -3.95,  -3.9,  -3.85, -3.8, -3.75, -3.7]
x_dimer_10, y_dimer_10, yerr_dimer_10 = bin_data(dimer_data_10['detuning'], 
								dimer_data_10['c5_scaledtransfer']*scaling, 
								dimer_data_10['em_c5_scaledtransfer']*scaling, 
								nbins=custom_bins_10)

dimer_fit_10 = pd.read_csv("Figures/data/Fig2/fit_2024-09-27_B_e.dat_sat_corr.csv")
dimer_fit_x_10 = dimer_fit_10['xs']/1e6
dimer_fit_y_10 = dimer_fit_10['ys'] *scaling

# subtract background offset
y_dimer_10 -= dimer_fit_y_10.min()
dimer_fit_y_10 -= dimer_fit_y_10.min()

# plot
ax_b = fig.add_subplot(gs[0,1])

ax_b.errorbar(x_dimer_640, y_dimer_640, yerr = yerr_dimer_640, **style640, 
				markersize=msize) 
ax_b.plot(dimer_fit_xs_640, dimer_fit_ys_640, color=color640, ls='-')

ax_b.errorbar(x_dimer_10, y_dimer_10, yerr=yerr_dimer_10, **style10, markersize=msize)
ax_b.plot(dimer_fit_x_10, dimer_fit_y_10, color=color10, ls='-')

# format axes and add labels
ax_b.set(xlim=[-4.15, -3.85],
	xlabel=r'$\omega$ [MHz]',
		ylabel=r'$\widetilde{\Gamma} \, \times \, 10^3$',
)
ax_b.set_yticks([0, 2, 4, 6, 8])

ax_b.text(0.2, 0.8, r'$t_\mathrm{rf} \ll \tau_F$', color=color640, 
		  fontsize=7, transform=ax_b.transAxes)
ax_b.text(0.05, 0.25, r'$t_\mathrm{rf} \approx \tau_F$', color=color10, 
		fontsize=7, transform=ax_b.transAxes)	

### =================================== 2 c) ===================================

my_color = '#8D6E63'
my_style = {'color':my_color,
				'mec':adjust_lightness(my_color, 0.3),
				'mfc':my_color,
				'mew':1,
				'marker':'o',
				'markersize':5,
				'ls':'none'}
msize = 5
overall_scaling = 100

# process 10us and 640us transfer points
tauF_640 = 1/EF_data_640/2/pi
trat_640 = 640/tauF_640

GammaPeak640 = dimer_fit_ys_640.max()/scaling
Id640 = GammaPeak640/640/EF_data_640*2 
tcurve = np.linspace(1, 640, 100)
Idcurve = GammaPeak640/tcurve/EF_data_640

tauF_10 = 1/EF_data_10/2/pi
trat_10 = 10/tauF_10

t640 = float(1/trat_640)
I640 = float(Id640) * overall_scaling
I640_std = I640*0.06 

GammaPeak10 = dimer_fit_y_10.max()/scaling
Id10 = GammaPeak10/10/EF_data_10*2 
t10 = float(1/trat_10)
I10 = float(Id10) * overall_scaling
I10_std = I10*0.08 

# load and process dimer transfer data
data = pd.read_csv("Figures/data/Fig2/veryshort_df.csv")

data['scaledtime']=1/data['scaledtime'] # tau_F/t_rf
data[['Id', 'em_Id']] = data[['Id', 'em_Id']] * overall_scaling

# average points with multiple entries, then remove from overall dataset
df = data[data['time'] == 0.003].copy()
t3us = df['scaledtime'].values[0]
I3us = df['Id'].mean()
I3us_std = np.sqrt(df['em_Id'].values[0]**2 + df['em_Id'].values[1]**2)

df = data[data['time'] == 0.020].copy()
t20us = df['scaledtime'].values[0]
I20us = df['Id'].mean()
I20us_std = np.sqrt(df['em_Id'].values[0]**2 + df['em_Id'].values[1]**2)

data = data[data['time']!= 0.020]
data = data[data['time'] > 0.0031]

x_dimer = data['scaledtime']
y_dimer = data['Id'] 
yerr_dimer = data['em_Id'] 

# estimated dimer transfer
scale_est = pd.read_csv("Figures/data/Fig2/time_scaling_estimate_avg_rescaled.csv")

# get average transfer of all points on plot
t_add = [t20us, t640, t10]
I_add = [I20us, I640, I10]
Ierr_add = [I20us_std, I640_std, I10_std]

t_all = pd.concat([x_dimer, pd.Series(t_add)])
I_all = pd.concat([y_dimer, pd.Series(I_add)])
Ierr_all = pd.concat([yerr_dimer, pd.Series(Ierr_add)])

avg_Id = np.mean(I_all.iloc[-4:])
e_avg_Id = np.sqrt(Ierr_all.iloc[-4]**2 + Ierr_all.iloc[-3]**2 + \
				   Ierr_all.iloc[-2]**2 + Ierr_all.iloc[-1]**2)/4

# plot
ax=fig.add_subplot(gs[1,:])

ax.errorbar(x_dimer, y_dimer, yerr_dimer, **my_style)
ax.errorbar([t20us], [I20us], yerr=[I20us_std], **my_style)

ax.errorbar([t640], [I640], yerr=[I640_std], **style640, markersize=msize)
ax.errorbar([t10], [I10], yerr=[I10_std], **style10, markersize=msize)

ax.hlines(avg_Id, 0, t_all.max(), ls='--', color=my_color)
ax.fill_between([0, t_all.max()], avg_Id - e_avg_Id, avg_Id + e_avg_Id, 
				color = adjust_lightness(my_color, 1.5))

ax.plot(scale_est['ftimes'], scale_est['peaks']*overall_scaling,color=color640, ls='-', marker='')

# format axes
ax.text(0.07, 0.0155*overall_scaling, r'$I_d$', color=my_color)
ax.set(xlabel = r'$\tau_F/t_\mathrm{rf}$',
		ylabel=r'$4 \alpha / \Omega_{23}^2  t_\mathrm{rf}^2  \, \times \, 10^2$',
		ylim = [-0.1, 2.5],
		xlim = [-0.1,2.1],
		xscale='linear',
		)
ax.set_xticks([0, 0.5, 1, 1.5, 2])