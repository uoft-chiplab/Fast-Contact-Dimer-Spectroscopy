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
from scipy.interpolate import UnivariateSpline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
general_files_path = os.path.join(parent_dir, 'General')
if general_files_path not in sys.path:
	sys.path.append(general_files_path)

from plot_settings import *
plt.rcParams.update(paper_settings)

data_path = os.path.join(current_dir, 'manuscript_data')

# data binning
def bin_data(x, y, yerr, nbins, xerr=None):

	if np.any(yerr == 0):
		avg_nonzero_yerr = np.mean(yerr[yerr>0])
		yerr[yerr==0] = avg_nonzero_yerr

	n, _ = np.histogram(x, bins=nbins)
	sy, _ = np.histogram(x, bins=nbins, weights=y/(yerr*yerr))
	syerr2, _ = np.histogram(x, bins=nbins, weights=1/(yerr*yerr))
	sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
	mean = sy / syerr2
	sem = np.sqrt(sy2/n - mean*mean)/np.sqrt(n)
	e_mean = 1/np.sqrt(syerr2)
	xbins = (_[1:] + _[:-1])/2 # mid points between bin edges
	
	# set error as yerr if n=1 for bin
	for i, num_in_bin in enumerate(n):
		if num_in_bin == 1:
			for j in range(len(y)):
				if mean[i] == y[j]:
					sem[i] += yerr[j]
					e_mean[i] = yerr[j]
					xbins[i] = x[j]
					break
		else:
			continue
		
	# average xerr
	if xerr is not None:
		sxerr, _ = np.histogram(x, bins=nbins, weights=xerr)
		mean_xerr = sxerr / n
		return xbins, mean, e_mean, mean_xerr
	
	else:
		return xbins, mean, e_mean


################# PLOT DIMER BINDING ENERGY VS FIELD
from matplotlib.gridspec import GridSpec

def format_axes(fig):
	for i, ax in enumerate(fig.axes):
		ax.tick_params(labelbottom=False, labelleft=False)

fig = plt.figure(layout='constrained', figsize=(3.4,2.8))
gs = GridSpec(2, 2, figure=fig, hspace=0.1)

yparam='ScaledTransfer'
# Eb vs field
Ebs = pd.read_pickle(os.path.join(data_path, 'Ebs.pkl'))
ExpEbs = pd.read_excel(os.path.join(data_path, 'Eb_results.xlsx'))
ExpEbs = ExpEbs.sort_values(by='B')
SqW = pd.read_excel(os.path.join(data_path, 'sqw_theory_line.xlsx'))
Tmat = pd.read_excel(os.path.join(data_path, 't_matrix_theory_line.xlsx'))
CCC = pd.read_csv(os.path.join(data_path, 'ac_s_Eb_vs_B_220-225G.dat'), header=None, names=['B','E'], delimiter='\s')

Eb_color =  '#1b9e77'
Eb_style= {'color':Eb_color,
				'mec':adjust_lightness(Eb_color, 0.3),
				'mfc':Eb_color,
				'mew':1,
				'marker':'o',
				'markersize':3}

colornaive = '#000000'
colorT = '#f20470'
colorSqW = '#23d197'
colorCC = '#f20470'
ax = fig.add_subplot(gs[0, 0])
ax.plot(Ebs['B'], Ebs['Ebs_naive'], ls='dotted', color=colornaive, marker='',  label=r'$1/a_{13}^2$')
ax.plot(SqW['Magnetic Field (G)'], SqW['Energy (MHz)'], color=colorSqW, marker='', ls = '--')
ax.plot(CCC['B'],CCC['E'], marker='', ls='-' , color = colorCC)
binx, biny, binxerr, binyerr = bin_data(ExpEbs['B'], ExpEbs['Eb'], xerr=np.ones(len(ExpEbs['B'])), yerr= np.ones(len(ExpEbs['Eb'])), nbins=25)
ax.plot(binx, biny, binyerr, **Eb_style)
#2ebdff
xlabel=r'$B$ [G]'
ylabel = r'$\omega_d/2\pi$ (MHz)'
ax.set(xlabel=xlabel, ylabel=ylabel,
	xlim=[199, 210],
	ylim = [-4.5, -1.5]
	)

ms = 3
color1 = "#7eb0d5"
style1 = {'color':color1,
				'mec':adjust_lightness(color1, 0.3),
				'mfc':color1,
				'mew':1,
				'marker':'s',
				'markersize':ms}
color2 = "#ffb55a"
style2 = {'color':color2,
				'mec':adjust_lightness(color2, 0.3),
				'mfc':color2,
				'mew':1,
				'marker':'p',
				'markersize':ms}
color3 = "#666699"
style3 = {'color':color3,
				'mec':adjust_lightness(color3, 0.3),
				'mfc':color3,
				'mew':1,
				'marker':'d',
				'markersize':ms} 
mystyles = [style1, style2, style3]
mycolors = [color1, color2, color3]
Bfield = [202.1, 203.1, 209]

ax = fig.add_subplot(gs[0,1])

color640 = '#4093ff'
style640 = {'color':color640,
				'mec':adjust_lightness(color640, 0.3),
				'mfc':color640,
				'mew':1,
				'marker':'o',
				'markersize':3}
color10 = '#ff5447'
style10 = {'color':color10,
			'mec':adjust_lightness(color10, 0.3),
			'mfc':color10,
			'mew':1,
			'marker':'s',
			'markersize':3}
file = '2024-07-17_J_e.dat_sat_corr.pkl'
data = pd.read_pickle(os.path.join(data_path, file))
scaling = 1000
data = data.sort_values(by='detuning')

custom_bins = [-4.05,  -4.03, -4.02, -4.01, -4.005, -4, -3.995, -3.990, -3.985, -3.98, -3.97, -3.96, -3.94]
x_dimer, y_dimer, yerr_dimer = bin_data(data['detuning'], data['c5_scaledtransfer']*scaling, data['em_c5_scaledtransfer']*scaling, nbins=custom_bins)
fit = pd.read_pickle(os.path.join(data_path, 'fit_'+file))
xs = fit['xs']/1e6
ys = fit['ys'] *scaling
EF_data = 0.0133
offs = ys.min()
ys = ys-offs
y_dimer = y_dimer - offs

json_file = 'lineshape_2024-07-17_J_e_backup.json'

with open(os.path.join(data_path, json_file)) as f:
	data_load = json.load(f)
	x_load = data_load['x']
	y_load = data_load['y']
	lineshape = interpolate.interp1d(x_load, y_load, 'linear', bounds_error=False, fill_value='extrapolate')

fitWithOffset = False
if fitWithOffset:
	guess_FDG = [0.01, -3.98/EF_data, 0]
	bounds = ([0, -600, -np.inf],[np.inf, 0, np.inf])
	def convls(x, A, x0, C):
		return A*lineshape(x-x0)+C
else:
	def convls(x, A, x0):
		return A*lineshape(x-x0)
	guess_FDG = [0.00001, -3.98/EF_data]
	bounds = ([0, -600],[1*scaling, 0])

# fit the lineshape onto the data
popt, pcov = curve_fit(convls, x_dimer/EF_data, y_dimer, sigma=yerr_dimer, p0=guess_FDG, bounds=bounds)
perr = np.sqrt(np.diag(pcov))

xx = np.linspace(-4.3/EF_data, -3.7/EF_data,1000)
yyconvls640 = convls(xx, *[popt[0], popt[1]])
ax.errorbar(x_dimer, y_dimer, yerr = yerr_dimer, **style640, ls='',  label=r'$\sigma = 28\,\mathrm{kHz} \approx 1.4\,E_F$') 
ax.plot(xx * EF_data, yyconvls640, marker='', ls='-', color=color640)
# axis settings
ax.set(xlim=[-4.04, -3.96],
		#ylim=[-0.05, 2.5],
		#xlabel=r'$\omega$ [MHz]',
		ylabel=r'$\widetilde{\Gamma} \; \times \; 10^3$',
		)

tauF1 = 1/EF_data/2/pi
trat1 = 640/tauF1

spline = UnivariateSpline(xs, ys-np.max(ys)/2, s=0)
r1, r2 = spline.roots()
FWHM = np.abs(r1-r2) # EF
print(f'FWHM={FWHM} MHz, or {FWHM/EF_data} EF')
ax.text(0.2, 0.8, r'$t_\mathrm{rf} \ll \tau_F$', color=color640, fontsize=7, transform=ax.transAxes)
GammaPeak = ys.max()/scaling
Id640 = GammaPeak/640/EF_data*2 
tcurve = np.linspace(1, 640, 100)
Idcurve = GammaPeak/tcurve/EF_data
print(f'640 us Id = {Id640}')

file2='2024-09-27_B_e.dat_sat_corr.pkl'
data = pd.read_pickle(os.path.join(data_path, file2))
data = data.sort_values(by='detuning')
scaling = 1000

custom_bins = [-4.25,  -4.15, -4.1, -4.05, -4.025, -4, -3.975, -3.95,  -3.9,  -3.85, -3.8, -3.75, -3.7]
print(data['c5_scaledtransfer'])
x_dimer2, y_dimer2, yerr_dimer2 = bin_data(data['detuning'], data['c5_scaledtransfer']*scaling, data['em_c5_scaledtransfer']*scaling, nbins=custom_bins)
fit2 = pd.read_pickle(os.path.join(data_path, 'fit_'+file2))
xs2 = fit2['xs']/1e6
ys2 = fit2['ys'] *scaling
EF_data = 0.0182
offs = ys2.min()
ys2 = ys2-offs
y_dimer2 = y_dimer2 - offs

ax.errorbar(x_dimer2, y_dimer2, yerr=yerr_dimer2, **style10, ls='',label=r'$\sigma = 100\,\mathrm{kHz} \approx 5 E_F$')
ax.plot(xs2, ys2, color=color10, marker='', ls='-')

# numerically figure out FWHM
from scipy.interpolate import UnivariateSpline
spline = UnivariateSpline(xs2, ys2-np.max(ys2)/2, s=0)
r1, r2 = spline.roots()
FWHM = np.abs(r1-r2) 
print(f'FWHM={FWHM} MHz, or {FWHM/EF_data} EF')

GammaPeak = ys2.max()/scaling
Id10 = GammaPeak/10/EF_data*2 #
print(f'10 us Id = {Id10}')

ax.set(xlim=[-4.15, -3.85],
	xlabel=r'$\omega$ [MHz]',
		ylabel=r'$\widetilde{\Gamma} \, \times \, 10^3$',
)

yticks = [0, 2, 4, 6, 8]
yticklabels = [str(x) for x in yticks]
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

tauF2 = 1/EF_data/2/pi
trat2 = 10/tauF2
ax.text(0.05, 0.25, r'$t_\mathrm{rf} \approx \tau_F$', color=color10, fontsize=7, transform=ax.transAxes)	

ax=fig.add_subplot(gs[1,:])
overall_scaling = 100
file = 'veryshort_df.xlsx'
data = pd.read_excel(os.path.join(data_path, file))
data = data.sort_values(by='scaledtime', ascending=True)

data['scaledtime']=1/data['scaledtime'] # tau_F/t_rf
data['Id'] = data['Id'] * overall_scaling
data['em_Id'] = data['em_Id'] * overall_scaling
df = data[data['time'] == 0.003].copy()
t3us = df['scaledtime'].values[0]
I3us = df['Id'].mean()
I3us_std = np.sqrt(df['em_Id'].values[0]**2 + df['em_Id'].values[1]**2)

df = data[data['time'] == 0.020].copy()
t20us = df['scaledtime'].values[0]
I20us = df['Id'].mean()
I20us_std = np.sqrt(df['em_Id'].values[0]**2 + df['em_Id'].values[1]**2)

t640us = float(1/trat1)
I640us = float(Id640) * overall_scaling
I640us_std = I640us*0.06 
t10us = float(1/trat2)
I10us = float(Id10) * overall_scaling
I10us_std = I10us*0.08 
data =data[data['time']!=0.020]
data= data[data['time']>0.0031]
my_color = '#8D6E63'
my_style = {'color':my_color,
				'mec':adjust_lightness(my_color, 0.3),
				'mfc':my_color,
				'mew':1,
				'marker':'o',
				'markersize':5,
				'ls':'none'}

x_dimer = data['scaledtime']
y_dimer = data['Id'] 
yerr_dimer = data['em_Id'] 
ax.errorbar(x_dimer, y_dimer, yerr_dimer, **my_style)
ax.errorbar([t3us], [I3us], yerr=[I3us_std], **my_style)
ax.errorbar([t20us], [I20us], yerr=[I20us_std], **my_style)
color640 = '#4093ff'
style640big = {'color':color640,
				'mec':adjust_lightness(color640, 0.3),
				'mfc':color640,
				'mew':1,
				'marker':'o',
				'markersize':5}
color10 = '#ff5447'
style10big = {'color':color10,
			'mec':adjust_lightness(color10, 0.3),
			'mfc':color10,
			'mew':1,
			'marker':'s',
			'markersize':5}
ax.errorbar([t640us], [I640us], yerr=[I640us_std], **style640big)
ax.errorbar([t10us], [I10us], yerr=[I10us_std], **style10big)

t_add = [t20us, t640us, t10us]
I_add = [I20us, I640us, I10us]
Ierr_add = [I20us_std, I640us_std, I10us_std]
t_all = pd.concat([x_dimer, pd.Series(t_add)]).sort_values(ascending=True)
I_all = pd.concat([y_dimer, pd.Series(I_add)]).sort_values(ascending=True)
Ierr_all = pd.concat([yerr_dimer, pd.Series(Ierr_add)]).sort_values(ascending=True)
avg_Id = np.mean(I_all.iloc[-4:])
e_avg_Id = np.sqrt(Ierr_all.iloc[-4]**2 + Ierr_all.iloc[-3]**2 + Ierr_all.iloc[-2]**2 + Ierr_all.iloc[-1]**2)/4

xs = np.linspace(0, t_all.max(), 100)
ax.hlines(avg_Id, 0,xs.max(), ls='--', color=my_color)
ax.fill_between(xs, avg_Id - e_avg_Id, avg_Id + e_avg_Id, color = adjust_lightness(my_color, 1.5))

scale_est = pd.read_excel(os.path.join(data_path, 'time_scaling_estimate_avg_rescaled.xlsx'))
ax.plot(scale_est['ftimes'], scale_est['peaks']*overall_scaling,color=color640, ls='-', marker='')

ax.text(0.07, 0.0155*overall_scaling, r'$I_d$', color=my_color)
ax.set(xlabel = r'$\tau_F/t_\mathrm{rf}$',
		ylabel=r'$4 \alpha / \Omega_{23}^2  t_\mathrm{rf}^2  \, \times \, 10^2$',
		ylim = [-0.1, 2.5],
		xlim = [-0.1,2.1],
		xscale='linear',
		)

xticks = [0, 0.5, 1, 1.5, 2]
xticklabels = [str(x) for x in xticks]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)