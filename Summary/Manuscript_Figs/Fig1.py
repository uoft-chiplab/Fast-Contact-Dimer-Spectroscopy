# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:34:41 2025

These are a collection of select figures for drafting the clock shift manuscript.
Typically it pulls data from manuscript_data/ and saves in manuscript_figures/

@author: coldatoms
"""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
general_files_path = os.path.join(parent_dir, 'General')
# Add the parent directory to sys.path
if general_files_path not in sys.path:
	sys.path.append(general_files_path)

from plot_settings import *
plt.rcParams.update(paper_settings)

from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
data_path = os.path.join(proj_path, 'manuscript_data')

# options
Show = True

# plot shading
tintshade = 0.6

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

from mpl_toolkits.axes_grid1 import make_axes_locatable
##another version
from matplotlib.gridspec import GridSpec

def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        #ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)


mysize = 5
dimer_color = '#1b9e77'
dimer_style = generate_plt_styles([dimer_color], ts=tintshade)[0]
dimer_style['marker'] = 'o'
dimer_style['markersize']=mysize
loss_color = '#d95f02'
loss_style = generate_plt_styles([loss_color], ts=tintshade)[0]
loss_style['marker']='s'
loss_style['markersize']=mysize

res_color = '#0F1AF0'
#res_color = '#FF000C'
res_style = generate_plt_styles([res_color], ts= 0.4)[0]
res_style['marker']= 'D'
res_style['markersize']=mysize


fig = plt.figure(layout="constrained", figsize=(4, 3))
gs = GridSpec(2, 4, figure=fig)
#ax1 = fig.add_subplot(gs[0, :])
gs0=gs[0, :].subgridspec(1, 2, wspace=0.05, hspace=0)
ax1 = fig.add_subplot(gs0[0])
ax1_2 = fig.add_subplot(gs0[1], sharey=ax1) # see: https://stackoverflow.com/questions/32185411/break-in-x-axis-of-matplotlib for broken x-axis plotting
plt.setp(ax1_2.get_yticklabels(), visible=False)
#ax_bl = fig.add_subplot(gs[1, 0:2])
#ax4 = fig.add_subplot(gs[1, 1]) # this plot used to contain a histogram of errorbar sizes
ax_br = fig.add_subplot(gs[1, :])

yparam = 'ScaledTransfer' #'ScaledTransfer' or 'transfer'
# dimer spectrum, long pulse

#file = '2024-07-17_J_e_ratio95.pkl'
file = '2025-03-19_G_e_pulsetime=0.64.dat.pkl'
data = pd.read_pickle(os.path.join(data_path, file))
x_dimer = data['detuning']
y_dimer = data['c5_scaledtransfer']
yerr_dimer = data['em_c5_scaledtransfer']
fit = pd.read_pickle(os.path.join(data_path, 'fit_'+file))
xs = fit['xs']/1e6
ys=fit['ys']
print(xs)
#fit_Eb = fit['Eb'][0]/1000
#fit_e_Eb = fit['e_Eb'][0]/1000


# dimer plot (left)
peakindex = np.where(ys==ys.max())
xpeak = xs[peakindex]
filt = 0.028 # arbitrarily chosen so that the plotted lineshape doesn't have sinc^2 sidebands
#filt=1
xs_filt = xs[(xs > (xpeak-filt)) & (xs < (xpeak+filt))]
ys_filt = ys[(xs > (xpeak-filt)) & (xs < (xpeak+filt))]

ax1.plot(xs_filt, ys_filt, ls='-',  lw= 1, marker='', color=dimer_color)
ax1.fill_between(xs_filt, ys_filt,0, color=adjust_lightness(dimer_color,1.8))
# binx, biny, binyerr, binxerr = bin_data(x_dimer, y_dimer, yerr=np.ones(len(y_dimer)), nbins=6, xerr=np.ones(len(x_dimer)))
# ax1.plot(binx, biny, **dimer_style)
# custom bin range
#bin_edges = [-4.05, -4.03, -4.01, -3.99, -3.97, -3.95, -3.93, -3.91]
bin_edges = np.arange(-4.08, -3.93, 0.015)
bin_edges = [-4.06, -4.05, -4.04, -4.03,  -4, -3.98, -3.97,-3.96, -3.95, -3.94, -3.93]
#bin_edges = [-4.06, -4.04, -4.03, -4.00, -3.97, -3.94]
data['value_bin'] = pd.cut(x_dimer, bins=bin_edges)
biny = data.groupby('value_bin')['c5_scaledtransfer'].mean()
binx = data.groupby('value_bin')['detuning'].mean()
ax1.plot(binx, biny, **dimer_style)
ax1.set(xlim=[-6.2, -3.8])
ax1.set_yscale('log')
ax1.set(
    #xlabel=r'$\omega$ [MHz]',
    ylabel=r'$\widetilde{\Gamma}$'
)
ax1.xaxis.set_label_coords(0.7, -0.2)
xticks = [-6, -5, -4]
ax1.set_xticks(xticks)


# HFT spectrum for ax1
file = 'HFT_2MHz_spectra.csv'
data = pd.read_csv(os.path.join(data_path, file))
x_name = 'detuning'
y_name = 'loss_ScaledTransfer'
yerr_name = 'loss_e_ScaledTransfer'
# y_name='ScaledTransfer'
# yerr_name='e_ScaledTransfer'
#data = data[data[x_name] > -1]
data[x_name] = data[x_name]/1000 # MHz
cutoff =2.1 # cutoff because really high frequencies have bad signal and don't filter well
data = data[data[x_name] < cutoff]
x_all = data[x_name]
y_all = data[y_name]
yerr_all = data[yerr_name]
res_bound = 0.05

x_res = data[data[x_name] <= res_bound][x_name]
y_res = data[data[x_name] <= res_bound][y_name]
res_bound_adjust = 0.01
x_HFT = data[data[x_name] > res_bound-res_bound_adjust][x_name]
y_HFT = data[data[x_name] > res_bound-res_bound_adjust][y_name]


# HFT plot (right)
def transfer_function(f, a):
    # note the EFs are so similar in the datasets I've baked in the average
    # EF here to make this analysis a little easier.
    EF_avg = 19.2
    Eb=3980
    return a*f**(-3/2)/(1+f*EF_avg/Eb)  # binding energy in kHz

x_ress = np.linspace(min(x_res), max(x_res), 30)
#x_ress = np.linspace(0, max(x_res),30)
y_ress = np.interp(x_ress, x_res, y_res)
y_ress_smooth = savgol_filter(y_ress, 5, 4)
popt, pcov = curve_fit(transfer_function, x_HFT, y_HFT)

x_HFTs = np.linspace(min(x_HFT), max(x_HFT),30)
y_HFTs = np.interp(x_HFTs, x_HFT, y_HFT)
ax1_2.plot(x_HFTs, transfer_function(x_HFTs, *popt), ls='-', lw= 1, marker='', color=loss_color)
ax1_2.fill_between(x_HFTs, transfer_function(x_HFTs, *popt), 0, color=adjust_lightness(loss_color,2.0))

ax1_2.plot(x_ress, y_ress_smooth, ls='-',  lw= 1,marker='', color=res_color)
ax1_2.fill_between(x_ress, 0, y_ress_smooth, color=adjust_lightness(res_color,1.8))

binx, biny, binyerr, binxerr = bin_data(x_res, y_res, yerr=np.ones(len(y_res)), nbins=4, xerr=np.ones(len(x_res)))
ax1_2.plot(binx, biny, **res_style)
# x_HFT.index = x_HFT.index - x_HFT.index[0]
# y_HFT.index = y_HFT.index - y_HFT.index[0
# binx, biny, binyerr, binxerr = bin_data(x_HFT, y_HFT, yerr=np.ones(len(y_HFT)), nbins=10, xerr=np.ones(len(x_HFT)))
ax1_2.plot(x_HFT, y_HFT, **loss_style)

# inset axis
left, bottom, width, height = [0.78, 0.82, 0.15, 0.11]
axi = fig.add_axes([left, bottom, width, height])
axi.plot(x_ress, y_ress_smooth, ls='-', marker='', color=res_color)
axi.plot(x_res, y_res, **res_style)
axi.fill_between(x_ress, 0, y_ress_smooth, color=adjust_lightness(res_color, 1.8))
axi.set(
    xlim=[-0.011, 0.011],
    ylim=[0, 0.35],
)
axi.tick_params(labelsize=6)
axi.set_xticks([-0.01,0,0.01])
axi.set_xticklabels(['-0.01','0','0.01'])

ax1_2.set(xlim=[-0.2, cutoff+0.1], ylim=[0.5e-5, 7e-1])
xticks = [0, 1, 2]
ax1_2.set_xticks(xticks)
#ax1_2.set(xlim=[-0.1, cutoff], ylim=[0, 0.01])
ax1_2.set_yscale('log')


ax1.spines['right'].set_visible(False)
ax1_2.spines['left'].set_visible(False)
ax1.yaxis.tick_left()
ax1_2.yaxis.tick_right()
#ax1_2.yick_params(axis='y', which='both', length=0)
plt.setp(ax1_2.get_yticklabels(), visible=False)
ax1.minorticks_off()
yticks=[1e-5,  1e-3, 1e-1]
ax1.set_yticks(yticks)

#### ZOOM-IN HFT SPECTRUM IN BOTTOM RIGHT
loss_color = '#d95f02'
loss_style = generate_plt_styles([loss_color], ts=tintshade)[0]
loss_style['marker']='s'

transfer_color = '#2877dd'
transfer_style = generate_plt_styles([transfer_color], ts=0.3)[0]
transfer_style['marker'] = 'o'

filter_by_Ut = False
trap_depth = 200.0 # estimate
EF_avg=19.2
#file = '2024-09-10_L_e.pkl'
file = 'HFT_2MHz_spectra.csv'

data = pd.read_csv(os.path.join(data_path, file))
x_name = 'ScaledDetuning'
if filter_by_Ut:
    data_below = data[(data[x_name] < trap_depth/EF_avg) & (data[x_name] > 0)]
    data_above = data[data[x_name] > trap_depth/EF_avg]
    
y_name='ScaledTransfer'
yerr_name = 'e_ScaledTransfer'
if filter_by_Ut:
    x = np.array(data_below[x_name])
    y = np.array(data_below[y_name])
    yerr = np.array(data_below[yerr_name])
    #ax_br.errorbar(x, y, yerr=yerr, linestyle='', **sty, label=r'$\alpha_3 = N_3/N_\mathrm{tot}$')
else: 
    x= np.array(data[x_name])
    y = np.array(data[y_name])
    yerr = np.array(data[yerr_name])
    #ax_br.errorbar(x, y, yerr=yerr, linestyle='', **sty, label=r'$\alpha_3 = N_3/N_\mathrm{tot}$')

# fit to both forms of the transfer rate equation, w/wout Final State Effect
def transfer_function(f, a):
    # note the EFs are so similar in the datasets I've baked in the average
    # EF here to make this analysis a little easier.
    EF_avg = 19.2
    Eb=3980
    return a*f**(-3/2)/(1+f*EF_avg/Eb)  # binding energy in kHz

def transfer_function_no_FSE(f, a):
    return a*f**(-3/2)


popt, pcov = curve_fit(transfer_function_no_FSE, x, y, sigma=yerr, p0=[0.05])
perr = np.sqrt(np.diag(pcov))
popt_2, pcov_2 = curve_fit(transfer_function, x, y, sigma=yerr, p0=[0.05])
perr_2 = np.sqrt(np.diag(pcov_2))

xs = np.linspace(0.5, max(x), 100)

#ax_br.plot(xs, transfer_function_no_FSE(xs, *popt), '-', color=colors[0])
#ax_br.plot(xs, transfer_function(xs, *popt_2), '--', color=colors[0])

C_FSE = popt[0] * 2*np.sqrt(2)*np.pi**2
e_C_FSE = perr[0] * 2*np.sqrt(2)*np.pi**2

C = popt_2[0] * 2*np.sqrt(2)*np.pi**2
e_C = perr_2[0] * 2*np.sqrt(2)*np.pi**2

print("Contact from tranfser with FSE is {:.2f}({:.0f})".format(C_FSE, e_C_FSE*1e2))
print("Contact from transfer w/out FSE is {:.2f}({:.0f})".format(C, e_C*1e2))

# transfer above trap depth
if filter_by_Ut:
    x = np.array(data_above[x_name])
    y = np.array(data_above[y_name])
    yerr = np.array(data_above[yerr_name])

    sty = styles[0].copy()
    sty['mfc'] = 'w'
    ax_br.errorbar(x, y, yerr=yerr, linestyle='', **sty)

# loss
y_name = 'loss_ScaledTransfer'
yerr_name = 'loss_e_ScaledTransfer'
x = np.array(data[x_name])
y_loss = np.array(data[y_name])
yerr_loss = np.array(data[yerr_name])

sty = loss_style
ax_br.errorbar(x, y_loss, yerr=yerr_loss, linestyle= '', **sty, label=r'$\alpha_2=(N_2^{\mathrm{bg}}-N_2)/N_\mathrm{tot}$')

df_fit = data.loc[data[x_name] > 0]
x = df_fit[x_name]
y = df_fit[y_name]
yerr = df_fit[yerr_name]

# fit to both forms of the transfer rate equation, w/wout Final State Effect
popt, pcov = curve_fit(transfer_function_no_FSE, x, y, sigma=yerr, p0=[0.05])
perr = np.sqrt(np.diag(pcov))
popt_2, pcov_2 = curve_fit(transfer_function, x, y, sigma=yerr, p0=[0.05])
perr_2 = np.sqrt(np.diag(pcov_2))

xs = np.linspace(0.5, max(x)+500, 500)

ax_br.plot(xs, transfer_function_no_FSE(xs, *popt), '-', color=colors[1])
#ax_br.plot(xs, transfer_function(xs, *popt_2), '--', color=colors[1])

C_loss_FSE = popt[0] * 2*np.sqrt(2)*np.pi**2
e_C_loss_FSE = perr[0] * 2*np.sqrt(2)*np.pi**2

C_loss = popt_2[0] * 2*np.sqrt(2)*np.pi**2
e_C_loss = perr_2[0] * 2*np.sqrt(2)*np.pi**2

print("Contact from loss with FSE is {:.2f}({:.0f})".format(C_loss_FSE, e_C_loss_FSE*1e2))
print("Contact from loss w/out FSE is {:.2f}({:.0f})".format(C_loss, e_C_loss*1e2))


data = pd.read_csv(os.path.join(data_path, file))
x_name = 'ScaledDetuning'
if filter_by_Ut:
    data_below = data[(data[x_name] < trap_depth/EF_avg) & (data[x_name] > 0)]
    data_above = data[data[x_name] > trap_depth/EF_avg]
    
y_name='ScaledTransfer'
yerr_name = 'e_ScaledTransfer'
if filter_by_Ut:
    x = np.array(data_below[x_name])
    y = np.array(data_below[y_name])
    yerr = np.array(data_below[yerr_name])
    sty=transfer_style
    ax_br.errorbar(x, y, yerr=yerr, linestyle='', **sty, label=r'$\alpha_3 = N_3/N_\mathrm{tot}$')
else: 
    x= np.array(data[x_name])
    y = np.array(data[y_name])
    yerr = np.array(data[yerr_name])
    sty = transfer_style
    ax_br.errorbar(x, y, yerr=yerr, linestyle='', **sty, label=r'$\alpha_3 = N_3/N_\mathrm{tot}$')


ax_br.vlines(trap_depth/EF_avg, 0, 1.0, color='k', linestyle='--') 

# Create a Rectangle patch
noisefloor=1e-5
rect = patches.Rectangle((0,0), 250, noisefloor, linewidth=3, facecolor='red', fill=True, alpha = 0.2)
# Add the patch to the Axes
ax_br.add_patch(rect)
# horizontal line for noise floor?
#ax_br.plot([0, 150], [noisefloor, noisefloor], color='red', marker='', ls = '-')
# vertical line for trap dept
ax_br.vlines(trap_depth/EF_avg, ymin=0, ymax=1.5,ls='dashed', color='black')

ax_br.set(xlabel=r'Detuning $\tilde{\omega}\,[E_F]$',
        ylabel = r'$\widetilde{\Gamma}_\mathrm{HFT}$',
        yscale='log', 
        xscale='log',
        xlim=[x.min()-0.5, x.max()+50],
        ylim=[5e-9, 20e-1])
ax_br.minorticks_off()
yticks = [10e-8, 10e-6, 10e-4, 10e-2]
ax_br.set_yticks(yticks)
# add text

#ax_br.text(1.5, 1e-1, r'$\omega^{-3/2}$')
#ax_br.text(80, 1e-2, r'$\frac{\omega^{-3/2}}  {\frac{1}{1+\omega/\omega^*}}$')
ax_br.text(7.5, 5e-2, r'$U_t$')
#ax_br.legend(fontsize=8, loc='lower left')

ax_br2 = ax_br.twiny()
x_trans = x * EF_avg / 1000
ax_br2.plot(x_trans, y, alpha=0)

# sync axis limits
ax_br2.set_xlim(ax_br.get_xlim()[0] * EF_avg/1000, ax_br.get_xlim()[1]*EF_avg/1000)
ax_br2.set_xscale('log')

# Set only top axis visible
ax_br2.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=True,
    labelbottom=False,
    labeltop=True
)
#ax_br2.minorticks_off()
ax_br2.set_xticklabels(['-','-','0.01','0.1','1'])
# Hide all axis elements except top x-axis
ax_br2.spines['bottom'].set_visible(False)
ax_br2.spines['left'].set_visible(False)
ax_br2.spines['right'].set_visible(False)
ax_br2.yaxis.set_visible(False)
ax_br2.set_xlabel(r'Detuning $\omega$ [MHz]')

fig.tight_layout()

