# -*- coding: utf-8 -*-
# figure 1: plot dimer and HFT on logscale, with noise floor and 5/2 tail region

from plot_settings import colors, adjust_lightness, paper_settings, generate_plt_styles, bin_data
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

plt.rcParams.update(paper_settings)

msize = 5
tintshade=0.6

# global parameters 

Ut = 200.0/19.2 # estimated trap depth/average E_F
EF_avg = 19.2
Eb=3980 # binding energy in kHz

def transfer_function(f, a):
    return a*f**(-3/2)/(1+f*EF_avg/Eb)  

def transfer_function_no_FSE(f, a):
    return a*f**(-3/2)

# initialize axes
fig = plt.figure(layout="constrained", figsize=(4, 3))
gs = GridSpec(2, 4, figure=fig)

gs0=gs[0, :].subgridspec(1, 2, wspace=0.05, hspace=0)
ax1 = fig.add_subplot(gs0[0])
ax1_2 = fig.add_subplot(gs0[1], sharey=ax1) # see: https://stackoverflow.com/questions/32185411/break-in-x-axis-of-matplotlib for broken x-axis plotting

plt.setp(ax1_2.get_yticklabels(), visible=False)
ax_br = fig.add_subplot(gs[1, :])

### =================================== 1 a) ===================================

dimer_color = '#1b9e77'
dimer_style = generate_plt_styles([dimer_color], ts=tintshade)[0]
dimer_style['marker'] = 'o'
dimer_style['markersize']=msize

loss_color = '#d95f02'
loss_style = generate_plt_styles([loss_color], ts=tintshade)[0]
loss_style['marker']='s'
loss_style['markersize']=msize

res_color = '#0F1AF0'
res_style = generate_plt_styles([res_color], ts= 0.4)[0]
res_style['marker']= 'D'
res_style['markersize']=msize

# cutoff because really high frequencies have bad signal and don't filter well
cutoff = 2.1
res_bound = 0.05 # upper bound on res transfer
hft_bound = res_bound - 0.01 # lower bound on hft
filt = 0.028 # arbitrarily chosen so that the plotted lineshape doesn't have sinc^2 sidebands

### load data
# dimer
dimer_data_fname = 'Figures/data/2025-03-19_G_e_pulsetime=0.64_avgdata.csv'
dimer_data = pd.read_csv(dimer_data_fname)
x_dimer, y_dimer, yerr_dimer = dimer_data[['detuning', 'c5', 'e_c5']].values.T
# bin_edges = [-4.06, -4.05, -4.04, -4.03,  -4, -3.98, -3.97,-3.96, -3.95, -3.94, -3.93]
# dimer_data['value_bin'] = pd.cut(x_dimer, bins=bin_edges)
# x_dimer_bin = dimer_data.groupby('value_bin')['c5_scaledtransfer'].mean()
# y_dimer_bin = dimer_data.groupby('value_bin')['detuning'].mean()

# transfer
fname = 'Figures/data/HFT_2MHz_spectra.csv' # transfer
data = pd.read_csv(fname)
data['detuning'] /= 1000 # convert to MHz
data = data[data['detuning'] < cutoff]

x_res, y_res = data.loc[data.detuning <= res_bound]\
                [['detuning', 'loss_ScaledTransfer']].values.T
x_res_bin, y_res_bin, __, __ = bin_data(x_res, y_res, yerr=np.ones(len(y_res)), 
                                        nbins=4, xerr=np.ones(len(x_res)))

x_HFT, y_HFT, yerr_HFT = data.loc[data.detuning > hft_bound]\
                [['detuning', 'loss_ScaledTransfer', 'loss_e_ScaledTransfer']].values.T

### make fit
# dimer
# #load x,y values for dimer fit from precomputed points
fit_data_fname = 'Figures/data/2025-03-19_G_e_pulsetime=0.64_fit.csv'
dimer_fit = pd.read_csv(fit_data_fname)
xpeak = dimer_fit['x'][dimer_fit['y'].argmax()]
dimer_fit['x'] /= 1e6
# crop around peak to avoid sinc^2 side bands
dimer_fit = dimer_fit[(dimer_fit.x > (xpeak - filt)) & \
                      (dimer_fit.x < (xpeak + filt))]
x_dimer_fit, y_dimer_fit = dimer_fit.values.T

# resonant feature
# create interpolated function
x_res_fit = np.linspace(min(x_res), max(x_res), 30)
y_res_fit = savgol_filter(np.interp(x_res_fit, x_res, y_res), 5, 4)

# HFT 
popt_HFT, __ = curve_fit(transfer_function, x_HFT, y_HFT)

### plot
# dimer plot
ax1.plot(x_dimer_fit, y_dimer_fit, ls='-', color=dimer_color)
ax1.fill_between(x_dimer_fit, y_dimer_fit, 
                 0, color=adjust_lightness(dimer_color,1.8))
ax1.plot(x_dimer, y_dimer, **dimer_style)

# HFT plot (right)
x_HFTs = np.linspace(min(x_HFT), max(x_HFT),30)
y_HFTs = np.interp(x_HFTs, x_HFT, y_HFT)
ax1_2.plot(x_HFTs, transfer_function(x_HFTs, *popt_HFT), ls='-', color=loss_color)
ax1_2.fill_between(x_HFTs, transfer_function(x_HFTs, *popt_HFT), 
                   0, color=adjust_lightness(loss_color, 2.0))

ax1_2.plot(x_res_fit, y_res_fit, ls='-', color=res_color)
ax1_2.fill_between(x_res_fit, 0, y_res_fit, color=adjust_lightness(res_color,1.8))

ax1_2.plot(x_res_bin, y_res_bin, **res_style)
ax1_2.plot(x_HFT, y_HFT, **loss_style)

# inset axis
left, bottom, width, height = [0.78, 0.82, 0.15, 0.11]
axi = fig.add_axes([left, bottom, width, height])
axi.plot(x_res_fit, y_res_fit, ls='-', marker='', color=res_color)
axi.plot(x_res, y_res, **res_style)
axi.fill_between(x_res_fit, 0, y_res_fit, color=adjust_lightness(res_color, 1.8))

# set axes ticks and limits
axi.set(
    xlim=[-0.011, 0.011],
    ylim=[0, 0.35],
)
axi.tick_params(labelsize=6)
axi.set_xticks([-0.01,0,0.01])
axi.set_xticklabels(['-0.01','0','0.01'])

ax1.set(xlim=[-6.2, -3.8])
ax1.set( ylabel=r'$\widetilde{\Gamma}$' )
ax1.xaxis.set_label_coords(0.7, -0.2)
ax1.set_xticks([-6, -5, -4])

ax1_2.set(xlim=[-0.2, cutoff+0.1], ylim=[0.5e-5, 7e-1])
ax1_2.set_xticks([0, 1, 2])
ax1_2.set_yscale('log')

ax1.spines['right'].set_visible(False)
ax1_2.spines['left'].set_visible(False)
ax1.yaxis.tick_left()
ax1_2.yaxis.tick_right()
plt.setp(ax1_2.get_yticklabels(), visible=False)
ax1.minorticks_off()
ax1.set_yticks([1e-5,  1e-3, 1e-1])

# ==================================== 1 b) ====================================

loss_color = '#d95f02'
loss_style = generate_plt_styles([loss_color], ts=tintshade)[0]
loss_style['marker']='s'

transfer_color = '#2877dd'
transfer_style = generate_plt_styles([transfer_color], ts=0.3)[0]
transfer_style['marker'] = 'o'

# get data
x_loss, y_loss, yerr_loss = data[['ScaledDetuning', 'loss_ScaledTransfer', 
                                  'loss_e_ScaledTransfer']].values.T

x_transf, y_transf, yerr_transf = data[['ScaledDetuning', 'ScaledTransfer', 
                                  'e_ScaledTransfer']].values.T

# fit loss
x_fit, y_fit, yerr_fit = x_loss[x_loss > 0], y_loss[x_loss > 0], yerr_loss[x_loss > 0]
popt, pcov = curve_fit(transfer_function_no_FSE, x_fit, y_fit, sigma=yerr_fit, p0=[0.05])
perr = np.sqrt(np.diag(pcov))

# plot
ax_br.errorbar(x_loss, y_loss, yerr=yerr_loss, **loss_style)

xs = np.linspace(0.5, max(x_fit)+500, 500)
ax_br.plot(xs, transfer_function_no_FSE(xs, *popt), '-', color=colors[1])

ax_br.errorbar(x_transf, y_transf, yerr=yerr_transf, **transfer_style)

# format axes
ax_br.vlines(Ut, 0, 1.0, color='k', linestyle='--') 

# Create a Rectangle patch
noisefloor=1e-5
rect = patches.Rectangle((0,0), 250, noisefloor, linewidth=3, facecolor='red', fill=True, alpha = 0.2)
ax_br.add_patch(rect)
ax_br.vlines(Ut, ymin=0, ymax=1.5,ls='dashed', color='black')

ax_br.set(xlabel=r'Detuning $\tilde{\omega}\,[E_F]$',
        ylabel = r'$\widetilde{\Gamma}_\mathrm{HFT}$',
        yscale='log', 
        xscale='log',
        xlim=[x_transf.min()+0.7, x_transf.max()+50],
        ylim=[5e-9, 20e-1])
ax_br.minorticks_off()
ax_br.set_yticks([10e-8, 10e-6, 10e-4, 10e-2])

ax_br.text(7.5, 5e-2, r'$U_t$')

ax_br2 = ax_br.twiny()
ax_br2.plot(x_transf * EF_avg / 1000, y_transf, alpha=0)

# sync axis limits
ax_br2.set_xlim(ax_br.get_xlim()[0] * EF_avg/1000, 
                ax_br.get_xlim()[1] * EF_avg/1000)
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

ax_br2.set_xticks([0.01, 0.1, 1])
ax_br2.set_xticklabels(['0.01','0.1','1'])
ax_br2.set_xlabel(r'Detuning $\omega$ [MHz]')

fig.tight_layout()