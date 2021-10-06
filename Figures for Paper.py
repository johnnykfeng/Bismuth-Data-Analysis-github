import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import find_peaks
from scipy.signal import argrelmin
from scipy import interpolate
from matplotlib.font_manager import FontProperties
import time
import csv
from ScanDictionary import scanlist
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Boolean, \
    ARRAY, Numeric, Float, ForeignKey, ForeignKeyConstraint, column
from sqlalchemy import insert, update, select, inspect

pixel2Ghkl = 2.126e-3  # converts CCD pixels to G_hkl = 1/d_hkl 'scattering vector'
convTT = 0.0046926 # Kam's value for CCD pixels to scatterin vector
def pixel2scatt_vector(x):
    return x * pixel2Ghkl
def scatt_vector2pixel(x):
    return x / pixel2Ghkl

# sqlalchemy connection to postgresql database
engine = create_engine('postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project')
metadata = MetaData()
connection = engine.connect()
# convert sql to pandas dataframe
scandict_df = pd.read_sql_table('scandictionary', 'postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project')


def figure1(timepoint_on, timepoint_off):
    bigscan23nm_df = pd.read_sql_table('Big_Scan_23nm', 'postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project')

    scankey = 2 # for fluence=1.3 with 23nm sample
    fig, ax = plt.subplots()
    fluence = scandict_df[scandict_df['scan_id'] == scankey]['fluence'].values
    fig_str_label = 'scankey: ' + str(scankey) + ', fluence: ' +str(fluence)
    ax.set_title(fig_str_label)

    scandf = bigscan23nm_df[bigscan23nm_df['scan_key'].isin([scankey])] # filters for scankey, partition the df
    tp = np.array(scandf['timepoint'])     # extract the 'timepoint' column

    radavgflat = scandf['radavg_flattened']    # extract the 'radavg_flattened' column   # extract the 'radavg_flattened' column
    radavgflat_off = scandf[scandf['timepoint'] == timepoint_off]['radavg_flattened'].values[0]

    radavgflat_on = scandf[scandf['timepoint'] == timepoint_on]['radavg_flattened'].values[0]

    radavgdiff = scandf[scandf['timepoint'] == timepoint_on]['radavg_diff'].values[0]
    # modify radavgdiff to fit better in the plot
    y_scaling = 5
    radavgdiff = [element*y_scaling for element in radavgdiff]

    axis_pixels = np.arange(len(radavgflat_off))
    axis_Ghkl = axis_pixels*pixel2Ghkl
    # axis_Ghkl = axis_pixels
    # axis_Ghkl = axis_pixels*convTT
    ax.plot(axis_Ghkl, radavgflat_off, alpha = 0.8, color = 'blue', linestyle = 'dotted', label = 'T = -3 ps')  # first timepoint
    ax.plot(axis_Ghkl, radavgflat_on, alpha = 0.8, color = 'red', linestyle = 'solid', label = 'T = 10 ps')  # last timepoint

    axis_pixels = np.arange(len(radavgdiff))
    axis_Ghkl = (axis_pixels-17)*pixel2Ghkl
    # axis_Ghkl = axis_pixels
    ax.plot(axis_Ghkl, radavgdiff, alpha = 0.8, color='black', linestyle = 'solid', label = 'Difference')  # last timepoint

    ax.axhline(y=0, color='black', alpha=0.5, label='y = 0')

    plt.xlim(0.2, 0.9)
    plt.ylim(1.2*min(radavgdiff), 9000)

    ax.axes.yaxis.set_ticks([])
    ax.set_ylabel('Intensity (a.u.)')
    ax.set_xlabel(r'$q$ ($\AA^{-1}$)')

    ax.legend(loc='upper right', prop={'size': 10}, frameon=False)

    plt.tight_layout()
    # plt.grid(True)
    plt.show()

# figure1(timepoint_on = 15.0, timepoint_off =-3.0)

# def figure2():
#------------------------------------------------------------------------------




peaksum_23nm_df = pd.read_sql_table('Peak_sum_23nm_adjusted',
                                    'postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project')
liquidrise_23nm_df = pd.read_sql_table('liquidrise_23nm',
                                       'postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project')
exp_fit_23nm_df = pd.read_sql_table('exp_fit_variables_23nm',
                                    'postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project')


peak_cols =['peak1','peak2','peak3','peak4','peak5','peak6','peak7',
'peak8','peak9','peak10','peak11','peak12','peak13','peak14','peak15']

base_cols = ['base1', 'base2', 'base3', 'base4', 'base5', 'base6', 'base7']

scankey = 2 # Only calling the 1.3 mJ/cm2 data set

peak_df = peaksum_23nm_df[peaksum_23nm_df['scan_id'] == scankey]
lr_df = liquidrise_23nm_df[liquidrise_23nm_df['scan_pk'] == scankey]
fit_df = exp_fit_23nm_df[exp_fit_23nm_df['scan_id'] == scankey]

tp = peak_df['timepoint']

# peak_choice = 0, 1, 2, 3, 4, 5
peak_choice = 0, 1, 4, 5
peak_colors = cm.turbo(np.linspace(0, 1, len(peak_choice)))

fig1, ax = plt.subplots()

for p, peak_index in enumerate(peak_choice):  # loop over peaks
    print('peak_index: ' + str(peak_index))

    tp = peak_df['timepoint']  # x-data

    peak = peak_df[peak_cols[peak_index]]  # get raw peak intensity
    peak_norm = peak / np.mean(peak[:7])  # y-data

    base = lr_df[base_cols[peak_index + 1]]
    base_zeroed = base - np.mean(base.iloc[0:5])
    peak_adjusted = peak - base_zeroed
    peak_adjusted_norm = peak_adjusted / np.mean(peak_adjusted[:7])

    from exponential_fit import Exp_Fit
    fit_vars_peak = fit_df[fit_df['peak_id']== peak_index+1]
    print(fit_vars_peak)
    a, tau, t0, c = fit_vars_peak['a'], fit_vars_peak['tau'], fit_vars_peak['t0'], fit_vars_peak['c']
    x_fit, y_fit, fit_var, pcov = Exp_Fit(tp, peak_norm, a, tau, t0, c, 'pltlabel')

    pltlabel = 'scan' + str(scankey) + '-' + peak_cols[peak_index]

    ax.plot(tp, peak_norm, '-o', label=peak_cols[peak_index], color=peak_colors[p], linewidth=0.5)
    ax.plot(x_fit - fit_var[2], y_fit, '--', color=peak_colors[p])

    # ax1.plot(tp, peak_adjusted_norm, marker='^', linestyle='--', label=base_cols[peak_index],
    #          color=peak_colors[peak_index])
    ax.set_ylabel(r'$\Delta I$ / $I_{o}$', labelpad=5)
    ax.set_xlabel('Time Delay (ps)')

ax.legend(loc='lower left', prop={'size': 10})
plt.tight_layout()
plt.show()