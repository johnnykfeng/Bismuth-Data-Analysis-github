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

#region Initialize
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
#endregion

def figure1(timepoint_on = 15.0, timepoint_off =-3.0, scankey = 2, savefigure = 0):
    bigscan23nm_df = pd.read_sql_table('Big_Scan_23nm', 'postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project')

    scandf = bigscan23nm_df[bigscan23nm_df['scan_key'].isin([scankey])] # filters for scankey, partition the df

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

    fig, ax = plt.subplots()
    # fluence = scandict_df[scandict_df['scan_id'] == scankey]['fluence'].values
    fluence = scandict_df.loc[scankey-1, 'fluence']
    print(fluence)
    fig_str_label = 'scankey: ' + str(scankey) + ', fluence= ' +str(fluence) +' mJ/cm2'

    # axis_Ghkl = axis_pixels

    ax.plot(axis_Ghkl, radavgflat_off, alpha = 0.8, color = 'blue', linestyle = 'dotted',
            label = 'T = '+str(timepoint_off)+ ' ps')  # first timepoint
    ax.plot(axis_Ghkl, radavgflat_on, alpha = 0.8, color = 'red', linestyle = 'solid',
            label = 'T = '+str(timepoint_on)+ ' ps')  # last timepoint

    axis_pixels = np.arange(len(radavgdiff))
    axis_Ghkl = (axis_pixels-17)*pixel2Ghkl
    ax.plot(axis_Ghkl, radavgdiff, alpha = 0.8, color='black', linestyle = 'solid', label = 'Difference (x 5)')  # last timepoint

    ax.axhline(y=0, color='black', alpha=0.5, label='y = 0')

    ax.axes.yaxis.set_ticks([])
    ax.set_title(fig_str_label)
    ax.set_ylabel('Intensity (a.u.)', labelpad=5, fontsize=12)
    ax.set_xlabel(r'$k$ ($\AA^{-1}$)', fontsize=12)

    ax.tick_params(axis='both', labelsize=12)
    plt.xlim(0.2, 0.9)
    plt.ylim(1.2*min(radavgdiff), 9000)
    ax.legend(loc='upper right', prop={'size': 10}, frameon=False)
    plt.tight_layout()
    # plt.grid(True)
    plt.show()

    if savefigure:
        saveDirectory = 'D:\\Bismuth Project\\Figures for paper\\'
        figureName = 'RadialAvgExample'
        plt.savefig(saveDirectory + figureName +'.pdf', format='pdf')
        plt.savefig(saveDirectory + figureName +'.svg', format='svg', dpi = 600)
        plt.savefig(saveDirectory + figureName +'.png', format='png', dpi = 600)

# figure1(timepoint_on = 15.0, timepoint_off =-3.0, scankey=2, savefigure = 0)

#------------------------------------------------------------------------------

def figure2(scankey = 2, savefigure = 0):
    peaksum_23nm_df = pd.read_sql_table('Peak_sum_23nm_adjusted',
                                        'postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project')
    liquidrise_23nm_df = pd.read_sql_table('liquidrise_23nm',
                                           'postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project')
    exp_fit_23nm_df = pd.read_sql_table('exp_fit_variables_23nm',
                                        'postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project')
    diff_index_df = pd.read_csv('Peak Indexing by Saeed.csv')
    print(diff_index_df)
    print(str(diff_index_df['Index']))

    peak_cols =['peak1','peak2','peak3','peak4','peak5','peak6','peak7',
    'peak8','peak9','peak10','peak11','peak12','peak13','peak14','peak15']

    base_cols = ['base1', 'base2', 'base3', 'base4', 'base5', 'base6', 'base7']

    peak_df = peaksum_23nm_df[peaksum_23nm_df['scan_id'] == scankey]
    lr_df = liquidrise_23nm_df[liquidrise_23nm_df['scan_pk'] == scankey]
    fit_df = exp_fit_23nm_df[exp_fit_23nm_df['scan_id'] == scankey]

    tp = peak_df['timepoint']

    # peak_choice = 0, 1, 2, 3, 4, 5
    peak_choice = 0, 1, 4, 5, 6
    peak_colors = cm.turbo(np.linspace(0, 1, len(peak_choice)))

    fig, ax = plt.subplots()

    for p, peak_index in enumerate(peak_choice):  # loop over peaks
        print('peak_index: ' + str(peak_index))

        tp = peak_df['timepoint']  # x-data

        peak = peak_df[peak_cols[peak_index]]  # get raw peak intensity
        peak_norm = peak / np.mean(peak[:7])  # y-data

        # base = lr_df[base_cols[peak_index + 1]]
        # base_zeroed = base - np.mean(base.iloc[0:5])
        # peak_adjusted = peak - base_zeroed
        # peak_adjusted_norm = peak_adjusted / np.mean(peak_adjusted[:7])

        from exponential_fit import Exp_Fit
        fit_vars_peak = fit_df[fit_df['peak_id']== peak_index+1]
        # extracting fit variables
        a, tau, t0, c = fit_vars_peak['a'], fit_vars_peak['tau'], fit_vars_peak['t0'], fit_vars_peak['c']
        x_fit, y_fit, fit_var, pcov = Exp_Fit(tp, peak_norm, a, tau, t0, c, 'pltlabel')

        # index_label = peak_cols[peak_index] + ' (' + str(diff_index_df.loc[p,'Index']) + ')'
        index_label = '(' + str(diff_index_df.loc[p,'Index']) + ')'

        # ax.plot(tp, peak_norm, '-o', label=peak_;cols[peak_index], color=peak_colors[p], linewidth=0.5)
        ax.plot(tp, peak_norm, '-o', color=peak_colors[p], linewidth=0.5, label=index_label)
        ax.plot(x_fit - fit_var[2], y_fit, '--', color=peak_colors[p])
        ax.axvline(x=0, color='grey', alpha=0.4)
        ax.axhline(y = a.item()+ c.item(), alpha=0.7, color=peak_colors[p])

        # ax1.plot(tp, peak_adjusted_norm, marker='^', linestyle='--', label=base_cols[peak_index],
        #          color=peak_colors[peak_index])

    # NOTE: default font size is 10
    ax.set_ylabel('Normalized Peak Intensity,  $\Delta I$ / $I_{o}$', labelpad=5, fontsize=12)
    ax.set_xlabel('Time Delay (ps)', fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='lower left', prop={'size': 12}, frameon = True)
    plt.tight_layout()

    if savefigure:
        saveDirectory = 'D:\\Bismuth Project\\Figures for paper'
        plt.savefig(saveDirectory + '\\PeakTraceFigure.pdf', format='pdf')
        plt.savefig(saveDirectory + '\\PeakTraceFigure.svg', format='svg', dpi = 600)
        plt.savefig(saveDirectory + '\\PeakTraceFigure.png', format='png', dpi = 600)

    plt.show()

figure2(scankey = 2, savefigure = 0)

def figure_DebyeWaller(linearplot = True):
    import Linear_fit
    peaksum_23nm_df = pd.read_sql_table('Peak_sum_23nm_adjusted',
                                        'postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project')
    peak_cols = ['peak1', 'peak2', 'peak3', 'peak4', 'peak5', 'peak6', 'peak7',
                 'peak8', 'peak9', 'peak10', 'peak11', 'peak12', 'peak13', 'peak14', 'peak15']
    fluence_label = ['0.78','1.3', '2.6', '5.2', '7.8', '10.4', '15.6']
    fig, ax = plt.subplots()

    if linearplot:
        scankey_loop = np.arange(1,4)
    else:
        scankey_loop = np.arange(1,6)

    print(scankey_loop)
    colors_fluence = cm.Set1(np.linspace(0, 1, len(fluence_label)))
    for scanindex, scankey in enumerate(scankey_loop):
        peak_df = peaksum_23nm_df[peaksum_23nm_df['scan_id'] == scankey]
        peak_choice = 0, 1, 4, 5
        DBy_list = []
        DBx_list = []
        peakposns = np.array([125, 186, 218, 263, 302, 333, 388, 430, 451, 479]) # peakposns are accurate to +/- 1 pixel
        peakposn_k = peakposns*pixel2Ghkl

        for p, peak_index in enumerate(peak_choice):  # loop over peaks
            peak = peak_df.loc[:, peak_cols[peak_index]]
            peak_norm = peak / np.mean(peak[:6])
            DBy = -np.log(np.mean(peak_norm[-5:]))   # average the last 5 peak_norm
            DBx = (peakposn_k[peak_index])
            DBx_list.append(DBx)
            DBy_list.append(DBy)


        axlabel = fluence_label[scanindex] + ' mJ/cm$^2$'
        print(axlabel)
        ax.plot(DBx_list, DBy_list, marker = 'o',markersize = 6, linewidth=1, label= axlabel, color = colors_fluence[scanindex])

        if linearplot:
            dbx_fit, dby_fit, popt, pcov = Linear_fit.mainfit(x=DBx_list, y=DBy_list,
                                                              slope_guess=.4, intercept_guess=0, include_intercept=True)
            lowerbound = 26
            ax.plot(dbx_fit[lowerbound:], dby_fit[lowerbound:], linestyle = '--', color = colors_fluence[scanindex])
            ax.set_xlim(0.25, 0.75)
            ax.set_ylim(-0.05, 0.40)
        else:
            ax.set_xlim(0.25, 0.75)
            ax.set_ylim(-0.05, 1.8)

        ax.set_ylabel('-ln($I(t)/I_{t0}$)', labelpad=5, fontsize=12)
        ax.set_xlabel('k$^2$' + ' ($A^{-2}$) ', fontsize=12)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(loc='upper left', prop={'size': 10}, frameon=False)

    plt.tight_layout()
    # plt.grid(True)
    plt.show()

# figure_DebyeWaller(linearplot=False)
# figure_DebyeWaller(linearplot=True)


def figure_colorbar_flatradavg():
    bigscan23nm_df = pd.read_sql_table('Big_Scan_23nm', 'postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project')
    scan_loop_arange = np.arange(1,8,1) # for 23nm scan
    # scan_loop_arange = np.arange(1,3,1) # for 23nm scan
    print(len(scan_loop_arange))
    liquid_rise_list = [None] * len(scan_loop_arange)
    peak_choices = 0, 1, 4, 5


    for scanindex, scankey in enumerate(scan_loop_arange):  # loop over scans
        fluence = scandict_df[scandict_df['scan_id'] == scankey]['fluence'].values

        # fig, ax = plt.subplots()
        fig_str_label = 'scankey: ' + str(scankey) + ', fluence: ' +str(fluence)
        plt.figure(fig_str_label)

        # plt.title(fig_str_label)
        # scandf = bigscan14nm_df[bigscan14nm_df['scan_key'].isin([scankey])] # filters for scankey, partition the df
        scandf = bigscan23nm_df[bigscan23nm_df['scan_key'].isin([scankey])] # filters for scankey, partition the df
        tp = np.array(scandf['timepoint'])     # extract the 'timepoint' column
        radavgflat = scandf['radavg_flattened']    # extract the 'radavg_flattened' column

        colors = cm.jet(np.linspace(0, 1, len(tp)))    # colormap for plotting

        sm = plt.cm.ScalarMappable(cmap='jet',
                                   norm=plt.Normalize(vmin=np.min(tp), vmax=np.max(tp)))

        for j, radavgflat_plot in enumerate(radavgflat): # this loops over every timepoint

            radavgflat_plot = np.array(radavgflat_plot)  # radavgflat_plot is an array for one timepoint
            plt.plot(radavgflat_plot, color=colors[j], label=str(tp[j]))

            if j == 0:  # for the first timepoint, get peakposn and bases
                peakposn, properties = find_peaks(radavgflat_plot, threshold=None, distance=20, prominence=2, width=2)
                bases = argrelmin(radavgflat_plot, order=20)[0]
                # plt.plot(peakposn, radavgflat_plot[peakposn], 'x')  # mark the peaks with x
                # plt.plot(bases, radavgflat_plot[bases], 'o')  # mark the bases with o

        cbar = plt.colorbar(sm)
        cbar.set_label('Time Delay (ps)')

        plt.xlim(90,370)
        plt.ylim(-300,8500)
        # plt.yticks([])
        plt.ylabel('Intensity (a.u.)', labelpad=5, fontsize=12)
        plt.xlabel(r'k ($\AA^{-1}$)', fontsize=12)
        # plt.legend()
        # plt.grid(True)

    plt.show()

# figure_colorbar_flatradavg()

def figure_colorbar_off():
    bigscan23nm_df = pd.read_sql_table('Big_Scan_23nm', 'postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project')
    scan_loop_arange = np.arange(7,8,1) # for 23nm scan
    # scan_loop_arange = np.arange(1,3,1) # for 23nm scan
    print(len(scan_loop_arange))
    liquid_rise_list = [None] * len(scan_loop_arange)
    peak_choices = 0, 1, 4, 5


    for scanindex, scankey in enumerate(scan_loop_arange):  # loop over scans
        fluence = scandict_df[scandict_df['scan_id'] == scankey]['fluence'].values

        # fig, ax = plt.subplots()
        fig_str_label = 'scankey: ' + str(scankey) + ', fluence: ' +str(fluence)
        plt.figure(fig_str_label)

        # plt.title(fig_str_label)
        # scandf = bigscan14nm_df[bigscan14nm_df['scan_key'].isin([scankey])] # filters for scankey, partition the df
        scandf = bigscan23nm_df[bigscan23nm_df['scan_key'].isin([scankey])] # filters for scankey, partition the df
        tp = np.array(scandf['timepoint'])     # extract the 'timepoint' column
        radavgflat = scandf['radavg_off']    # extract the 'radavg_flattened' column

        colors = cm.jet(np.linspace(0, 1, len(tp)))    # colormap for plotting

        sm = plt.cm.ScalarMappable(cmap='jet',
                                   norm=plt.Normalize(vmin=np.min(tp), vmax=np.max(tp)))

        for j, radavgflat_plot in enumerate(radavgflat): # this loops over every timepoint

            radavgflat_plot = np.array(radavgflat_plot)  # radavgflat_plot is an array for one timepoint
            px = np.arange(0,len(radavgflat_plot))
            x_plot = px*pixel2Ghkl
            plt.plot(x_plot, radavgflat_plot, color=colors[j], label=str(tp[j]))

            if j == 0:  # for the first timepoint, get peakposn and bases
                peakposn, properties = find_peaks(radavgflat_plot, threshold=None, distance=20, prominence=2, width=2)
                bases = argrelmin(radavgflat_plot, order=20)[0]
                # plt.plot(peakposn, radavgflat_plot[peakposn], 'x')  # mark the peaks with x
                # plt.plot(bases, radavgflat_plot[bases], 'o')  # mark the bases with o

        cbar = plt.colorbar(sm)
        cbar.set_label('Time Delay (ps)')

        plt.xlim(0.23, 0.9)
        # plt.ylim(-300,8500)
        plt.ylim(500,9500)
        plt.yticks([])  # empty y-ticks for arbituary units
        plt.ylabel('Intensity (a.u.)', labelpad=5, fontsize=12)
        plt.xlabel(r'k ($\AA^{-1}$)', fontsize=12)
        # plt.legend()
        # plt.grid(True)
    plt.tight_layout()
    plt.show()

figure_colorbar_off()