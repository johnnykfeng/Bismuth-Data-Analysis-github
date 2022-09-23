import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
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

def figure_fluence_scan(savefigure=False):

    fitvar_df = pd.read_csv('All_fit_constants_w_err.csv')
    print(fitvar_df.tail())
    print(fitvar_df.info())

    df = pd.read_csv('Normalized_peaks.csv')
    df.drop(df[df.scan_id == 7].index, inplace=True)
    print(df.tail())
    print(df.info())

    fluence_dict = {1:'0.78', 2:'1.3', 3:'2.6', 4:'5.2', 5:'7.8', 6:'10.4'}
    df['fluence'] = df['scan_id'].map(fluence_dict)
    print(df.tail())
    print(df.info())

    scan_colors = cm.plasma(np.linspace(0, 1, 8))
    print('scan_colors: ')
    print(scan_colors)

    from exponential_fit import Exp_Fit

    fig, ax = plt.subplots()

    for i, fluence in enumerate(df['fluence'].unique()):
        print('fluence: ', fluence)

        fitvar_scan = fitvar_df[(fitvar_df['fluence']==np.float64(fluence)) & (fitvar_df['peak_id']==6) ]
        print(fitvar_scan)
        df_scan = df[df['fluence'] == fluence]
        tp = df_scan['timepoint']
        peak_norm = df_scan['peak6_norm']

        # extracting fit variables
        a, tau, t0, c = fitvar_scan['a'].values, fitvar_scan['tau'].values, fitvar_scan['t0'].values, fitvar_scan['c'].values
        print(a, tau, t0, c)
        x_fit, y_fit, fit_var, pcov = Exp_Fit(tp, peak_norm, a, tau, t0, c)
        ax.plot(x_fit - fit_var[2], y_fit, '--', color=scan_colors[i])
        ax.plot(tp - fit_var[2], peak_norm, '-o', linewidth=0.5, color=scan_colors[i], label = str(fluence) )

    ax.set_ylabel('Normalized Intensity of (022) peak,  $\Delta I$ / $I_{o}$', labelpad=5, fontsize=12)
    ax.set_xlabel('Time Delay (ps)', fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='lower left', prop={'size': 12}, frameon = True, title= 'Fluence ($mJ/cm^2$)')
    ax.grid(linestyle='--', linewidth = 0.8)
    plt.tight_layout()

    if savefigure:
        print('Saving figure.')
        saveDirectory = 'D:\\Bismuth Project\\Figures for paper'
        plt.savefig(saveDirectory + '\\FluenceScanPeak6.pdf', format='pdf')
        plt.savefig(saveDirectory + '\\FluenceScanPeak6.svg', format='svg', dpi = 600)
        plt.savefig(saveDirectory + '\\FluenceScanPeak6.png', format='png', dpi = 600)

    plt.show()

figure_fluence_scan(savefigure=True)

def TimeZeroAccuracy(savefigure=False):

    fluence = [0.78, 1.3, 2.6, 5.2, 7.8, 10.4]
    peak1_T0err = [0.230, 0.159, 0.169, 0.100, 0.097, 0.114]
    peak2_T0err = [0.435, 0.399, 0.403, 0.191, 0.167, 0.224]
    peak5_T0err = [0.923, 0.733, 0.594, 0.218, 0.258, 0.280]
    peak6_T0err = [0.166, 0.154, 0.127, 0.059, 0.056, 0.063]

    avg_T0err = [0.439, 0.361, 0.323, 0.142, 0.144, 0.170]

    fig, ax = plt.subplots()
    ax.plot(fluence, peak1_T0err, '--o', alpha=0.5, label='(01-1) peak')
    ax.plot(fluence, peak2_T0err, '--o', alpha=0.5, label='(002) peak')
    ax.plot(fluence, peak5_T0err, '--o', alpha=0.5, label='(011) peak')
    ax.plot(fluence, peak6_T0err, '--o', alpha=0.5, label='(022) peak')
    # ax.plot(fluence[1:], avg_T0err[1:], '--o')
    ax.plot(fluence, avg_T0err, '-o', label='Average all peaks', ms=10, linewidth=5)

    ax.set_ylabel('$T_0$ accuracy (ps)', labelpad=5, fontsize=12)
    ax.set_xlabel('Fluence ($mJ/cm^2$)', fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='upper right', prop={'size': 12}, frameon=True)
    # ax.grid(linestyle='--', linewidth=0.8)
    plt.tight_layout()


    if savefigure:
        print('Saving figure.')
        saveDirectory = 'D:\\Bismuth Project\\Figures for paper'
        plt.savefig(saveDirectory + '\\TimeZeroAccuracy.pdf', format='pdf')
        plt.savefig(saveDirectory + '\\TimeZeroAccuracy.svg', format='svg', dpi=600)
        plt.savefig(saveDirectory + '\\TimeZeroAccuracy.png', format='png', dpi=600)
    plt.show()

# TimeZeroAccuracy(savefigure=True)


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

#figure2(scankey = 2, savefigure = 0)

def figure_DebyeWaller(linearplot = True):
    import Linear_fit
    peaksum_23nm_df = pd.read_sql_table('Peak_sum_23nm_adjusted',
                                        'postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project')
    peak_cols = ['peak1', 'peak2', 'peak3', 'peak4', 'peak5', 'peak6', 'peak7',
                 'peak8', 'peak9', 'peak10', 'peak11', 'peak12', 'peak13', 'peak14', 'peak15']
    fluence_label = ['0.78','1.3', '2.6', '5.2', '7.8', '10.4', '15.6']
    fig, ax = plt.subplots(figsize = (7.5,5.5))

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
        ax.legend(loc='upper left', prop={'size': 10}, frameon=True)

    plt.tight_layout()
    # plt.grid(True)
    plt.show()

# figure_DebyeWaller(linearplot = True)

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

# def figure_animate():

def animation():
    from matplotlib import animation
    import time
    bigscan23nm_df = pd.read_sql_table('Big_Scan_23nm', 'postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project')

    scan_loop_arange = np.arange(3,4,1) # for 23nm scan
    print(scan_loop_arange)

    for scanindex, scankey in enumerate(scan_loop_arange):  # loop over scans
        fluence = scandict_df[scandict_df['scan_id'] == scankey]['fluence'].values  # extract fluence value
        fig_str_label = 'scankey: ' + str(scankey) + ', fluence: ' +str(fluence)
        # fig = plt.figure(fig_str_label)

        # fig = plt.figure(figsize=(8,6))
        # ax = fig.add_subplot()

        scandf = bigscan23nm_df[bigscan23nm_df['scan_key'] == scankey] # filters for scankey, partition the df
        tp = np.array(scandf['timepoint'])     # extract the 'timepoint' column
        radavgflat = scandf['radavg_flattened']    # extract the 'radavg_flattened' column
        # print(radavgflat.loc[80])
        print(tp)

        colors = cm.jet(np.linspace(0, 1, len(tp)))    # colormap for plotting
        sm = plt.cm.ScalarMappable(cmap='jet',
                                   norm=plt.Normalize(vmin=np.min(tp), vmax=np.max(tp)))

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot()
        ax.set_xlim(0.23, 0.75)
        ax.set_ylim(0, 1.0e4)
        # ax.set_yticks([])  # empty y-ticks for arbituary units
        # ax.set_ylabel('Intensity (a.u.)', labelpad=5, fontsize=12)
        # ax.set_xlabel(r'k ($\AA^{-1}$)', fontsize=12)
        ax.grid(True)

        # time_template = 'time = %d'
        time_template = 'time delay = %.1f ps'
        time_text = ax.text(0.5, 0.9, '', transform=ax.transAxes)

        line, = ax.plot([],[], '-')

        def animate_scan(i):
            print(f'{i}, {tp[i]}' )
            radavgflat_plot = np.array(radavgflat.loc[80+i])
            x_plot = np.arange(0,len(radavgflat_plot))*pixel2Ghkl

            # line.set_data = (x_plot, radavgflat_plot)
            # line = ax.plot(x_plot, radavgflat_plot, color=colors[i], label=str(tp[i]))
            line.set_data(x_plot, radavgflat_plot)
            time_text.set_text(time_template % tp[i])
            return line, time_text


    anim = animation.FuncAnimation(fig, animate_scan, frames = len(tp), interval=250, blit=1)
    plt.show()

    # figure_animate()

def figure_colorbar_off(savefigure=False):
    bigscan23nm_df = pd.read_sql_table('Big_Scan_23nm', 'postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project')
    scan_loop_arange = np.arange(7,8,1) # for 23nm scan
    # scan_loop_arange = np.arange(1,3,1) # for 23nm scan
    print(scan_loop_arange)

    for scanindex, scankey in enumerate(scan_loop_arange):  # loop over scans
        fluence = scandict_df[scandict_df['scan_id'] == scankey]['fluence'].values
        # fig, ax = plt.subplots()
        fig_str_label = 'scankey: ' + str(scankey) + ', fluence: ' +str(fluence)
        plt.figure(fig_str_label)

        scandf = bigscan23nm_df[bigscan23nm_df['scan_key'].isin([scankey])] # filters for scankey, partition the df
        tp = np.array(scandf['timepoint'])     # extract the 'timepoint' column
        radavgflat = scandf['radavg_off']    # extract the 'radavg_flattened' column

        colors = cm.jet(np.linspace(0, 1, len(tp)))    # colormap for plotting
        sm = plt.cm.ScalarMappable(cmap='jet',
                                   norm=plt.Normalize(vmin=np.min(tp), vmax=np.max(tp)))

        for j, radavgflat_plot in enumerate(radavgflat): # this loops over every timepoint

            radavgflat_plot = np.array(radavgflat_plot)  # radavgflat_plot is an array for one timepoint
            x_plot = np.arange(0,len(radavgflat_plot))*pixel2Ghkl  # converts the pixels to scattering vector k
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

    if savefigure:
        saveDirectory = 'D:\\Bismuth Project\\Figures for paper\\'
        figureName = 'OffRadAvg_colorbar'
        plt.savefig(saveDirectory + figureName +'.pdf', format='pdf', dpi = 1200)
        plt.savefig(saveDirectory + figureName +'.svg', format='svg', dpi = 1200)
        plt.savefig(saveDirectory + figureName +'.png', format='png', dpi = 1200)
        plt.savefig(saveDirectory + figureName +'.eps', format='eps', dpi = 1200)

#figure_colorbar_off(savefigure = True)

def figure_liquid_rise():
    liquidrise_df = pd.read_sql_table('liquidrise_23nm', 'postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project')

    base_columns = ['base1', 'base2', 'base3', 'base4', 'base5', 'base6', 'base7']
    colors = cm.jet(np.linspace(0, 1, len(base_columns)))  # colormap for plotting

    scan_loop_arange = np.arange(1,6,1)
    # plotting the Liquid Rise from the data frame
    for scanindex, scankey in enumerate(scan_loop_arange):  # loop over scans
        plt.figure('LR scankey: ' + str(scankey))

        for b_index, base_col in enumerate(base_columns):
            x = liquidrise_df[liquidrise_df['scan_pk'] == scankey]['timepoint']
            y = liquidrise_df[liquidrise_df['scan_pk'] == scankey][base_col]

            y_norm = y- np.mean(y.iloc[0:5])
            plt.plot(x, y_norm, label=base_col, color=colors[b_index], marker='o')

        plt.grid(True)
        plt.legend()
    plt.show()

def figure_liquid_rise2():
    liquidrise_df = pd.read_sql_table('liquidrise_23nm', 'postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project')

    base_columns = ['base1', 'base2', 'base3', 'base4', 'base5', 'base6', 'base7']
    fluence_labels = ['0.78', '1.3', '2.6', '5.2', '7.8', '10.4', '15.6']
    # colors = cm.jet(np.linspace(0, 1, len(base_columns)))  # colormap for plotting

    scan_loop_arange = np.arange(1,6,1)
    colors = cm.jet(np.linspace(0, 1, 7))  # colormap for plotting
    # plotting the Liquid Rise from the data frame
    for b_index, base_col in enumerate(base_columns):

        # plt.figure('Scankey: ' + str(scankey) +' Fluence: '+fluence_labels[scanindex])
        plt.figure(base_columns[b_index])

        for scanindex, scankey in enumerate(scan_loop_arange):  # loop over scans
            x = liquidrise_df[liquidrise_df['scan_pk'] == scankey]['timepoint']
            y = liquidrise_df[liquidrise_df['scan_pk'] == scankey][base_col]

            y_norm = y- np.mean(y.iloc[0:5])
            plt.plot(x, y_norm, color=colors[scanindex],
                     marker='o', label =fluence_labels[scanindex])

        plt.grid(True)
        plt.legend()
    plt.show()

# figure_liquid_rise2()

def fit_variables_calc():
    fit_var = pd.read_sql_table('exp_fit_variables_23nm', 'postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project')
    # fit_var.info()

# fit_variables_calc()

