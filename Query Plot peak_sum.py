import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.cm as cm
from scipy.signal import find_peaks
from scipy import interpolate
from matplotlib.font_manager import FontProperties
import time
import csv
from ScanDictionary import scanlist
from Bi_exponential_fit import biexp_fit
from exponential_fit import Exp_Fit
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Boolean, \
    ARRAY, Numeric, Float, ForeignKey, ForeignKeyConstraint, column
from sqlalchemy import insert, update, select, inspect

csv_direc = 'D:\\Bismuth Project\\Bismuth-Data-Analysis-github\\peak_sum_sql_export.csv'
peak_sum_df = pd.read_csv(csv_direc, index_col=[0])

# scan_index = 5
# scan1 = peak_sum_df.loc[[1]]
# scan2 = peak_sum_df.loc[[2]]
# scan3 = peak_sum_df.loc[[3]]
# scan4 = peak_sum_df.loc[[4]]
# scan5 = peak_sum_df.loc[[5]]
# scan6 = peak_sum_df.loc[[6]]
# scan7 = peak_sum_df.loc[[7]]

# scan_df = scan1, scan2, scan3, scan4, scan5, scan6, scan7

cols =['peak1','peak2','peak3','peak4','peak5','peak6','peak7',
'peak8','peak9','peak10','peak11','peak12','peak13','peak14','peak15']

exp_fit_df = pd.DataFrame(columns = ['scan_id', 'peak_id','a', 'tau', 't0', 'c'])
biexp_fit_df = pd.DataFrame(columns = ['scan_id', 'peak_id', 'a1', 'a2', 'tau1', 'tau2', 't0', 'c'])

peak_choice = 0, 1, 2, 3, 4, 5, 6
peak_colors = cm.turbo(np.linspace(0, 1, len(peak_choice)))

for scan_index in np.arange(1,4,1):  # loop over scans
    print('scan_index: ' + str(scan_index))
    fig1, ax1 = plt.subplots()
    # fig1.suptitle('scan_id: ' + str(scan_index))
    # peak_choice = 5, 6

    for peak_index in peak_choice:  # loop over peaks
        print('peak_index: ' + str(peak_index))
        # extract the data I want from the dataframe
        scandf = peak_sum_df.loc[[scan_index]] # partition a block of the dataframe to scandf
        tp = scandf['timepoint']   # x-data
        peak = scandf[cols[peak_index]]  # get raw peak intensity
        peak_norm = peak/np.mean(peak[:7]) # y-data

        pltlabel = 'scan'+str(scan_index)+'-'+cols[peak_index]
        ax1.plot(tp, peak_norm, '-o', label = cols[peak_index], color = peak_colors[peak_index])


        #region BI_EXP_FIT
        # a1, a2, tau1, tau2, t0, c = -0.4, -0.2, 2, 100, 0, 1
        # x_fit, y_fit, fit_var, pcov = biexp_fit(tp, peak_norm, a1, a2, tau1, tau2, t0, c, pltlabel)
        # biexp_new_row = {'scan_id':scan_index, 'peak_id':(peak_index+1),
        #               'a1':fit_var[0], 'a2':fit_var[1], 'tau1':fit_var[2], 'tau2':fit_var[3],
        #               't0':fit_var[4], 'c':fit_var[5]}
        #
        # biexp_fit_df = biexp_fit_df.append(biexp_new_row, ignore_index=True)
        #endregion


        #region Mono EXP FIT
        a, tau, t0, c = -0.2, 2.0, 0, 1

        x_fit, y_fit, fit_var, pcov = Exp_Fit(tp, peak_norm, a, tau, t0, c, pltlabel)
        expfit_new_row = {'scan_id':scan_index, 'peak_id':(peak_index+1),
                      'a':fit_var[0], 'tau':fit_var[1], 't0':fit_var[2], 'c':fit_var[3]}
        exp_fit_df = exp_fit_df.append(expfit_new_row, ignore_index=True)

        ax1.plot(x_fit, y_fit, '--', color = peak_colors[peak_index])
        #endregion


    fig1.suptitle('scan_id: ' + str(scan_index))
    plt.legend()
    plt.grid(True)

plt.show()

# arraytest = tp.to_numpy()
# print(arraytest)
# plt.plot(arraytest)
# plt.show()

# biexp_fit(x, y, a1, a2, tau1_guess, tau2_guess, t0_guess, c_guess, plotlabel)

using_sql = 0
if using_sql:
    # sqlalchemy connection to postgresql database
    engine = create_engine('postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project')
    metadata = MetaData()

    connection = engine.connect()

    scandict = Table('scandictionary', metadata, autoload=True, autoload_with=engine, extend_existing=True)

    peaksum = Table('Peak_sum', metadata, autoload=True, autoload_with=engine, extend_existing=True)

    peak_scan_list = []
    timepoint_list = []

    col = [peaksum.c.peak1, peaksum.c.peak2, peaksum.c.peak3, peaksum.c.peak4, peaksum.c.peak5, peaksum.c.peak6,
    peaksum.c.peak7, peaksum.c.peak8, peaksum.c.peak9, peaksum.c.peak10, peaksum.c.peak11, peaksum.c.peak12,
    peaksum.c.peak13, peaksum.c.peak14, peaksum.c.peak15]

    for scan_select in np.arange(1,7):
        print('scan_select: ' + str(scan_select))
        peak_num_list = []

        for peak_select in np.arange(0,10):

            stmt = select(col[peak_select]).select_from(peaksum.join(scandict)) # how does this join work?
            stmt = stmt.where(scandict.c.scan_id == str(scan_select))
            peak = connection.execute(stmt).fetchall()
            peak_num_list.append(peak)

        peak_scan_list.append(peak_num_list)

        stmt = select(peaksum.c.timepoint)
        stmt = stmt.select_from(peaksum.join(scandict))
        stmt = stmt.where(scandict.c.scan_id == str(scan_select))
        timepoint = connection.execute(stmt).fetchall()
        timepoint_list.append(timepoint)

    for scan_select in np.arange(1,7):
        fig1, ax1 = plt.subplots()
        for peak_select in np.arange(0,10):

            ax1.plot(timepoint_list[scan_select-1], peak_scan_list[scan_select-1][peak_select], '-o')

        plt.grid(True)
    plt.show()


