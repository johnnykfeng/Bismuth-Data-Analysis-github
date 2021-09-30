import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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


# sqlalchemy connection to postgresql database
engine = create_engine('postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project')
metadata = MetaData()
connection = engine.connect()
# connect to sql database
#scandict = Table('scandictionary', metadata, autoload=True, autoload_with=engine, extend_existing=True)
#bigscan14nm = Table('Big_Scan_14nm', metadata, autoload=True, autoload_with=engine, extend_existing=True)
# convert sql to pandas dataframe
scandict_df = pd.read_sql_table('scandictionary', 'postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project')
bigscan14nm_df = pd.read_sql_table('Big_Scan_14nm', 'postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project')
bigscan23nm_df = pd.read_sql_table('Big_Scan_23nm', 'postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project')

peak_choices = 1,2,3,4,5,6,7

sum_range = 6
#scan_loop_arange = np.arange(8, 13, 1) # for 14nm scan
scan_loop_arange = np.arange(1,8,1) # for 23nm scan
# print(len(scan_loop_arange))
liquid_rise_list = [None] * len(scan_loop_arange)

# list of column names for he
dict_columns = ['scan_pk', 'timepoint', 'base1', 'base2', 'base3', 'base4', 'base5', 'base6', 'base7']

lr_dict = []

for scanindex, scankey in enumerate(scan_loop_arange):  # loop over scans

    plt.figure('scankey: ' + str(scankey))
    # scandf = bigscan14nm_df[bigscan14nm_df['scan_key'].isin([scankey])] # filters for scankey, partition the df
    scandf = bigscan23nm_df[bigscan23nm_df['scan_key'].isin([scankey])] # filters for scankey, partition the df
    tp = np.array(scandf['timepoint'])     # extract the 'timepoint' column
    radavgflat = scandf['radavg_flattened']    # extract the 'radavg_flattened' column

    colors = cm.jet(np.linspace(0, 1, len(tp)))    # colormap for plotting

    liquid_rise = np.zeros((len(peak_choices), (len(tp))))
    print(len(tp))
    for j, radavgflat_plot in enumerate(radavgflat): # this loops over every timepoint

        radavgflat_plot = np.array(radavgflat_plot)  # radavgflat_plot is an array for one timepoint
        plt.plot(radavgflat_plot, color=colors[j], label=str(tp[j]))

        if j == 0:  # for the first timepoint, get peakposn and bases
            peakposn, properties = find_peaks(radavgflat_plot, threshold=None, distance=20, prominence=2, width=2)
            bases = argrelmin(radavgflat_plot, order=20)[0]
            plt.plot(peakposn, radavgflat_plot[peakposn], 'x')  # mark the peaks with x
            plt.plot(bases, radavgflat_plot[bases], 'o')  # mark the bases with o

            for pc in peak_choices:
                plt.axvline(x=(bases[pc] - sum_range), linestyle = '--')
                plt.axvline(x=(bases[pc] + sum_range), linestyle = '--')

        for p_index, p in enumerate(peak_choices):
            liquid_rise[p_index, j] = sum(radavgflat_plot[bases[p] - sum_range : bases[p]+ sum_range])

        #region Organize the liquid rise data into dictionary
        liquid_rise_list[scanindex]= liquid_rise
        row_values = [scankey, tp[j], liquid_rise[0, j], liquid_rise[1, j], liquid_rise[2, j],
                      liquid_rise[3, j], liquid_rise[4, j], liquid_rise[5, j], liquid_rise[6, j]]

        dictrow = {dict_columns[a]: row_values[a] for a in np.arange(9)}
        lr_dict.append(dictrow)
        #endregion

    plt.legend()
    plt.grid(True)
plt.show()

# turn this into a proper dataframe?!?!?!?!
liquidrise_DF = pd.DataFrame(lr_dict, columns = dict_columns)

# exports liquidrise_DF to PostgreSQL
#liquidrise_DF.to_sql('liquidrise_14nm', con=engine)

# liquid_rise_list[scanindex][peakchoice, timepoint]
base_columns = ['base1', 'base2', 'base3', 'base4', 'base5', 'base6', 'base7']
colors = cm.jet(np.linspace(0, 1, len(base_columns)))  # colormap for plotting

# plotting the Liquid Rise from the data frame
for scanindex, scankey in enumerate(scan_loop_arange):  # loop over scans
    # scandf = liquidrise_DF[liquidrise_DF['scan_key'].isin([scankey])

    plt.figure('LR scankey: ' + str(scankey))

    for b_index, base_col in enumerate(base_columns):
        x = liquidrise_DF[liquidrise_DF['scan_pk'] == scankey]['timepoint']
        y = liquidrise_DF[liquidrise_DF['scan_pk'] == scankey][base_col]

        y_norm = y- np.mean(y.iloc[0:5])
        plt.plot(x, y_norm, label= base_col, color = colors[b_index], marker='o')


    plt.grid(True)
    plt.legend()

plt.show()

