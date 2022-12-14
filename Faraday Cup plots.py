import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns



def intensity_laserpower_plot(savefigure=False):
    df = pd.read_csv('FaradayCup_data_for_thesis_sheet1.csv', usecols=[0,1,2,3,4,5])

    # print(df.head())
    print(df.info())
    # print(df.voltage.describe())

    voltages = [30,60,90]

    fig, ax = plt.subplots(figsize=(10,6))
    for v in voltages:
        x = df[df['voltage']==v]['laser power']
        y = df[df['voltage']==v]['intensity']*1e-6
        x_err = df[df['voltage']==v]['lp_error']
        y_err = df[df['voltage']==v]['intensity_err']*1e-6
        ax.errorbar(x, y, xerr=x_err, yerr=y_err, label=str(v)+' kV', linestyle='-', lw=1)

    ax.legend(loc='upper right', prop={'size': 11}, frameon = True)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(True, alpha = 0.6)
    ax.set_ylabel('Net Pixel Intensity (a.u.)', labelpad=5, fontsize=11)
    ax.set_xlabel('Laser Power ($\mu W$)', fontsize=11)

    if savefigure:
        print('Saving figure...')
        saveDirectory = 'D:\\Bismuth Project\\Figures for paper'
        plt.savefig(saveDirectory + '\\IntensityVsLaserPower.png', format='png', dpi = 400)

    plt.show()

# intensity_laserpower_plot(savefigure=True)

def spotsize_laserpower_plot(savefigure=False):
    df = pd.read_csv('FaradayCup_data_for_thesis_sheet2.csv')
    print(df.head())
    print(df.info())

    fig, ax = plt.subplots(figsize=(8,3))
    x = df['laser power']
    y = df['mean diameter']
    x_err = df['laser fluctuation']
    y_err = y*x_err/x
    ax.errorbar(x, y, xerr=x_err, yerr=y_err*0.5, lw=0, elinewidth = 1.5, marker='o', ecolor='r', capsize=2)
    ax.grid(True, alpha = 0.6)

    ax.set_ylabel('FWHM spot size at detector (mm)', labelpad=5, fontsize=11)
    ax.set_xlabel('Laser Power ($\mu W$)', fontsize=10)

    ax.set_ylim(0.5, 2.5)
    ax.set_xlim(0, 1000)

    if savefigure:
        print('Saving figure...')
        saveDirectory = 'D:\\Bismuth Project\\Figures for paper'
        plt.savefig(saveDirectory + '\\SpotSizeVsLaserPower.png', format='png', dpi = 400)

    plt.show()

spotsize_laserpower_plot(savefigure=True)