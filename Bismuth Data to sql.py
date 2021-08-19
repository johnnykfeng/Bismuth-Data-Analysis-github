"""
No fucking clue what I'm doing
"""
import os
import glob
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import find_peaks
from scipy import interpolate
from matplotlib.font_manager import FontProperties
import findCenters
import exponential_fit as exp_fit
import Linear_fit
from astropy.table import QTable, Table, Column
from astropy import units as u
from ScanDictionary import scanlist
import time
import csv
from BaseSubtraction import BaseSubtraction
from Bi_exponential_fit import biexp_fit
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Boolean, ARRAY, Numeric, Float
from sqlalchemy import insert, update, select

# sqlalchemy connection to postgresql database
engine = create_engine('postgresql://postgres:sodapop1@localhost:7981/Bismuth_Project')
metadata = MetaData()

# region MAIN SCAN LOOP
scan_index_length = 1
scan_start = 1
fit_var = []

for scan_index, scan_number in enumerate(np.arange(scan_start, scan_start + scan_index_length, 1)):
    print('+-+-+-+-+-+-+-+- scan index: ' + str(scan_index))
    print('+-+-+-+-+-+-+-+- scan number: ' + str(scan_number))

    # region Data Directory and dictionary structure
    # scan_number = 11
    dataDirec = 'D:\\Bismuth Project\\New Bismuth Data\\' + scanlist[scan_number]['date'] + '\\scans\\scan' + \
                scanlist[scan_number]['scan']
    print('Data Directory: ' + dataDirec)
    plot_title = 'thickness= ' + scanlist[scan_number]['thickness'] + ' $nm$,  ' + \
                 'fluence= ' + scanlist[scan_number]['fluence'] + ' $mJ/cm^2$,  ' + \
                 'exposure= ' + scanlist[scan_number]['exposure'] + ' $s$'
    plot_title_short = 'thickness= ' + scanlist[scan_number]['thickness'] + ' $nm$,  ' + \
                       'fluence= ' + scanlist[scan_number]['fluence'] + ' $mJ/cm^2$'
    # endregion

    # region Toggle Show Plots
    # region pixel2Ghkl functions
    pixel2Ghkl = 2.126e-3  # converts CCD pixels to G_hkl = 1/d_hkl 'scattering vector'


    def pixel2scatt_vector(x):
        return x * pixel2Ghkl


    def scatt_vector2pixel(x):
        return x / pixel2Ghkl


    # endregion
    # ----------------------------------------------------------------
    show_centerfinder = 0
    show_all_diffImg = 0  # toggles diff_Img plot
    show_vertical_lines = 1  # toggles vertical integration lines in diff_Img plot
    show_on_img = 0
    show_off_img = 0
    show_liquid_rise = 0
    show_expfits = 0
    normalize_intensity = 1  # normalize intensity I(t)/I_t0 in expfits
    show_flattened_on_img = 1
    plot_find_peaks = 0
    debye_waller_analysis = 0
    write_csv_file = 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # endregion

    # region Initializing variables for data collection
    onList = []  # for collecting ON images filenames
    offList = []  # for collecting OFF images filenames
    diffList = []  # for collecting DIFF images
    darkBg = np.zeros((1000, 1148))
    scatterBg = np.zeros((1000, 1148))
    onImg = np.zeros((1000, 1148))
    offImg = np.zeros((1000, 1148))
    sum_all_img = np.zeros((1000, 1148))
    t_list = []  # converts timepoint data to ps
    # for plotting the liquid rise
    lr_range = (148, 190)
    lr_data = []
    dby = []  # collecting y data for debye-waller analysis
    dbx = []  # collecting x data for debye-waller analysis
    y_peaks = []
    # endregion

    # region Image File Extraction - Computing On, Off, Diff Img

    # region Loading data images from Directory
    for file in glob.glob(os.path.join(dataDirec, '*.tiff')):
        # Iterate through file, find the ones we want.
        if (os.path.basename(file).split('_')[5] == 'On'):
            onList.append(file)
        if (os.path.basename(file).split('_')[5] == 'Off'):
            offList.append(file)
    print('loading on and off images successful')
    try:
        darkBg = io.imread(os.path.join(dataDirec, 'darkExp.TIF'))
        scatterBg = io.imread(os.path.join(dataDirec, 'pumpExp.TIF'))
        print('reading darkBg and scatterBg successful')
    except:
        print('dark images not okay!')
    print('Number of data points: ' + str(len(onList)))
    # endregion
    for j in range(len(onList)):
        # timepoint = os.path.basename(onList[j]).split('_')[3]  # reads time delay from file name
        # imgnum = os.path.basename(onList[j]).split('_')[1]  # not really used anymore
        onImg = io.imread(onList[j]) - scatterBg
        # onImg = io.imread(onList[j])
        offImg = io.imread(offList[j]) - darkBg
        # offImg = io.imread(offList[j])
        diffImg = (onImg - offImg)
        diffList.append(diffImg)
        sum_all_img = sum_all_img + offImg  # this variable sometimes isn't used
    # endregion

    # region FindCenter control
    start_timer = time.time()
    initCenter = (598, 543)
    step = 8;
    radiusAvgMin = 140;
    radiusAvgMax = 150
    center0 = findCenters.meanAroundRing(sum_all_img, initCenter, step, radiusAvgMin, radiusAvgMax, show_centerfinder)
    find_center_runtime = time.time() - start_timer
    print('Find Center Runtime = ' + str(find_center_runtime))
    # endregion

    # region New Peak Finding using BaseSubtraction.py
    image_index = 1  # first ON image, usually before t0
    rad_avg_img = findCenters.radialProfile(io.imread(onList[image_index]), center0)  # image
    flattened_rad_avg, peakposition, properties, bases1, bases2, bases3, spline_bg1, spline_bg2 = BaseSubtraction(
        rad_avg_img, 1, 0)
    plt.figure('Demonstrate peak and base finding, flattened_rad_avg')
    plt.plot(flattened_rad_avg)
    pp = peakposition[:-2]
    plt.plot(pp, flattened_rad_avg[pp], 'x')
    plt.plot(bases3, flattened_rad_avg[bases3], 'o')
    plt.grid(True)
    # endregion

    # region Control plotting and integration variables

    # plotting_index basically is indexing the timepoints I want to plot
    plotting_index = np.arange(1, len(onList))
    # plotting_index = np.arange(1, 20, 2)

    peak_choices = (1, 2, 5, 6)
    # peak_choices = (1, 2, 3, 4, 5, 6, 7, 11)
    peak_step = 6
    peak_sum = np.zeros((len(peak_choices), len(plotting_index),
                         scan_index_length))  # empty array for collecting the integrated peak data
    # endregion

    # region Plotting colors and fonts
    colors = cm.jet(np.linspace(0, 1, len(onList)))
    peak_colors = cm.turbo(np.linspace(0, 1, len(peak_choices)))
    colors_fluence = cm.Set1(np.linspace(0, 1, scan_index_length))
    fontP = FontProperties()
    fontP.set_size('x-small')
    # endregion

    # Using sqlalchemy to create table for storing radial average data
    connection = engine.connect()

    # delete_table = Table('test', metadata)
    # delete_table.drop(engine)
    # print(delete_table.exists(engine))

    scan_table = Table('test2', metadata,
                       Column('timepoint', Numeric()),
                       Column('radavg_off', ARRAY(Float())),
                       Column('radavg_on', ARRAY(Float())),
                       Column('radavg_diff', ARRAY(Float())),
                       Column('radavg_flattened', ARRAY(Float())),
                       extend_existing=True)


    metadata.create_all(engine)
    # engine.table_names()   # should this be printed?
    print(scan_table.exists(engine))

    # region Main Loop for Computing and Plotting Radial Average
    start_timer = time.time()
    for i in plotting_index:
        # print(i)
        timepoint = os.path.basename(onList[i]).split('_')[3]
        imgnum = os.path.basename(onList[i]).split('_')[1]
        t_ps = int(timepoint) * 1e-3
        t_list.append(t_ps)  # accumulates the timepoint as ps
        # center0 = findCenters.meanAroundRing(io.imread(offList[i]), initCenter, 3, radiusAvgMin, radiusAvgMax, 0)
        rad_avg_diff = findCenters.radialProfile(diffList[i], center0)
        rad_avg_on = findCenters.radialProfile(io.imread(onList[i]), center0)
        rad_avg_off = findCenters.radialProfile(io.imread(offList[i]), center0)
        flattened_rad_avg_on = rad_avg_on[min(bases1):max(bases1)] - spline_bg1
        flattened_rad_avg_on = flattened_rad_avg_on[min(bases2):max(bases2)] - spline_bg2
        lr_data.append(sum(rad_avg_diff[lr_range[0]:lr_range[1]]))  # integrates pixels for liquid rise analysis

        insert_stmt = insert(scan_table).values(timepoint=t_ps, radavg_off=rad_avg_off,
                                                radavg_on=rad_avg_on, radavg_diff=rad_avg_diff,
                                                radavg_flattened=flattened_rad_avg_on)
        result_proxy = connection.execute(insert_stmt)


        if show_all_diffImg:
            plt.figure(
                '[diffImg] ' + 'fluence:' + scanlist[scan_number]['fluence'] + ', thickness:' + scanlist[scan_number][
                    'thickness'])
            if i % 2 == 1:
                # plt.plot(radial_distance*radius2dhkl, rad_avg_diff, color=colors[i], linewidth=2, linestyle='-', label=str(t_ps) + 'ps ')
                plt.plot(rad_avg_diff, color=colors[i], linewidth=2, linestyle='-', label=str(t_ps) + 'ps ')
            else:
                # plt.plot(radial_distance*radius2dhkl, rad_avg_diff, color=colors[i], linewidth=2, linestyle='--', label=str(t_ps) + 'ps ')
                plt.plot(rad_avg_diff, color=colors[i], linewidth=2, linestyle='--', label=str(t_ps) + 'ps ')
            if show_vertical_lines:
                for p in peak_choices:
                    plt.axvline(x=(peakposition[p] - peak_step), linestyle='--')
                    plt.axvline(x=(peakposition[p] + peak_step), linestyle='--')
            # if show_vertical_lines:
            #     for p in peak_choices:
            #         plt.axvline(x_bases=(peakposition[p]-peak_step), linestyle='--')
            #         plt.axvline(x_bases=(peakposition[p]+peak_step), linestyle='--')

            plt.legend(title='timepoints', bbox_to_anchor=(1, 1), loc='upper left', prop=fontP)
            plt.xlim(125, 600)
            plt.title(plot_title)
            plt.xlabel('Radius from center (pixel)')
            plt.ylabel('Radial average difference intensity')
            plt.grid(True)

        if show_off_img:
            plt.figure(
                'OFF - ' + 'fluence:' + scanlist[scan_number]['fluence'] + ', thickness:' + scanlist[scan_number][
                    'thickness'])
            if i % 2 == 1:
                plt.plot(rad_avg_off, color=colors[i], linewidth=2, linestyle='-', label=str(t_ps) + 'ps ')
            else:
                plt.plot(rad_avg_off, color=colors[i], linewidth=2, linestyle='--', label=str(t_ps) + 'ps ')

            if show_vertical_lines:
                for p in peak_choices:
                    plt.axvline(x=(peakposition[p] - peak_step), linestyle='--')
                    plt.axvline(x=(peakposition[p] + peak_step), linestyle='--')

            plt.legend(title='timepoints', bbox_to_anchor=(1, 1), loc='upper left', prop=fontP)
            plt.xlim(125, 400)
            plt.ylim(0, 12500)
            plt.title(plot_title)
            plt.xlabel('Radius from center (pixel)')
            plt.ylabel('Radial average difference intensity')
            plt.grid(True)  #

        if show_on_img:
            plt.figure('ON Images')
            if i % 2 == 1:
                plt.plot(rad_avg_on, color=colors[i], linewidth=2, linestyle='-', label=str(t_ps) + 'ps ')
            else:
                plt.plot(rad_avg_on, color=colors[i], linewidth=2, linestyle='--', label=str(t_ps) + 'ps ')

            plt.legend(title='timepoints', bbox_to_anchor=(1, 1), loc='upper left', prop=fontP)
            plt.xlim(min(bases), max(bases))

            plt.title(plot_title)
            plt.xlabel('Radius from center (pixel)')
            plt.ylabel('Radial average difference intensity')
            plt.grid(True)

        # flattened_rad_avg_on = rad_avg_on[min(bases):max(bases)] - spline_bg
        # show_flattened_on_img = 0
        if show_flattened_on_img:
            plt.figure(
                'fluence:' + scanlist[scan_number]['fluence'] + ', thickness:' + scanlist[scan_number]['thickness'])

            if i % 2 == 1:
                # axs[scan_index].plot(flattened_rad_avg_on , color=colors[i], linewidth=2, linestyle='-', label=str(t_ps) + 'ps ')
                plt.plot(flattened_rad_avg_on, color=colors[i], linewidth=2, linestyle='-', label=str(t_ps) + 'ps ')
            else:
                # axs[scan_index].plot(flattened_rad_avg_on , color=colors[i], linewidth=2, linestyle='--', label=str(t_ps) + 'ps ')
                plt.plot(flattened_rad_avg_on, color=colors[i], linewidth=2, linestyle='--', label=str(t_ps) + 'ps ')

            if show_vertical_lines:
                for p in peak_choices:
                    plt.axvline(x=(peakposition[p] - peak_step), linestyle='--')
                    plt.axvline(x=(peakposition[p] + peak_step), linestyle='--')
                # LABEL peaks in the plots
            for t in range(len(peakposition)):
                peakposn_text = str(peakposition[t])
                plt.text(peakposition[t], flattened_rad_avg[peakposition][t] - 50, str(t),
                         fontweight='bold')  # labels the peak index number

            plt.legend(title='timepoints', bbox_to_anchor=(1, 1), loc='upper left', prop=fontP)
            plt.xlim(100, 400)
            plt.ylim(-500, 8500)
            plt.title(plot_title)
            plt.xlabel('Radius from center (pixel)')
            plt.ylabel('Radial average difference intensity')
            plt.grid(True)

        # Integrates the peaks in the radial average and accumulates them in the 'peak_sum' array

    for time_index in plotting_index:
        for p_index, p in enumerate(peak_choices):
            peak_sum[p_index, time_index - 1, scan_index] = sum(
                flattened_rad_avg_on[peakposition[p] - peak_step: peakposition[p] + peak_step])

    radialprofile_runtime = time.time() - start_timer
    print('radialprofile_runtime = ' + str(radialprofile_runtime))
    # endregion

plt.show()
