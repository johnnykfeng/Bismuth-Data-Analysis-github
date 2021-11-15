"""
Last edited: 2021-02-19 
"""
import os
import glob
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import findCenters
from matplotlib.font_manager import FontProperties
import exponential_fit as exp_fit
from astropy.table import QTable, Table, Column
from astropy import units as u

from scipy.signal import find_peaks
from scipy import interpolate
#---------------INITIALIZE VARIABLES -----------------------------#

plt.close('all')
dataDirec = 'D:\\Bismuth Project\\New Bismuth Data\\2021-02-19\\scans\\SingleShotScans'
print(dataDirec)

show_diff_img = 0
show_on_img = 0
show_off_img = 0
show_integrated_peaks = 0
show_centerfinding = 0
show_flattened_on_img = 1
show_vertical_lines = 1
show_expfits = 1

peak1_range = (125, 155)
peak1_data = []
peak2_range = (190, 210)
peak2_data = []

#----------------------------------------------------------------------#
# region pixel2Ghkl functions
pixel2Ghkl = 2.126e-3  # converts CCD pixels to G_hkl = 1/d_hkl 'scattering vector'
def pixel2scatt_vector(x):
    return x * pixel2Ghkl
def scatt_vector2pixel(x):
    return x / pixel2Ghkl
# endregion

onList = []
offList = []
pumpExpList = []
darkExpList = []
onImgList = []
offImgList = []
diffImgList = []
time_integers=[]
t_list = []
lr_data= []
lr_range = (148, 190)
onImg = np.zeros((1000,1148))
offImg = np.zeros((1000,1148))
darkExp = np.zeros((1000,1148))
pumpExp = np.zeros((1000,1148))

for file in glob.glob(os.path.join(dataDirec, '*.tiff')):
    # Iterate through file, find the ones we want.
    if (os.path.basename(file).split('_')[5] == 'On'):
        onList.append(file)
    if (os.path.basename(file).split('_')[5] == 'Off'):
        offList.append(file)
print('loading image files successful!')

bgDirec = 'D:\\Bismuth Project\\New Bismuth Data\\2021-02-19\\scans\\SingleShotScans\\background'
# for file in glob.glob(os.path.join(bgDirec, '*.tiff')):
#     if (os.path.basename(file).split('_')[5] == 'On'):
#         pumpExpList.append(io.imread(file))
#     if (os.path.basename(file).split('_')[5] == 'Off'):
#         darkExpList.append(io.imread(file))
# pumpExp = sum(pumpExpList)
# darkExp = sum(darkExpList)

try:
    darkExp = io.imread(os.path.join(bgDirec, 'darkExp.tiff'))
    pumpExp = io.imread(os.path.join(bgDirec, 'pumpExp.tiff'))
    print('loading background successful!')
except:
    print('dark images not okay!')

Nshots = 4 # not used in this code

avgImg = np.zeros((1000, 1148))
sumImg = np.zeros((1000, 1148))
sumOff = np.zeros((1000, 1148))
sumOn = np.zeros((1000, 1148))
sumDiff = np.zeros((1000, 1148))

# script for generating timepoints in correct string format
# timepoint = np.arange(-1000,4000,500).astype(int)
timepoint = np.array([-5000,-2000,-1000,0,500,1000,1500,2000,2500,3000,3500,4000,5000,6000,8000])
extra=np.array([15000])
# extra = np.array([5000, 6000, 8000, 10000, 12000, 15000, 20000, 50000, 100000, 500000])
timepointfinal = np.sort(np.concatenate((timepoint,extra)))
timepoint_str=[]
for i in range(len(timepointfinal)):
    timepoint_str.append("{:06d}".format(timepointfinal[i]))
print(timepoint_str)

print('Total number of timepoints =' + str(len(timepoint_str)))
print('Number of of total ON images=' + str(len(onList)))
for n, time in enumerate(timepoint_str):
    print('Processing images in timepoint:' + str(time))
    time_integers.append(int(time)/1000)  # converts the timepoints into integer units of ps
    sumOff = np.zeros((1000, 1148))
    sumOn = np.zeros((1000, 1148))
    sumDiff = np.zeros((1000, 1148))

    #loop for summing the images of each timepoint
    for i in range(len(onList)):
        if (os.path.basename(onList[i]).split('_')[3] == time):  #reads the ON img for each timepoint
            # onImg = io.imread(onList[i]) - pumpExp
            onImg = io.imread(onList[i])
        if (os.path.basename(offList[i]).split('_')[3] == time): #reads the OFF img for each timepoint
            # offImg = io.imread(offList[i]) - darkExp
            offImg = io.imread(offList[i])

        sumOn = sumOn + onImg  # accumulates the ON images
        sumOff = sumOff + offImg # accumulates the OFF images
        sumDiff = sumOn - sumOff # calculates the DIFF images

    perform_averaging=0   # just divides the summed images by Nshots, not necessary
    if perform_averaging:
        sumOn = sumOn/Nshots
        sumOff = sumOff/Nshots
        sumDiff = sumDiff/Nshots

    # collects the summed images in lists
    # this should be the largest memory used
    diffImgList.append(sumDiff)
    onImgList.append(sumOn)
    offImgList.append(sumOff)

allOff = sum(offImgList)

#region FindCenter control
initCenter = (600,545)
step = 6
radiusAvgMin = 130
radiusAvgMax = 155
center0 = findCenters.meanAroundRing(allOff, initCenter, step, radiusAvgMin, radiusAvgMax, show_centerfinding)
#endregion

# region Control plotting and integration variables
plotting_index = np.arange(0, len(timepoint_str))  # skip certain timepoints in the scan, such as the first
# peak_choices = (1, 2, 5, 6)
peak_choices = (1, 2)
peak_step = 4
peak_sum = np.zeros((len(peak_choices), len(plotting_index)))  # empty array for collecting the integrated peak data
# endregion

# region Plotting colors and fonts
colors = cm.jet(np.linspace(0, 1, len(onList)))
peak_colors = cm.turbo(np.linspace(0, 1, len(peak_choices)))
fontP = FontProperties()
fontP.set_size('x-small')
# endregion

FIND_PEAKS = 1
if FIND_PEAKS:
    image_index = 1  # first ON image, usually before t0
    timepoint = os.path.basename(onList[image_index]).split('_')[3]
    rad_avg_peakimg = findCenters.radialProfile(onImgList[image_index], center0)
    # ----------- MAIN PEAK FINDING LINE OF CODE ------------ #
    peakposn, properties = find_peaks(rad_avg_peakimg, threshold=None, distance=20, prominence=1, width=2)
    bases = properties['left_bases']  # finds the bases for each peak
    bases = np.append(bases, [375, 420])
    f = interpolate.interp1d(bases, rad_avg_peakimg[bases])

    x_bases = np.arange(min(bases), max(bases), 1)
    spline_bg = f(x_bases)  # spline of bases, to be subtracted as background
    flattened_rad_avg = rad_avg_peakimg[min(bases): max(bases)] - spline_bg
    # flattened_rad_avg = flattened_rad_avg - min(flattened_rad_avg[100:])
    peakposition, properties = find_peaks(flattened_rad_avg, threshold=None, distance=20, prominence=5, width=5)

    plot_find_peaks = 1
    if plot_find_peaks:
        fig, ax = plt.subplots()

        ax.plot(rad_avg_peakimg, color=colors[-image_index], linewidth=1, linestyle='--', label='raw rad avg')
        ax.plot(peakposn, rad_avg_peakimg[peakposn], "x", markersize=5, color='blue')

        ax.plot(bases, rad_avg_peakimg[bases], 'o', color='red')  # show bases as red circle
        ax.plot(spline_bg, label='fitted background')

        fig2, ax2 = plt.subplots()
        # --- FIND PEAKS FOR THE FLATTENED RADIAL AVERAGE ----- #
        ax2.plot(flattened_rad_avg, color=colors[image_index], linewidth=2, linestyle='-', label='flattened rad avg ')
        ax2.plot(peakposition, flattened_rad_avg[peakposition], "x", markersize=5, color='green')
        # LABEL peaks in the plots
        for t in range(len(peakposition)):
            peakposn_text = str(peakposition[t]) + ', ' + str(np.round(pixel2Ghkl * peakposition[t], 3)) + '$A^{-1}$ '
            plt.text(peakposition[t] - 10, flattened_rad_avg[peakposition][t] + 100,
                     peakposn_text)  # labels the peak positions
            plt.text(peakposition[t] + 8, flattened_rad_avg[peakposition][t] - 50, str(t),
                     fontweight='bold')  # labels the peak index number

        ax2.set_xlabel('CCD pixels')
        secax = ax2.secondary_xaxis('top', functions=(pixel2scatt_vector, scatt_vector2pixel))
        secax.set_xlabel('scattering vector (1/A)')
        plt.legend()
        plt.grid(True)
        # plt.title(plot_title)


############################################
######### PLOTTING #########################
############################################

fontP = FontProperties()
fontP.set_size('small')

colors = cm.jet(np.linspace(0,1,len(timepoint_str)))  # colormap for plotting

fig_flat = plt.figure('flattened radial average')
ax_flat = fig_flat.add_subplot(111)

for k in np.arange(1, len(timepoint_str)):
    #region Setting up radial average arrays
    timepoint = os.path.basename(onList[k]).split('_')[3]
    t_ps = int(timepoint) * 1e-3
    t_list.append(t_ps)  # accumulates the timepoint as ps

    rad_avg_diff = findCenters.radialProfile(diffImgList[k], center0)
    rad_avg_on = findCenters.radialProfile(onImgList[k], center0)
    rad_avg_off = findCenters.radialProfile(offImgList[k], center0)
    peak1_data.append(sum(rad_avg_diff[peak1_range[0]:peak1_range[1]]))
    peak2_data.append(sum(rad_avg_diff[peak2_range[0]:peak2_range[1]]))

    flattened_rad_avg_on = rad_avg_on[min(bases):max(bases)] - spline_bg
    # flattened_rad_avg_on = flattened_rad_avg_on - min(flattened_rad_avg_on[100:])
    lr_data.append(sum(rad_avg_diff[lr_range[0]:lr_range[1]]))  # integrates pixels for liquid rise analysis
    #endregion

    if show_diff_img:
        plt.figure(20)
        plt.plot(rad_avg_diff, label = str(time_integers[k])+' ps', color=colors[k])
        plt.axvline(x=peak1_range[0], color='r', linestyle='--')
        plt.axvline(x=peak1_range[1], color='r', linestyle='--')
        plt.axvline(x=peak2_range[0], color='b', linestyle='--')
        plt.axvline(x=peak2_range[1], color='b', linestyle='--')
        plt.xlim(20, 600)
        plt.legend(title='timepoints', bbox_to_anchor=(1, 1), loc='upper left', prop=fontP)
        plt.grid(True)
        plt.title(dataDirec)

    if show_on_img:
        plt.figure(21)
        plt.plot(rad_avg_on, label =str(time_integers[k])+' ps', color=colors[k], ls='-')
        plt.xlim(20, 600)
        plt.legend(title='timepoints', bbox_to_anchor=(1, 1), loc='upper left', prop=fontP)
        plt.grid(True)
        plt.title("ON image  - " + dataDirec)

    if show_off_img:
        plt.figure(22)
        plt.plot(rad_avg_off, label =str(time_integers[k])+' ps', color=colors[k], ls='--')
        plt.xlim(20, 600)
        plt.legend(title='timepoints', bbox_to_anchor=(1, 1), loc='upper left', prop=fontP)
        plt.grid(True)
        plt.title("OFF image  - " + dataDirec)

    if show_flattened_on_img:



        if i % 2 == 1:
            # axs[scan_index].plot(flattened_rad_avg_on , color=colors[i], linewidth=2, linestyle='-', label=str(t_ps) + 'ps ')
            plt.plot(flattened_rad_avg_on , color=colors[k], linewidth=2, linestyle='-', label=str(time_integers[k]) + ' ps ')
        else:
            # axs[scan_index].plot(flattened_rad_avg_on , color=colors[i], linewidth=2, linestyle='--', label=str(t_ps) + 'ps ')
            plt.plot(flattened_rad_avg_on, color=colors[k], linewidth=2, linestyle='--', label=str(time_integers[k]) + ' ps ')

        show_vertical_lines = 0
        if show_vertical_lines:
            for p in peak_choices:
                plt.axvline(x=(peakposition[p]-peak_step), linestyle='--')
                plt.axvline(x=(peakposition[p]+peak_step), linestyle='--')
            # LABEL peaks in the plots
        # for t in range(len(peakposition)):
        #     peakposn_text = str(peakposition[t])
        #     plt.text(peakposition[t] + 10, flattened_rad_avg[peakposition][t] + 100, str(t),
        #              fontweight='bold')  # labels the peak index number

        # plt.legend(title='timepoints', bbox_to_anchor=(1, 1), loc='upper left', prop=fontP)
        plt.legend(title='Time Delays', loc='upper right', prop=fontP)
        plt.xlim(80, 300)
        plt.ylim(-100, 1600)
        # plt.title(plot_title)
        ax_flat.set_xlabel('Distance from center (pixel)', fontsize=12)
        ax_flat.set_ylabel('Intensity (a.u.)', fontsize=12)
        ax_flat.axes.xaxis.set_ticks([])
        ax_flat.axes.yaxis.set_ticks([])
        # ax.tick_params(axis='both', labelsize=0, length=0)
        # plt.grid(True)
        # plt.title("Background subtracted")

    for p_index, p in enumerate(peak_choices):
        peak_sum[p_index, k-1] = sum(flattened_rad_avg_on[peakposition[p]-peak_step: peakposition[p]+peak_step])

savefigure_flatradialaverage = 1
if savefigure_flatradialaverage:
    saveDirectory = 'D:\\Bismuth Project\\Figures for paper'
    fig_flat.savefig(saveDirectory + '\\SingleShot_RadialAverage.pdf', format='pdf')
    fig_flat.savefig(saveDirectory + '\\SingleShot_RadialAverage.svg', format='svg', dpi=800)
    fig_flat.savefig(saveDirectory + '\\SingleShot_RadialAverage.png', format='png', dpi=800)
    fig_flat.savefig(saveDirectory + '\\SingleShot_RadialAverage.eps', format='eps', dpi=800)

plt.show()

if show_expfits:

    fitted_variables = np.zeros((len(peak_choices), 5))

    fig, ax1 = plt.subplots()
    # fig2, ax2 = plt.subplots()

    skipfirst = 0
    skiplast = 1
    time_integers.pop(0)
    x_peaks = np.array(time_integers)  # picks timepoints for the exponential fitting
    # Does the exponential fit for each peak

    for p_index, p in enumerate(peak_choices):
        print(p_index, p)
        a, tau, t0, c = -5000, 2.0, 0.1, 9000
        if p_index == 1:
            a, tau, t0, c  = -1100, 1.0, 0.1, 1600

        y_peaks = (peak_sum[p_index, skipfirst:-skiplast])
        # y_peaks = (peak_sum[p_index, :])  # keep all time points
        # x_fit, y_fit, popt, pcov = exp_fit.mainfitting(x_peaks, y_peaks, a, c, tau, t0, 'Peak ' + str(p))
        x_fit, y_fit, popt, pcov = exp_fit.Exp_Fit(x_peaks, y_peaks, a, tau, t0, c,
                                                   sigma=0.35, plotlabel='Peak ' + str(p))
        fitted_variables[p_index, :] = p, popt[0], popt[1], popt[2], popt[3]  # stick the fitted variables here to make a table

        normalize_intensity = True
        if normalize_intensity:
            # y_fit = np.true_divide(y_fit, popt[0] + popt[3])
            y_fit_norm = np.true_divide(y_fit, popt[3])
            # y_peaks = np.true_divide(y_peakspt[0] + popt[3])
            y_peaks_norm = np.true_divide(y_peaks, popt[3])

        t_zero_correction = True
        if t_zero_correction:
            x_peaks_zeroed = x_peaks - popt[2]
            x_fit_zeroed = x_fit - popt[2]

        marks = '^', 's'
        peaklabels = '(011) peak trace', '(01-1) peak trace'
        fit_labels = '(011) exponential fit', '(01-1) exponential fit'
        ax1.plot(x_peaks_zeroed, y_peaks_norm, linestyle='solid', color=peak_colors[p_index],
                 label=peaklabels[p_index], linewidth=0.8, marker = marks[p_index], markersize = 7 ,alpha = 0.8)

        ax1.plot(x_fit_zeroed, y_fit_norm, '--', color=peak_colors[p_index], linewidth=2.5, label=fit_labels[p_index])

        ax1.tick_params(axis='y')
        ax1.set_ylabel('Normalized Peak Intensity,  $\Delta I$ / $I_{o}$', labelpad=5, fontsize=12)
        ax1.set_xlabel('Time Delay (ps)', fontsize=12)
        ax1.tick_params(axis='both', labelsize=12)
        ax1.legend(loc='upper right', prop={'size': 13})
        ax1.grid(linestyle='--', linewidth=0.8)
        # ax1.set_title('Single Shot scan, 23 nm Bi layer, 35 mJ/cm^2 ')

        savefigure = 0
        if savefigure:
            saveDirectory = 'D:\\Bismuth Project\\Figures for paper'
            plt.savefig(saveDirectory + '\\SingleShot_PeakTraceFigure_HD.pdf', format='pdf')
            plt.savefig(saveDirectory + '\\SingleShot_PeakTraceFigure_HD.svg', format='svg', dpi=1200)
            plt.savefig(saveDirectory + '\\SingleShot_PeakTraceFigure_HD.png', format='png', dpi=1200)
            plt.savefig(saveDirectory + '\\SingleShot_PeakTraceFigure_HD.eps', format='eps', dpi=1200)

        # ax2.plot(x_peaks, y_peaks, '-o')
        # ax2.grid(True)

    output_qtable=1
    if output_qtable:
        qtable = QTable()
        qtable['Peak #'] = np.round(fitted_variables[:,0], 0)
        qtable['Tau'] = np.round(fitted_variables[:,2] , 2) *u.ps
        qtable['t0'] = np.round(fitted_variables[:,3] , 2)*u.ps
        qtable['a'] = np.round(fitted_variables[:, 1], 2)
        qtable['c'] = np.round(fitted_variables[:, 4], 2)
        print(qtable)


# if show_integrated_peaks:
#     plt.figure()
#     plt.plot(time_integers, peak1_data, '-o', color='r', label="Peak 1")
#     plt.plot(time_integers, peak2_data, '-o', color='b', label="Peak 2")
#     plt.grid(True)
#     plt.legend()




# plt.figure(30)

# fig,ax1 = plt.subplots()
#
# x = time_integers
#
# a, c, tau, t0 = 1000, -1000, 2.0, 0
# y = peak1_data
# x_out, y_out, popt, pcov =exp_fit.mainfitting(x, y, a, c, tau, t0, 'Peak 1')
# ax1.plot(x,y, '-o', color = 'r', label = "Peak 1")
# ax1.plot(x_out,y_out,'--')
#
# y = peak2_data
# x_out, y_out, popt, pcov =exp_fit.mainfitting(x, y, a, c, tau, t0, 'Peak 2')
# ax1.plot(x,y, '-o', color = 'b', label = "Peak 2")
# ax1.plot(x_out,y_out,'--')
#
# ax1.grid(True)
# ax1.legend(loc='lower left')
# ax1.xlabel('Time delay (ps)')
# ax1.ylabel('Integrated difference peak intensities')


plt.show()