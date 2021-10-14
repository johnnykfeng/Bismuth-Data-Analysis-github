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
import exponential_fit as exp_fit
#---------------INITIALIZE VARIABLES -----------------------------#

plt.close('all')
dataDirec = 'E:\\New Bismuth Data\\2021-02-18\\scans\\scan100'
print(dataDirec)

show_diff_img = 1
show_on_img = 1
show_off_img = 1
show_integrated_peaks = 0
show_centerfinding = 1

initCenter = (592, 543)
step = 6
radiusAvgMin = 130
radiusAvgMax = 155

peak1_range = (195, 210)
peak1_data = []

#----------------------------------------------------------------------#

onList = []
offList = []
onImgList = []
offImgList = []
diffImgList = []
time_integers=[]
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

try:
    darkExp = io.imread(os.path.join(dataDirec, 'darkExp.TIF'))
    pumpExp = io.imread(os.path.join(dataDirec, 'pumpExp.TIF'))
    print('loading background successful!')
except:
    print('dark images not okay!')

try:
    offImg_alt = io.imread(os.path.join(dataDirec, 'off_image.TIF'))
    alternate_offImg = 1
    print('Alternate off image accepted!')
except:
    print('No alternate off image')
    alternate_offImg = 0

Nshots = 10 # not used in this code

avgImg = np.zeros((1000,1148))
sumImg = np.zeros((1000,1148))

# script for generating timepoints in correct string format
timepoint = np.arange(0,10000,400).astype(int)
# extra=np.array([-5000, -3000, -1500, -1000, -500, 12000, 15000, 25000, 50000, 100000])
extra = np.array([-5000, 12000, 15000, 25000, 50000, 100000])
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
            onImg = io.imread(onList[i]) - pumpExp
        if (os.path.basename(offList[i]).split('_')[3] == time): #reads the OFF img for each timepoint
            offImg = io.imread(offList[i]) - darkExp
        # if alternate_offImg:
        #     offImg = offImg_alt
        # else:
        #     if (os.path.basename(offList[i]).split('_')[3] == time):
        #         offImg = io.imread(offList[i]) - darkExp

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

############################################
######### PLOTTING #########################
############################################
center0 = findCenters.meanAroundRing(allOff, initCenter, step, radiusAvgMin, radiusAvgMax, show_centerfinding)
colors = cm.jet(np.linspace(0,1,len(timepoint_str)))  # colormap for plotting
for k in range(len(timepoint_str)):
    rad_avg_diff = findCenters.radialProfile(diffImgList[k], center0)
    rad_avg_on = findCenters.radialProfile(onImgList[k], center0)
    rad_avg_off = findCenters.radialProfile(offImgList[k], center0)
    peak1_data.append(sum(rad_avg_diff[peak1_range[0]:peak1_range[1]]))

    if show_diff_img:
        plt.figure(20)
        plt.plot(rad_avg_diff, label = timepoint_str[k], color=colors[k])
        plt.axvline(x=peak1_range[0], color='r', linestyle='--')
        plt.axvline(x=peak1_range[1], color='r', linestyle='--')
        plt.xlim(20, 600)
        plt.legend()
        plt.grid(True)
        plt.title(dataDirec)

    if show_on_img:
        plt.figure(21)
        plt.plot(rad_avg_on, label =timepoint_str[k], color=colors[k], ls='-')
        plt.xlim(20, 600)
        plt.legend()
        plt.grid(True)
        plt.title("ON image  - " + dataDirec)

    if show_off_img:
        plt.figure(22)
        plt.plot(rad_avg_off, label =timepoint_str[k], color=colors[k], ls='--')
        plt.xlim(20, 600)
        plt.legend()
        plt.grid(True)
        plt.title("OFF image  - " + dataDirec)

if show_integrated_peaks:
    plt.figure()
    plt.plot(time_integers, peak1_data, '-o', color='r', label="Peak 1")
    plt.grid(True)
    plt.legend()

a,c,tau,t0 = 30000,-30000,7.5,2
exp_fit.mainfitting(time_integers, peak1_data ,a ,c ,tau ,t0,'peak1')

# >>> plot(x, y, 'go--', linewidth=2, markersize=12)
# >>> plot(x, y, color='green', marker='o', linestyle='dashed',
# ...      linewidth=2, markersize=12)

# plt.figure()
# plt.imshow(sumDiff)
# plt.title('Difference')
# plt.colorbar()
#
# plt.figure()
# plt.imshow(sumOff)
# plt.title('Off')
# plt.colorbar()
#
# plt.figure()
# plt.imshow(sumOn)
# plt.title('On')
# plt.colorbar()

plt.show()