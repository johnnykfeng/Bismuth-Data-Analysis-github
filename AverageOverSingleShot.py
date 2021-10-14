import os
import glob
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import findCenters

plt.close('all')  # this doesn't work in PyCharm
dataDirec =  'E:\\New Bismuth Data\\2021-02-18\\scans\\scan100'
# dataDirec = 'E:\\Exp\\2021-02-09\\scans\\scan11'
print(dataDirec)

onList = []  # for collecting ON images
offList = [] # for collecting OFF images
diffList = [] # for collecting DIFF images

for file in glob.glob(os.path.join(dataDirec, '*.tiff')):
    # Iterate through file, find the ones we want.
    if (os.path.basename(file).split('_')[5] == 'On'):
        onList.append(file)
    if (os.path.basename(file).split('_')[5] == 'Off'):
        offList.append(file)

print('loading on and off images successful')