
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

pixel2Ghkl = 2.126e-3  # converts CCD pixels to G_hkl = 1/d_hkl 'scattering vector'
def pixel2scatt_vector(x):
    return x * pixel2Ghkl
def scatt_vector2pixel(x):
    return x / pixel2Ghkl

image_name = 'SiN_5s_exposure.TIF'
image_name2 = 'Bi_5s_exposure.TIF'
imageDirec = "D:\\Bismuth Project\\Older Bismuth Data\\2021-01-25\\scans\\" + image_name
imageDirec2 = "D:\\Bismuth Project\\Older Bismuth Data\\2021-01-25\\scans\\" + image_name2

SiN_img = io.imread(imageDirec)

initCenter = 605,541
step = 4
radiusAvgMin = 300
radiusAvgMax = 380
center0 = findCenters.meanAroundRing(SiN_img, initCenter, step, radiusAvgMin, radiusAvgMax, 1)

rad_avg_SiN = findCenters.radialProfile(SiN_img, center0)
#-------------------------------------------------

Bi_img = io.imread(imageDirec2)
initCenter = 605,541
step = 2
radiusAvgMin = 300
radiusAvgMax = 380
center0 = findCenters.meanAroundRing(Bi_img, initCenter, step, radiusAvgMin, radiusAvgMax, 1)

rad_avg_Bi = findCenters.radialProfile(Bi_img, center0)
rad_avg_Bi_norm = rad_avg_Bi/max(rad_avg_Bi)
x = np.arange(len(rad_avg_Bi))
x = x *pixel2Ghkl

# rad_avg_data = []
# rad_avg_data.append(x)
# rad_avg_data.append(rad_avg_Bi)
rad_avg_data = np.stack((x, rad_avg_Bi*1e3), axis = 1)

np.savetxt('D:\\Bismuth Project\\Software\\rad_avg_data.txt', rad_avg_data)

# plt.figure()
# plt.imshow(SiN_img)
# plt.clim(700,5000)

plt.figure()
# plt.plot(rad_avg_SiN)
plt.plot(x, rad_avg_Bi_norm)
plt.grid(True)


R3m_profile_txt = 'D:\\Bismuth Project\\Software\\Bismuth R-3m Profile.txt'
R3m_profile= np.loadtxt(R3m_profile_txt)
AMS_profile_txt = 'D:\\Bismuth Project\\Software\\AMS_DATA_bimuth2 Profile.txt'
AMS_profile= np.loadtxt(AMS_profile_txt)


scale_x = 1.0
scale_y = 1.0/max(R3m_profile[:,1])
plt.figure('R3-m Bismuth diffraction')
plt.plot(R3m_profile[:,0]*scale_x, R3m_profile[:,1]*scale_y)
plt.plot(x, rad_avg_Bi_norm)
plt.grid(True)

scale_x = 1.0
scale_y = 1.0/max(AMS_profile[:,1])
plt.figure('AMS Bismuth diffraction')
plt.plot(AMS_profile[:,0]*scale_x, AMS_profile[:,1]*scale_y)
plt.plot(x, rad_avg_Bi_norm)
plt.grid(True)


plt.show()
