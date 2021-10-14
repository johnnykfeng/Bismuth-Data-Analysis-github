# -*- coding: utf-8 -*-

import os
import glob
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import findCenters

#darkBg = io.imread(os.path.join('C:\Users\johnn\Desktop\Bismuth Scan 2021-01-13\scans\scan51', 'darkExp.TIF'))
#scatterBg = io.imread(os.path.join('C:\Users\johnn\Desktop\Bismuth Scan 2021-01-13\scans\scan51', 'pumpExp.TIF'))
plt.close('all')
#plt.close()

folderDirec = 'C:\Users\johnn\Desktop\Bismuth Scan 2021-01-13\scans\scan'
Scan0 = np.arange(59,62,2)
Scan1 = np.arange(10,29,2) # [-1.5ps: 0.5ps: 3ps]
Scan2 = np.concatenate([np.arange(31,40,2)]) # [3.5ps: 0.5ps: 6ps]
Scan3 = np.array([52,54,57]) # [10ps, 15ps, 20ps]
ScanDirectoryNumber = np.concatenate([Scan0,Scan1,Scan2,Scan3])

peaks_list = []
t_list=[]

print('ScanDirectoryNumber:')
print(ScanDirectoryNumber)
print(' ')
for j, scanNum in enumerate(ScanDirectoryNumber):
    print('----------------------------------')
    print('j= ' +str(j))
    print('scanNum= ' + str(scanNum))    
    # Creating empty arrays and lists for processing data
    onList = []
    offList = []
    darkExp_List = []
    pumpExp_List= []
    
    onImg = np.zeros((1000,1148)) 
    offImg = np.zeros((1000,1148))  
    darkExp_Img = np.zeros((1000,1148))  
    pumpExp_Img = np.zeros((1000,1148))
    sumImg = np.zeros((1000, 1148)) 
    sumOnImg = np.zeros((1000, 1148)) 
    
    # collects the On and Off images
    dataDirec = folderDirec+ str(scanNum)
    print(dataDirec)
    for file in glob.glob(os.path.join(dataDirec, '*.tiff')):
        # Iterate through file, find the ones we want.
            if (os.path.basename(file).split('_')[5] == 'On'):
                onList.append(file)
            if (os.path.basename(file).split('_')[5] == 'Off'):
                offList.append(file)
                
    # collects the corresponding darkExp and pumpExp images
    dataDirec2 = folderDirec+ str(scanNum + 1 )
    print(dataDirec2)
    for file in glob.glob(os.path.join(dataDirec2, '*.tiff')):
        # Iterate through file, find the ones we want.
            if (os.path.basename(file).split('_')[5] == 'On'):
                pumpExp_List.append(file)
            if (os.path.basename(file).split('_')[5] == 'Off'):
                darkExp_List.append(file)
    
    #---- Reads time step information from file name ---#
    imgfile = glob.glob(os.path.join(dataDirec,'*.tiff'))
    timepoint = os.path.basename(imgfile[0]).split('_')[3]  # extracts timepoint from filename
    t_ps =int(timepoint)*1e-3    
    t_list.append(t_ps)    
    print('t= '+str(timepoint))    
    #---------------------------------------------------#
    
    
    #::::::::: Average all the shots in one scan folder ::::::::::#
    Nshots = (len(os.listdir(dataDirec)))/2 #number of shots per scan
    print('Nshots = '+str(Nshots))
    
    for i in np.arange(Nshots):  
        onImg = io.imread(onList[i])
        offImg = io.imread(offList[i])
        pumpExp_Img = io.imread(pumpExp_List[i])
        darkExp_Img = io.imread(darkExp_List[i])
#    
#        plt.figure()
#        ax1= plt.subplot(2,2,1)
#        ax2= plt.subplot(2,2,2)
#        ax3= plt.subplot(2,2,3)
#        ax4= plt.subplot(2,2,4)
#        ax1.imshow(onImg)
#        ax2.imshow(offImg)
#        ax3.imshow(onImg-pumpExp_Img)
#        ax4.imshow(offImg-darkExp_Img)
#        plt.show()
        
        # Take difference.
#        diffImg = onImg - offImg
        diffImg = (onImg - pumpExp_Img) - (offImg - darkExp_Img)        
        sumImg = sumImg + diffImg    
        sumOnImg = sumOnImg + (onImg)
    #    plt.figure(i)
    #    plt.imshow(onImg - offImg)
    #    plt.clim(-1000,1000)
    #    plt.show() 
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
    
    
    avgImg = sumImg/Nshots
    
    
    initCenter=(600, 543)
    step = 8
    radiusMin = 120
    radiusMax = 160
#    center = findCenters.meanAroundRing(sumOnImg, initCenter, step, radiusMin, radiusMax, 1)
    center = initCenter
    rad_avg = findCenters.radialProfile(avgImg, center)
    
    peak = sum(rad_avg[radiusMin:radiusMax])    
#    print peak
    peaks_list.append(peak)
#    

    plot_rad_avg = 1
    if plot_rad_avg:    
        plt.figure(100)
        colors = cm.gist_rainbow(np.linspace(0,1,len(ScanDirectoryNumber)))
        plt.plot(rad_avg, color=colors[j], label=str(t_ps)+'ps', linewidth=1)

plt.legend()        
plt.xlim(100,300)  
plt.ylim(25000,45000)      
plt.axvline(x=radiusMin, color='r', linestyle='--') #vertical line at radiusMin
plt.axvline(x=radiusMax, color='r', linestyle='--') #vertical line at radiusMax
plt.ylabel('Arbitrary pixel intensity')
plt.xlabel('Radial distance from center (pixel)')
plt.title('Radial Average plot')
plt.grid(True)

plt.figure()
plt.plot(t_list, peaks_list, '-o')
plt.xlabel('time delay (ps)')
plt.ylabel('peak intensity')
plt.grid(True)

plt.show()
    
    
    
    
#    plt.figure(scanNum)
#    plt.title('scan'+str(scanNum)+ '  t='+str(timepoint)+'fs')
##    plt.imshow(avgImg)  
#    plt.imshow(avgImg, cmap=plt.cm.BuPu_r)
#    plt.colorbar()
#    plt.clim(1e3,5e4)
#    plt.show()
#    


    