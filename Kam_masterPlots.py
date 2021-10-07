import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Figure 1 is a separate SVG.

# Figure 2 final.
def figure1():

    # Set up figure stuff.
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['figure.figsize'] = (3.25, 3.25)

    # Set up the plot.
    # fig, (ax1, ax2) = plt.subplots(1,2)
    fig, (ax1) = plt.subplots(1,1)

    # Master functions for Figure 1.
    # Run to generate diffractogram of select dot size.
    def plotDiffractogram(dotSize, ax, diff):

        import math
        # Control variables here.
        plotDiff = diff

        def convertTwoTheta(twoTheta):

            # This function converts an x-axis of two-theta in to Q.
            lambdaElec = 0.038061
            qArray = []
            for i in twoTheta:
                radVal = (i) * (np.pi / 180)
                result = ((4 * np.pi) / lambdaElec) * math.sin((radVal) / 2)
                qArray.append(result)

            return np.asarray(qArray)

        if dotSize == 'Medium':
            # Global data.
            convTT = 0.0046926

            # Load in data for off images.
            dataDirec = 'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\2020-02-04\\scans\scan5\\bgRem\\averagedImages'
            dataLength = len(np.load(os.path.join(dataDirec, 'dataOff_000000.npy')))
            masterData = np.zeros(dataLength)
            counter = 0

            for file in glob.glob(os.path.join(dataDirec, '*.npy')):
                if os.path.basename(file).split('_')[0] == 'dataOff':
                    data = np.load(file)
                    masterData += data
                    counter += 1

            # Finalize, and plot.
            masterData = masterData / counter

        # Plot results.
        xRange = np.load(os.path.join(dataDirec, 'xRange.npy'))
        xRange = xRange * convTT
        xRange = convertTwoTheta(xRange)

        # If including difference at time t = 30 ps?
        onData = np.load(os.path.join(dataDirec, 'dataOn_015000.npy'))

        # Calculate difference.
        diffData = onData - masterData

        hfont = {'fontname': 'Helvetica'}

        if plotDiff:
            ax.plot(xRange, diffData * 10, alpha=0.7, linewidth=1.5, label='Difference (x 10)', color='black', zorder=1)
            ax.plot(xRange, masterData, alpha=0.7, linewidth=1.5, linestyle='solid', label='No Excitation', color='C0', zorder=-1)
            ax.plot(xRange, onData, alpha=0.9, linewidth=1.5, linestyle='dotted', label='Photoexcitation', color='red', zorder=0)
            ax.axhline(y=0, color='black', alpha=0.5, label='y = 0')
            ax.set_ylabel('Intensity (a.u.)')
            ax.set_xlabel(r'q ($\AA^{-1})$')
            ax.set_xlim([xRange[0], xRange[-1]])
            # ax.text(-0.3, 0.95, 'a)', transform=ax.transAxes, size=12, weight='bold', **hfont)
            ax.legend(loc='upper right', prop={'size': 7}, frameon=False)
        if not plotDiff:
            ax.plot(xRange, masterData, alpha=1, linewidth=1.5, color='C0')
            ax.fill_between(xRange, -15, masterData, alpha=0.5)
            ax.set_xlim([xRange[0], xRange[-1]])
            # ax.text(-0.3, 0.95, 'b)', transform=ax.transAxes, size=12, weight='bold', **hfont)
            # ax.legend(loc='upper right', prop={'size': 12}, frameon=False)
            ax.set_ylim([-10, max(masterData) + 20])
            ax.set_ylabel('Intensity (a.u.)')
            ax.set_xlabel(r'2$q$ $(\AA})^{-1}$')
        #if plotDiff:
        ax.axes.yaxis.set_ticks([])

    # Do the plotting.
    plotDiffractogram('Medium', ax1, True)
    # plotDiffractogram('Medium', ax1, False)
    plt.tight_layout()
    plt.savefig('E:\\Ediff Samples\\PbS Paper - Final Folder\\Final Push\\Changed Figures\\Figure2v3.pdf')
    plt.show()

    # Save figure.

# Figure 3 final.
def figure2():

    plt.rcParams.update({'font.size': 10})
    plt.rcParams['figure.figsize'] = (7, 6)

    # Set up the plot.
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

    # All figure functions defined below.
    def plotTimeTraces(dotSize, ax):

        import fitTimeTraceModels
        hfont = {'fontname': 'Helvetica'}

        # Select dot size and fit params.
        if dotSize == 'Small':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            peaksToDraw = ['(311) and (222)', '(400)', '(331) and (420)']
            fitResults = [['Tri', 'Tri', 'Bi'],
                          [(3.1, 50, 585, -0.30, 0.17, -0.049), (2.0, 40, 707, -1.09, 0.43, -0.25),
                           (12.6, 47, 0.40, -0.31)]]
            masterData = np.load(
                'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\Master Data\\masterData_Int_Small.npy')
        if dotSize == 'Medium':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            peaksToDraw = ['(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            fitResults = [['Bi', 'Bi', 'Bi', 'Mono', 'Tri'],
                          [(0.86, 554, -0.037, -0.076), (3.81, 586, -0.095, -0.066), (5.89, 260, -0.635, 0.119),
                           (2.15, -0.104), (0.42, 46.87, 749.31, -0.149, -0.012, -0.094)]]
            masterData = np.load(
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\Master Data\\masterData_Int_Medium.npy')
        if dotSize == 'Large':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            peaksToDraw = ['(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)']
            fitResults = [['Mono', 'Mono', 'Mono', 'Bi', 'Mono'],
                          [(2.39, -0.024), (2.41, -0.095), (3.51, -0.195), (3.5, 882, -0.512, -0.348), (2.30, -0.164)]]
            masterData = np.load('E:\\Ediff Samples\PbS Data\\8 nm Dots Pumped\\Master Data\\masterData_Int_Large.npy')
        if dotSize == 'Shelled':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            peaksToDraw = ['(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)']
            fitResults = [['Mono', 'Tri', 'Tri', 'Mono', 'Tri'],
                          [(4.83, -0.034), (1.04, 22.62, 359, -0.17, 0.06, -0.07),
                           (2.48, 12, 339, -0.303, 0.0971, -0.047), (3.05, -0.392),
                           (1.32, 13.43, 252.20, -0.44, 0.24, -0.10)]]
            masterData = np.load(
                'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\Master Data\\masterData_Int_Shelled.npy')

        # Plot parameters.
        colors = plt.cm.plasma(np.linspace(0, 1, len(peaksToDraw)))

        # Set up evaluation.
        tpRange = np.arange(-300, 1000, 1)
        fitEval = []

        tpVals = masterData[0][0]
        xRange = [min(tpVals), max(tpVals)]

        # Go through, plot and evaluate for each peak.
        for ind, peak in enumerate(peaksToDraw):

            # Which peak?
            peakInd = peakLabels.index(peak)

            # Plot the data.
            ax.plot(masterData[peakInd][0], masterData[peakInd][1], marker='.', alpha=0.6, label=peak,
                     color=colors[ind])

            # Evaluate fit.
            fitEval = []
            for x in tpRange:

                if fitResults[0][ind] == 'Mono':
                    fitEval.append(
                        fitTimeTraceModels.ExponentialIntensityDecay(x, fitResults[1][ind][0], fitResults[1][ind][1]))

                if fitResults[0][ind] == 'Bi':
                    fitEval.append(
                        fitTimeTraceModels.BiExponentialIntensityDecay(x, fitResults[1][ind][0], fitResults[1][ind][1],
                                                                       fitResults[1][ind][2], fitResults[1][ind][3]))

                if fitResults[0][ind] == 'Tri':
                    fitEval.append(
                        fitTimeTraceModels.TriExponentialIntensityDecay(x, fitResults[1][ind][0], fitResults[1][ind][1],
                                                                        fitResults[1][ind][2], fitResults[1][ind][3],
                                                                        fitResults[1][ind][4], fitResults[1][ind][5]))

            # Plot the fit.
            ax.plot(tpRange, fitEval, linestyle='--', color=colors[ind])
        ax.set_xlim(xRange)
        ax.set_ylabel(r'$\Delta I$ / $I_{o}$', labelpad=-3)
        ax.set_xlabel('Timepoint (ps)', labelpad=1)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        # ax.text(-0.3, 0.95, 'a)', transform=ax.transAxes, size=12, weight='bold', **hfont)
        ax.legend(loc='lower right', prop={'size': 7}, frameon=False)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    def debyeWallerPE(dotSize, ax):
        import numpy as np
        import math
        import os
        import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit

        hfont = {'fontname': 'Helvetica'}

        # Global constants.
        # TODO: Populate these suckers.
        verbose = True
        selectAverage = False
        selectWindow = True
        selectSingle = False
        showTitle = False
        peaksToAvoid = ['(111)']

        # Global functions and tools.
        def convertTwoTheta(twoTheta):

            # This function converts an x-axis of two-theta in to Q.
            lambdaElec = 0.038061
            qArray = []
            for i in twoTheta:
                radVal = (i) * (np.pi / 180)
                result = ((4 * np.pi) / lambdaElec) * math.sin((radVal) / 2)
                qArray.append(result)

            return np.asarray(qArray)

        def findNearest(array, value):

            # Returns nearest index value to specified value in array.
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx

        # Dot parameters defined.
        if dotSize == 'Large':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            dataDirec = 'E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\Master Data'
            convTT = 0.0047808
        elif dotSize == 'Medium':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            dataDirec = 'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\Master Data'
            convTT = 0.0046926
        elif dotSize == 'Small':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            dataDirec = 'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\Master Data'
            convTT = 0.0047412
        elif dotSize == 'Shelled':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            dataDirec = 'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\Master Data'
            convTT = 0.0047023

        # Begin by loading in data.
        fileName = 'masterData_Int2_%s.npy' % dotSize
        masterData = np.load(os.path.join(dataDirec, fileName))

        # Step 1: determine peak positions by loading data for selected folder.
        # It is best to use a directory with lots of data.

        # peakDataLoc = 'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\2020-08-06\\scans\\scan14\\bgRem\\averagedImages\\fitData' # Smalls
        peakDataLoc = 'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\2020-02-04\\scans\\scan12\\bgRem\\averagedImages\\fitData'

        # Iterate through and average all peak positions for each peak, convert to 1/d.
        peakPos = []
        peakPosError = []
        for ind, peak in enumerate(peakLabels):

            if peak not in peaksToAvoid:
                # Load in peak data for specified peak.
                peakData = np.load(os.path.join(peakDataLoc, '%s_peakData.npy' % peak))

                # Average all values in the 2nd row (peak off positions).
                peakPositions = peakData[1, :]
                peakPos.append(np.average(peakPositions))
                peakPosError.append(np.std(peakPositions))

        # Convert to s, then convert to Q. Peak data is now available.
        peakPos = np.asarray(peakPos)
        peakPosS = peakPos * convTT
        peakPosQ = convertTwoTheta(peakPosS)
        peakPosError = np.asarray(peakPosError)
        peakPosSError = peakPosError * convTT
        peakPosQError = convertTwoTheta(peakPosSError)

        # Start D-W stuff. All master data has first row as timepoints, second row as -ln(I/I_o).
        if selectAverage:
            dwChanges = []
            dwErrors = []
            for ind, peak in enumerate(peakLabels):
                # Load in integrated data.
                plt.plot(masterData[ind][0], masterData[ind][1], label=peak)
            plt.legend(loc='upper right')
            plt.show()

            print("Available timepoints:\n")
            print(masterData[0][0])
            tpSel = int(input("Please select at which timepoint to start integrating:\n"))

            # Find matching index, average everything after that, and append to master list.
            tpInd = findNearest(masterData[0][0], tpSel)

            # Start the averaging.
            for ind, peak in enumerate(peakLabels):

                if peak not in peaksToAvoid:
                    if verbose:
                        print("Now averaging data past %s ps for peak %s." % (masterData[0][0][tpInd], peak))

                    # Get all data after selected timepoint.
                    dwData = masterData[ind][1][tpInd:]

                    # Average the data for that peak and append to array.
                    dwChanges.append(np.average(dwData))
                    dwErrors.append(np.std(dwData))

        if selectWindow:
            dwChanges = []
            dwErrors = []
            # for ind, peak in enumerate(peakLabels):
            #     # Load in integrated data.
            #     plt.plot(masterData[ind][0], masterData[ind][1], label=peak)
            # plt.legend(loc='upper right')
            # plt.show()

            print("Available timepoints:\n")
            print(masterData[0][0])
            tpSel = int(input("Please select at which timepoint to start integrating:\n"))
            tpSelEnd = int(input("Please select at which timepoint to stop integrating:\n"))

            # Find matching index, average everything after that, and append to master list.
            tpInd = findNearest(masterData[0][0], tpSel)
            tpIndEnd = findNearest(masterData[0][0], tpSelEnd)

            # Start the averaging.
            for ind, peak in enumerate(peakLabels):

                if peak not in peaksToAvoid:
                    if verbose:
                        print("Now averaging data past %s ps for peak %s." % (masterData[0][0][tpInd], peak))

                    # Get all data after selected timepoint.
                    dwData = masterData[ind][1][tpInd:tpIndEnd]

                    # Average the data for that peak and append to array.
                    dwChanges.append(np.average(dwData))
                    dwErrors.append(np.std(dwData))

        if selectSingle:
            dwChanges = []
            dwErrors = []
            for ind, peak in enumerate(peakLabels):
                # Load in integrated data.
                plt.plot(masterData[ind][0], masterData[ind][1], label=peak)
            plt.legend(loc='upper right')
            plt.show()

            print("Available timepoints:\n")
            print(masterData[0][0])
            tpSel = int(input("Please select a timepoint to view:\n"))

            # Find matching index, average everything after that, and append to master list.
            tpInd = findNearest(masterData[0][0], tpSel)

            # Start the averaging.
            for ind, peak in enumerate(peakLabels):

                if peak not in peaksToAvoid:
                    if verbose:
                        print("Now averaging data past %s ps for peak %s." % (masterData[0][0][tpInd], peak))

                    # Get all data after selected timepoint.
                    dwData = masterData[ind][1][tpInd]

                    # Average the data for that peak and append to array.
                    dwChanges.append(dwData)
                    dwErrors.append(np.std(dwData))

        # # Fit data to a line.
        # def linearFit(m, x):
        #     return m * x
        #
        # popt, _ = curve_fit(linearFit, peakPosQ**2, dwChanges)
        # slope= popt
        # xRange = np.arange(peakPosQ[0]**2, peakPosQ[-1]**2, 0.1)
        # yEval = linearFit(slope, xRange)

        # Plot the results.
        ax.errorbar(peakPosQ ** 2, dwChanges, linestyle='None', yerr=dwErrors, capsize=5,
                     marker='o', ecolor='black', zorder=0)

        # Draw square to end highlighting region of localized disorder exceeding the D-W behaviour.
        print(peakPosQ ** 2)
        print(dwChanges)

        # Begin determining fitting for linear D-W portion of response.
        peaksToFit = ['(200)', '(220)', '(311) and (222)']
        yFit = []
        xFit = []
        for peak in peaksToFit:
            # Extract the relevant peaks.
            ind = peakLabels.index(peak) - 1
            xFit.append(peakPosQ[ind])
            yFit.append(dwChanges[ind])

        # Fit data to a line.
        def linearFit(x, m, b):
            return m * x + b

        yFit = np.asarray(yFit)
        xFit = np.asarray(xFit)
        popt, pcov = curve_fit(linearFit, xFit ** 2, yFit)
        slope = popt[0]
        intercept = popt[1]
        xRange = np.arange(peakPosQ[0] ** 2, peakPosQ[-1] ** 2, 0.1)
        yEval = linearFit(slope, xRange, intercept)

        ax.plot(xRange, yEval, alpha=0.6, linewidth=4, color='C0', zorder=-1)
        # if showTitle:
        #     if selectAverage:
        #         plt.title('Photoexcited DW - Averaged > %.0f ps (%s)' % (tpSel, dotSize))
        #     if selectSingle:
        #         plt.title('Photoexcited DW - At %.0f  ps (%s)' % (tpSel, dotSize))
        #     if selectWindow:
        #         plt.title('Photoexcited DW - %.0f to %.0f ps (%s)' % (tpSel, tpSelEnd, dotSize))
        ax.set_ylabel(r'$-ln(I/I_{o})$', labelpad=1)
        ax.set_xlabel(r'$q^{2}$ ($\AA^{-2}$)', labelpad=0)
        ax.axvspan(peakPosQ[3] ** 2, (peakPosQ[-1] + 2) ** 2, alpha=0.4, zorder=-2, facecolor='grey', lw=0)
        ax.set_xlim([peakPosQ[0] ** 2 - 0.5, peakPosQ[-1] ** 2 + 0.5])
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        #ax.text(-0.3, 0.95, 'b)', transform=ax.transAxes, size=12, weight='bold', **hfont)

    def plotPeakShiftThermal(dotSize, ax):
        import numpy as np
        import glob
        import sys
        import os
        import matplotlib.pyplot as plt
        from tempHelperFunctions import convertResistanceToTemp
        import fitDiffPeaks

        hfont = {'fontname': 'Helvetica'}

        # Global variables here.
        pixelToInvAng = 0.0019673
        tempDirec = 'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\Thermal Run\\fitData'
        peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)',
                      '(422)']  # Medium dot labels.
        # peakLabels = ['(111)', '(200)', '(220)', '(311)', '(222)', '(400)', '(331) and (420)', '(422)'] # Large dot labels.
        tempPoints = [54.4, 68.3, 81.3, 93.4, 104.8, 115.5, 125.3, 134.7]  # Medium dot temp points.
        # tempPoints = [54.1, 67.6, 81.1, 93.0, 104.4, 114.5, 125.9, 133.6] # Large dot temp points (run 1).
        # tempPoints = [52.0, 67.4, 80.2, 92.8, 104.1, 114.6, 125.6, 133.5] # Large dot temp points (run 2).
        # tempPoints = [52.0, 67.4, 80.2, 92.8, 104.1, 114.6, 125.6, 133.5]

        # Load in peak centers.
        peakCentersMaster = np.load(os.path.join(tempDirec, 'peakCenters.npy')) * pixelToInvAng

        # Convert resistances to Kelvin.
        tempsKelvin = []
        for t in tempPoints:
            tempsKelvin.append(convertResistanceToTemp(t))
        tempsKelvin = np.asarray(tempsKelvin)

        print(tempsKelvin)

        # Plot peak shift for each peak, and fit to a line.
        diffCenters = []
        for i in range(0, len(tempPoints)):
            diffCenters.append((peakCentersMaster[i, :] - peakCentersMaster[0, :]) / peakCentersMaster[0, :])
        diffCenters = np.asarray(diffCenters)

        # Plot results, if verbose.
        # if verbosePeakShift:
        #     for ind, peak in enumerate(peakLabels):
        #         plt.plot(tempsKelvin, diffCenters[:, ind], label=peak)
        #     plt.title('Peak Shift')
        #     plt.legend(loc='upper right')
        #     plt.show()

        # Fit each line, show results. Then take fit data to create a line, and then average.
        avgSlope = 0
        avgInt = 0
        counter = 0
        tempEvalRange = np.arange(tempsKelvin[0], tempsKelvin[-1], 1)
        print(tempEvalRange)
        for ind, peak in enumerate(peakLabels):

            # Feed in data for fit, evaluate.
            thermalFit = fitDiffPeaks.performLinearFit(tempsKelvin, diffCenters[:, ind], verbose=True)

            # Pull out slope and intercept values.
            slope = thermalFit[0].params['ThermalLine_slope'].value
            intercept = thermalFit[0].params['ThermalLine_intercept'].value

            # Evaluate over x temperature range.
            fitEval = []
            for x in tempEvalRange:
                fitEval.append(thermalFit[0].eval(x=x))
            fitEval = np.asarray(fitEval)

            # Plot results of fit.
            # ax.plot(tempsKelvin, diffCenters[:, ind], label=peak)
            # ax.plot(tempEvalRange, fitEval, linestyle='--', label='Fitted Eval')
            # #plt.title('Slope: %s, Int: %s' % (slope, intercept))
            # plt.legend(loc='upper right')

            # Add to average slope and intercept.
            avgSlope += slope
            avgInt += intercept

            # Increase counter.
            counter += 1

        # Average out slope.
        avgSlope = avgSlope / counter
        avgInt = avgInt / counter

        # Plot final result.
        avgLineEval = []
        for x in tempEvalRange:
            avgLineEval.append(avgSlope * x + avgInt)
        avgLineEval = np.asarray(avgLineEval)

        colors = plt.cm.plasma(np.linspace(0, 1, len(peakLabels)))

        for ind, peak in enumerate(peakLabels):
            ax.plot(tempsKelvin, diffCenters[:, ind], color=colors[ind], alpha=0.5, marker='.', label=peak)
        ax.plot(tempEvalRange, avgLineEval, color='grey', linewidth=2, linestyle='--', alpha=1, label='Averaged Shift')
        ax.legend(loc='lower left', frameon=False, prop={'size': 7})
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.set_ylabel(r'$\Delta$a', labelpad=0)
        ax.set_xlabel('Temperature (K)', labelpad=1)
        # ax.text(-0.3, 0.95, 'c)', transform=ax.transAxes, size=12, weight='bold', **hfont)

    def plotDW(dotSize, ax):

        import numpy as np
        import matplotlib.pyplot as plt
        import os
        import easygui
        import sys
        from tempHelperFunctions import convertResistanceToTemp
        from pdfTools import convertTwoTheta
        from scipy.optimize import curve_fit

        hfont = {'fontname': 'Helvetica'}

        # Global constants.
        # resistances = [68.3, 81.3, 93.4, 104.8, 115.5, 125.3, 134.7]
        resistances = [54.1, 67.6, 81.1, 93.0, 104.4, 114.5, 125.9, 133.6]
        temps = []
        for r in resistances:
            temps.append(convertResistanceToTemp(r))
        peaksToAvoid = ['(111)']

        # Define dot information here.
        if dotSize == 'Large':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            peakDataLoc = 'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\2020-02-04\\scans\\scan12\\bgRem\\averagedImages\\fitData'
            convTT = 0.0047808
        elif dotSize == 'Medium':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            peakDataLoc = 'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\2020-02-04\\scans\\scan12\\bgRem\\averagedImages\\fitData'
            fileDirec = 'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\Thermal Run\\Raw Data\\2020-03-17\\scans\\Temperature Scans\\temperatureData\\newProcessed\\finalData'
            convTT = 0.0046926
        elif dotSize == 'Small':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            peakDataLoc = 'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\2020-02-04\\scans\\scan12\\bgRem\\averagedImages\\fitData'
            convTT = 0.0047412
        elif dotSize == 'Shelled':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            peakDataLoc = 'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\2020-02-04\\scans\\scan12\\bgRem\\averagedImages\\fitData'
            convTT = 0.0047023

        # Ask user to find relevant integrated data for their dot.
        # fileDirec = easygui.diropenbox("Please select master integrated temperature data (thermal DW).")
        # if fileDirec is None:
        #     sys.exit("Please select a file to load.")

        # Iterate through and average all peak positions for each peak, convert to 1/d.
        peakPos = []
        peakPosError = []
        for ind, peak in enumerate(peakLabels):

            if peak not in peaksToAvoid:
                # Load in peak data for specified peak.
                peakData = np.load(os.path.join(peakDataLoc, '%s_peakData.npy' % peak))

                # Average all values in the 2nd row (peak off positions).
                peakPositions = peakData[1, :]
                peakPos.append(np.average(peakPositions))
                peakPosError.append(np.std(peakPositions))

        # Convert to s, then convert to Q. Peak data is now available.
        peakPos = np.asarray(peakPos)
        peakPosS = peakPos * convTT
        peakPosQ = convertTwoTheta(peakPosS)
        peakPosError = np.asarray(peakPosError)
        peakPosSError = peakPosError * convTT
        peakPosQError = convertTwoTheta(peakPosSError)

        # Load in master temperature intensity data.

        mainDataDirec = os.path.join(fileDirec, 'master%sIntData.npy' % dotSize)

        masterData = np.load(mainDataDirec)
        masterError = np.load(os.path.join(fileDirec, 'master%sIntErrorData.npy' % dotSize))

        dwChanges = []
        errors = []
        tempDiff = []

        for ind, t in enumerate(temps):

            if ind != 0:
                # Calculate difference between said timepoint and lowest temperature.
                delDW = -1 * np.log(masterData[ind] / masterData[0])
                dwChanges.append(delDW)

                # Capture errors.
                errorResult = np.sqrt((masterError[ind] / masterData[ind]) ** 2 + (masterError[0] / masterData[0]) ** 2)
                errors.append(errorResult[1:])

                # Save temperature difference result.
                tempDiff.append(t - temps[0])

        # # Plot all results.
        # for ind, val in enumerate(dwChanges):
        #
        #     plt.plot(peakPosQ**2, val, marker='o', label=tempDiff[ind])
        # plt.legend(loc='upper left')
        # plt.show()

        # Fit to data.
        def linearFit(x, m, b):
            return m * x + b

        yFit = np.asarray(dwChanges[3][1:])
        xFit = np.asarray(peakPosQ ** 2)

        popt, pcov = curve_fit(linearFit, xFit, yFit)
        slope = popt[0]
        intercept = popt[1]
        xRange = np.arange(peakPosQ[0] ** 2, peakPosQ[-1] ** 2, 0.1)
        yEval = linearFit(slope, xRange, intercept)

        # Plot select result.
        ax.plot(xRange, yEval, linewidth=4, alpha=0.6, color='C0', zorder=-1, label='Linear DW Fit')
        ax.errorbar(peakPosQ ** 2, dwChanges[3][1:], yerr=errors[3], marker='o', linestyle='None', color='C0',
                     zorder=0, ecolor='black', capsize=5)
        ax.set_ylabel(r'$-ln(I/I_{o})$', labelpad=0)
        ax.set_xlabel(r'$q^{2}$ ($\AA^{-2}$)', labelpad=0)
        # ax.text(-0.3, 0.95, 'd)', transform=ax.transAxes, size=12, weight='bold', **hfont)
        ax.legend(loc='upper left', prop={'size': 7}, frameon=False)

        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # Do the plotting.
    plotTimeTraces('Medium', ax3)
    debyeWallerPE('Medium', ax4)
    plotPeakShiftThermal('Medium', ax1)
    plotDW('Medium', ax2)

    #fig.tight_layout(pad=0.1, w_pad=0.05, h_pad=0.05)
    fig.tight_layout()
    fig.align_ylabels((ax1, ax2, ax3, ax4))

    fig.savefig('E:\\Ediff Samples\\PbS Paper - Final Folder\\Final Push\\Changed Figures\\Figure3_Final.pdf')
    #plt.show()

def figure3():

    # Set up figure stuff.
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['figure.figsize'] = (7, 6)

    # Set up the plot.
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

    # Define all used figures below.
    def plotTimeTraces(dotSize, ax):

        color400 = 'C0'

        import fitTimeTraceModels
        hfont = {'fontname': 'Helvetica'}

        # Select dot size and fit params.
        if dotSize == 'Small':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            peaksToDraw = ['(311) and (222)', '(400)', '(331) and (420)']
            letter = 'a)'
            fitResults = [['Tri', 'Tri', 'Bi'],
                          [(3.1, 50, 585, -0.30, 0.17, -0.049), (2.0, 40, 707, -1.09, 0.43, -0.25),
                           (12.6, 47, 0.40, -0.31)]]
            masterData = np.load(
                'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\Master Data\\masterData_Int_Small.npy')
        if dotSize == 'Medium':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            peaksToDraw = ['(220)', '(311) and (222)', '(400)', '(331) and (420)']
            fitResults = [['Bi', 'Bi', 'Bi', 'Mono'],
                          [(0.86, 554, -0.037, -0.076), (3.81, 586, -0.095, -0.066), (5.89, 260, -0.635, 0.119),
                           (2.15, -0.104)]]
            masterData = np.load(
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\Master Data\\masterData_Int_Medium.npy')
        if dotSize == 'Large':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            peaksToDraw = ['(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)']
            fitResults = [['Mono', 'Mono', 'Mono', 'Bi', 'Mono'],
                          [(2.39, -0.024), (2.41, -0.095), (3.51, -0.195), (3.5, 882, -0.512, -0.348), (2.30, -0.164)]]
            masterData = np.load('E:\\Ediff Samples\PbS Data\\8 nm Dots Pumped\\Master Data\\masterData_Int_Large.npy')
            letter = 'b)'
        if dotSize == 'Shelled':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            peaksToDraw = ['(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)']
            fitResults = [['Mono', 'Tri', 'Tri', 'Mono', 'Tri'],
                          [(4.83, -0.034), (1.04, 22.62, 359, -0.17, 0.06, -0.07),
                           (2.48, 12, 339, -0.303, 0.0971, -0.047), (3.05, -0.392),
                           (1.32, 13.43, 252.20, -0.44, 0.24, -0.10)]]
            masterData = np.load(
                'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\Master Data\\masterData_Int_Shelled.npy')

        # Plot parameters.
        colors = plt.cm.plasma(np.linspace(0, 1, len(peaksToDraw)))

        # Set up evaluation.
        tpRange = np.arange(-300, 1000, 1)
        fitEval = []

        tpVals = masterData[0][0]
        xRange = [min(tpVals), max(tpVals)]

        # Go through, plot and evaluate for each peak.
        for ind, peak in enumerate(peaksToDraw):

            # Which peak?
            peakInd = peakLabels.index(peak)

            # Plot the data.
            if peak == '(400)':
                ax.plot(masterData[peakInd][0], masterData[peakInd][1], marker='.', alpha=0.6, label=peak,
                        color=color400)
            else:
                ax.plot(masterData[peakInd][0], masterData[peakInd][1], marker='.', alpha=0.6, label=peak,
                     color=colors[ind])

            # Evaluate fit.
            fitEval = []
            for x in tpRange:

                if fitResults[0][ind] == 'Mono':
                    fitEval.append(
                        fitTimeTraceModels.ExponentialIntensityDecay(x, fitResults[1][ind][0], fitResults[1][ind][1]))

                if fitResults[0][ind] == 'Bi':
                    fitEval.append(
                        fitTimeTraceModels.BiExponentialIntensityDecay(x, fitResults[1][ind][0], fitResults[1][ind][1],
                                                                       fitResults[1][ind][2], fitResults[1][ind][3]))

                if fitResults[0][ind] == 'Tri':
                    fitEval.append(
                        fitTimeTraceModels.TriExponentialIntensityDecay(x, fitResults[1][ind][0], fitResults[1][ind][1],
                                                                        fitResults[1][ind][2], fitResults[1][ind][3],
                                                                        fitResults[1][ind][4], fitResults[1][ind][5]))

            # Plot the fit.
            if peak == '(400)':
                ax.plot(tpRange, fitEval, linestyle='--', color=color400)
            else:
                ax.plot(tpRange, fitEval, linestyle='--', color=colors[ind])
        ax.set_xlim(xRange)
        ax.set_ylabel(r'$\Delta I$ / $I_{o}$', labelpad=5)
        ax.set_xlabel('Timepoint (ps)', labelpad=5)
        ax.text(-0.2, 0.95, letter, transform=ax.transAxes, size=12, weight='bold', **hfont)
        ax.legend(loc='lower right', prop={'size': 7})
        if dotSize == 'Small':
            xticks = ax.xaxis.get_major_ticks()
            # xticks[0].label1.set_visible(False)
            xticks[-1].label1.set_visible(False)

    def plotPEDWTwice(ax):
        import numpy as np
        import math
        import os
        import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit

        hfont = {'fontname': 'Helvetica'}

        # Global constants.
        # TODO: Populate these suckers.
        verbose = True
        selectAverage = False
        selectWindow = True
        selectSingle = False
        showTitle = False
        peaksToAvoid = ['(111)']
        dotsToPlot = ['Small', 'Large']

        # Global functions and tools.
        def convertTwoTheta(twoTheta):

            # This function converts an x-axis of two-theta in to Q.
            lambdaElec = 0.038061
            qArray = []
            for i in twoTheta:
                radVal = (i) * (np.pi / 180)
                result = ((4 * np.pi) / lambdaElec) * math.sin((radVal) / 2)
                qArray.append(result)

            return np.asarray(qArray)

        def findNearest(array, value):

            # Returns nearest index value to specified value in array.
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx

        for pInd, dot in enumerate(dotsToPlot):

            cmap = plt.get_cmap("tab10")

            # Dot parameters defined.
            dotSize = dot
            if dotSize == 'Large':
                peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
                dataDirec = 'E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\Master Data'
                convTT = 0.0046926
            elif dotSize == 'Medium':
                peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
                dataDirec = 'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\Master Data'
                convTT = 0.0046926
            elif dotSize == 'Small':
                peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
                dataDirec = 'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\Master Data'
                convTT = 0.0046926
            elif dotSize == 'Shelled':
                peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
                dataDirec = 'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\Master Data'
                convTT = 0.0046926

            # Begin by loading in data.
            fileName = 'masterData_Int2_%s.npy' % dotSize
            masterData = np.load(os.path.join(dataDirec, fileName))

            # Step 1: determine peak positions by loading data for selected folder.
            # It is best to use a directory with lots of data.

            # peakDataLoc = 'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\2020-08-06\\scans\\scan14\\bgRem\\averagedImages\\fitData' # Smalls
            peakDataLoc = 'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\2020-02-04\\scans\\scan12\\bgRem\\averagedImages\\fitData'

            # Iterate through and average all peak positions for each peak, convert to 1/d.
            peakPos = []
            peakPosError = []
            for ind, peak in enumerate(peakLabels):

                if peak not in peaksToAvoid:
                    # Load in peak data for specified peak.
                    peakData = np.load(os.path.join(peakDataLoc, '%s_peakData.npy' % peak))

                    # Average all values in the 2nd row (peak off positions).
                    peakPositions = peakData[1, :]
                    peakPos.append(np.average(peakPositions))
                    peakPosError.append(np.std(peakPositions))

            # Convert to s, then convert to Q. Peak data is now available.
            peakPos = np.asarray(peakPos)
            peakPosS = peakPos * convTT
            peakPosQ = convertTwoTheta(peakPosS)
            peakPosError = np.asarray(peakPosError)
            peakPosSError = peakPosError * convTT
            peakPosQError = convertTwoTheta(peakPosSError)

            # Start D-W stuff. All master data has first row as timepoints, second row as -ln(I/I_o).
            if selectAverage:
                dwChanges = []
                dwErrors = []
                # for ind, peak in enumerate(peakLabels):

                # Load in integrated data.
                #     plt.plot(masterData[ind][0], masterData[ind][1], label=peak)
                # plt.legend(loc='upper right')
                # plt.show()

                print("Available timepoints:\n")
                print(masterData[0][0])
                tpSel = int(input("Please select at which timepoint to start integrating:\n"))

                # Find matching index, average everything after that, and append to master list.
                tpInd = findNearest(masterData[0][0], tpSel)

                # Start the averaging.
                for ind, peak in enumerate(peakLabels):

                    if peak not in peaksToAvoid:
                        if verbose:
                            print("Now averaging data past %s ps for peak %s." % (masterData[0][0][tpInd], peak))

                        # Get all data after selected timepoint.
                        dwData = masterData[ind][1][tpInd:]

                        # Average the data for that peak and append to array.
                        dwChanges.append(np.average(dwData))
                        dwErrors.append(np.std(dwData))

            if selectWindow:
                dwChanges = []
                dwErrors = []
                # for ind, peak in enumerate(peakLabels):

                # Load in integrated data.
                #     plt.plot(masterData[ind][0], masterData[ind][1], label=peak)
                # plt.legend(loc='upper right')
                # plt.show()

                print("Available timepoints:\n")
                print(masterData[0][0])
                tpSel = int(input("Please select at which timepoint to start integrating:\n"))
                tpSelEnd = int(input("Please select at which timepoint to stop integrating:\n"))

                # Find matching index, average everything after that, and append to master list.
                tpInd = findNearest(masterData[0][0], tpSel)
                tpIndEnd = findNearest(masterData[0][0], tpSelEnd)

                # Start the averaging.
                for ind, peak in enumerate(peakLabels):

                    if peak not in peaksToAvoid:
                        if verbose:
                            print("Now averaging data past %s ps for peak %s." % (masterData[0][0][tpInd], peak))

                        # Get all data after selected timepoint.
                        dwData = masterData[ind][1][tpInd:tpIndEnd]

                        # Average the data for that peak and append to array.
                        dwChanges.append(np.average(dwData))
                        dwErrors.append(np.std(dwData))

            if selectSingle:
                dwChanges = []
                dwErrors = []
                # for ind, peak in enumerate(peakLabels):

                # Load in integrated data.
                #     plt.plot(masterData[ind][0], masterData[ind][1], label=peak)
                # plt.legend(loc='upper right')
                # plt.show()

                print("Available timepoints:\n")
                print(masterData[0][0])
                tpSel = int(input("Please select a timepoint to view:\n"))

                # Find matching index, average everything after that, and append to master list.
                tpInd = findNearest(masterData[0][0], tpSel)

                # Start the averaging.
                for ind, peak in enumerate(peakLabels):

                    if peak not in peaksToAvoid:
                        if verbose:
                            print("Now averaging data past %s ps for peak %s." % (masterData[0][0][tpInd], peak))

                        # Get all data after selected timepoint.
                        dwData = masterData[ind][1][tpInd]

                        # Average the data for that peak and append to array.
                        dwChanges.append(dwData)
                        dwErrors.append(np.std(dwData))

            # # Fit data to a line.
            # def linearFit(m, x):
            #     return m * x
            #
            # popt, _ = curve_fit(linearFit, peakPosQ**2, dwChanges)
            # slope= popt
            # xRange = np.arange(peakPosQ[0]**2, peakPosQ[-1]**2, 0.1)
            # yEval = linearFit(slope, xRange)

            # Plot the results.
            ax.errorbar(peakPosQ ** 2, dwChanges, linestyle='None', yerr=dwErrors, capsize=5, marker='o',
                         color=cmap(pInd), ecolor='black', zorder=0, label='%s QD' % dot)

            # Draw square to end highlighting region of localized disorder exceeding the D-W behaviour.

            # Begin determining fitting for linear D-W portion of response.
            peaksToFit = ['(200)', '(220)', '(311) and (222)']
            yFit = []
            xFit = []
            for peak in peaksToFit:
                # Extract the relevant peaks.
                ind = peakLabels.index(peak) - 1
                xFit.append(peakPosQ[ind])
                yFit.append(dwChanges[ind])

            # Fit data to a line.
            def linearFit(x, m, b):
                return m * x + b

            yFit = np.asarray(yFit)
            xFit = np.asarray(xFit)
            popt, pcov = curve_fit(linearFit, xFit ** 2, yFit)
            slope = popt[0]
            intercept = popt[1]
            xRange = np.arange(peakPosQ[0] ** 2, peakPosQ[-1] ** 2, 0.1)
            yEval = linearFit(slope, xRange, intercept)

            ax.plot(xRange, yEval, alpha=0.6, linewidth=7, zorder=-1, color=cmap(pInd),
                     label='%s (Thermal)' % dot)
            if showTitle:
                if selectAverage:
                    plt.title('Photoexcited DW - Averaged > %.0f ps (%s)' % (tpSel, dotSize))
                if selectSingle:
                    plt.title('Photoexcited DW - At %.0f  ps (%s)' % (tpSel, dotSize))
                if selectWindow:
                    plt.title('Photoexcited DW - %.0f to %.0f ps (%s)' % (tpSel, tpSelEnd, dotSize))
            ax.set_ylabel(r'$-ln(I/I_{o})$', labelpad=5)
            ax.set_xlabel(r'$q^{2}$ ($\AA^{-2}$)', labelpad=5)
            if pInd == 0:
                ax.axvspan(peakPosQ[3] ** 2, (peakPosQ[-1] + 2) ** 2, alpha=0.4, zorder=-2, facecolor='grey', lw=0)
            ax.set_xlim([peakPosQ[0] ** 2 - 0.5, peakPosQ[-1] ** 2 + 0.5])

        ax.text(-0.2, 0.95, 'c)', transform=ax.transAxes, size=12, weight='bold', **hfont)
        ax.legend(loc='upper left', prop={'size': 7}, frameon=False)

    def generateSizeDependenceCurve(ax):

        hfont = {'fontname': 'Helvetica'}

        import fitDiffPeaks

        # Include the shelled dot as well.
        includeShelled = True

        # Define data.
        xData = np.asarray([2.41, 5.47, 8.73])
        yData = np.asarray([-1.208906673, -0.524616148, -0.378001409])
        xLims = (2, 9)

        # Apply 1/r fit, plot.
        # xRange = np.arange(xData[0], xData[-1], 0.01)
        xRange = np.arange(2, 9, 0.01)

        # Apply fit.
        fitResult = fitDiffPeaks.performOneOverR(xData, yData)

        # Error in fit.
        Aval = fitResult[0].params['Fit_A'].value
        Aerror = fitResult[0].params['Fit_A'].stderr

        # Evaluate the fit and plot.
        fitEvalLow = []
        fitEvalHigh = []
        fitEval = []
        for x in xRange:
            fitEval.append(fitResult[0].eval(x=x))
            fitEvalLow.append(fitDiffPeaks.OneOverR(x, Aval - Aerror))
            fitEvalHigh.append(fitDiffPeaks.OneOverR(x, Aval + Aerror))

        fitEvalHigh = np.asarray(fitEvalHigh)
        fitEvalLow = np.asarray(fitEvalLow)
        fitEval = np.asarray(fitEval)

        # Plot the result.
        ax.plot(xData, yData, marker='o', linestyle='None', zorder=0, label='Unshelled PbS QD')
        if includeShelled:
            ax.plot(5.3, -0.31, marker='o', linestyle='None', zorder=0, color='orange', label='Shelled PbS QD')
        ax.plot(xRange, fitEval, linestyle='--', color='grey', zorder=-1, label=r'$1/r$ fit')
        ax.fill_between(xRange, fitEval - Aerror, fitEval + Aerror, color='grey', alpha=0.3, linestyle='None',
                         zorder=-2)
        ax.set_xlabel('QD Diameter (mm)', labelpad=5)
        ax.set_ylabel(r'$\Delta$ $-ln(I/I_{o})$', labelpad=5)
        ax.set_xlim(xLims)
        ax.legend(loc='lower right', prop={'size': 8}, frameon=False)
        ax.text(-0.2, 0.95, 'd)', transform=ax.transAxes, size=12, weight='bold', **hfont)

    # Do the plot stuff.
    plotTimeTraces('Small', ax1)
    plotTimeTraces('Large', ax2)
    plotPEDWTwice(ax3)
    generateSizeDependenceCurve(ax4)

    plt.tight_layout(pad=0.1, w_pad=0.05, h_pad=0.05)
    plt.show()


def figure4():

    # Set up figure stuff.
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['figure.figsize'] = (7, 3)

    # Set up the plot.
    fig, (ax1, ax2) = plt.subplots(1,2)

    # Define functions, again.
    def plotTimeTraces(dotSize, ax):

        import fitTimeTraceModels
        hfont = {'fontname': 'Helvetica'}

        # Select dot size and fit params.
        if dotSize == 'Small':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            peaksToDraw = ['(311) and (222)', '(400)', '(331) and (420)']
            letter = 'a)'
            fitResults = [['Tri', 'Tri', 'Bi'],
                          [(3.1, 50, 585, -0.30, 0.17, -0.049), (2.0, 40, 707, -1.09, 0.43, -0.25),
                           (12.6, 47, 0.40, -0.31)]]
            masterData = np.load(
                'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\Master Data\\masterData_Int_Small.npy')
        if dotSize == 'Medium':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            peaksToDraw = ['(220)', '(311) and (222)', '(400)', '(331) and (420)']
            fitResults = [['Bi', 'Bi', 'Bi', 'Mono'],
                          [(0.86, 554, -0.037, -0.076), (3.81, 586, -0.095, -0.066), (5.89, 260, -0.635, 0.119),
                           (2.15, -0.104)]]
            masterData = np.load(
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\Master Data\\masterData_Int_Medium.npy')
        if dotSize == 'Large':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            peaksToDraw = ['(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)']
            fitResults = [['Mono', 'Mono', 'Mono', 'Bi', 'Mono'],
                          [(2.39, -0.024), (2.41, -0.095), (3.51, -0.195), (3.5, 882, -0.512, -0.348), (2.30, -0.164)]]
            masterData = np.load('E:\\Ediff Samples\PbS Data\\8 nm Dots Pumped\\Master Data\\masterData_Int_Large.npy')
            letter = 'b)'
        if dotSize == 'Shelled':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            peaksToDraw = ['(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)']
            fitResults = [['Mono', 'Tri', 'Tri', 'Mono', 'Tri'],
                          [(4.83, -0.034), (1.04, 22.62, 359, -0.17, 0.06, -0.07),
                           (2.48, 12, 339, -0.303, 0.0971, -0.047), (3.05, -0.392),
                           (1.32, 13.43, 252.20, -0.44, 0.24, -0.10)]]
            masterData = np.load(
                'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\Master Data\\masterData_Int_Shelled.npy')
            letter='a)'

        # Plot parameters.
        colors = plt.cm.plasma(np.linspace(0, 1, len(peaksToDraw)))

        # Set up evaluation.
        tpRange = np.arange(-300, 1000, 1)
        fitEval = []

        tpVals = masterData[0][0]
        xRange = [min(tpVals), max(tpVals)]

        # Go through, plot and evaluate for each peak.
        for ind, peak in enumerate(peaksToDraw):

            # Which peak?
            peakInd = peakLabels.index(peak)

            # Plot the data.
            ax.plot(masterData[peakInd][0], masterData[peakInd][1], marker='.', alpha=0.6, label=peak,
                     color=colors[ind])

            # Evaluate fit.
            fitEval = []
            for x in tpRange:

                if fitResults[0][ind] == 'Mono':
                    fitEval.append(
                        fitTimeTraceModels.ExponentialIntensityDecay(x, fitResults[1][ind][0], fitResults[1][ind][1]))

                if fitResults[0][ind] == 'Bi':
                    fitEval.append(
                        fitTimeTraceModels.BiExponentialIntensityDecay(x, fitResults[1][ind][0], fitResults[1][ind][1],
                                                                       fitResults[1][ind][2], fitResults[1][ind][3]))

                if fitResults[0][ind] == 'Tri':
                    fitEval.append(
                        fitTimeTraceModels.TriExponentialIntensityDecay(x, fitResults[1][ind][0], fitResults[1][ind][1],
                                                                        fitResults[1][ind][2], fitResults[1][ind][3],
                                                                        fitResults[1][ind][4], fitResults[1][ind][5]))

            # Plot the fit.
            ax.plot(tpRange, fitEval, linestyle='--', color=colors[ind])
        ax.set_xlim(xRange)
        ax.set_ylabel(r'$\Delta I$ / $I_{o}$', labelpad=5)
        ax.set_xlabel('Timepoint (ps)', labelpad=5)
        ax.text(-0.3, 0.95, letter, transform=ax.transAxes, size=12, weight='bold', **hfont)
        ax.legend(loc='lower right', prop={'size': 7})

    def debyeWallerPE(dotSize, ax):
        import numpy as np
        import math
        import os
        import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit

        hfont = {'fontname': 'Helvetica'}

        # Global constants.
        # TODO: Populate these suckers.
        verbose = True
        selectAverage = False
        selectWindow = True
        selectSingle = False
        showTitle = False
        peaksToAvoid = ['(111)']

        # Global functions and tools.
        def convertTwoTheta(twoTheta):

            # This function converts an x-axis of two-theta in to Q.
            lambdaElec = 0.038061
            qArray = []
            for i in twoTheta:
                radVal = (i) * (np.pi / 180)
                result = ((4 * np.pi) / lambdaElec) * math.sin((radVal) / 2)
                qArray.append(result)

            return np.asarray(qArray)

        def findNearest(array, value):

            # Returns nearest index value to specified value in array.
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx

        # Dot parameters defined.
        if dotSize == 'Large':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            dataDirec = 'E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\Master Data'
            convTT = 0.0047808
        elif dotSize == 'Medium':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            dataDirec = 'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\Master Data'
            convTT = 0.0046926
        elif dotSize == 'Small':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            dataDirec = 'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\Master Data'
            convTT = 0.0047412
        elif dotSize == 'Shelled':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            dataDirec = 'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\Master Data'
            convTT = 0.0047023

        # Begin by loading in data.
        fileName = 'masterData_Int2_%s.npy' % dotSize
        masterData = np.load(os.path.join(dataDirec, fileName))

        # Step 1: determine peak positions by loading data for selected folder.
        # It is best to use a directory with lots of data.

        # peakDataLoc = 'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\2020-08-06\\scans\\scan14\\bgRem\\averagedImages\\fitData' # Smalls
        peakDataLoc = 'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\2020-02-04\\scans\\scan12\\bgRem\\averagedImages\\fitData'

        # Iterate through and average all peak positions for each peak, convert to 1/d.
        peakPos = []
        peakPosError = []
        for ind, peak in enumerate(peakLabels):

            if peak not in peaksToAvoid:
                # Load in peak data for specified peak.
                peakData = np.load(os.path.join(peakDataLoc, '%s_peakData.npy' % peak))

                # Average all values in the 2nd row (peak off positions).
                peakPositions = peakData[1, :]
                peakPos.append(np.average(peakPositions))
                peakPosError.append(np.std(peakPositions))

        # Convert to s, then convert to Q. Peak data is now available.
        peakPos = np.asarray(peakPos)
        peakPosS = peakPos * convTT
        peakPosQ = convertTwoTheta(peakPosS)
        peakPosError = np.asarray(peakPosError)
        peakPosSError = peakPosError * convTT
        peakPosQError = convertTwoTheta(peakPosSError)

        # Start D-W stuff. All master data has first row as timepoints, second row as -ln(I/I_o).
        if selectAverage:
            dwChanges = []
            dwErrors = []
            for ind, peak in enumerate(peakLabels):
                # Load in integrated data.
                plt.plot(masterData[ind][0], masterData[ind][1], label=peak)
            plt.legend(loc='upper right')
            plt.show()

            print("Available timepoints:\n")
            print(masterData[0][0])
            tpSel = int(input("Please select at which timepoint to start integrating:\n"))

            # Find matching index, average everything after that, and append to master list.
            tpInd = findNearest(masterData[0][0], tpSel)

            # Start the averaging.
            for ind, peak in enumerate(peakLabels):

                if peak not in peaksToAvoid:
                    if verbose:
                        print("Now averaging data past %s ps for peak %s." % (masterData[0][0][tpInd], peak))

                    # Get all data after selected timepoint.
                    dwData = masterData[ind][1][tpInd:]

                    # Average the data for that peak and append to array.
                    dwChanges.append(np.average(dwData))
                    dwErrors.append(np.std(dwData))

        if selectWindow:
            dwChanges = []
            dwErrors = []
            # for ind, peak in enumerate(peakLabels):
            #     # Load in integrated data.
            #     plt.plot(masterData[ind][0], masterData[ind][1], label=peak)
            # plt.legend(loc='upper right')
            # plt.show()

            print("Available timepoints:\n")
            print(masterData[0][0])
            tpSel = int(input("Please select at which timepoint to start integrating:\n"))
            tpSelEnd = int(input("Please select at which timepoint to stop integrating:\n"))

            # Find matching index, average everything after that, and append to master list.
            tpInd = findNearest(masterData[0][0], tpSel)
            tpIndEnd = findNearest(masterData[0][0], tpSelEnd)

            # Start the averaging.
            for ind, peak in enumerate(peakLabels):

                if peak not in peaksToAvoid:
                    if verbose:
                        print("Now averaging data past %s ps for peak %s." % (masterData[0][0][tpInd], peak))

                    # Get all data after selected timepoint.
                    dwData = masterData[ind][1][tpInd:tpIndEnd]

                    # Average the data for that peak and append to array.
                    dwChanges.append(np.average(dwData))
                    dwErrors.append(np.std(dwData))

        if selectSingle:
            dwChanges = []
            dwErrors = []
            for ind, peak in enumerate(peakLabels):
                # Load in integrated data.
                plt.plot(masterData[ind][0], masterData[ind][1], label=peak)
            plt.legend(loc='upper right')
            plt.show()

            print("Available timepoints:\n")
            print(masterData[0][0])
            tpSel = int(input("Please select a timepoint to view:\n"))

            # Find matching index, average everything after that, and append to master list.
            tpInd = findNearest(masterData[0][0], tpSel)

            # Start the averaging.
            for ind, peak in enumerate(peakLabels):

                if peak not in peaksToAvoid:
                    if verbose:
                        print("Now averaging data past %s ps for peak %s." % (masterData[0][0][tpInd], peak))

                    # Get all data after selected timepoint.
                    dwData = masterData[ind][1][tpInd]

                    # Average the data for that peak and append to array.
                    dwChanges.append(dwData)
                    dwErrors.append(np.std(dwData))

        # # Fit data to a line.
        # def linearFit(m, x):
        #     return m * x
        #
        # popt, _ = curve_fit(linearFit, peakPosQ**2, dwChanges)
        # slope= popt
        # xRange = np.arange(peakPosQ[0]**2, peakPosQ[-1]**2, 0.1)
        # yEval = linearFit(slope, xRange)

        # Plot the results.
        ax.errorbar(peakPosQ ** 2, dwChanges, linestyle='None', yerr=dwErrors, capsize=5,
                     marker='.', ecolor='black', zorder=0)

        # Draw square to end highlighting region of localized disorder exceeding the D-W behaviour.
        print(peakPosQ ** 2)
        print(dwChanges)

        # Begin determining fitting for linear D-W portion of response.
        peaksToFit = ['(200)', '(220)', '(311) and (222)']
        yFit = []
        xFit = []
        for peak in peaksToFit:
            # Extract the relevant peaks.
            ind = peakLabels.index(peak) - 1
            xFit.append(peakPosQ[ind])
            yFit.append(dwChanges[ind])

        # Fit data to a line.
        def linearFit(x, m, b):
            return m * x + b

        yFit = np.asarray(yFit)
        xFit = np.asarray(xFit)
        popt, pcov = curve_fit(linearFit, xFit ** 2, yFit)
        slope = popt[0]
        intercept = popt[1]
        xRange = np.arange(peakPosQ[0] ** 2, peakPosQ[-1] ** 2, 0.1)
        yEval = linearFit(slope, xRange, intercept)

        ax.plot(xRange, yEval, alpha=0.6, linewidth=7, color='C0', zorder=-1)
        # if showTitle:
        #     if selectAverage:
        #         plt.title('Photoexcited DW - Averaged > %.0f ps (%s)' % (tpSel, dotSize))
        #     if selectSingle:
        #         plt.title('Photoexcited DW - At %.0f  ps (%s)' % (tpSel, dotSize))
        #     if selectWindow:
        #         plt.title('Photoexcited DW - %.0f to %.0f ps (%s)' % (tpSel, tpSelEnd, dotSize))
        ax.set_ylabel(r'$-ln(I/I_{o})$', labelpad=5)
        ax.set_xlabel(r'$q^{2}$ ($\AA^{-2}$)', labelpad=5)
        ax.axvspan(peakPosQ[3] ** 2, (peakPosQ[-1] + 2) ** 2, alpha=0.4, zorder=-2, facecolor='grey', lw=0)
        ax.set_xlim([peakPosQ[0] ** 2 - 0.5, peakPosQ[-1] ** 2 + 0.5])
        ax.text(-0.3, 0.95, 'b)', transform=ax.transAxes, size=12, weight='bold', **hfont)

    # Do the plot stuff.
    plotTimeTraces('Shelled', ax1)
    debyeWallerPE('Shelled', ax2)
    plt.tight_layout()
    plt.show()

def figure5():

    # Set up figure stuff.
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['figure.figsize'] = (7, 6)

    # Set up the plot.
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

    # Define things for the last time.
    def plotPDF(dotSize, ax, label):
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        import glob

        hfont = {'fontname': 'Helvetica'}

        def valueToIndex(list, value):
            index = np.abs(np.array(list) - value).argmin()
            return index

        # Define global variables here.
        plotRange = [2.4, 14]
        # dotSize = 'Shelled'
        if dotSize == 'Small':
            dataDirecs = ['D:\\PDFs\\PDF Patterns Small\\scan6_\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Small\\scan7\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Small\\scan9\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Small\\scan9_\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Small\\scan10\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Small\\scan10_\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Small\\scan11_\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Small\\scan12\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Small\\scan13\\PDFs Long Q and R',
                          ]
        if dotSize == 'Medium':
            dataDirecs = ['E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan5\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan6\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan7\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan9\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan10\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan11\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan12\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan17\\PDFs Long Q and R',
                          ]
        if dotSize == 'Large':
            dataDirecs = ['D:\\PDFs\\PDF Patterns Large\\scan13\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Large\\scan16\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Large\\scan17\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Large\\scan18\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Large\\scan19\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Large\\scan1315\\PDFs Long Q and R',
                          ]
        if dotSize == 'Shelled':
            dataDirecs = ['D:\\PDFs\\PDF Patterns Shelled\\scan6\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Shelled\\scan8\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Shelled\\scan9\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Shelled\\scan10\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Shelled\\scan11\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Shelled\\scan12\\PDFs Long Q and R',
                          ]

        # Data structures for initial figure (comparison of on/off G(r)).
        pdfOff = np.zeros(2000)
        pdfOn = np.zeros(2000)
        rRange = np.arange(0, 20, 0.01)
        onCounter = 0
        offCounter = 0

        # Additional data structures for figures.
        offPreT0 = np.zeros(2000)
        onPreT0 = np.zeros(2000)
        # zeroTo5On = np.zeros(1200)
        # zeroTo5Off = np.zeros(1200)
        # fiveTo15On = np.zeros(1200)
        # fiveTo15Off = np.zeros(1200)
        # fifteenTo50On = np.zeros(1200)
        # fifteenTo50Off = np.zeros(1200)
        # fiftyTo100On = np.zeros(1200)
        # fiftyTo100Off = np.zeros(1200)
        # hundredTo200On = np.zeros(1200)
        # hundredTo200Off = np.zeros(1200)
        offPreT0Counter = 0
        onPreT0Counter = 0
        # zeroFiveOffCounter = 0
        # zeroFiveOnCounter = 0

        # Begin by loading in and storing data.
        for pdfDirec in dataDirecs:
            for pdf in glob.glob(os.path.join(pdfDirec, '*.npy')):

                # Extract data config from filename.
                tp = int(os.path.basename(pdf).split('_')[1].split('.')[0])
                state = os.path.basename(pdf).split('_')[0]

                # Split process for ons and offs, check if ons are post-t0 to average.
                if state == 'Goff':

                    # Load in data, append to pdfOffs and adjust counter accordingly.
                    data = np.load(pdf)
                    pdfOff += data
                    offCounter += 1

                    # If it is OFF and before t0:
                    if tp < 0:
                        # Add to off pre-t0 data.
                        offPreT0 += data
                        offPreT0Counter += 1

                if state == 'Gon':

                    # Load in data.
                    data = np.load(pdf)

                    if tp > 0:
                        # Load in data after t0, append to pdfOns and just counter accordingly.
                        pdfOn += data
                        onCounter += 1

                    if tp < 0:
                        # Add to ON pre-t0 data.
                        onPreT0 += data
                        onPreT0Counter += 1

        # Average everything, and begin plotting.
        avgOffs = pdfOff / offCounter
        avgOns = pdfOn / onCounter
        onPreT0 = onPreT0 / onPreT0Counter
        offPreT0 = offPreT0 / offPreT0Counter
        # zeroTo5On = zeroTo5On / zeroFiveOnCounter
        # zeroTo5Off = zeroTo5Off / zeroFiveOffCounter

        # Setup the results and shit.

        # Plot the results.
        ax.plot(rRange, avgOffs, color='black', label='g(r) Off', linewidth=2)
        ax.plot(rRange, avgOns, color='C3', label='g(r) On', linewidth=2)
        ax.set_xlim(plotRange)
        ax.legend(loc='upper left', prop={'size': 7}, frameon=False)
        ax.set_ylabel(r'g (r)', labelpad=5)
        ax.set_xlabel(r'r ($\AA$)', labelpad=5)
        # plt.title('%s Dots - PDF On vs Off' % dotSize)

        startPoint = valueToIndex(rRange, plotRange[0])
        endPoint = valueToIndex(rRange, plotRange[-1])

        # Get minima for plot x axis limits?
        maxY = int(max(avgOffs[startPoint:endPoint]))
        minY = int(min(avgOffs[startPoint:endPoint]))
        ax.set_ylim((minY - 2.5, maxY + 2.5))

        # Draw arrows in the plot.
        # ax1.annotate(s='', xy=(2.7953, minY - 1), xytext=(2.7953, minY - 3), arrowprops=dict(arrowstyle='->'))
        # ax1.arrow(2.7953, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax1.arrow(2.9697, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        ax.arrow(2.9697, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        ax.text(3.0697, minY - 1, 'Pb-S', rotation=90, size=10)
        # ax1.annotate('Pb-S', xy=(2.9697, minY - 1))
        ax.arrow(4.1999, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        ax.arrow(5.1438, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        ax.arrow(5.9395, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        ax.text(6.0395, minY - 1, 'a-axis', rotation=90, size=10)
        ax.arrow(6.6406, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        ax.arrow(7.2744, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        ax.arrow(8.3997, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        ax.arrow(8.9093, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        ax.arrow(9.3912, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        ax.arrow(10.2880, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        ax.arrow(11.1056, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        ax.text(-0.3, 0.95, label, transform=ax.transAxes, size=12, weight='bold', **hfont)

        # ax1.axvline(x=2.7953, color='gray', alpha=0.6)
        # ax1.axvline(x=2.9697, color='yellow', alpha=0.6)
        # ax1.axvline(x=3.2185, color='gray', alpha=0.6)
        # ax1.axvline(x=4.1999, color='black', alpha=0.6)
        # ax1.axvline(x=5.1438, color='yellow', alpha=0.6)
        # ax1.axvline(x=5.9395, color='black', alpha=0.6)
        # ax1.axvline(x=6.6406, color='yellow', alpha=0.6)
        # ax1.axvline(x=7.2744, color='black', alpha=0.6)
        # ax1.axvline(x=8.3997, color='black', alpha=0.6)
        # ax1.axvline(x=8.9093, color='yellow', alpha=0.6)
        # ax1.axvline(x=9.3912, color='black', alpha=0.6)
        # ax1.axvline(x=10.2880, color='yellow', alpha=0.6)

        # plt.show()

        # Extract ranges and store master data.
        ranges = [[0], [0, 5000], [5000, 25000], [25000, 100000], [100000, 1000000]]
        masterChanges = []

        for r in ranges:

            # Global data struct.
            finalDataOff = np.zeros(2000)
            finalDataOn = np.zeros(2000)
            finalCounterOff = 0
            finalCounterOn = 0

            for pdfDirec in dataDirecs:
                for pdf in glob.glob(os.path.join(pdfDirec, '*.npy')):

                    # Extract info.
                    tp = int(os.path.basename(pdf).split('_')[1].split('.')[0])
                    state = os.path.basename(pdf).split('_')[0]

                    data = np.load(pdf)

                    # If pre-t0:
                    if len(r) == 1:
                        if state == 'Goff':
                            if tp < 0:
                                finalDataOff += data
                                finalCounterOff += 1

                        if state == 'Gon':
                            if tp < 0:
                                finalDataOn += data
                                finalCounterOn += 1

                    # Now go over the rest of the ranges.
                    if len(r) > 1:
                        if state == 'Goff':
                            if (tp > r[0]) and (tp <= r[1]):
                                finalDataOff += data
                                finalCounterOff += 1

                        if state == 'Gon':
                            if (tp > r[0]) and (tp <= r[1]):
                                finalDataOn += data
                                finalCounterOn += 1

            # Finish processing data and append to master data array.
            finalDataOn = finalDataOn / finalCounterOn
            finalDataOff = finalDataOff / finalCounterOff
            masterChanges.append(finalDataOn - finalDataOff)

        masterChanges = np.asarray(masterChanges)

        # Twin the axis.
        ax2 = ax.twinx()

        # Plot the results.
        colors = plt.cm.plasma(np.linspace(0, 1, len(ranges) - 1))
        for ind, range in enumerate(ranges):
            if len(range) == 1:
                labelTitle = '< 0 ps'
                ax2.plot(rRange, masterChanges[ind], color='grey', label=labelTitle, linewidth=1, alpha=0.5)
            if len(range) > 1:
                labelTitle = '%0.0f - %0.0f ps' % (range[0] / 1000, range[1] / 1000)
                ax2.plot(rRange, masterChanges[ind], color=colors[ind - 1], label=labelTitle, linewidth=1, alpha=0.5)
        ax2.legend(loc='upper right', prop={'size': 7}, frameon=False)
        ax2.set_ylabel(r'$\Delta$g (r)', labelpad=5)
        # plt.xlabel(r'r ($\AA$)')
        # ax2.xlim(plotRange)

        maxY = max(masterChanges[-1][valueToIndex(rRange, plotRange[0]):valueToIndex(rRange, plotRange[1])]) + 1
        minY = min(masterChanges[-1][valueToIndex(rRange, plotRange[0]):valueToIndex(rRange, plotRange[1])]) - 1

        ax2.set_ylim([-1.5, 1.5])

        # plt.axvline(x=2.7953, color='gray', alpha=0.6)
        # plt.axvline(x=2.9697, color='yellow', alpha=0.6)
        # plt.axvline(x=3.2185, color='gray', alpha=0.6)
        # plt.axvline(x=4.1999, color='black', alpha=0.6)
        # plt.axvline(x=5.1438, color='yellow', alpha=0.6)
        # plt.axvline(x=5.9395, color='black', alpha=0.6)
        # plt.axvline(x=6.6406, color='yellow', alpha=0.6)
        # plt.axvline(x=7.2744, color='black', alpha=0.6)
        # plt.axvline(x=8.3997, color='black', alpha=0.6)
        # plt.axvline(x=8.9093, color='yellow', alpha=0.6)
        # plt.axvline(x=9.3912, color='black', alpha=0.6)
        # plt.axvline(x=10.2880, color='yellow', alpha=0.6)

        # Plot the pre-t0 difference.
        # plt.plot(rRange, onPreT0 - offPreT0, label="< 0 ps")
        # plt.plot(rRange, zeroTo5On - zeroTo5Off, label="0 - 5 ps")
        # plt.legend(loc='upper right')
        # plt.show()

    # Do the plot stuff.
    plotPDF('Small', ax1, 'a)')
    plotPDF('Medium', ax2, 'b)')
    plotPDF('Large', ax3, 'c)')
    plotPDF('Shelled', ax4, 'd)')

    plt.tight_layout(pad=0.1, w_pad=0.05, h_pad=0.05)
    plt.show()

def figure5alt():

    # Set up figure stuff.
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['figure.figsize'] = (7, 6)

    # Set up the axes and figures and stuff.
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
    (ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col', sharey='row')

    # Alt plotting function for the PDFs.
    def plotPDFdiffs(dotSize, ax, label):
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        import glob

        hfont = {'fontname': 'Helvetica'}

        def valueToIndex(list, value):
            index = np.abs(np.array(list) - value).argmin()
            return index

        # Define global variables here.
        plotRange = [2.4, 15]
        # dotSize = 'Shelled'
        if dotSize == 'Small':
            dataDirecs = ['D:\\PDFs\\PDF Patterns Small\\scan6_\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Small\\scan7\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Small\\scan9\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Small\\scan9_\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Small\\scan10\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Small\\scan10_\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Small\\scan11_\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Small\\scan12\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Small\\scan13\\PDFs Long Q and R',
                          ]
        if dotSize == 'Medium':
            dataDirecs = ['E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan5\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan6\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan7\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan9\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan10\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan11\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan12\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan17\\PDFs Long Q and R',
                          ]
        if dotSize == 'Large':
            dataDirecs = ['D:\\PDFs\\PDF Patterns Large\\scan13\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Large\\scan16\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Large\\scan17\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Large\\scan18\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Large\\scan19\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Large\\scan1315\\PDFs Long Q and R',
                          ]
        if dotSize == 'Shelled':
            dataDirecs = ['D:\\PDFs\\PDF Patterns Shelled\\scan6\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Shelled\\scan8\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Shelled\\scan9\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Shelled\\scan10\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Shelled\\scan11\\PDFs Long Q and R',
                          'D:\\PDFs\\PDF Patterns Shelled\\scan12\\PDFs Long Q and R',
                          ]

        # Data structures for initial figure (comparison of on/off G(r)).
        pdfOff = np.zeros(2000)
        pdfOn = np.zeros(2000)
        rRange = np.arange(0, 20, 0.01)
        onCounter = 0
        offCounter = 0

        # Additional data structures for figures.
        offPreT0 = np.zeros(2000)
        onPreT0 = np.zeros(2000)
        # zeroTo5On = np.zeros(1200)
        # zeroTo5Off = np.zeros(1200)
        # fiveTo15On = np.zeros(1200)
        # fiveTo15Off = np.zeros(1200)
        # fifteenTo50On = np.zeros(1200)
        # fifteenTo50Off = np.zeros(1200)
        # fiftyTo100On = np.zeros(1200)
        # fiftyTo100Off = np.zeros(1200)
        # hundredTo200On = np.zeros(1200)
        # hundredTo200Off = np.zeros(1200)
        offPreT0Counter = 0
        onPreT0Counter = 0
        # zeroFiveOffCounter = 0
        # zeroFiveOnCounter = 0

        # Begin by loading in and storing data.
        for pdfDirec in dataDirecs:
            for pdf in glob.glob(os.path.join(pdfDirec, '*.npy')):

                # Extract data config from filename.
                tp = int(os.path.basename(pdf).split('_')[1].split('.')[0])
                state = os.path.basename(pdf).split('_')[0]

                # Split process for ons and offs, check if ons are post-t0 to average.
                if state == 'Goff':

                    # Load in data, append to pdfOffs and adjust counter accordingly.
                    data = np.load(pdf)
                    pdfOff += data
                    offCounter += 1

                    # If it is OFF and before t0:
                    if tp < 0:
                        # Add to off pre-t0 data.
                        offPreT0 += data
                        offPreT0Counter += 1

                if state == 'Gon':

                    # Load in data.
                    data = np.load(pdf)

                    if tp > 0:
                        # Load in data after t0, append to pdfOns and just counter accordingly.
                        pdfOn += data
                        onCounter += 1

                    if tp < 0:
                        # Add to ON pre-t0 data.
                        onPreT0 += data
                        onPreT0Counter += 1

        # Average everything, and begin plotting.
        avgOffs = pdfOff / offCounter
        avgOns = pdfOn / onCounter
        onPreT0 = onPreT0 / onPreT0Counter
        offPreT0 = offPreT0 / offPreT0Counter
        # zeroTo5On = zeroTo5On / zeroFiveOnCounter
        # zeroTo5Off = zeroTo5Off / zeroFiveOffCounter

        # Setup the results and shit.

        # Extract ranges and store master data.
        ranges = [[0], [0, 5000], [5000, 25000], [25000, 100000], [100000, 1000000]]
        masterChanges = []

        for r in ranges:

            # Global data struct.
            finalDataOff = np.zeros(2000)
            finalDataOn = np.zeros(2000)
            finalCounterOff = 0
            finalCounterOn = 0

            for pdfDirec in dataDirecs:
                for pdf in glob.glob(os.path.join(pdfDirec, '*.npy')):

                    # Extract info.
                    tp = int(os.path.basename(pdf).split('_')[1].split('.')[0])
                    state = os.path.basename(pdf).split('_')[0]

                    data = np.load(pdf)

                    # If pre-t0:
                    if len(r) == 1:
                        if state == 'Goff':
                            if tp < 0:
                                finalDataOff += data
                                finalCounterOff += 1

                        if state == 'Gon':
                            if tp < 0:
                                finalDataOn += data
                                finalCounterOn += 1

                    # Now go over the rest of the ranges.
                    if len(r) > 1:
                        if state == 'Goff':
                            if (tp > r[0]) and (tp <= r[1]):
                                finalDataOff += data
                                finalCounterOff += 1

                        if state == 'Gon':
                            if (tp > r[0]) and (tp <= r[1]):
                                finalDataOn += data
                                finalCounterOn += 1

            # Finish processing data and append to master data array.
            finalDataOn = finalDataOn / finalCounterOn
            finalDataOff = finalDataOff / finalCounterOff
            masterChanges.append(finalDataOn - finalDataOff)

        masterChanges = np.asarray(masterChanges)

        # Plot the results.
        colors = plt.cm.plasma(np.linspace(0, 1, len(ranges) - 1))
        for ind, range in enumerate(ranges):
            if len(range) == 1:
                labelTitle = '< 0 ps'
                ax.plot(rRange, masterChanges[ind], color='grey', label=labelTitle, linewidth=1, alpha=0.8)
            if len(range) > 1:
                labelTitle = '%0.0f - %0.0f ps' % (range[0] / 1000, range[1] / 1000)
                ax.plot(rRange, masterChanges[ind], color=colors[ind - 1], label=labelTitle, linewidth=1, alpha=0.8)
        if label == 'b)':
            ax.legend(loc='upper right', prop={'size': 7}, frameon=False)
        if (label == 'b)') or (label == 'd)'):
            ax.axes.yaxis.set_visible(False)
        # ax.set_ylabel(r'$\Delta$g (r)', labelpad=5)
        # plt.xlabel(r'r ($\AA$)')
        # ax2.xlim(plotRange)

        maxY = max(masterChanges[-1][valueToIndex(rRange, plotRange[0]):valueToIndex(rRange, plotRange[1])])
        minY = min(masterChanges[-1][valueToIndex(rRange, plotRange[0]):valueToIndex(rRange, plotRange[1])])

        ax.text(0.025, 0.925, label, transform=ax.transAxes, size=12, weight='bold', **hfont)

        ax.set_ylim([minY - 0.5, maxY + 0.5])
        ax.set_xlim([plotRange[0], plotRange[1]])

        # # Draw arrows in the plot.
        # # ax1.annotate(s='', xy=(2.7953, minY - 1), xytext=(2.7953, minY - 3), arrowprops=dict(arrowstyle='->'))
        # # ax1.arrow(2.7953, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # # ax1.arrow(2.9697, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.arrow(2.9697, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        # ax.text(3.0697, minY - 1, 'Pb-S', rotation=90, size=10)
        # # ax1.annotate('Pb-S', xy=(2.9697, minY - 1))
        # ax.arrow(4.1999, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.arrow(5.1438, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        # ax.arrow(5.9395, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.text(6.0395, minY - 1, 'a-axis', rotation=90, size=10)
        # ax.arrow(6.6406, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        # ax.arrow(7.2744, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.arrow(8.3997, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.arrow(8.9093, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        # ax.arrow(9.3912, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.arrow(10.2880, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        # ax.arrow(11.1056, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.text(-0.3, 0.95, label, transform=ax.transAxes, size=12, weight='bold', **hfont)

    plotPDFdiffs('Small', ax1, 'a)')
    plotPDFdiffs('Medium', ax2, 'b)')
    plotPDFdiffs('Large', ax3, 'c)')
    plotPDFdiffs('Shelled', ax4, 'd)')

    shadowaxes = fig.add_subplot(111, xticks=[], yticks=[], frame_on=False)

    shadowaxes.set_xlabel(r'r ($\AA$)')
    shadowaxes.set_ylabel(r'$\Delta$g (r)')
    shadowaxes.xaxis.labelpad = 25
    shadowaxes.yaxis.labelpad = 35

    plt.tight_layout()


    plt.show()

def figure5extra():

    # Set up figure stuff.
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['figure.figsize'] = (3.33, 3)

    # Set up the plot.
    fig, ax1 = plt.subplots(1, 1)

    def plotPDF(dotSize, ax):
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        import glob

        hfont = {'fontname': 'Helvetica'}

        def valueToIndex(list, value):
            index = np.abs(np.array(list) - value).argmin()
            return index

        # Define global variables here.
        plotRange = [2.4, 12]
        # dotSize = 'Shelled'
        if dotSize == 'Small':
            dataDirecs = ['E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan6_\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan7\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan9\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan9_\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan10\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan10_\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan11_\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan12\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan13\\PDFs',
                          ]
        if dotSize == 'Medium':
            dataDirecs = ['E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan5\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan6\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan8\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan9\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
                          ]
        if dotSize == 'Large':
            dataDirecs = ['E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\PDF\\PDF Patterns Large\\scan1315\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\PDF\\PDF Patterns Large\\scan14\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\PDF\\PDF Patterns Large\\scan16\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\PDF\\PDF Patterns Large\\scan17\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\PDF\\PDF Patterns Large\\scan18\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\PDF\\PDF Patterns Large\\scan19\\PDFs',
                          ]
        if dotSize == 'Shelled':
            dataDirecs = dataDirecs = ['E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\PDF\\scan6\\PDFs',
                                       'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\PDF\\scan8\\PDFs',
                                       'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\PDF\\scan9\\PDFs',
                                       'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\PDF\\scan10\\PDFs',
                                       'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\PDF\\scan11\\PDFs',
                                       'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\PDF\\scan12\\PDFs',
                                       ]

        # Data structures for initial figure (comparison of on/off G(r)).
        pdfOff = np.zeros(1200)
        pdfOn = np.zeros(1200)
        pdfOnRange = np.zeros(1200)
        pdfOffRange = np.zeros(1200)
        rRange = np.arange(0, 12, 0.01)
        onCounter = 0
        offCounter = 0

        # Additional data structures for figures.
        offPreT0 = np.zeros(1200)
        onPreT0 = np.zeros(1200)
        # zeroTo5On = np.zeros(1200)
        # zeroTo5Off = np.zeros(1200)
        # fiveTo15On = np.zeros(1200)
        # fiveTo15Off = np.zeros(1200)
        # fifteenTo50On = np.zeros(1200)
        # fifteenTo50Off = np.zeros(1200)
        # fiftyTo100On = np.zeros(1200)
        # fiftyTo100Off = np.zeros(1200)
        # hundredTo200On = np.zeros(1200)
        # hundredTo200Off = np.zeros(1200)
        offPreT0Counter = 0
        onPreT0Counter = 0
        # zeroFiveOffCounter = 0
        # zeroFiveOnCounter = 0

        # Begin by loading in and storing data.
        for pdfDirec in dataDirecs:
            for pdf in glob.glob(os.path.join(pdfDirec, '*.npy')):

                # Extract data config from filename.
                tp = int(os.path.basename(pdf).split('_')[1].split('.')[0])
                state = os.path.basename(pdf).split('_')[0]

                # Split process for ons and offs, check if ons are post-t0 to average.
                if state == 'Goff':

                    # Load in data, append to pdfOffs and adjust counter accordingly.
                    data = np.load(pdf)
                    pdfOff += data
                    offCounter += 1

                    # If it is OFF and before t0:
                    if tp < 0:
                        # Add to off pre-t0 data.
                        offPreT0 += data
                        offPreT0Counter += 1

                if state == 'Gon':

                    # Load in data.
                    data = np.load(pdf)

                    if tp > 0:
                        # Load in data after t0, append to pdfOns and just counter accordingly.
                        pdfOn += data
                        onCounter += 1

                    if tp < 0:
                        # Add to ON pre-t0 data.
                        onPreT0 += data
                        onPreT0Counter += 1

        # Average everything, and begin plotting.
        avgOffs = pdfOff / offCounter
        avgOns = pdfOn / onCounter
        onPreT0 = onPreT0 / onPreT0Counter
        offPreT0 = offPreT0 / offPreT0Counter
        # zeroTo5On = zeroTo5On / zeroFiveOnCounter
        # zeroTo5Off = zeroTo5Off / zeroFiveOffCounter

        # Setup the results and shit.
        diff = avgOns - avgOffs

        # Plot the results.
        ax.plot(rRange, avgOffs, color='black', label='g(r) Off', linewidth=2)
        ax.plot(rRange, avgOns, color='C3', label='g(r) On', linewidth=2)
        ax.plot(rRange, diff * 5, color='grey', linestyle='dotted', label='Difference', linewidth=2, zorder=2)
        ax.set_xlim(plotRange)
        ax.legend(loc='upper right', prop={'size': 7}, frameon=False)
        ax.set_ylabel(r'g (r)', labelpad=1)
        ax.set_xlabel(r'r ($\AA$)', labelpad=1)
        # plt.title('%s Dots - PDF On vs Off' % dotSize)

        startPoint = valueToIndex(rRange, plotRange[0])
        endPoint = valueToIndex(rRange, plotRange[-1])

        # Get minima for plot x axis limits?
        maxY = int(max(avgOffs[startPoint:endPoint]))
        minY = int(min(avgOffs[startPoint:endPoint]))
        ax.set_ylim((minY - 1.5, maxY + 1.5))

        # Draw arrows in the plot.
        # ax1.annotate(s='', xy=(2.7953, minY - 1), xytext=(2.7953, minY - 3), arrowprops=dict(arrowstyle='->'))
        # ax1.arrow(2.7953, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax1.arrow(2.9697, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        ax.arrow(2.9697, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        ax.text(3.0697, minY - 1, 'Pb-S', rotation=90, size=10)
        # ax1.annotate('Pb-S', xy=(2.9697, minY - 1))
        ax.arrow(4.1999, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        ax.arrow(5.1438, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        ax.arrow(5.9395, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        ax.text(6.0395, minY - 1, 'a-axis', rotation=90, size=10)
        ax.arrow(6.6406, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        ax.arrow(7.2744, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        ax.arrow(8.3997, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        ax.arrow(8.9093, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        ax.arrow(9.3912, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        ax.arrow(10.2880, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        ax.arrow(11.1056, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)

    # Do the plot stuff.
    plotPDF('Small', ax1)

    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    # plt.tight_layout()
    plt.show()

def figure5extrav2():

    # Set up figure stuff.
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['figure.figsize'] = (3.33, 3)

    # Set up the plot.
    fig, ax1 = plt.subplots(1, 1)

    def plotPDF(dotSize, ax):
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        import glob

        hfont = {'fontname': 'Helvetica'}

        def valueToIndex(list, value):
            index = np.abs(np.array(list) - value).argmin()
            return index

        # Define global variables here.
        plotRange = [2.4, 12]
        # dotSize = 'Shelled'
        if dotSize == 'Small':
            dataDirecs = ['E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan6_\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan7\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan9\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan9_\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan10\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan10_\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan11_\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan12\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan13\\PDFs',
                          ]
        if dotSize == 'Medium':
            dataDirecs = ['E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan5\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan6\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan8\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan9\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
                          ]
        if dotSize == 'Large':
            dataDirecs = ['E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\PDF\\PDF Patterns Large\\scan1315\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\PDF\\PDF Patterns Large\\scan14\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\PDF\\PDF Patterns Large\\scan16\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\PDF\\PDF Patterns Large\\scan17\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\PDF\\PDF Patterns Large\\scan18\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\PDF\\PDF Patterns Large\\scan19\\PDFs',
                          ]
        if dotSize == 'Shelled':
            dataDirecs = dataDirecs = ['E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\PDF\\scan6\\PDFs',
                                       'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\PDF\\scan8\\PDFs',
                                       'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\PDF\\scan9\\PDFs',
                                       'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\PDF\\scan10\\PDFs',
                                       'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\PDF\\scan11\\PDFs',
                                       'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\PDF\\scan12\\PDFs',
                                       ]

        # Data structures for initial figure (comparison of on/off G(r)).
        pdfOff = np.zeros(1200)
        pdfOn = np.zeros(1200)
        pdfOnRange = np.zeros(1200)
        pdfOffRange = np.zeros(1200)
        rRange = np.arange(0, 12, 0.01)
        onCounter = 0
        offCounter = 0
        onRangeCounter = 0
        offRangeCounter = 0

        # Additional data structures for figures.
        offPreT0 = np.zeros(1200)
        onPreT0 = np.zeros(1200)
        # zeroTo5On = np.zeros(1200)
        # zeroTo5Off = np.zeros(1200)
        # fiveTo15On = np.zeros(1200)
        # fiveTo15Off = np.zeros(1200)
        # fifteenTo50On = np.zeros(1200)
        # fifteenTo50Off = np.zeros(1200)
        # fiftyTo100On = np.zeros(1200)
        # fiftyTo100Off = np.zeros(1200)
        # hundredTo200On = np.zeros(1200)
        # hundredTo200Off = np.zeros(1200)
        offPreT0Counter = 0
        onPreT0Counter = 0
        # zeroFiveOffCounter = 0
        # zeroFiveOnCounter = 0

        # Begin by loading in and storing data.
        for pdfDirec in dataDirecs:
            for pdf in glob.glob(os.path.join(pdfDirec, '*.npy')):

                # Extract data config from filename.
                tp = int(os.path.basename(pdf).split('_')[1].split('.')[0])
                state = os.path.basename(pdf).split('_')[0]

                # Split process for ons and offs, check if ons are post-t0 to average.
                if state == 'Goff':

                    # Load in data, append to pdfOffs and adjust counter accordingly.
                    data = np.load(pdf)
                    pdfOff += data
                    offCounter += 1

                    # If it is OFF and before t0:
                    if tp < 0:
                        # Add to off pre-t0 data.
                        offPreT0 += data
                        offPreT0Counter += 1

                    if (tp <= 25000) and (tp >= 5000):
                        pdfOffRange += data
                        offRangeCounter += 1

                if state == 'Gon':

                    # Load in data.
                    data = np.load(pdf)

                    if tp > 0:
                        # Load in data after t0, append to pdfOns and just counter accordingly.
                        pdfOn += data
                        onCounter += 1

                    if (tp <= 25000) and (tp >= 5000):
                        pdfOnRange += data
                        onRangeCounter += 1

                    if tp < 0:
                        # Add to ON pre-t0 data.
                        onPreT0 += data
                        onPreT0Counter += 1

        # Average everything, and begin plotting.
        avgOffs = pdfOff / offCounter
        avgOns = pdfOn / onCounter
        avgOnsRange = pdfOnRange / onRangeCounter
        avgOffsRange = pdfOffRange / offRangeCounter
        onPreT0 = onPreT0 / onPreT0Counter
        offPreT0 = offPreT0 / offPreT0Counter
        # zeroTo5On = zeroTo5On / zeroFiveOnCounter
        # zeroTo5Off = zeroTo5Off / zeroFiveOffCounter

        # Setup the results and shit.
        # diff = avgOns - avgOffs
        diff = avgOnsRange - avgOffsRange

        # Plot the results.
        ax.plot(rRange, avgOffsRange, color='black', label='g(r) Off', linewidth=2)
        ax.plot(rRange, avgOnsRange, color='C3', label='g(r) On', linewidth=2)
        ax.plot(rRange, diff*3.5, color='grey', linestyle='dotted', label='Difference', linewidth=2, zorder=2)
        ax.set_xlim(plotRange)
        ax.legend(loc='upper right', prop={'size': 7}, frameon=False)
        ax.set_ylabel(r'g (r)', labelpad=1)
        ax.set_xlabel(r'r ($\AA$)', labelpad=1)
        # plt.title('%s Dots - PDF On vs Off' % dotSize)

        startPoint = valueToIndex(rRange, plotRange[0])
        endPoint = valueToIndex(rRange, plotRange[-1])

        # Get minima for plot x axis limits?
        maxY = int(max(avgOffs[startPoint:endPoint]))
        minY = int(min(avgOffs[startPoint:endPoint]))
        ax.set_ylim((minY - 1.5, maxY + 1.5))

        # Draw arrows in the plot.
        # ax1.annotate(s='', xy=(2.7953, minY - 1), xytext=(2.7953, minY - 3), arrowprops=dict(arrowstyle='->'))
        # ax1.arrow(2.7953, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax1.arrow(2.9697, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        ax.arrow(2.9697, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        ax.text(3.0697, minY - 1, 'Pb-S', rotation=90, size=10)
        # ax1.annotate('Pb-S', xy=(2.9697, minY - 1))
        ax.arrow(4.1999, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        ax.arrow(5.1438, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        ax.arrow(5.9395, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        ax.text(6.0395, minY - 1, 'a-axis', rotation=90, size=10)
        ax.arrow(6.6406, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        ax.arrow(7.2744, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        ax.arrow(8.3997, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        ax.arrow(8.9093, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        ax.arrow(9.3912, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        ax.arrow(10.2880, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        ax.arrow(11.1056, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)

    # Do the plot stuff.
    plotPDF('Small', ax1)

    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    # plt.tight_layout()
    plt.show()

def figure5v2():

    # Set up figure stuff.
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['figure.figsize'] = (6.4, 4.8)

    # Set up the plot.
    fig, (ax1, ax2) = plt.subplots(1,2)

    # Alt plotting function for the PDFs.
    def plotPDFdiffs(dotSize, ax, label):
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        import glob

        hfont = {'fontname': 'Helvetica'}

        def valueToIndex(list, value):
            index = np.abs(np.array(list) - value).argmin()
            return index

        # Define global variables here.
        plotRange = [2.4, 12]
        # dotSize = 'Shelled'
        if dotSize == 'Small':
            dataDirecs = ['E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan6_\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan7\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan9\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan9_\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan10\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan10_\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan11_\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan12\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan13\\PDFs',
                          ]
        if dotSize == 'Medium':
            dataDirecs = ['E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan5\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan6\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan8\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan9\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
                          ]
        if dotSize == 'Large':
            dataDirecs = ['E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\PDF\\PDF Patterns Large\\scan1315\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\PDF\\PDF Patterns Large\\scan14\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\PDF\\PDF Patterns Large\\scan16\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\PDF\\PDF Patterns Large\\scan17\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\PDF\\PDF Patterns Large\\scan18\\PDFs',
                          'E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\PDF\\PDF Patterns Large\\scan19\\PDFs',
                          ]
        if dotSize == 'Shelled':
            dataDirecs = dataDirecs = ['E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\PDF\\scan6\\PDFs',
                                       'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\PDF\\scan8\\PDFs',
                                       'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\PDF\\scan9\\PDFs',
                                       'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\PDF\\scan10\\PDFs',
                                       'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\PDF\\scan11\\PDFs',
                                       'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\PDF\\scan12\\PDFs',
                                       ]

        # Data structures for initial figure (comparison of on/off G(r)).
        pdfOff = np.zeros(1200)
        pdfOn = np.zeros(1200)
        rRange = np.arange(0, 12, 0.01)
        onCounter = 0
        offCounter = 0

        # Additional data structures for figures.
        offPreT0 = np.zeros(1200)
        onPreT0 = np.zeros(1200)
        # zeroTo5On = np.zeros(1200)
        # zeroTo5Off = np.zeros(1200)
        # fiveTo15On = np.zeros(1200)
        # fiveTo15Off = np.zeros(1200)
        # fifteenTo50On = np.zeros(1200)
        # fifteenTo50Off = np.zeros(1200)
        # fiftyTo100On = np.zeros(1200)
        # fiftyTo100Off = np.zeros(1200)
        # hundredTo200On = np.zeros(1200)
        # hundredTo200Off = np.zeros(1200)
        offPreT0Counter = 0
        onPreT0Counter = 0
        # zeroFiveOffCounter = 0
        # zeroFiveOnCounter = 0

        # Begin by loading in and storing data.
        for pdfDirec in dataDirecs:
            for pdf in glob.glob(os.path.join(pdfDirec, '*.npy')):

                # Extract data config from filename.
                tp = int(os.path.basename(pdf).split('_')[1].split('.')[0])
                state = os.path.basename(pdf).split('_')[0]

                # Split process for ons and offs, check if ons are post-t0 to average.
                if state == 'Goff':

                    # Load in data, append to pdfOffs and adjust counter accordingly.
                    data = np.load(pdf)
                    pdfOff += data
                    offCounter += 1

                    # If it is OFF and before t0:
                    if tp < 0:
                        # Add to off pre-t0 data.
                        offPreT0 += data
                        offPreT0Counter += 1

                if state == 'Gon':

                    # Load in data.
                    data = np.load(pdf)

                    if tp > 0:
                        # Load in data after t0, append to pdfOns and just counter accordingly.
                        pdfOn += data
                        onCounter += 1

                    if tp < 0:
                        # Add to ON pre-t0 data.
                        onPreT0 += data
                        onPreT0Counter += 1

        # Average everything, and begin plotting.
        avgOffs = pdfOff / offCounter
        avgOns = pdfOn / onCounter
        onPreT0 = onPreT0 / onPreT0Counter
        offPreT0 = offPreT0 / offPreT0Counter
        # zeroTo5On = zeroTo5On / zeroFiveOnCounter
        # zeroTo5Off = zeroTo5Off / zeroFiveOffCounter

        # Setup the results and shit.

        # Extract ranges and store master data.
        ranges = [[0], [0, 5000], [5000, 25000], [25000, 100000], [100000, 1000000]]
        masterChanges = []

        for r in ranges:

            # Global data struct.
            finalDataOff = np.zeros(1200)
            finalDataOn = np.zeros(1200)
            finalCounterOff = 0
            finalCounterOn = 0

            for pdfDirec in dataDirecs:
                for pdf in glob.glob(os.path.join(pdfDirec, '*.npy')):

                    # Extract info.
                    tp = int(os.path.basename(pdf).split('_')[1].split('.')[0])
                    state = os.path.basename(pdf).split('_')[0]

                    data = np.load(pdf)

                    # If pre-t0:
                    if len(r) == 1:
                        if state == 'Goff':
                            if tp < 0:
                                finalDataOff += data
                                finalCounterOff += 1

                        if state == 'Gon':
                            if tp < 0:
                                finalDataOn += data
                                finalCounterOn += 1

                    # Now go over the rest of the ranges.
                    if len(r) > 1:
                        if state == 'Goff':
                            if (tp > r[0]) and (tp <= r[1]):
                                finalDataOff += data
                                finalCounterOff += 1

                        if state == 'Gon':
                            if (tp > r[0]) and (tp <= r[1]):
                                finalDataOn += data
                                finalCounterOn += 1

            # Finish processing data and append to master data array.
            finalDataOn = finalDataOn / finalCounterOn
            finalDataOff = finalDataOff / finalCounterOff
            masterChanges.append(finalDataOn - finalDataOff)

        masterChanges = np.asarray(masterChanges)

        # Plot the results.
        colors = plt.cm.plasma(np.linspace(0, 1, len(ranges) - 1))
        for ind, range in enumerate(ranges):
            if len(range) == 1:
                labelTitle = '< 0 ps'
                ax.plot(rRange, masterChanges[ind], color='grey', label=labelTitle, linewidth=1, alpha=0.8)
            if len(range) > 1:
                labelTitle = '%0.0f - %0.0f ps' % (range[0] / 1000, range[1] / 1000)
                ax.plot(rRange, masterChanges[ind], color=colors[ind - 1], label=labelTitle, linewidth=1, alpha=0.8)
        if label == 'b)':
            ax.legend(loc='upper right', prop={'size': 7}, frameon=False)
        if (label == 'b)') or (label == 'd)'):
            ax.axes.yaxis.set_visible(False)
        # ax.set_ylabel(r'$\Delta$g (r)', labelpad=5)
        # plt.xlabel(r'r ($\AA$)')
        # ax2.xlim(plotRange)

        ax.set_ylabel(r'$\Delta$g (r)', labelpad=5)
        ax.set_xlabel(r'r ($\AA$)', labelpad=5)
        ax.legend(loc='upper right', prop={'size': 7}, frameon=False)

        maxY = max(masterChanges[-1][valueToIndex(rRange, plotRange[0]):valueToIndex(rRange, plotRange[1])])
        minY = min(masterChanges[-1][valueToIndex(rRange, plotRange[0]):valueToIndex(rRange, plotRange[1])])

        ax.text(0.025, 0.925, label, transform=ax.transAxes, size=12, weight='bold', **hfont)

        ax.set_ylim([minY - 0.25, maxY + 0.25])
        ax.set_xlim([plotRange[0], plotRange[1]])

        # # Draw arrows in the plot.
        # # ax1.annotate(s='', xy=(2.7953, minY - 1), xytext=(2.7953, minY - 3), arrowprops=dict(arrowstyle='->'))
        # # ax1.arrow(2.7953, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # # ax1.arrow(2.9697, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.arrow(2.9697, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        # ax.text(3.0697, minY - 1, 'Pb-S', rotation=90, size=10)
        # # ax1.annotate('Pb-S', xy=(2.9697, minY - 1))
        # ax.arrow(4.1999, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.arrow(5.1438, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        # ax.arrow(5.9395, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.text(6.0395, minY - 1, 'a-axis', rotation=90, size=10)
        # ax.arrow(6.6406, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        # ax.arrow(7.2744, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.arrow(8.3997, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.arrow(8.9093, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        # ax.arrow(9.3912, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.arrow(10.2880, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        # ax.arrow(11.1056, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.text(-0.3, 0.95, label, transform=ax.transAxes, size=12, weight='bold', **hfont)

    def plotPDFThermal(dotSize, ax, label):

        if dotSize == 'Medium':
            thermalPDFDirec = 'E:\\Ediff Samples\\PbS Paper - Final Folder\\Thermal PDF Stuff\\Medium Thermal\\PDF'
            tempValues = [155, 191, 225, 256, 286, 313, 339, 363]
            rRange = np.arange(0,12,0.01)
            limits = [2.4, 12]

        colors = plt.cm.plasma(np.linspace(0, 1, len(tempValues)))

        # Load in each data point, plot.
        for ind, temperature in enumerate(tempValues):

            # Load in data.
            filename = str(temperature) + '_avg.npy'
            baseTemp = np.load(os.path.join(thermalPDFDirec, '155_avg.npy'))
            data = np.load(os.path.join(thermalPDFDirec, filename))

            diff = data - baseTemp

            # Plot it.
            if ind > 0:
                ax.plot(rRange, diff[0:1200], label='%s K' % temperature, alpha=0.8, color=colors[ind])
                # ax.legend(loc='upper right', prop={'size': 7}, frameon=False)
                ax.legend(loc='upper right', prop={'size': 7})
                ax.set_ylabel(r'$\Delta$g (r)', labelpad=5)
                ax.set_xlabel(r'r ($\AA$)', labelpad=5)

    # Do the plot stuff.
    plotPDFdiffs('Medium', ax1, 'a)')
    plotPDFThermal('Medium', ax2, 'b)')

    # plt.tight_layout(pad=0.1, w_pad=0.05, h_pad=0.05)
    plt.tight_layout()
    plt.show()

def figure5v3(size):

    # Scaling stuff for figure b.
    p0 = (4.2345, 4.1923)
    p1 = (5.0539, 5.0605)
    p2 = (5.8951, 5.8708)
    p3 = (7.2552, 7.2636)
    p4 = (8.1148, 8.2500)
    p5 = (9.2866, 9.3218)
    p6 = (10.2095, 10.2806)
    p7 = (10.9391, 11.0610)
    p8 = (12.3565, 12.1730)
    p9 = (13.5166, 13.4950)

    photoPeakOff0 = []
    photoPeakOff1 = []
    photoPeakOff2 = []
    photoPeakOff3 = []
    photoPeakOff4 = []
    photoPeakOff5 = []
    photoPeakOff6 = []
    photoPeakOff7 = []
    photoPeakOff8 = []
    photoPeakOff9 = []

    photoPeakOn0 = []
    photoPeakOn1 = []
    photoPeakOn2 = []
    photoPeakOn3 = []
    photoPeakOn4 = []
    photoPeakOn5 = []
    photoPeakOn6 = []
    photoPeakOn7 = []
    photoPeakOn8 = []
    photoPeakOn9 = []

    def valueToIndex(list, value):
        index = np.abs(np.array(list) - value).argmin()
        return index

    rScaling = 0.99
    dGScaling = 4
    baseT = '155'
    endT = '363'

    rRangePhoto = np.arange(0, 20, 0.01) * rScaling
    rRangeThermal = np.arange(0, 20, 0.01)
    photoChanges = []
    thermalChanges = []

    enableSave = True

    # Set up figure stuff.
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['figure.figsize'] = (7, 6)

    # Set up the axes and figures and stuff.
    fig = plt.figure()
    gs = fig.add_gridspec(3, 1, hspace=0, wspace=0)
    (ax1, ax2, ax3) = gs.subplots(sharex='col', sharey='row')
    # (ax1, ax2) = gs.subplots(sharex='col')

    # Alt plotting function for the PDFs.
    def plotPDFdiffs(dotSize, ax, label):
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        import glob

        hfont = {'fontname': 'Helvetica'}

        def valueToIndex(list, value):
            index = np.abs(np.array(list) - value).argmin()
            return index

        # Define global variables here.
        plotRange = [2.4, 15]
        # dotSize = 'Shelled'
        if dotSize == 'Small':
            dataDirecs = ['D:\\PDF Patterns Small\\scan6_\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan7\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan9\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan9_\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan10\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan10_\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan11_\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan12\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan13\\PDFs Long Q and R',
                          ]
        # if dotSize == 'Medium':
        #     dataDirecs = ['E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan5\\PDFs',
        #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan6\\PDFs',
        #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan8\\PDFs',
        #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan9\\PDFs',
        #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
        #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
        #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
        #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
        #                   ]
        if dotSize == 'Medium':
            dataDirecs = ['E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan5\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan6\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan7\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan9\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan10\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan11\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan12\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan17\\PDFs Long Q and R',
                          ]
        if dotSize == 'Large':
            dataDirecs = ['D:\\PDF Patterns Small\\scan13\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan16\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan17\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan18\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan19\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan1315\\PDFs Long Q and R',
                          ]
        if dotSize == 'Shelled':
            dataDirecs = ['D:\\PDF Patterns Shelled\\scan6\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan8\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan9\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan10\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan11\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan12\\PDFs Long Q and R',
                          ]

        # Data structures for initial figure (comparison of on/off G(r)).
        pdfOff = np.zeros(2000)
        pdfOn = np.zeros(2000)
        rRange = np.arange(0, 20, 0.01)
        # pdfOff = np.zeros(2000)
        # pdfOn = np.zeros(2000)
        # rRange = np.arange(0, 20, 0.01)
        onCounter = 0
        offCounter = 0

        # Additional data structures for figures.
        offPreT0 = np.zeros(2000)
        onPreT0 = np.zeros(2000)
        # zeroTo5On = np.zeros(1200)
        # zeroTo5Off = np.zeros(1200)
        # fiveTo15On = np.zeros(1200)
        # fiveTo15Off = np.zeros(1200)
        # fifteenTo50On = np.zeros(1200)
        # fifteenTo50Off = np.zeros(1200)
        # fiftyTo100On = np.zeros(1200)
        # fiftyTo100Off = np.zeros(1200)
        # hundredTo200On = np.zeros(1200)
        # hundredTo200Off = np.zeros(1200)
        offPreT0Counter = 0
        onPreT0Counter = 0
        # zeroFiveOffCounter = 0
        # zeroFiveOnCounter = 0

        # Begin by loading in and storing data.
        for pdfDirec in dataDirecs:
            for pdf in glob.glob(os.path.join(pdfDirec, '*.npy')):

                # Extract data config from filename.
                tp = int(os.path.basename(pdf).split('_')[1].split('.')[0])
                state = os.path.basename(pdf).split('_')[0]

                # Split process for ons and offs, check if ons are post-t0 to average.
                if state == 'Goff':

                    # Load in data, append to pdfOffs and adjust counter accordingly.
                    data = np.load(pdf)
                    pdfOff += data
                    offCounter += 1

                    # If it is OFF and before t0:
                    if tp < 0:
                        # Add to off pre-t0 data.
                        offPreT0 += data
                        offPreT0Counter += 1

                if state == 'Gon':

                    # Load in data.
                    data = np.load(pdf)

                    if tp > 0:
                        # Load in data after t0, append to pdfOns and just counter accordingly.
                        pdfOn += data
                        onCounter += 1

                    if tp < 0:
                        # Add to ON pre-t0 data.
                        onPreT0 += data
                        onPreT0Counter += 1

        # Average everything, and begin plotting.
        avgOffs = pdfOff / offCounter
        avgOns = pdfOn / onCounter
        onPreT0 = onPreT0 / onPreT0Counter
        offPreT0 = offPreT0 / offPreT0Counter
        # zeroTo5On = zeroTo5On / zeroFiveOnCounter
        # zeroTo5Off = zeroTo5Off / zeroFiveOffCounter

        # Setup the results and shit.

        # Extract ranges and store master data.
        ranges = [[0], [0, 5000], [5000, 25000], [25000, 100000], [100000, 1000000]]
        masterChanges = []

        for r in ranges:

            # Global data struct.
            finalDataOff = np.zeros(2000)
            finalDataOn = np.zeros(2000)
            finalCounterOff = 0
            finalCounterOn = 0

            for pdfDirec in dataDirecs:
                for pdf in glob.glob(os.path.join(pdfDirec, '*.npy')):

                    # Extract info.
                    tp = int(os.path.basename(pdf).split('_')[1].split('.')[0])
                    state = os.path.basename(pdf).split('_')[0]

                    data = np.load(pdf)

                    # If pre-t0:
                    if len(r) == 1:
                        if state == 'Goff':
                            if tp < 0:
                                finalDataOff += data
                                finalCounterOff += 1

                        if state == 'Gon':
                            if tp < 0:
                                finalDataOn += data
                                finalCounterOn += 1

                    # Now go over the rest of the ranges.
                    if len(r) > 1:
                        if state == 'Goff':
                            if (tp > r[0]) and (tp <= r[1]):
                                finalDataOff += data
                                finalCounterOff += 1

                        if state == 'Gon':
                            if (tp > r[0]) and (tp <= r[1]):
                                finalDataOn += data
                                finalCounterOn += 1

            # Finish processing data and append to master data array.
            finalDataOn = finalDataOn / finalCounterOn
            finalDataOff = finalDataOff / finalCounterOff
            masterChanges.append(finalDataOn - finalDataOff)

        masterChanges = np.asarray(masterChanges)

        # Plot the results.
        colors = plt.cm.plasma(np.linspace(0, 1, len(ranges) - 1))
        for ind, range in enumerate(ranges):
            if len(range) == 1:
                labelTitle = '< 0 ps'
                ax.plot(rRange, masterChanges[ind], color='grey', label=labelTitle, linewidth=2, alpha=0.8)
            if len(range) > 1:
                labelTitle = '%0.0f - %0.0f ps' % (range[0] / 1000, range[1] / 1000)
                ax.plot(rRange, masterChanges[ind], color=colors[ind - 1], label=labelTitle, linewidth=2, alpha=0.8)
        if label == 'b)':
            ax.legend(loc='upper right', prop={'size': 7}, frameon=False)
        if (label == 'b)') or (label == 'd)'):
            ax.axes.yaxis.set_visible(False)
        # ax.set_ylabel(r'$\Delta$g (r)', labelpad=5)
        # plt.xlabel(r'r ($\AA$)')
        # ax2.xlim(plotRange)

        ax.set_ylabel(r'$\Delta$g (r)', labelpad=5)
        ax.set_xlabel(r'r ($\AA$)', labelpad=5)
        ax.legend(loc='upper right', prop={'size': 7}, frameon=True)

        maxY = max(masterChanges[-1][valueToIndex(rRange, plotRange[0]):valueToIndex(rRange, plotRange[1])])
        minY = min(masterChanges[-1][valueToIndex(rRange, plotRange[0]):valueToIndex(rRange, plotRange[1])])

        ax.text(0.005, 0.900, label, transform=ax.transAxes, size=12, weight='bold', **hfont)

        #ax.set_ylim([minY - 0.1, maxY + 0.1])
        ax.set_xlim([3,15])

        # xticks = ax.xaxis.get_major_ticks()
        # xticks[-1].label1.set_visible(False)
        # xticks[-1].label1.set_visible(False)

        # # Draw arrows in the plot.
        # # ax1.annotate(s='', xy=(2.7953, minY - 1), xytext=(2.7953, minY - 3), arrowprops=dict(arrowstyle='->'))
        # # ax1.arrow(2.7953, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # # ax1.arrow(2.9697, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.arrow(2.9697, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        # ax.text(3.0697, minY - 1, 'Pb-S', rotation=90, size=10)
        # # ax1.annotate('Pb-S', xy=(2.9697, minY - 1))
        # ax.arrow(4.1999, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.arrow(5.1438, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        # ax.arrow(5.9395, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.text(6.0395, minY - 1, 'a-axis', rotation=90, size=10)
        # ax.arrow(6.6406, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        # ax.arrow(7.2744, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.arrow(8.3997, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.arrow(8.9093, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        # ax.arrow(9.3912, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.arrow(10.2880, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        # ax.arrow(11.1056, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.text(-0.3, 0.95, label, transform=ax.transAxes, size=12, weight='bold', **hfont)

    # def plotPDFThermalComparsion(dotSize, ax, label):
    #
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     import os
    #     import glob
    #
    #     hfont = {'fontname': 'Helvetica'}
    #
    #     def valueToIndex(list, value):
    #         index = np.abs(np.array(list) - value).argmin()
    #         return index
    #
    #     # Define global variables here.
    #     plotRange = [2.4, 15]
    #     # dotSize = 'Shelled'
    #     if dotSize == 'Small':
    #         dataDirecs = ['E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan6_\\PDFs',
    #                       'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan7\\PDFs',
    #                       'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan9\\PDFs',
    #                       'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan9_\\PDFs',
    #                       'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan10\\PDFs',
    #                       'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan10_\\PDFs',
    #                       'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan11_\\PDFs',
    #                       'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan12\\PDFs',
    #                       'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\PDF\\scan13\\PDFs',
    #                       ]
    #     # if dotSize == 'Medium':
    #     #     dataDirecs = ['E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan5\\PDFs',
    #     #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan6\\PDFs',
    #     #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan8\\PDFs',
    #     #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan9\\PDFs',
    #     #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
    #     #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
    #     #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
    #     #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
    #     #                   ]
    #     if dotSize == 'Medium':
    #         dataDirecs = ['E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan5\\PDFs Long Q and R',
    #                       'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan6\\PDFs Long Q and R',
    #                       'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan7\\PDFs Long Q and R',
    #                       'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan9\\PDFs Long Q and R',
    #                       'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan10\\PDFs Long Q and R',
    #                       'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan11\\PDFs Long Q and R',
    #                       'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan12\\PDFs Long Q and R',
    #                       'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan17\\PDFs Long Q and R',
    #                       ]
    #     if dotSize == 'Large':
    #         dataDirecs = ['E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\PDF\\PDF Patterns Large\\scan1315\\PDFs',
    #                       'E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\PDF\\PDF Patterns Large\\scan14\\PDFs',
    #                       'E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\PDF\\PDF Patterns Large\\scan16\\PDFs',
    #                       'E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\PDF\\PDF Patterns Large\\scan17\\PDFs',
    #                       'E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\PDF\\PDF Patterns Large\\scan18\\PDFs',
    #                       'E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\PDF\\PDF Patterns Large\\scan19\\PDFs',
    #                       ]
    #     if dotSize == 'Shelled':
    #         dataDirecs = dataDirecs = ['E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\PDF\\scan6\\PDFs',
    #                                    'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\PDF\\scan8\\PDFs',
    #                                    'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\PDF\\scan9\\PDFs',
    #                                    'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\PDF\\scan10\\PDFs',
    #                                    'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\PDF\\scan11\\PDFs',
    #                                    'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\PDF\\scan12\\PDFs',
    #                                    ]
    #
    #     # Data structures for initial figure (comparison of on/off G(r)).
    #     pdfOff = np.zeros(2000)
    #     pdfOn = np.zeros(2000)
    #     rRange = np.arange(0, 20, 0.01)
    #     onCounter = 0
    #     offCounter = 0
    #
    #     # Additional data structures for figures.
    #     offPreT0 = np.zeros(2000)
    #     onPreT0 = np.zeros(2000)
    #     # zeroTo5On = np.zeros(1200)
    #     # zeroTo5Off = np.zeros(1200)
    #     # fiveTo15On = np.zeros(1200)
    #     # fiveTo15Off = np.zeros(1200)
    #     # fifteenTo50On = np.zeros(1200)
    #     # fifteenTo50Off = np.zeros(1200)
    #     # fiftyTo100On = np.zeros(1200)
    #     # fiftyTo100Off = np.zeros(1200)
    #     # hundredTo200On = np.zeros(1200)
    #     # hundredTo200Off = np.zeros(1200)
    #     offPreT0Counter = 0
    #     onPreT0Counter = 0
    #     # zeroFiveOffCounter = 0
    #     # zeroFiveOnCounter = 0
    #
    #     # Begin by loading in and storing data.
    #     for pdfDirec in dataDirecs:
    #         for pdf in glob.glob(os.path.join(pdfDirec, '*.npy')):
    #
    #             # Extract data config from filename.
    #             tp = int(os.path.basename(pdf).split('_')[1].split('.')[0])
    #             state = os.path.basename(pdf).split('_')[0]
    #
    #             # Split process for ons and offs, check if ons are post-t0 to average.
    #             if state == 'Goff':
    #
    #                 # Load in data, append to pdfOffs and adjust counter accordingly.
    #                 data = np.load(pdf)
    #                 pdfOff += data
    #                 offCounter += 1
    #
    #                 # If it is OFF and before t0:
    #                 if tp < 0:
    #                     # Add to off pre-t0 data.
    #                     offPreT0 += data
    #                     offPreT0Counter += 1
    #
    #             if state == 'Gon':
    #
    #                 # Load in data.
    #                 data = np.load(pdf)
    #
    #                 if tp > 0:
    #                     # Load in data after t0, append to pdfOns and just counter accordingly.
    #                     pdfOn += data
    #                     onCounter += 1
    #
    #                 if tp < 0:
    #                     # Add to ON pre-t0 data.
    #                     onPreT0 += data
    #                     onPreT0Counter += 1
    #
    #     # Average everything, and begin plotting.
    #     avgOffs = pdfOff / offCounter
    #     avgOns = pdfOn / onCounter
    #     onPreT0 = onPreT0 / onPreT0Counter
    #     offPreT0 = offPreT0 / offPreT0Counter
    #     # zeroTo5On = zeroTo5On / zeroFiveOnCounter
    #     # zeroTo5Off = zeroTo5Off / zeroFiveOffCounter
    #
    #     # Setup the results and shit.
    #
    #     # Extract ranges and store master data.
    #     #ranges = [[0], [0, 5000], [5000, 25000], [25000, 100000], [100000, 1000000]]
    #     ranges = [[5000, 25000]]
    #     masterChanges = []
    #
    #     for r in ranges:
    #
    #         # Global data struct.
    #         finalDataOff = np.zeros(2000)
    #         finalDataOn = np.zeros(2000)
    #         finalCounterOff = 0
    #         finalCounterOn = 0
    #
    #         for pdfDirec in dataDirecs:
    #             for pdf in glob.glob(os.path.join(pdfDirec, '*.npy')):
    #
    #                 # Extract info.
    #                 tp = int(os.path.basename(pdf).split('_')[1].split('.')[0])
    #                 state = os.path.basename(pdf).split('_')[0]
    #
    #                 data = np.load(pdf)
    #
    #                 # If pre-t0:
    #                 if len(r) == 1:
    #                     if state == 'Goff':
    #                         if tp < 0:
    #                             finalDataOff += data
    #                             finalCounterOff += 1
    #
    #                     if state == 'Gon':
    #                         if tp < 0:
    #                             finalDataOn += data
    #                             finalCounterOn += 1
    #
    #                 # Now go over the rest of the ranges.
    #                 if len(r) > 1:
    #                     if state == 'Goff':
    #                         if (tp > r[0]) and (tp <= r[1]):
    #                             finalDataOff += data
    #                             finalCounterOff += 1
    #
    #                     if state == 'Gon':
    #                         if (tp > r[0]) and (tp <= r[1]):
    #                             finalDataOn += data
    #                             finalCounterOn += 1
    #
    #         # Finish processing data and append to master data array.
    #         finalDataOn = finalDataOn / finalCounterOn
    #         finalDataOff = finalDataOff / finalCounterOff
    #         masterChanges.append(finalDataOn - finalDataOff)
    #
    #     masterChanges = np.asarray(masterChanges)
    #
    #     # Plot the results.
    #     colors = plt.cm.plasma(np.linspace(0, 1, len(ranges)))
    #     for ind, range in enumerate(ranges):
    #         if len(range) == 1:
    #             labelTitle = '< 0 ps'
    #             ax.plot(rRange, masterChanges[ind], color='grey', label=labelTitle, linewidth=2, alpha=0.8)
    #         if len(range) > 1:
    #             labelTitle = '%0.0f - %0.0f ps' % (range[0] / 1000, range[1] / 1000)
    #             ax.plot(rRange*0.986, masterChanges[ind], color='C0', label=labelTitle, linewidth=2, alpha=0.8, zorder=0)
    #     if label == 'b)':
    #         ax.legend(loc='upper right', prop={'size': 7}, frameon=False)
    #     # if (label == 'b)') or (label == 'd)'):
    #         # ax.axes.yaxis.set_visible(False)
    #     # ax.set_ylabel(r'$\Delta$g (r)', labelpad=5)
    #     # plt.xlabel(r'r ($\AA$)')
    #     # ax2.xlim(plotRange)
    #
    #     ax.set_ylabel(r'$\Delta$g (r)', labelpad=5)
    #     ax.set_xlabel(r'r ($\AA$)', labelpad=5)
    #     ax.legend(loc='upper right', prop={'size': 7}, frameon=True)
    #
    #     # get all the labels of this axis
    #     # xticks = ax.xaxis.get_major_ticks()
    #     # xticks[-1].label1.set_visible(False)
    #     # xticks[0].label1.set_visible(False)
    #
    #     maxY = max(masterChanges[-1][valueToIndex(rRange, plotRange[0]):valueToIndex(rRange, plotRange[1])])
    #     minY = min(masterChanges[-1][valueToIndex(rRange, plotRange[0]):valueToIndex(rRange, plotRange[1])])
    #
    #     ax.text(0.005, 0.900, label, transform=ax.transAxes, size=12, weight='bold', **hfont)
    #
    #     # ax3 = ax.twinx()
    #
    #     if dotSize == 'Medium':
    #         # thermalPDFDirec = 'E:\\Ediff Samples\\PbS Paper - Final Folder\\Thermal PDF Stuff\\Medium Thermal\\PDF'
    #         thermalPDFDirec = 'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\Temperature Dependence\\PDF Long Q and R'
    #         tempValues = [155, 191, 225, 256, 286, 313, 339, 363]
    #         rRange = np.arange(0,20,0.01)
    #         # limits = [2.4, 12]
    #         limits = [2.4, 15]
    #     if dotSize == 'Small':
    #         thermalPDFDirec = 'E:\\Ediff Samples\\PbS Paper - Final Folder\\Thermal PDF Stuff\\Medium Thermal\\PDF'
    #         tempValues = [155, 191, 225, 256, 286, 313, 339, 363]
    #         rRange = np.arange(0,20,0.01)
    #         limits = [2.4, 15]
    #
    #     colors = plt.cm.plasma(np.linspace(0, 1, len(tempValues)))
    #
    #     baseTemp = np.load(os.path.join(thermalPDFDirec, '191_avg.npy'))
    #     data = np.load(os.path.join(thermalPDFDirec, '225_avg.npy'))
    #
    #     diff = data - baseTemp
    #
    #     ax.plot(rRange, diff[0:2000], alpha=0.8, color='black', linewidth=2, linestyle='dotted', label='Thermal PDF', zorder=1)
    #     ax.legend(loc='upper right', prop={'size': 7}, frameon=False)
    #
    #     ax.axvspan(5.3, 6.3, alpha=0.2, color='C2')
    #     ax.axvspan(3.92, 4.48, alpha=0.2)
    #     ax.axvspan(8.97, 9.53, alpha=0.2)
    #     ax.axvspan(11.8, 12.8, alpha=0.2, color='C2')
    #     ax.text(0.25, 0.1, 'a-axis', transform=ax.transAxes, size=12, weight='bold', **hfont)
    #
    #     # ax.text(6.0395, 0, 'a-axis', rotation=90, size=10)
    #
    #     ax.legend(loc='upper right', prop={'size': 7}, frameon=True)
    #     # ax.set_ylabel(r'$\Delta$g (r, T)', labelpad=5)
    #     ax.set_xlim([3, 15])

    def plotPDFThermalComparsion(dotSize, ax, label):

        import numpy as np
        import matplotlib.pyplot as plt
        import os
        import glob

        if dotSize == 'Small':
            # Small scaling factor.
            rScaling = 0.991
            dGScaling = 6.2
            baseT = '155'
            endT = '363'
        if dotSize == 'Medium':
            # Medium scaling factor.
            rScaling = 0.99
            dGScaling = 4
            baseT = '155'
            endT = '363'
        if dotSize == 'Large':
            # Large scaling factor.
            rScaling = 1.003
            dGScaling = 5.1
            baseT = '155'
            endT = '363'
        if dotSize == 'Shelled':
            # Shelled scaling factor.
            rScaling = 0.999
            dGScaling = 2.92
            baseT = '155'
            endT = '363'

        hfont = {'fontname': 'Helvetica'}

        def valueToIndex(list, value):
            index = np.abs(np.array(list) - value).argmin()
            return index

        # Define global variables here.
        plotRange = [2.4, 15]
        # dotSize = 'Shelled'
        if dotSize == 'Small':
            dataDirecs = ['D:\\PDF Patterns Small\\scan6_\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan7\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan9\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan9_\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan10\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan10_\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan11_\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan12\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan13\\PDFs Long Q and R',
                          ]
        # if dotSize == 'Medium':
        #     dataDirecs = ['E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan5\\PDFs',
        #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan6\\PDFs',
        #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan8\\PDFs',
        #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan9\\PDFs',
        #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
        #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
        #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
        #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
        #                   ]
        if dotSize == 'Medium':
            dataDirecs = [
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan5\\PDFs Long Q and R',
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan6\\PDFs Long Q and R',
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan7\\PDFs Long Q and R',
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan9\\PDFs Long Q and R',
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan10\\PDFs Long Q and R',
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan11\\PDFs Long Q and R',
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan12\\PDFs Long Q and R',
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan17\\PDFs Long Q and R',
                ]
        if dotSize == 'Large':
            dataDirecs = ['D:\\PDF Patterns Small\\scan13\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan16\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan17\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan18\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan19\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan1315\\PDFs Long Q and R',
                          ]
        if dotSize == 'Shelled':
            dataDirecs = ['D:\\PDF Patterns Shelled\\scan6\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan8\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan9\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan10\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan11\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan12\\PDFs Long Q and R',
                          ]

        # Data structures for initial figure (comparison of on/off G(r)).
        pdfOff = np.zeros(2000)
        pdfOn = np.zeros(2000)
        rRange = np.arange(0, 20, 0.01)
        onCounter = 0
        offCounter = 0

        # Additional data structures for figures.
        offPreT0 = np.zeros(2000)
        onPreT0 = np.zeros(2000)
        # zeroTo5On = np.zeros(1200)
        # zeroTo5Off = np.zeros(1200)
        # fiveTo15On = np.zeros(1200)
        # fiveTo15Off = np.zeros(1200)
        # fifteenTo50On = np.zeros(1200)
        # fifteenTo50Off = np.zeros(1200)
        # fiftyTo100On = np.zeros(1200)
        # fiftyTo100Off = np.zeros(1200)
        # hundredTo200On = np.zeros(1200)
        # hundredTo200Off = np.zeros(1200)
        offPreT0Counter = 0
        onPreT0Counter = 0
        # zeroFiveOffCounter = 0
        # zeroFiveOnCounter = 0

        # Begin by loading in and storing data.
        for pdfDirec in dataDirecs:
            for pdf in glob.glob(os.path.join(pdfDirec, '*.npy')):

                # Extract data config from filename.
                tp = int(os.path.basename(pdf).split('_')[1].split('.')[0])
                state = os.path.basename(pdf).split('_')[0]

                # Split process for ons and offs, check if ons are post-t0 to average.
                if state == 'Goff':

                    # Load in data, append to pdfOffs and adjust counter accordingly.
                    data = np.load(pdf)
                    pdfOff += data
                    offCounter += 1

                    # If it is OFF and before t0:
                    if tp < 0:
                        # Add to off pre-t0 data.
                        offPreT0 += data
                        offPreT0Counter += 1

                if state == 'Gon':

                    # Load in data.
                    data = np.load(pdf)

                    if tp > 0:
                        # Load in data after t0, append to pdfOns and just counter accordingly.
                        pdfOn += data
                        onCounter += 1

                    if tp < 0:
                        # Add to ON pre-t0 data.
                        onPreT0 += data
                        onPreT0Counter += 1

        # Average everything, and begin plotting.
        avgOffs = pdfOff / offCounter
        avgOns = pdfOn / onCounter
        onPreT0 = onPreT0 / onPreT0Counter
        offPreT0 = offPreT0 / offPreT0Counter
        # zeroTo5On = zeroTo5On / zeroFiveOnCounter
        # zeroTo5Off = zeroTo5Off / zeroFiveOffCounter

        # Setup the results and shit.

        # Extract ranges and store master data.
        #ranges = [[0], [0, 5000], [5000, 25000], [25000, 100000], [100000, 1000000]]
        ranges = [[5000, 25000]]
        masterChanges = []

        for r in ranges:

            # Global data struct.
            finalDataOff = np.zeros(2000)
            finalDataOn = np.zeros(2000)
            finalCounterOff = 0
            finalCounterOn = 0

            for pdfDirec in dataDirecs:
                for pdf in glob.glob(os.path.join(pdfDirec, '*.npy')):

                    # Extract info.
                    tp = int(os.path.basename(pdf).split('_')[1].split('.')[0])
                    state = os.path.basename(pdf).split('_')[0]

                    data = np.load(pdf)

                    # If pre-t0:
                    if len(r) == 1:
                        if state == 'Goff':
                            if tp < 0:
                                finalDataOff += data
                                finalCounterOff += 1

                        if state == 'Gon':
                            if tp < 0:
                                finalDataOn += data
                                finalCounterOn += 1

                    # Now go over the rest of the ranges.
                    if len(r) > 1:
                        if state == 'Goff':
                            if (tp > r[0]) and (tp <= r[1]):
                                finalDataOff += data
                                finalCounterOff += 1
                                photoPeakOff0.append(data[valueToIndex(rRangePhoto, p0[0])])
                                photoPeakOff1.append(data[valueToIndex(rRangePhoto, p1[0])])
                                photoPeakOff2.append(data[valueToIndex(rRangePhoto, p2[0])])
                                photoPeakOff3.append(data[valueToIndex(rRangePhoto, p3[0])])
                                photoPeakOff4.append(data[valueToIndex(rRangePhoto, p4[0])])
                                photoPeakOff5.append(data[valueToIndex(rRangePhoto, p5[0])])
                                photoPeakOff6.append(data[valueToIndex(rRangePhoto, p6[0])])
                                photoPeakOff7.append(data[valueToIndex(rRangePhoto, p7[0])])
                                photoPeakOff8.append(data[valueToIndex(rRangePhoto, p8[0])])
                                photoPeakOff9.append(data[valueToIndex(rRangePhoto, p9[0])])

                        if state == 'Gon':
                            if (tp > r[0]) and (tp <= r[1]):
                                finalDataOn += data
                                finalCounterOn += 1
                                photoPeakOn0.append(data[valueToIndex(rRangePhoto, p0[0])])
                                photoPeakOn1.append(data[valueToIndex(rRangePhoto, p1[0])])
                                photoPeakOn2.append(data[valueToIndex(rRangePhoto, p2[0])])
                                photoPeakOn3.append(data[valueToIndex(rRangePhoto, p3[0])])
                                photoPeakOn4.append(data[valueToIndex(rRangePhoto, p4[0])])
                                photoPeakOn5.append(data[valueToIndex(rRangePhoto, p5[0])])
                                photoPeakOn6.append(data[valueToIndex(rRangePhoto, p6[0])])
                                photoPeakOn7.append(data[valueToIndex(rRangePhoto, p7[0])])
                                photoPeakOn8.append(data[valueToIndex(rRangePhoto, p8[0])])
                                photoPeakOn9.append(data[valueToIndex(rRangePhoto, p9[0])])

            # Finish processing data and append to master data array.
            finalDataOn = finalDataOn / finalCounterOn
            finalDataOff = finalDataOff / finalCounterOff
            masterChanges.append(finalDataOn - finalDataOff)
            photoChanges.append(finalDataOn - finalDataOff)

        masterChanges = np.asarray(masterChanges)

        # Plot the results.
        colors = plt.cm.plasma(np.linspace(0, 1, len(ranges)))
        for ind, range in enumerate(ranges):
            if len(range) == 1:
                labelTitle = '< 0 ps'
                ax.plot(rRange, masterChanges[ind], color='grey', label=labelTitle, linewidth=2, alpha=0.8)
            if len(range) > 1:
                labelTitle = '%0.0f - %0.0f ps' % (range[0] / 1000, range[1] / 1000)
                ax.plot(rRange*rScaling, masterChanges[ind], color='C0', label=labelTitle, linewidth=2, alpha=0.8, zorder=0)
        if label == 'b)':
            ax.legend(loc='upper right', prop={'size': 7}, frameon=False)
        # if (label == 'b)') or (label == 'd)'):
            # ax.axes.yaxis.set_visible(False)
        # ax.set_ylabel(r'$\Delta$g (r)', labelpad=5)
        # plt.xlabel(r'r ($\AA$)')
        # ax2.xlim(plotRange)

        ax.set_ylabel(r'$\Delta$g (r)', labelpad=5)
        ax.set_xlabel(r'r ($\AA$)', labelpad=5)
        ax.legend(loc='upper right', prop={'size': 7}, frameon=True)

        # get all the labels of this axis
        # xticks = ax.xaxis.get_major_ticks()
        # xticks[-1].label1.set_visible(False)
        # xticks[0].label1.set_visible(False)

        maxY = max(masterChanges[-1][valueToIndex(rRange, plotRange[0]):valueToIndex(rRange, plotRange[1])])
        minY = min(masterChanges[-1][valueToIndex(rRange, plotRange[0]):valueToIndex(rRange, plotRange[1])])

        ax.text(0.005, 0.925, label, transform=ax.transAxes, size=12, weight='bold', **hfont)

        # ax3 = ax.twinx()

        if dotSize == 'Medium':
            thermalPDFDirec = 'E:\\Ediff Samples\\PbS Paper - Final Folder\\Thermal PDF Stuff\\Medium Thermal\\PDF'
            #thermalPDFDirec = 'D:\\Documents\\Graduate School\\Research\\PhD Electron Diffraction\\PbS\\Processed Data\\5 nm Dots\\PDF Patterns Medium\\Temperature Dependence\\PDF Long Q and R'
            tempValues = [155, 191, 225, 256, 286, 313, 339, 363]
            rRange = np.arange(0,20,0.01)
            # limits = [2.4, 12]
            limits = [2.4, 15]
        if (dotSize == 'Small') or (dotSize == 'Large') or (dotSize == 'Shelled'):
            thermalPDFDirec = 'E:\\Ediff Samples\\PbS Paper - Final Folder\\Thermal PDF Stuff\\Medium Thermal\\PDF'
            tempValues = [155, 191, 225, 256, 286, 313, 339, 363]
            rRange = np.arange(0,20,0.01)
            limits = [2.4, 15]

        colors = plt.cm.plasma(np.linspace(0, 1, len(tempValues)))

        baseTemp = np.load(os.path.join(thermalPDFDirec, baseT + '_avg.npy'))
        data = np.load(os.path.join(thermalPDFDirec, endT + '_avg.npy'))

        diff = data - baseTemp
        diffNorm = diff/dGScaling
        thermalChanges.append(diffNorm)

        ax.plot(rRange, diffNorm[0:2000], alpha=0.8, color='black', linewidth=2, linestyle='dotted', label='Thermal PDF', zorder=1)
        ax.legend(loc='upper right', prop={'size': 7}, frameon=False)

        ax.axvspan(5.3, 6.3, alpha=0.2, color='C2')
        ax.axvspan(3.92, 4.48, alpha=0.2)
        ax.axvspan(8.97, 9.53, alpha=0.2)
        ax.axvspan(11.8, 12.8, alpha=0.2, color='C2')
        ax.text(0.25, 0.1, 'a-axis', transform=ax.transAxes, size=12, weight='bold', **hfont)

        # ax.text(6.0395, 0, 'a-axis', rotation=90, size=10)

        ax.legend(loc='upper right', prop={'size': 7}, frameon=True)
        # ax.set_ylabel(r'$\Delta$g (r, T)', labelpad=5)
        ax.set_xlim([3.7, 14.1])

    def plotDifferences(dotSize, ax):
        # Peaks to plot.
        #xPlot = [4.2, 6.0, 7.3, 9.3, 10.3, 11.1, 12.5, 15.0, 16.2, 17.0]
        xPlot = [4.2, 5.1, 5.9, 7.3, 8.2, 9.3, 10.25, 11.0, 12.3, 13.5]

        hfont = {'fontname': 'Helvetica'}

        # Load in data.
        masterData = np.load('E:\\Ediff Samples\\PbS Paper - Final Folder\\Thermal PDF Stuff\\PDFpeakRatios_Medium2.npy')
        masterData = masterData.astype(float)

        # Extract data.
        # photoY = []
        # thermY = []
        # errorY = []
        # errorThY = []
        # for x in xPlot:
        #     result = np.where(masterData == x)
        #     photoY.append(float(masterData[1, result[1]]))
        #     thermY.append(float(masterData[2, result[1]]))
        #     errorY.append(float(masterData[3, result[1]]))
        #     errorThY.append(float(masterData[4, result[1]]))
        # photoY = np.asarray(photoY)
        # thermY = np.asarray(thermY)
        # errorY = np.asarray(errorY)
        # errorThY = np.asarray(errorThY)

        # errors = np.sqrt(errorY**2 + errorThY**2)
        # differences = photoY - thermY
        if dotSize == 'Small':
            xPlot = [4.0, 5.0, 5.9, 7.3, 7.9, 9.3, 10.5, 12.609, 13.6, 15.1]
            differences = [0.74762886, 0.00837607, -0.75256994, 0.14875261, -0.00463234, -0.48521589, -0.14322196, -0.23022123, -0.07301846, -0.10155377]
            errors = [0.07785568245393165, 0.0529550848777655, 0.13150789597589993, 0.05895071307620886, 0.1751670498728508, 0.08627888466801287, 0.061122615753901925, 0.07599522625813603, 0.05662285322537715, 0.17786726238452608]
        if dotSize == 'Medium':
            xPlot = [4.2, 5.1, 5.9, 7.3, 8.2, 9.3, 10.25, 11.0, 12.3, 13.5]
            differences = [0.1916145, 0.01655423, -0.20525484, -0.01753846, -0.0176356, -0.11191377, 0.04982928, 0.01248378, -0.14634654, 0.00726162]
            errors = [0.17697527218800005, 0.141845088467062, 0.0818508396062314, 0.08460032792194236, 0.0737535171970173, 0.1295514246571809, 0.10440307138292221, 0.09820517739904627, 0.092092960026445, 0.10401374275019658]
        if dotSize == 'Large':
            xPlot = [4.0, 5.0, 6.1, 7.3, 8.0, 9.4, 10.6, 11.1, 12.7, 13.9]
            differences = [7.13673795e-01, 3.18449835e-04, -7.07038961e-01, 4.57255788e-02, -5.85762095e-02, -5.09957759e-01, -6.54574697e-02, 1.36082871e-01, -1.17588911e-02, 1.97078697e-02]
            errors = [0.08634963873228226, 0.053497122129820704, 0.052089208468485304, 0.07281909410268327, 0.0374256320595185, 0.05174505966502137, 0.13616944958117452, 0.0662576470979757, 0.034044444470130455, 0.09079632846581837]
        if dotSize == 'Shelled':
            xPlot = [4.2, 5.1, 6.0, 7.3, 8.2, 9.3, 10.3, 11.1, 12.4, 13.6]
            differences = [0.17178773, 0.00552623, -0.58078254, -0.04629318, -0.11778402, -0.33773031, 0.05231041, -0.10604742, -0.27132612, -0.00949356]
            errors = [0.11379924105186576, 0.12716355308800725, 0.12544640076512817, 0.11148870541494516, 0.07708065527740685, 0.07986038578810352, 0.11596533305933658, 0.0829311243238656, 0.05793007248959473, 0.059735606761203076]

        # plt.bar(xPlot, differences, yerr=errors, capsize=5, zorder=1, color='C0', alpha=0.8)
        # plt.axhline(y=0, linestyle='--', color='black', alpha=0.5, zorder=0)

        # Crazy idea bits now.
        ax.errorbar(xPlot, differences, yerr=errors, marker='s', capsize=6, linestyle='None', color='C0',
                     markersize=10, alpha=1, zorder=1, ecolor='black')

        for ind, val in enumerate(differences):
            if val > 0:
                ax.errorbar(xPlot[ind], val, yerr=errors[ind], marker='s', capsize=6, linestyle='None', color='C0',
                             markersize=10, alpha=1, zorder=1, ecolor='black')

        for ind, val in enumerate(differences):
            if val > 0:
                colorRange = np.arange(0, val, 0.01)
                colors = plt.cm.OrRd(np.linspace(0, 1, len(colorRange)))
                for i, j in enumerate(colorRange):
                    plt.vlines(xPlot[ind], 0, colorRange[i], color=colors[i], zorder=-1*i, linewidth=7)
            if val < 0:
                colorRange = np.arange(0, -1*val, 0.01)
                colors = plt.cm.OrRd(np.linspace(0,1, len(colorRange)))
                for i, j in enumerate(colorRange):
                    plt.vlines(xPlot[ind], -1*colorRange[i], 0, color=colors[i], zorder=-1*i, linewidth=7)
        ax.axhline(y=0, linestyle='--', color='gray', alpha=0.5, zorder=0)

        ax.set_xlabel(r'r ($\AA$)')
        ax.set_ylabel(r'$\Delta g_{P.E.}(r)$ - $\Delta g_{T}(r)$')
        ax.text(0.005, 0.90, 'c)', transform=ax.transAxes, size=12, weight='bold', **hfont)

    # Do the plot stuff.
    plotPDFdiffs(size, ax1, 'a)')
    plotPDFThermalComparsion(size, ax2, 'b)')
    plotDifferences(size, ax3)

    # plt.tight_layout(pad=0.1, w_pad=0.05, h_pad=0.05)

    plt.tight_layout()
    fig.savefig('E:\\Ediff Samples\\PbS Paper - Final Folder\\Final Push\\Changed Figures\\Figure6_%s_New.pdf' % size)
    plt.show()

def figureNew():

    # Set up figure stuff.
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['figure.figsize'] = (3.33, 3)

    # Load in data.
    masterData = np.load('E:\\Ediff Samples\\PbS Paper - Final Folder\\Thermal PDF Stuff\\PDFpeakRatios_Medium.npy')
    masterData = masterData.astype(float)

    print(masterData)

    xVal = masterData[0]
    xVal = np.delete(xVal, 3)
    print(xVal)

    photoY = masterData[1]
    photoY = np.delete(photoY, 3)
    errorY = masterData[3]
    errorY = np.delete(errorY, 3)

    thermalY = masterData[2]
    thermalY = np.delete(thermalY, 3)
    errorThY = masterData[4]
    errorThY = np.delete(errorThY, 3)

    plt.errorbar(xVal[2:], photoY[2:], yerr=errorY[2:], linestyle="None", color='C0', marker='o', ecolor='C0', label='Photoexcited', capsize=5, alpha=0.8)
    plt.errorbar(xVal[2:], thermalY[2:], yerr=errorThY[2:], linestyle="None", color='C1', marker='^', ecolor='C1', label='Thermal', capsize=5, alpha=0.8)

    plt.axvspan(masterData[0][2] - 0.1, masterData[0][2] + 0.1, alpha=0.2)
    plt.axvspan(masterData[0][4] - 0.1, masterData[0][4] + 0.1, alpha=0.2)
    plt.axvspan(masterData[0][6] - 0.1, masterData[0][6] + 0.1, alpha=0.2)

    plt.annotate(s='', xy=(masterData[0][2], photoY[2]), xytext=(masterData[0][2], thermalY[2]), arrowprops=dict(arrowstyle='<->', color='C3', lw=2))
    plt.annotate(s='', xy=(masterData[0][4], photoY[3]), xytext=(masterData[0][4], thermalY[3]), arrowprops=dict(arrowstyle='<->', color='C3', lw=2))
    # plt.annotate(s='', xy=(masterData[0][5], photoY[4]), xytext=(masterData[0][5], thermalY[4]),
    #              arrowprops=dict(arrowstyle='<->', color='C3', lw=2))
    plt.annotate(s='', xy=(masterData[0][6], photoY[5]), xytext=(masterData[0][6], thermalY[5]), arrowprops=dict(arrowstyle='<->', color='C3', lw=2))
    # plt.annotate(s='', xy=(masterData[0][7], photoY[6]), xytext=(masterData[0][7], thermalY[6]),
    #              arrowprops=dict(arrowstyle='<->', color='C3', lw=2))
    # plt.annotate(s='', xy=(masterData[0][8], photoY[7]), xytext=(masterData[0][8], thermalY[7]),
    #              arrowprops=dict(arrowstyle='<->', color='C3', lw=2))


    plt.legend(loc='upper right', prop={'size': 8})

    plt.xlabel(r'r ($\AA$)', labelpad=1)
    plt.ylabel(r'$\Delta g (r)$ / $\Delta g (5.1 \AA)$', labelpad=1)
    plt.tight_layout()
    plt.show()

# New figures for the combined shelled/size series.
# Figure 4 in the final document.
def FigureAllDotsTogether():

    def plotTimeTraces(dotSize, ax):

        color400 = 'C0'

        import fitTimeTraceModels
        hfont = {'fontname': 'Helvetica'}

        # Select dot size and fit params.
        if dotSize == 'Small':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            peaksToDraw = ['(311) and (222)', '(400)', '(331) and (420)', '(422)']
            letter = 'a)'
            fitResults = [['Tri', 'Tri', 'Bi', 'Tri'],
                          [(3.1, 50, 585, -0.30, 0.17, -0.049), (2.0, 40, 707, -1.09, 0.43, -0.25),
                           (12.6, 47, 0.40, -0.31), (0.42, 12.19, 375.14, -0.083, 0.032, -0.003)]]
            masterData = np.load(
                'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\Master Data\\masterData_Int_Small.npy')
        if dotSize == 'Medium':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            peaksToDraw = ['(220)', '(311) and (222)', '(400)', '(331) and (420)']
            fitResults = [['Bi', 'Bi', 'Bi', 'Mono'],
                          [(0.86, 554, -0.037, -0.076), (3.81, 586, -0.095, -0.066), (5.89, 260, -0.635, 0.119),
                           (2.15, -0.104)]]
            masterData = np.load(
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\Master Data\\masterData_Int_Medium.npy')
        if dotSize == 'Large':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            peaksToDraw = ['(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            fitResults = [['Mono', 'Mono', 'Mono', 'Bi', 'Mono', 'Mono'],
                          [(2.39, -0.024), (2.41, -0.095), (3.51, -0.195), (3.5, 882, -0.512, -0.348), (2.30, -0.164), (1.08, -0.19)]]
            masterData = np.load('E:\\Ediff Samples\PbS Data\\8 nm Dots Pumped\\Master Data\\masterData_Int_Large.npy')
            letter = 'b)'
        if dotSize == 'Shelled':
            peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            peaksToDraw = ['(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
            fitResults = [['Mono', 'Tri', 'Tri', 'Mono', 'Tri', 'Tri'],
                          [(4.83, -0.034), (1.04, 22.62, 359, -0.17, 0.06, -0.07),
                           (2.48, 12, 339, -0.303, 0.0971, -0.047), (3.05, -0.392),
                           (1.32, 13.43, 252.20, -0.44, 0.24, -0.10),
                           (1.12, 10.27, 385.78, -0.74, 0.47, -0.14)]]
            masterData = np.load(
                'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\Master Data\\masterData_Int_Shelled.npy')

        # Plot parameters.
        colors = plt.cm.plasma(np.linspace(0, 1, len(peaksToDraw)))

        if dotSize == 'Large':
            print("Yo go fuck urself")
            print(colors)

        # Set up evaluation.
        tpRange = np.arange(-300, 1000, 1)
        fitEval = []

        tpVals = masterData[0][0]
        xRange = [min(tpVals), max(tpVals)]

        # Go through, plot and evaluate for each peak.
        for ind, peak in enumerate(peaksToDraw):

            # Which peak?
            peakInd = peakLabels.index(peak)

            # Plot the data.
            if peak == '(400)':
                ax.plot(masterData[peakInd][0], masterData[peakInd][1], marker='.', alpha=0.6, label=peak,
                        color=color400)
            else:
                if (dotSize == 'Small') and (peak == '(311) and (222)'):
                    ax.plot(masterData[peakInd][0], masterData[peakInd][1], marker='.', alpha=0.6, label=peak,
                     color=[0.798216, 0.280197, 0.469538, 1.      ])
                else:
                    ax.plot(masterData[peakInd][0], masterData[peakInd][1], marker='.', alpha=0.6, label=peak,
                            color=colors[ind])

            # Evaluate fit.
            fitEval = []
            for x in tpRange:

                if fitResults[0][ind] == 'Mono':
                    fitEval.append(
                        fitTimeTraceModels.ExponentialIntensityDecay(x, fitResults[1][ind][0], fitResults[1][ind][1]))

                if fitResults[0][ind] == 'Bi':
                    fitEval.append(
                        fitTimeTraceModels.BiExponentialIntensityDecay(x, fitResults[1][ind][0], fitResults[1][ind][1],
                                                                       fitResults[1][ind][2], fitResults[1][ind][3]))

                if fitResults[0][ind] == 'Tri':
                    fitEval.append(
                        fitTimeTraceModels.TriExponentialIntensityDecay(x, fitResults[1][ind][0], fitResults[1][ind][1],
                                                                        fitResults[1][ind][2], fitResults[1][ind][3],
                                                                        fitResults[1][ind][4], fitResults[1][ind][5]))

            # Plot the fit.
            if peak == '(400)':
                ax.plot(tpRange, fitEval, linestyle='--', color=color400)
            else:
                if (dotSize == 'Small') and (peak == '(311) and (222)'):
                    ax.plot(tpRange, fitEval, linestyle='--', color=[0.798216, 0.280197, 0.469538, 1.      ])
                else:
                    ax.plot(tpRange, fitEval, linestyle='--', color=colors[ind])

        ax.set_xlim(xRange)
        ax.set_ylabel(r'$\Delta I$ / $I_{o}$', labelpad=5)
        ax.set_xlabel('Timepoint (ps)', labelpad=5)
        # ax.text(-0.2, 0.95, letter, transform=ax.transAxes, size=12, weight='bold', **hfont)
        # ax.legend(loc='lower right', prop={'size': 7})
        # if dotSize == 'Small':
        #     xticks = ax.xaxis.get_major_ticks()
        #     # xticks[0].label1.set_visible(False)
        #     xticks[-1].label1.set_visible(False)

    from matplotlib import gridspec

    # Set up figure stuff.
    plt.rcParams.update({'font.size': 9})
    plt.rcParams['figure.figsize'] = (7, 3)

    # Set up the plot.
    fig, ((ax1, ax2, ax3)) = plt.subplots(1,3, sharey='row', sharex='row')

    plotTimeTraces('Small', ax1)
    plotTimeTraces('Shelled', ax2)
    plotTimeTraces('Large', ax3)
    #ax2.set_yticklabels([])
    ax2.set_ylabel('')
    ax1.set_xlabel('')
    ax3.set_xlabel('')
    ax1.tick_params(direction='in')
    ax2.tick_params(direction='in')
    #ax3.set_yticklabels([])
    ax3.set_ylabel('')
    ax3.tick_params(direction='in')

    ax2.legend(loc='lower center', ncol=1, prop={'size': 6})



    # fig.subplots_adjust(wspace=0)

    plt.tight_layout(w_pad=0)
    fig.savefig('E:\\Ediff Samples\\PbS Paper - Final Folder\\Final Push\\Changed Figures\\FigureX_Part1v2.pdf')
    plt.show()

def FigureAllDotsTogether2ElectricBoogaloo():

    # Set up figure stuff.
    plt.rcParams.update({'font.size': 9})
    plt.rcParams['figure.figsize'] = (3.5, 3.5)

    def plotPEDWTwice(ax):
        import numpy as np
        import math
        import os
        import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit

        hfont = {'fontname': 'Helvetica'}

        # Global constants.
        # TODO: Populate these suckers.
        verbose = True
        selectAverage = False
        selectWindow = True
        selectSingle = False
        showTitle = False
        peaksToAvoid = ['(111)']
        dotsToPlot = ['Small', 'Medium', 'Shelled', 'Large']
        markersPlot = ['o', '^', 's', 'X']

        # Global functions and tools.
        def convertTwoTheta(twoTheta):

            # This function converts an x-axis of two-theta in to Q.
            lambdaElec = 0.038061
            qArray = []
            for i in twoTheta:
                radVal = (i) * (np.pi / 180)
                result = ((4 * np.pi) / lambdaElec) * math.sin((radVal) / 2)
                qArray.append(result)

            return np.asarray(qArray)

        def findNearest(array, value):

            # Returns nearest index value to specified value in array.
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx

        for pInd, dot in enumerate(dotsToPlot):

            cmap = plt.get_cmap("tab10")

            # Dot parameters defined.
            dotSize = dot
            if dotSize == 'Large':
                peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
                dataDirec = 'E:\\Ediff Samples\\PbS Data\\8 nm Dots Pumped\\Master Data'
                convTT = 0.0046926
            elif dotSize == 'Medium':
                peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
                dataDirec = 'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\Master Data'
                convTT = 0.0046926
            elif dotSize == 'Small':
                peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
                dataDirec = 'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\Master Data'
                convTT = 0.0046926
            elif dotSize == 'Shelled':
                peakLabels = ['(111)', '(200)', '(220)', '(311) and (222)', '(400)', '(331) and (420)', '(422)']
                dataDirec = 'E:\\Ediff Samples\\PbS Data\\5 nm Shelled Dots\\Master Data'
                convTT = 0.0046926

            # Begin by loading in data.
            fileName = 'masterData_Int2_%s.npy' % dotSize
            masterData = np.load(os.path.join(dataDirec, fileName))

            # Step 1: determine peak positions by loading data for selected folder.
            # It is best to use a directory with lots of data.

            # peakDataLoc = 'E:\\Ediff Samples\\PbS Data\\2.5 nm Dots Pumped\\2020-08-06\\scans\\scan14\\bgRem\\averagedImages\\fitData' # Smalls
            peakDataLoc = 'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\2020-02-04\\scans\\scan12\\bgRem\\averagedImages\\fitData'

            # Iterate through and average all peak positions for each peak, convert to 1/d.
            peakPos = []
            peakPosError = []
            for ind, peak in enumerate(peakLabels):

                if peak not in peaksToAvoid:
                    # Load in peak data for specified peak.
                    peakData = np.load(os.path.join(peakDataLoc, '%s_peakData.npy' % peak))

                    # Average all values in the 2nd row (peak off positions).
                    peakPositions = peakData[1, :]
                    peakPos.append(np.average(peakPositions))
                    peakPosError.append(np.std(peakPositions))

            # Convert to s, then convert to Q. Peak data is now available.
            peakPos = np.asarray(peakPos)
            peakPosS = peakPos * convTT
            peakPosQ = convertTwoTheta(peakPosS)
            peakPosError = np.asarray(peakPosError)
            peakPosSError = peakPosError * convTT
            peakPosQError = convertTwoTheta(peakPosSError)

            # Start D-W stuff. All master data has first row as timepoints, second row as -ln(I/I_o).
            if selectAverage:
                dwChanges = []
                dwErrors = []
                # for ind, peak in enumerate(peakLabels):

                # Load in integrated data.
                #     plt.plot(masterData[ind][0], masterData[ind][1], label=peak)
                # plt.legend(loc='upper right')
                # plt.show()

                print("Available timepoints:\n")
                print(masterData[0][0])
                tpSel = int(input("Please select at which timepoint to start integrating:\n"))

                # Find matching index, average everything after that, and append to master list.
                tpInd = findNearest(masterData[0][0], tpSel)

                # Start the averaging.
                for ind, peak in enumerate(peakLabels):

                    if peak not in peaksToAvoid:
                        if verbose:
                            print("Now averaging data past %s ps for peak %s." % (masterData[0][0][tpInd], peak))

                        # Get all data after selected timepoint.
                        dwData = masterData[ind][1][tpInd:]

                        # Average the data for that peak and append to array.
                        dwChanges.append(np.average(dwData))
                        dwErrors.append(np.std(dwData))

            if selectWindow:
                dwChanges = []
                dwErrors = []
                # for ind, peak in enumerate(peakLabels):

                # Load in integrated data.
                #     plt.plot(masterData[ind][0], masterData[ind][1], label=peak)
                # plt.legend(loc='upper right')
                # plt.show()

                print("Available timepoints:\n")
                print(masterData[0][0])
                tpSel = int(input("Please select at which timepoint to start integrating:\n"))
                tpSelEnd = int(input("Please select at which timepoint to stop integrating:\n"))

                # Find matching index, average everything after that, and append to master list.
                tpInd = findNearest(masterData[0][0], tpSel)
                tpIndEnd = findNearest(masterData[0][0], tpSelEnd)

                # Start the averaging.
                for ind, peak in enumerate(peakLabels):

                    if peak not in peaksToAvoid:
                        if verbose:
                            print("Now averaging data past %s ps for peak %s." % (masterData[0][0][tpInd], peak))

                        # Get all data after selected timepoint.
                        dwData = masterData[ind][1][tpInd:tpIndEnd]

                        # Average the data for that peak and append to array.
                        dwChanges.append(np.average(dwData))
                        dwErrors.append(np.std(dwData))

            if selectSingle:
                dwChanges = []
                dwErrors = []
                # for ind, peak in enumerate(peakLabels):

                # Load in integrated data.
                #     plt.plot(masterData[ind][0], masterData[ind][1], label=peak)
                # plt.legend(loc='upper right')
                # plt.show()

                print("Available timepoints:\n")
                print(masterData[0][0])
                tpSel = int(input("Please select a timepoint to view:\n"))

                # Find matching index, average everything after that, and append to master list.
                tpInd = findNearest(masterData[0][0], tpSel)

                # Start the averaging.
                for ind, peak in enumerate(peakLabels):

                    if peak not in peaksToAvoid:
                        if verbose:
                            print("Now averaging data past %s ps for peak %s." % (masterData[0][0][tpInd], peak))

                        # Get all data after selected timepoint.
                        dwData = masterData[ind][1][tpInd]

                        # Average the data for that peak and append to array.
                        dwChanges.append(dwData)
                        dwErrors.append(np.std(dwData))

            # # Fit data to a line.
            # def linearFit(m, x):
            #     return m * x
            #
            # popt, _ = curve_fit(linearFit, peakPosQ**2, dwChanges)
            # slope= popt
            # xRange = np.arange(peakPosQ[0]**2, peakPosQ[-1]**2, 0.1)
            # yEval = linearFit(slope, xRange)


            # Plot the results.
            ax.errorbar(peakPosQ ** 2, dwChanges, linestyle='None', yerr=dwErrors, capsize=5, marker=markersPlot[pInd],
                         color=cmap(pInd), ecolor='black', zorder=0, label='%s QD' % dot)

            # Draw square to end highlighting region of localized disorder exceeding the D-W behaviour.

            # Begin determining fitting for linear D-W portion of response.
            peaksToFit = ['(200)', '(220)', '(311) and (222)']
            yFit = []
            xFit = []
            for peak in peaksToFit:
                # Extract the relevant peaks.
                ind = peakLabels.index(peak) - 1
                xFit.append(peakPosQ[ind])
                yFit.append(dwChanges[ind])

            # Fit data to a line.
            def linearFit(x, m, b):
                return m * x + b

            yFit = np.asarray(yFit)
            xFit = np.asarray(xFit)
            popt, pcov = curve_fit(linearFit, xFit ** 2, yFit)
            slope = popt[0]
            intercept = popt[1]
            xRange = np.arange(peakPosQ[0] ** 2, peakPosQ[-1] ** 2, 0.1)
            yEval = linearFit(slope, xRange, intercept)

            ax.plot(xRange, yEval, alpha=0.6, linewidth=3, zorder=-1, color=cmap(pInd),
                     label='%s (Thermal)' % dot)
            if showTitle:
                if selectAverage:
                    plt.title('Photoexcited DW - Averaged > %.0f ps (%s)' % (tpSel, dotSize))
                if selectSingle:
                    plt.title('Photoexcited DW - At %.0f  ps (%s)' % (tpSel, dotSize))
                if selectWindow:
                    plt.title('Photoexcited DW - %.0f to %.0f ps (%s)' % (tpSel, tpSelEnd, dotSize))
            ax.set_ylabel(r'$-ln(I/I_{o})$')
            ax.set_xlabel(r'$q^{2}$ ($\AA^{-2}$)')
            if pInd == 0:
                ax.axvspan(peakPosQ[3] ** 2, (peakPosQ[-1] + 2) ** 2, alpha=0.4, zorder=-2, facecolor='grey', lw=0)
            ax.set_xlim([peakPosQ[0] ** 2 - 0.5, peakPosQ[-1] ** 2 + 0.5])

        ax.legend(loc='upper left', prop={'size': 7}, frameon=False)

    fig, (ax1) = plt.subplots(1, 1)
    plotPEDWTwice(ax1)
    plt.tight_layout(w_pad=0)
    fig.savefig('E:\\Ediff Samples\\PbS Paper - Final Folder\\Final Push\\Changed Figures\\FigureX_Part2_fuck.pdf')
    # plt.show()

def SizeDepIndependent():

    # Set up figure stuff.
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['figure.figsize'] = (3.5, 3.5)

    def generateSizeDependenceCurve(ax):

        hfont = {'fontname': 'Helvetica'}

        import fitDiffPeaks

        # Include the shelled dot as well.
        includeShelled = True

        # Define data.
        xData = np.asarray([2.41, 5.47, 8.73])
        yData = np.asarray([-1.208906673, -0.524616148, -0.378001409])
        xLims = (2, 9)

        # Apply 1/r fit, plot.
        # xRange = np.arange(xData[0], xData[-1], 0.01)
        xRange = np.arange(2, 9, 0.01)

        # Apply fit.
        fitResult = fitDiffPeaks.performOneOverR(xData, yData)

        # Error in fit.
        Aval = fitResult[0].params['Fit_A'].value
        Aerror = fitResult[0].params['Fit_A'].stderr

        # Evaluate the fit and plot.
        fitEvalLow = []
        fitEvalHigh = []
        fitEval = []
        for x in xRange:
            fitEval.append(fitResult[0].eval(x=x))
            fitEvalLow.append(fitDiffPeaks.OneOverR(x, Aval - Aerror))
            fitEvalHigh.append(fitDiffPeaks.OneOverR(x, Aval + Aerror))

        fitEvalHigh = np.asarray(fitEvalHigh)
        fitEvalLow = np.asarray(fitEvalLow)
        fitEval = np.asarray(fitEval)

        # Plot the result.
        ax.plot(xData, yData, marker='o', linestyle='None', zorder=0, label='Unshelled PbS QD')
        if includeShelled:
            ax.plot(5.3, -0.31, marker='o', linestyle='None', zorder=0, color='orange', label='Shelled PbS QD')
        ax.plot(xRange, fitEval, linestyle='--', color='grey', zorder=-1, label=r'$1/r$ fit')
        ax.fill_between(xRange, fitEval - Aerror, fitEval + Aerror, color='grey', alpha=0.3, linestyle='None',
                         zorder=-2)
        ax.set_xlabel('QD Diameter (mm)', labelpad=5)
        ax.set_ylabel(r'$\Delta$ $-ln(I/I_{o})$', labelpad=5)
        ax.set_xlim(xLims)
        ax.legend(loc='lower right', prop={'size': 8}, frameon=False)
        # ax.text(-0.2, 0.95, 'd)', transform=ax.transAxes, size=12, weight='bold', **hfont)

    # Plot stuff.
    fig, ax1 = plt.subplots(1,1)

    generateSizeDependenceCurve(ax1)
    fig.tight_layout()
    fig.savefig('E:\\Ediff Samples\\PbS Paper - Final Folder\\Final Push\\Changed Figures\\FigureX2.pdf')
    #plt.show()

# End of figure 4 in final document.

#figure5v3()
# figure1()
#FigureAllDotsTogether()
# FigureAllDotsTogether2ElectricBoogaloo()
#SizeDepIndependent()

def testingPDF():

    # Set up figure stuff.
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['figure.figsize'] = (6.4, 4.8)

    fig, ax1 = plt.subplots(1,1)

    def plotPDFThermalComparsion(dotSize, ax, label):

        import numpy as np
        import matplotlib.pyplot as plt
        import os
        import glob

        hfont = {'fontname': 'Helvetica'}

        def valueToIndex(list, value):
            index = np.abs(np.array(list) - value).argmin()
            return index

        # Define global variables here.
        plotRange = [2.4, 20]
        # dotSize = 'Shelled'
        if dotSize == 'Small':
            dataDirecs = ['D:\\PDF Patterns Small\\scan6_\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan7\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan9\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan9_\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan10\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan10_\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan11_\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan12\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan13\\PDFs Long Q and R',
                          ]
        if dotSize == 'Medium':
            dataDirecs = [
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan5\\PDFs Long Q and R',
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan6\\PDFs Long Q and R',
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan7\\PDFs Long Q and R',
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan9\\PDFs Long Q and R',
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan10\\PDFs Long Q and R',
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan11\\PDFs Long Q and R',
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan12\\PDFs Long Q and R',
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan17\\PDFs Long Q and R',
                ]
        if dotSize == 'Large':
            dataDirecs = ['D:\\PDF Patterns Small\\scan13\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan16\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan17\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan18\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan19\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan1315\\PDFs Long Q and R',
                          ]
        if dotSize == 'Shelled':
            dataDirecs = ['D:\\PDF Patterns Shelled\\scan6\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan8\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan9\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan10\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan11\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan12\\PDFs Long Q and R',
                          ]

        # Data structures for initial figure (comparison of on/off G(r)).
        pdfOff = np.zeros(2000)
        pdfOn = np.zeros(2000)
        rRange = np.arange(0, 20, 0.01)
        onCounter = 0
        offCounter = 0

        # Additional data structures for figures.
        offPreT0 = np.zeros(2000)
        onPreT0 = np.zeros(2000)
        # zeroTo5On = np.zeros(1200)
        # zeroTo5Off = np.zeros(1200)
        # fiveTo15On = np.zeros(1200)
        # fiveTo15Off = np.zeros(1200)
        # fifteenTo50On = np.zeros(1200)
        # fifteenTo50Off = np.zeros(1200)
        # fiftyTo100On = np.zeros(1200)
        # fiftyTo100Off = np.zeros(1200)
        # hundredTo200On = np.zeros(1200)
        # hundredTo200Off = np.zeros(1200)
        offPreT0Counter = 0
        onPreT0Counter = 0
        # zeroFiveOffCounter = 0
        # zeroFiveOnCounter = 0

        # Begin by loading in and storing data.
        for pdfDirec in dataDirecs:
            for pdf in glob.glob(os.path.join(pdfDirec, '*.npy')):

                # Extract data config from filename.
                tp = int(os.path.basename(pdf).split('_')[1].split('.')[0])
                state = os.path.basename(pdf).split('_')[0]

                # Split process for ons and offs, check if ons are post-t0 to average.
                if state == 'Goff':

                    # Load in data, append to pdfOffs and adjust counter accordingly.
                    data = np.load(pdf)
                    pdfOff += data
                    offCounter += 1

                    # If it is OFF and before t0:
                    if tp < 0:
                        # Add to off pre-t0 data.
                        offPreT0 += data
                        offPreT0Counter += 1

                if state == 'Gon':

                    # Load in data.
                    data = np.load(pdf)

                    if tp > 0:
                        # Load in data after t0, append to pdfOns and just counter accordingly.
                        pdfOn += data
                        onCounter += 1

                    if tp < 0:
                        # Add to ON pre-t0 data.
                        onPreT0 += data
                        onPreT0Counter += 1

        # Average everything, and begin plotting.
        avgOffs = pdfOff / offCounter
        avgOns = pdfOn / onCounter
        onPreT0 = onPreT0 / onPreT0Counter
        offPreT0 = offPreT0 / offPreT0Counter
        # zeroTo5On = zeroTo5On / zeroFiveOnCounter
        # zeroTo5Off = zeroTo5Off / zeroFiveOffCounter

        # Setup the results and shit.

        # Extract ranges and store master data.
        #ranges = [[0], [0, 5000], [5000, 25000], [25000, 100000], [100000, 1000000]]
        ranges = [[5000, 25000]]
        # ranges = [[200000, 1000000]]
        masterChanges = []

        for r in ranges:

            # Global data struct.
            finalDataOff = np.zeros(2000)
            finalDataOn = np.zeros(2000)
            finalCounterOff = 0
            finalCounterOn = 0

            for pdfDirec in dataDirecs:
                for pdf in glob.glob(os.path.join(pdfDirec, '*.npy')):

                    # Extract info.
                    tp = int(os.path.basename(pdf).split('_')[1].split('.')[0])
                    state = os.path.basename(pdf).split('_')[0]

                    data = np.load(pdf)

                    # If pre-t0:
                    if len(r) == 1:
                        if state == 'Goff':
                            if tp < 0:
                                finalDataOff += data
                                finalCounterOff += 1

                        if state == 'Gon':
                            if tp < 0:
                                finalDataOn += data
                                finalCounterOn += 1

                    # Now go over the rest of the ranges.
                    if len(r) > 1:
                        if state == 'Goff':
                            if (tp > r[0]) and (tp <= r[1]):
                                finalDataOff += data
                                finalCounterOff += 1

                        if state == 'Gon':
                            if (tp > r[0]) and (tp <= r[1]):
                                finalDataOn += data
                                finalCounterOn += 1

            # Finish processing data and append to master data array.
            finalDataOn = finalDataOn / finalCounterOn
            finalDataOff = finalDataOff / finalCounterOff
            masterChanges.append(finalDataOn - finalDataOff)

        masterChanges = np.asarray(masterChanges)

        # Plot the results.
        colors = plt.cm.plasma(np.linspace(0, 1, len(ranges)))
        for ind, range in enumerate(ranges):
            if len(range) == 1:
                labelTitle = '< 0 ps'
                ax.plot(rRange, masterChanges[ind], color='grey', label=labelTitle, linewidth=2, alpha=0.8)
            if len(range) > 1:
                labelTitle = '%0.0f - %0.0f ps' % (range[0] / 1000, range[1] / 1000)
                #ax.plot(rRange*0.986, masterChanges[ind], color='C0', label=labelTitle, linewidth=2, alpha=0.8, zorder=0)
        if label == 'b)':
            ax.legend(loc='upper right', prop={'size': 7}, frameon=False)
        # if (label == 'b)') or (label == 'd)'):
            # ax.axes.yaxis.set_visible(False)
        # ax.set_ylabel(r'$\Delta$g (r)', labelpad=5)
        # plt.xlabel(r'r ($\AA$)')
        # ax2.xlim(plotRange)

        ax.set_ylabel(r'$\Delta$g (r)', labelpad=5)
        ax.set_xlabel(r'r ($\AA$)', labelpad=5)
        ax.legend(loc='upper right', prop={'size': 7}, frameon=True)

        # get all the labels of this axis
        # xticks = ax.xaxis.get_major_ticks()
        # xticks[-1].label1.set_visible(False)
        # xticks[0].label1.set_visible(False)

        maxY = max(masterChanges[-1][valueToIndex(rRange, plotRange[0]):valueToIndex(rRange, plotRange[1])])
        minY = min(masterChanges[-1][valueToIndex(rRange, plotRange[0]):valueToIndex(rRange, plotRange[1])])

        ax.text(0.005, 0.925, label, transform=ax.transAxes, size=12, weight='bold', **hfont)

        # ax3 = ax.twinx()

        if dotSize == 'Medium':
            # thermalPDFDirec = 'E:\\Ediff Samples\\PbS Paper - Final Folder\\Thermal PDF Stuff\\Medium Thermal\\PDF'
            thermalPDFDirec = 'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\Temperature Dependence\\PDF Long Q and R'
            tempValues = [155, 191, 225, 256, 286, 313, 339, 363]
            rRange = np.arange(0,20,0.01)
            # limits = [2.4, 12]
            limits = [2.4, 20]
        if dotSize == 'Small':
            thermalPDFDirec = 'E:\\Ediff Samples\\PbS Paper - Final Folder\\Thermal PDF Stuff\\Medium Thermal\\PDF'
            tempValues = [155, 191, 225, 256, 286, 313, 339, 363]
            rRange = np.arange(0,20,0.01)
            limits = [2.4, 20]

        colors = plt.cm.plasma(np.linspace(0, 1, len(tempValues)))

        baseTemp = np.load(os.path.join(thermalPDFDirec, '191_avg.npy'))
        data = np.load(os.path.join(thermalPDFDirec, '225_avg.npy'))

        diff = data - baseTemp

        #ax.plot(rRange, diff[0:2000], alpha=0.8, color='black', linewidth=2, linestyle='dotted', label='Thermal PDF', zorder=1)
        # ax.legend(loc='upper right', prop={'size': 7}, frameon=False)

        ax.axvspan(5.5, 6.5, alpha=0.2, color='C2')
        ax.axvspan(3.92, 4.48, alpha=0.2)
        ax.axvspan(8.97, 9.53, alpha=0.2)
        ax.axvspan(11.7, 12.7, alpha=0.2, color='C2')
        # ax.text(0.3, 0.1, 'a-axis', transform=ax.transAxes, size=12, weight='bold', **hfont)

        #ax.text(6.0395, 0, 'a-axis', rotation=90, size=10)

        ax.legend(loc='upper right', prop={'size': 7}, frameon=True)
        # ax.set_ylabel(r'$\Delta$g (r, T)', labelpad=5)
        ax.set_xlim([3, 15])

        ax.plot(rRange, masterChanges[0] - diff[0:2000])


    # Do the plot stuff.
    plotPDFThermalComparsion('Medium', ax1, 'a)')

    # plt.tight_layout(pad=0.1, w_pad=0.05, h_pad=0.05)
    plt.tight_layout()
    plt.show()

# testingPDF()

def figureNewTesting():

    import sys
    from matplotlib.collections import LineCollection

    # Set up figure stuff.
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['figure.figsize'] = (7, 3)

    hfont = {'fontname': 'Helvetica'}

    fig, ax1 = plt.subplots(1,1)

    def plotDifferences(ax):
        # Peaks to plot.
        xPlot = [4.2, 6.0, 7.3, 9.3, 10.3, 11.1, 12.5, 15.0, 16.2, 17.0]

        # Load in data.
        masterData = np.load('E:\\Ediff Samples\\PbS Paper - Final Folder\\Thermal PDF Stuff\\PDFpeakRatios_Medium2.npy')
        masterData = masterData.astype(float)

        # Extract data.
        photoY = []
        thermY = []
        errorY = []
        errorThY = []
        for x in xPlot:
            result = np.where(masterData == x)
            photoY.append(float(masterData[1, result[1]]))
            thermY.append(float(masterData[2, result[1]]))
            errorY.append(float(masterData[3, result[1]]))
            errorThY.append(float(masterData[4, result[1]]))
        photoY = np.asarray(photoY)
        thermY = np.asarray(thermY)
        errorY = np.asarray(errorY)
        errorThY = np.asarray(errorThY)

        errors = np.sqrt(errorY**2 + errorThY**2)
        differences = photoY - thermY

        print(errors)
        sys.exit()

        # plt.bar(xPlot, differences, yerr=errors, capsize=5, zorder=1, color='C0', alpha=0.8)
        # plt.axhline(y=0, linestyle='--', color='black', alpha=0.5, zorder=0)

        # Crazy idea bits now.
        ax.errorbar(xPlot, differences, yerr=errors, marker='s', capsize=6, linestyle='None', color='C0',
                     markersize=10, alpha=1, zorder=1, ecolor='black')

        for ind, val in enumerate(differences):
            if val > 0:
                ax.errorbar(xPlot[ind], val, yerr=errors[ind], marker='s', capsize=6, linestyle='None', color='C0',
                             markersize=10, alpha=1, zorder=1, ecolor='black')

        for ind, val in enumerate(differences):
            if val > 0:
                colorRange = np.arange(0, val, 0.01)
                colors = plt.cm.OrRd(np.linspace(0, 1, len(colorRange)))
                for i, j in enumerate(colorRange):
                    plt.vlines(xPlot[ind], 0, colorRange[i], color=colors[i], zorder=-1*i, linewidth=7)
            if val < 0:
                colorRange = np.arange(0, -1*val, 0.01)
                colors = plt.cm.OrRd(np.linspace(0,1, len(colorRange)))
                for i, j in enumerate(colorRange):
                    plt.vlines(xPlot[ind], -1*colorRange[i], 0, color=colors[i], zorder=-1*i, linewidth=7)
        ax.axhline(y=0, linestyle='--', color='gray', alpha=0.5, zorder=0)

        ax.set_xlabel(r'r ($\AA$)')
        ax.set_ylabel(r'$\Delta g_{photoexcited}(r)$ - $\Delta g_{thermal}(r)$')

    plotDifferences(ax1)
    plt.tight_layout()
    fig.savefig('E:\\Ediff Samples\\PbS Paper - Final Folder\\Final Push\\Changed Figures\\Figure6_part2.pdf')
    plt.show()

# figureNewTesting()

# figure5v3('Shelled')

def figure5v4(size):

    # Scaling stuff for figure b.
    p0 = (4.2345, 4.1923)
    p1 = (5.0539, 5.0605)
    p2 = (5.8951, 5.8708)
    p3 = (7.2552, 7.2636)
    p4 = (8.1148, 8.2500)
    p5 = (9.2866, 9.3218)
    p6 = (10.2095, 10.2806)
    p7 = (10.9391, 11.0610)
    p8 = (12.3565, 12.1730)
    p9 = (13.5166, 13.4950)

    photoPeakOff0 = []
    photoPeakOff1 = []
    photoPeakOff2 = []
    photoPeakOff3 = []
    photoPeakOff4 = []
    photoPeakOff5 = []
    photoPeakOff6 = []
    photoPeakOff7 = []
    photoPeakOff8 = []
    photoPeakOff9 = []

    photoPeakOn0 = []
    photoPeakOn1 = []
    photoPeakOn2 = []
    photoPeakOn3 = []
    photoPeakOn4 = []
    photoPeakOn5 = []
    photoPeakOn6 = []
    photoPeakOn7 = []
    photoPeakOn8 = []
    photoPeakOn9 = []

    def valueToIndex(list, value):
        index = np.abs(np.array(list) - value).argmin()
        return index

    rScaling = 0.99
    dGScaling = 4
    baseT = '155'
    endT = '363'

    rRangePhoto = np.arange(0, 20, 0.01) * rScaling
    rRangeThermal = np.arange(0, 20, 0.01)
    photoChanges = []
    thermalChanges = []

    enableSave = True

    # Set up figure stuff.
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['figure.figsize'] = (7, 8)

    # Set up the axes and figures and stuff.
    fig = plt.figure()
    gs = fig.add_gridspec(4, 1, hspace=0, wspace=0)
    (ax1, ax2, ax3, ax4) = gs.subplots(sharex='col', sharey='row')

    # Alt plotting function for the PDFs.
    def plotPDF(dotSize, ax):
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        import glob

        hfont = {'fontname': 'Helvetica'}

        def valueToIndex(list, value):
            index = np.abs(np.array(list) - value).argmin()
            return index

        # Define global variables here.
        plotRange = [2.4, 15]
        # dotSize = 'Shelled'
        if dotSize == 'Small':
            dataDirecs = ['D:\\PDF Patterns Small\\scan6_\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan7\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan9\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan9_\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan10\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan10_\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan11_\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan12\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan13\\PDFs Long Q and R',
                          ]
        if dotSize == 'Medium':
            dataDirecs = ['E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan5\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan6\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan7\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan9\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan10\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan11\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan12\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan17\\PDFs Long Q and R',
                          ]
        if dotSize == 'Large':
            dataDirecs = ['D:\\PDF Patterns Small\\scan13\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan16\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan17\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan18\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan19\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan1315\\PDFs Long Q and R',
                          ]
        if dotSize == 'Shelled':
            dataDirecs = ['D:\\PDF Patterns Shelled\\scan6\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan8\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan9\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan10\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan11\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan12\\PDFs Long Q and R',
                          ]

        # Data structures for initial figure (comparison of on/off G(r)).
        pdfOff = np.zeros(2000)
        pdfOn = np.zeros(2000)
        pdfOnRange = np.zeros(2000)
        pdfOffRange = np.zeros(2000)
        rRange = np.arange(0, 20, 0.01)
        onCounter = 0
        offCounter = 0
        onRangeCounter = 0
        offRangeCounter = 0

        # Additional data structures for figures.
        offPreT0 = np.zeros(2000)
        onPreT0 = np.zeros(2000)
        # zeroTo5On = np.zeros(1200)
        # zeroTo5Off = np.zeros(1200)
        # fiveTo15On = np.zeros(1200)
        # fiveTo15Off = np.zeros(1200)
        # fifteenTo50On = np.zeros(1200)
        # fifteenTo50Off = np.zeros(1200)
        # fiftyTo100On = np.zeros(1200)
        # fiftyTo100Off = np.zeros(1200)
        # hundredTo200On = np.zeros(1200)
        # hundredTo200Off = np.zeros(1200)
        offPreT0Counter = 0
        onPreT0Counter = 0
        # zeroFiveOffCounter = 0
        # zeroFiveOnCounter = 0

        # Begin by loading in and storing data.
        for pdfDirec in dataDirecs:
            for pdf in glob.glob(os.path.join(pdfDirec, '*.npy')):

                # Extract data config from filename.
                tp = int(os.path.basename(pdf).split('_')[1].split('.')[0])
                state = os.path.basename(pdf).split('_')[0]

                # Split process for ons and offs, check if ons are post-t0 to average.
                if state == 'Goff':

                    # Load in data, append to pdfOffs and adjust counter accordingly.
                    data = np.load(pdf)
                    pdfOff += data
                    offCounter += 1

                    # If it is OFF and before t0:
                    if tp < 0:
                        # Add to off pre-t0 data.
                        offPreT0 += data
                        offPreT0Counter += 1

                    if (tp <= 25000) and (tp >= 5000):
                        pdfOffRange += data
                        offRangeCounter += 1

                if state == 'Gon':

                    # Load in data.
                    data = np.load(pdf)

                    if tp > 0:
                        # Load in data after t0, append to pdfOns and just counter accordingly.
                        pdfOn += data
                        onCounter += 1

                    if (tp <= 25000) and (tp >= 5000):
                        pdfOnRange += data
                        onRangeCounter += 1

                    if tp < 0:
                        # Add to ON pre-t0 data.
                        onPreT0 += data
                        onPreT0Counter += 1

        # Average everything, and begin plotting.
        avgOffs = pdfOff / offCounter
        avgOns = pdfOn / onCounter
        avgOnsRange = pdfOnRange / onRangeCounter
        avgOffsRange = pdfOffRange / offRangeCounter
        onPreT0 = onPreT0 / onPreT0Counter
        offPreT0 = offPreT0 / offPreT0Counter
        # zeroTo5On = zeroTo5On / zeroFiveOnCounter
        # zeroTo5Off = zeroTo5Off / zeroFiveOffCounter

        # Setup the results and shit.
        # diff = avgOns - avgOffs
        diff = avgOnsRange - avgOffsRange

        # Plot the results.
        ax.plot(rRange, avgOffsRange, color='black', label='G(r) Off', linewidth=2)
        ax.plot(rRange, avgOnsRange, color='C3', label='G(r) On', linewidth=2)
        # ax.plot(rRange, diff*3.5, color='grey', linestyle='dotted', label='Difference', linewidth=2, zorder=2)
        ax.set_xlim(plotRange)
        ax.legend(loc='upper right', prop={'size': 7}, frameon=False)
        ax.set_ylabel(r'G (r)', labelpad=1)
        #ax.set_xlabel(r'r ($\AA$)', labelpad=1)
        # plt.title('%s Dots - PDF On vs Off' % dotSize)

        ax.text(0.005, 0.90, 'a)', transform=ax.transAxes, size=12, weight='bold', **hfont)

        startPoint = valueToIndex(rRange, plotRange[0])
        endPoint = valueToIndex(rRange, plotRange[-1])

        # Get minima for plot x axis limits?
        maxY = int(max(avgOffs[startPoint:endPoint]))
        minY = int(min(avgOffs[startPoint:endPoint]))
        ax.set_ylim((minY - 1.5, maxY + 1.5))

        # Draw arrows in the plot.
        # ax1.annotate(s='', xy=(2.7953, minY - 1), xytext=(2.7953, minY - 3), arrowprops=dict(arrowstyle='->'))
        # ax1.arrow(2.7953, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax1.arrow(2.9697, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        ax.arrow(2.9697, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        # ax.text(3.0697, minY - 1, 'Pb-S', rotation=90, size=10)
        # ax1.annotate('Pb-S', xy=(2.9697, minY - 1))
        ax.arrow(4.1999, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        ax.arrow(5.1438, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        ax.arrow(5.9395, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        ax.text(6.0395, minY - 1, 'a-axis', rotation=90, size=10)
        ax.arrow(6.6406, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        ax.arrow(7.2744, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        ax.arrow(8.3997, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        ax.arrow(8.9093, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        ax.arrow(9.3912, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        ax.arrow(10.2880, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        ax.arrow(11.1056, minY - 1.5, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)

    def plotPDFdiffs(dotSize, ax, label):
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        import glob

        hfont = {'fontname': 'Helvetica'}

        def valueToIndex(list, value):
            index = np.abs(np.array(list) - value).argmin()
            return index

        # Define global variables here.
        plotRange = [2.4, 15]
        # dotSize = 'Shelled'
        if dotSize == 'Small':
            dataDirecs = ['D:\\PDF Patterns Small\\scan6_\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan7\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan9\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan9_\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan10\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan10_\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan11_\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan12\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan13\\PDFs Long Q and R',
                          ]
        if dotSize == 'Medium':
            dataDirecs = ['E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan5\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan6\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan7\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan9\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan10\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan11\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan12\\PDFs Long Q and R',
                          'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan17\\PDFs Long Q and R',
                          ]
        if dotSize == 'Large':
            dataDirecs = ['D:\\PDF Patterns Small\\scan13\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan16\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan17\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan18\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan19\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan1315\\PDFs Long Q and R',
                          ]
        if dotSize == 'Shelled':
            dataDirecs = ['D:\\PDF Patterns Shelled\\scan6\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan8\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan9\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan10\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan11\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan12\\PDFs Long Q and R',
                          ]

        # Data structures for initial figure (comparison of on/off G(r)).
        pdfOff = np.zeros(2000)
        pdfOn = np.zeros(2000)
        rRange = np.arange(0, 20, 0.01)
        # pdfOff = np.zeros(2000)
        # pdfOn = np.zeros(2000)
        # rRange = np.arange(0, 20, 0.01)
        onCounter = 0
        offCounter = 0

        # Additional data structures for figures.
        offPreT0 = np.zeros(2000)
        onPreT0 = np.zeros(2000)
        # zeroTo5On = np.zeros(1200)
        # zeroTo5Off = np.zeros(1200)
        # fiveTo15On = np.zeros(1200)
        # fiveTo15Off = np.zeros(1200)
        # fifteenTo50On = np.zeros(1200)
        # fifteenTo50Off = np.zeros(1200)
        # fiftyTo100On = np.zeros(1200)
        # fiftyTo100Off = np.zeros(1200)
        # hundredTo200On = np.zeros(1200)
        # hundredTo200Off = np.zeros(1200)
        offPreT0Counter = 0
        onPreT0Counter = 0
        # zeroFiveOffCounter = 0
        # zeroFiveOnCounter = 0

        # Begin by loading in and storing data.
        for pdfDirec in dataDirecs:
            for pdf in glob.glob(os.path.join(pdfDirec, '*.npy')):

                # Extract data config from filename.
                tp = int(os.path.basename(pdf).split('_')[1].split('.')[0])
                state = os.path.basename(pdf).split('_')[0]

                # Split process for ons and offs, check if ons are post-t0 to average.
                if state == 'Goff':

                    # Load in data, append to pdfOffs and adjust counter accordingly.
                    data = np.load(pdf)
                    pdfOff += data
                    offCounter += 1

                    # If it is OFF and before t0:
                    if tp < 0:
                        # Add to off pre-t0 data.
                        offPreT0 += data
                        offPreT0Counter += 1

                if state == 'Gon':

                    # Load in data.
                    data = np.load(pdf)

                    if tp > 0:
                        # Load in data after t0, append to pdfOns and just counter accordingly.
                        pdfOn += data
                        onCounter += 1

                    if tp < 0:
                        # Add to ON pre-t0 data.
                        onPreT0 += data
                        onPreT0Counter += 1

        # Average everything, and begin plotting.
        avgOffs = pdfOff / offCounter
        avgOns = pdfOn / onCounter
        onPreT0 = onPreT0 / onPreT0Counter
        offPreT0 = offPreT0 / offPreT0Counter
        # zeroTo5On = zeroTo5On / zeroFiveOnCounter
        # zeroTo5Off = zeroTo5Off / zeroFiveOffCounter

        # Setup the results and shit.

        # Extract ranges and store master data.
        ranges = [[0], [0, 5000], [5000, 25000], [25000, 100000], [100000, 1000000]]
        masterChanges = []

        for r in ranges:

            # Global data struct.
            finalDataOff = np.zeros(2000)
            finalDataOn = np.zeros(2000)
            finalCounterOff = 0
            finalCounterOn = 0

            for pdfDirec in dataDirecs:
                for pdf in glob.glob(os.path.join(pdfDirec, '*.npy')):

                    # Extract info.
                    tp = int(os.path.basename(pdf).split('_')[1].split('.')[0])
                    state = os.path.basename(pdf).split('_')[0]

                    data = np.load(pdf)

                    # If pre-t0:
                    if len(r) == 1:
                        if state == 'Goff':
                            if tp < 0:
                                finalDataOff += data
                                finalCounterOff += 1

                        if state == 'Gon':
                            if tp < 0:
                                finalDataOn += data
                                finalCounterOn += 1

                    # Now go over the rest of the ranges.
                    if len(r) > 1:
                        if state == 'Goff':
                            if (tp > r[0]) and (tp <= r[1]):
                                finalDataOff += data
                                finalCounterOff += 1

                        if state == 'Gon':
                            if (tp > r[0]) and (tp <= r[1]):
                                finalDataOn += data
                                finalCounterOn += 1

            # Finish processing data and append to master data array.
            finalDataOn = finalDataOn / finalCounterOn
            finalDataOff = finalDataOff / finalCounterOff
            masterChanges.append(finalDataOn - finalDataOff)

        masterChanges = np.asarray(masterChanges)

        # Plot the results.
        colors = plt.cm.plasma(np.linspace(0, 1, len(ranges) - 1))
        for ind, range in enumerate(ranges):
            if len(range) == 1:
                labelTitle = '< 0 ps'
                ax.plot(rRange, masterChanges[ind], color='grey', label=labelTitle, linewidth=2, alpha=0.8)
            if len(range) > 1:
                labelTitle = '%0.0f - %0.0f ps' % (range[0] / 1000, range[1] / 1000)
                ax.plot(rRange, masterChanges[ind], color=colors[ind - 1], label=labelTitle, linewidth=2, alpha=0.8)
        if label == 'b)':
            ax.legend(loc='upper right', prop={'size': 7}, frameon=False)
        # if (label == 'b)') or (label == 'd)'):
        #     ax.axes.yaxis.set_visible(False)
        # ax.set_ylabel(r'$\Delta$g (r)', labelpad=5)
        # plt.xlabel(r'r ($\AA$)')
        # ax2.xlim(plotRange)

        ax.set_ylabel(r'$\Delta$G (r)', labelpad=5)
        ax.set_xlabel(r'r ($\AA$)', labelpad=5)
        ax.legend(loc='upper right', prop={'size': 7}, frameon=True)

        maxY = max(masterChanges[-1][valueToIndex(rRange, plotRange[0]):valueToIndex(rRange, plotRange[1])])
        minY = min(masterChanges[-1][valueToIndex(rRange, plotRange[0]):valueToIndex(rRange, plotRange[1])])

        ax.text(0.005, 0.900, label, transform=ax.transAxes, size=12, weight='bold', **hfont)

        #ax.set_ylim([minY - 0.1, maxY + 0.1])
        ax.set_xlim([3,15])

        # xticks = ax.xaxis.get_major_ticks()
        # xticks[-1].label1.set_visible(False)
        # xticks[-1].label1.set_visible(False)

        # # Draw arrows in the plot.
        # # ax1.annotate(s='', xy=(2.7953, minY - 1), xytext=(2.7953, minY - 3), arrowprops=dict(arrowstyle='->'))
        # # ax1.arrow(2.7953, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # # ax1.arrow(2.9697, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.arrow(2.9697, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        # ax.text(3.0697, minY - 1, 'Pb-S', rotation=90, size=10)
        # # ax1.annotate('Pb-S', xy=(2.9697, minY - 1))
        # ax.arrow(4.1999, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.arrow(5.1438, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        # ax.arrow(5.9395, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.text(6.0395, minY - 1, 'a-axis', rotation=90, size=10)
        # ax.arrow(6.6406, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        # ax.arrow(7.2744, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.arrow(8.3997, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.arrow(8.9093, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        # ax.arrow(9.3912, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.arrow(10.2880, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1, color='C0')
        # ax.arrow(11.1056, minY - 3, 0, 1.5, linewidth=1, head_width=0.1, head_length=0.1)
        # ax.text(-0.3, 0.95, label, transform=ax.transAxes, size=12, weight='bold', **hfont)

    def plotPDFThermalComparsion(dotSize, ax, label):

        import numpy as np
        import matplotlib.pyplot as plt
        import os
        import glob

        if dotSize == 'Small':
            # Small scaling factor.
            rScaling = 0.991
            dGScaling = 6.2
            baseT = '155'
            endT = '363'
        if dotSize == 'Medium':
            # Medium scaling factor.
            rScaling = 0.99
            dGScaling = 4
            baseT = '155'
            endT = '363'
        if dotSize == 'Large':
            # Large scaling factor.
            rScaling = 1.003
            dGScaling = 5.1
            baseT = '155'
            endT = '363'
        if dotSize == 'Shelled':
            # Shelled scaling factor.
            rScaling = 0.999
            dGScaling = 2.92
            baseT = '155'
            endT = '363'

        hfont = {'fontname': 'Helvetica'}

        def valueToIndex(list, value):
            index = np.abs(np.array(list) - value).argmin()
            return index

        # Define global variables here.
        plotRange = [2.4, 15]
        # dotSize = 'Shelled'
        if dotSize == 'Small':
            dataDirecs = ['D:\\PDF Patterns Small\\scan6_\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan7\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan9\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan9_\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan10\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan10_\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan11_\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan12\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan13\\PDFs Long Q and R',
                          ]
        # if dotSize == 'Medium':
        #     dataDirecs = ['E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan5\\PDFs',
        #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan6\\PDFs',
        #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan8\\PDFs',
        #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\scan9\\PDFs',
        #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
        #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
        #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
        #                   'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\PDF Patterns Medium\\PDFs',
        #                   ]
        if dotSize == 'Medium':
            dataDirecs = [
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan5\\PDFs Long Q and R',
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan6\\PDFs Long Q and R',
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan7\\PDFs Long Q and R',
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan9\\PDFs Long Q and R',
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan10\\PDFs Long Q and R',
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan11\\PDFs Long Q and R',
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan12\\PDFs Long Q and R',
                'E:\\Ediff Samples\\PbS Data\\5 nm Dots Pumped\\PDF\\Medium Long R PDF\\scan17\\PDFs Long Q and R',
                ]
        if dotSize == 'Large':
            dataDirecs = ['D:\\PDF Patterns Small\\scan13\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan16\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan17\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan18\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan19\\PDFs Long Q and R',
                          'D:\\PDF Patterns Small\\scan1315\\PDFs Long Q and R',
                          ]
        if dotSize == 'Shelled':
            dataDirecs = ['D:\\PDF Patterns Shelled\\scan6\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan8\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan9\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan10\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan11\\PDFs Long Q and R',
                          'D:\\PDF Patterns Shelled\\scan12\\PDFs Long Q and R',
                          ]

        # Data structures for initial figure (comparison of on/off G(r)).
        pdfOff = np.zeros(2000)
        pdfOn = np.zeros(2000)
        rRange = np.arange(0, 20, 0.01)
        onCounter = 0
        offCounter = 0

        # Additional data structures for figures.
        offPreT0 = np.zeros(2000)
        onPreT0 = np.zeros(2000)
        # zeroTo5On = np.zeros(1200)
        # zeroTo5Off = np.zeros(1200)
        # fiveTo15On = np.zeros(1200)
        # fiveTo15Off = np.zeros(1200)
        # fifteenTo50On = np.zeros(1200)
        # fifteenTo50Off = np.zeros(1200)
        # fiftyTo100On = np.zeros(1200)
        # fiftyTo100Off = np.zeros(1200)
        # hundredTo200On = np.zeros(1200)
        # hundredTo200Off = np.zeros(1200)
        offPreT0Counter = 0
        onPreT0Counter = 0
        # zeroFiveOffCounter = 0
        # zeroFiveOnCounter = 0

        # Begin by loading in and storing data.
        for pdfDirec in dataDirecs:
            for pdf in glob.glob(os.path.join(pdfDirec, '*.npy')):

                # Extract data config from filename.
                tp = int(os.path.basename(pdf).split('_')[1].split('.')[0])
                state = os.path.basename(pdf).split('_')[0]

                # Split process for ons and offs, check if ons are post-t0 to average.
                if state == 'Goff':

                    # Load in data, append to pdfOffs and adjust counter accordingly.
                    data = np.load(pdf)
                    pdfOff += data
                    offCounter += 1

                    # If it is OFF and before t0:
                    if tp < 0:
                        # Add to off pre-t0 data.
                        offPreT0 += data
                        offPreT0Counter += 1

                if state == 'Gon':

                    # Load in data.
                    data = np.load(pdf)

                    if tp > 0:
                        # Load in data after t0, append to pdfOns and just counter accordingly.
                        pdfOn += data
                        onCounter += 1

                    if tp < 0:
                        # Add to ON pre-t0 data.
                        onPreT0 += data
                        onPreT0Counter += 1

        # Average everything, and begin plotting.
        avgOffs = pdfOff / offCounter
        avgOns = pdfOn / onCounter
        onPreT0 = onPreT0 / onPreT0Counter
        offPreT0 = offPreT0 / offPreT0Counter
        # zeroTo5On = zeroTo5On / zeroFiveOnCounter
        # zeroTo5Off = zeroTo5Off / zeroFiveOffCounter

        # Setup the results and shit.

        # Extract ranges and store master data.
        #ranges = [[0], [0, 5000], [5000, 25000], [25000, 100000], [100000, 1000000]]
        ranges = [[5000, 25000]]
        masterChanges = []

        for r in ranges:

            # Global data struct.
            finalDataOff = np.zeros(2000)
            finalDataOn = np.zeros(2000)
            finalCounterOff = 0
            finalCounterOn = 0

            for pdfDirec in dataDirecs:
                for pdf in glob.glob(os.path.join(pdfDirec, '*.npy')):

                    # Extract info.
                    tp = int(os.path.basename(pdf).split('_')[1].split('.')[0])
                    state = os.path.basename(pdf).split('_')[0]

                    data = np.load(pdf)

                    # If pre-t0:
                    if len(r) == 1:
                        if state == 'Goff':
                            if tp < 0:
                                finalDataOff += data
                                finalCounterOff += 1

                        if state == 'Gon':
                            if tp < 0:
                                finalDataOn += data
                                finalCounterOn += 1

                    # Now go over the rest of the ranges.
                    if len(r) > 1:
                        if state == 'Goff':
                            if (tp > r[0]) and (tp <= r[1]):
                                finalDataOff += data
                                finalCounterOff += 1
                                photoPeakOff0.append(data[valueToIndex(rRangePhoto, p0[0])])
                                photoPeakOff1.append(data[valueToIndex(rRangePhoto, p1[0])])
                                photoPeakOff2.append(data[valueToIndex(rRangePhoto, p2[0])])
                                photoPeakOff3.append(data[valueToIndex(rRangePhoto, p3[0])])
                                photoPeakOff4.append(data[valueToIndex(rRangePhoto, p4[0])])
                                photoPeakOff5.append(data[valueToIndex(rRangePhoto, p5[0])])
                                photoPeakOff6.append(data[valueToIndex(rRangePhoto, p6[0])])
                                photoPeakOff7.append(data[valueToIndex(rRangePhoto, p7[0])])
                                photoPeakOff8.append(data[valueToIndex(rRangePhoto, p8[0])])
                                photoPeakOff9.append(data[valueToIndex(rRangePhoto, p9[0])])

                        if state == 'Gon':
                            if (tp > r[0]) and (tp <= r[1]):
                                finalDataOn += data
                                finalCounterOn += 1
                                photoPeakOn0.append(data[valueToIndex(rRangePhoto, p0[0])])
                                photoPeakOn1.append(data[valueToIndex(rRangePhoto, p1[0])])
                                photoPeakOn2.append(data[valueToIndex(rRangePhoto, p2[0])])
                                photoPeakOn3.append(data[valueToIndex(rRangePhoto, p3[0])])
                                photoPeakOn4.append(data[valueToIndex(rRangePhoto, p4[0])])
                                photoPeakOn5.append(data[valueToIndex(rRangePhoto, p5[0])])
                                photoPeakOn6.append(data[valueToIndex(rRangePhoto, p6[0])])
                                photoPeakOn7.append(data[valueToIndex(rRangePhoto, p7[0])])
                                photoPeakOn8.append(data[valueToIndex(rRangePhoto, p8[0])])
                                photoPeakOn9.append(data[valueToIndex(rRangePhoto, p9[0])])

            # Finish processing data and append to master data array.
            finalDataOn = finalDataOn / finalCounterOn
            finalDataOff = finalDataOff / finalCounterOff
            masterChanges.append(finalDataOn - finalDataOff)
            photoChanges.append(finalDataOn - finalDataOff)

        masterChanges = np.asarray(masterChanges)

        # Plot the results.
        colors = plt.cm.plasma(np.linspace(0, 1, len(ranges)))
        for ind, range in enumerate(ranges):
            if len(range) == 1:
                labelTitle = '< 0 ps'
                ax.plot(rRange, masterChanges[ind], color='grey', label=labelTitle, linewidth=2, alpha=0.8)
            if len(range) > 1:
                labelTitle = '%0.0f - %0.0f ps' % (range[0] / 1000, range[1] / 1000)
                ax.plot(rRange*rScaling, masterChanges[ind], color='C0', label=labelTitle, linewidth=2, alpha=0.8, zorder=0)
        if label == 'b)':
            ax.legend(loc='upper right', prop={'size': 7}, frameon=False)
        # if (label == 'b)') or (label == 'd)'):
            # ax.axes.yaxis.set_visible(False)
        # ax.set_ylabel(r'$\Delta$g (r)', labelpad=5)
        # plt.xlabel(r'r ($\AA$)')
        # ax2.xlim(plotRange)

        ax.set_ylabel(r'$\Delta$G (r)', labelpad=5)
        ax.set_xlabel(r'r ($\AA$)', labelpad=5)
        ax.legend(loc='upper right', prop={'size': 7}, frameon=True)

        # get all the labels of this axis
        # xticks = ax.xaxis.get_major_ticks()
        # xticks[-1].label1.set_visible(False)
        # xticks[0].label1.set_visible(False)

        maxY = max(masterChanges[-1][valueToIndex(rRange, plotRange[0]):valueToIndex(rRange, plotRange[1])])
        minY = min(masterChanges[-1][valueToIndex(rRange, plotRange[0]):valueToIndex(rRange, plotRange[1])])

        ax.text(0.005, 0.925, label, transform=ax.transAxes, size=12, weight='bold', **hfont)

        # ax3 = ax.twinx()

        if dotSize == 'Medium':
            thermalPDFDirec = 'E:\\Ediff Samples\\PbS Paper - Final Folder\\Thermal PDF Stuff\\Medium Thermal\\PDF'
            #thermalPDFDirec = 'D:\\Documents\\Graduate School\\Research\\PhD Electron Diffraction\\PbS\\Processed Data\\5 nm Dots\\PDF Patterns Medium\\Temperature Dependence\\PDF Long Q and R'
            tempValues = [155, 191, 225, 256, 286, 313, 339, 363]
            rRange = np.arange(0,20,0.01)
            # limits = [2.4, 12]
            limits = [2.4, 15]
        if (dotSize == 'Small') or (dotSize == 'Large') or (dotSize == 'Shelled'):
            thermalPDFDirec = 'E:\\Ediff Samples\\PbS Paper - Final Folder\\Thermal PDF Stuff\\Medium Thermal\\PDF'
            tempValues = [155, 191, 225, 256, 286, 313, 339, 363]
            rRange = np.arange(0,20,0.01)
            limits = [2.4, 15]

        colors = plt.cm.plasma(np.linspace(0, 1, len(tempValues)))

        baseTemp = np.load(os.path.join(thermalPDFDirec, baseT + '_avg.npy'))
        data = np.load(os.path.join(thermalPDFDirec, endT + '_avg.npy'))

        diff = data - baseTemp
        diffNorm = diff/dGScaling
        thermalChanges.append(diffNorm)

        ax.plot(rRange, diffNorm[0:2000], alpha=0.8, color='black', linewidth=2, linestyle='dotted', label='Thermal PDF', zorder=1)
        ax.legend(loc='upper right', prop={'size': 7}, frameon=False)

        ax.axvspan(5.3, 6.3, alpha=0.2, color='C2')
        ax.axvspan(3.92, 4.48, alpha=0.2)
        ax.axvspan(8.97, 9.53, alpha=0.2)
        ax.axvspan(11.8, 12.8, alpha=0.2, color='C2')
        ax.text(0.25, 0.1, 'a-axis', transform=ax.transAxes, size=12, weight='bold', **hfont)

        # ax.text(6.0395, 0, 'a-axis', rotation=90, size=10)

        ax.legend(loc='upper right', prop={'size': 7}, frameon=True)
        # ax.set_ylabel(r'$\Delta$g (r, T)', labelpad=5)
        ax.set_xlim([3.7, 14.1])

    def plotDifferences(dotSize, ax):
        # Peaks to plot.
        #xPlot = [4.2, 6.0, 7.3, 9.3, 10.3, 11.1, 12.5, 15.0, 16.2, 17.0]
        xPlot = [4.2, 5.1, 5.9, 7.3, 8.2, 9.3, 10.25, 11.0, 12.3, 13.5]

        hfont = {'fontname': 'Helvetica'}

        # Load in data.
        masterData = np.load('E:\\Ediff Samples\\PbS Paper - Final Folder\\Thermal PDF Stuff\\PDFpeakRatios_Medium2.npy')
        masterData = masterData.astype(float)

        # Extract data.
        # photoY = []
        # thermY = []
        # errorY = []
        # errorThY = []
        # for x in xPlot:
        #     result = np.where(masterData == x)
        #     photoY.append(float(masterData[1, result[1]]))
        #     thermY.append(float(masterData[2, result[1]]))
        #     errorY.append(float(masterData[3, result[1]]))
        #     errorThY.append(float(masterData[4, result[1]]))
        # photoY = np.asarray(photoY)
        # thermY = np.asarray(thermY)
        # errorY = np.asarray(errorY)
        # errorThY = np.asarray(errorThY)

        # errors = np.sqrt(errorY**2 + errorThY**2)
        # differences = photoY - thermY
        if dotSize == 'Small':
            xPlot = [4.0, 5.0, 5.9, 7.3, 7.9, 9.3, 10.5, 12.609, 13.6, 15.1]
            differences = [0.74762886, 0.00837607, -0.75256994, 0.14875261, -0.00463234, -0.48521589, -0.14322196, -0.23022123, -0.07301846, -0.10155377]
            errors = [0.07785568245393165, 0.0529550848777655, 0.13150789597589993, 0.05895071307620886, 0.1751670498728508, 0.08627888466801287, 0.061122615753901925, 0.07599522625813603, 0.05662285322537715, 0.17786726238452608]
        if dotSize == 'Medium':
            xPlot = [4.2, 5.1, 5.9, 7.3, 8.2, 9.3, 10.25, 11.0, 12.3, 13.5]
            differences = [0.1916145, 0.01655423, -0.20525484, -0.01753846, -0.0176356, -0.11191377, 0.04982928, 0.01248378, -0.14634654, 0.00726162]
            errors = [0.17697527218800005, 0.141845088467062, 0.0818508396062314, 0.08460032792194236, 0.0737535171970173, 0.1295514246571809, 0.10440307138292221, 0.09820517739904627, 0.092092960026445, 0.10401374275019658]
        if dotSize == 'Large':
            xPlot = [4.0, 5.0, 6.1, 7.3, 8.0, 9.4, 10.6, 11.1, 12.7, 13.9]
            differences = [7.13673795e-01, 3.18449835e-04, -7.07038961e-01, 4.57255788e-02, -5.85762095e-02, -5.09957759e-01, -6.54574697e-02, 1.36082871e-01, -1.17588911e-02, 1.97078697e-02]
            errors = [0.08634963873228226, 0.053497122129820704, 0.052089208468485304, 0.07281909410268327, 0.0374256320595185, 0.05174505966502137, 0.13616944958117452, 0.0662576470979757, 0.034044444470130455, 0.09079632846581837]
        if dotSize == 'Shelled':
            xPlot = [4.2, 5.1, 6.0, 7.3, 8.2, 9.3, 10.3, 11.1, 12.4, 13.6]
            differences = [0.17178773, 0.00552623, -0.58078254, -0.04629318, -0.11778402, -0.33773031, 0.05231041, -0.10604742, -0.27132612, -0.00949356]
            errors = [0.11379924105186576, 0.12716355308800725, 0.12544640076512817, 0.11148870541494516, 0.07708065527740685, 0.07986038578810352, 0.11596533305933658, 0.0829311243238656, 0.05793007248959473, 0.059735606761203076]

        # plt.bar(xPlot, differences, yerr=errors, capsize=5, zorder=1, color='C0', alpha=0.8)
        # plt.axhline(y=0, linestyle='--', color='black', alpha=0.5, zorder=0)

        # Crazy idea bits now.
        ax.errorbar(xPlot, differences, yerr=errors, marker='s', capsize=6, linestyle='None', color='C0',
                     markersize=10, alpha=1, zorder=1, ecolor='black')

        for ind, val in enumerate(differences):
            if val > 0:
                ax.errorbar(xPlot[ind], val, yerr=errors[ind], marker='s', capsize=6, linestyle='None', color='C0',
                             markersize=10, alpha=1, zorder=1, ecolor='black')

        for ind, val in enumerate(differences):
            if val > 0:
                colorRange = np.arange(0, val, 0.01)
                colors = plt.cm.OrRd(np.linspace(0, 1, len(colorRange)))
                for i, j in enumerate(colorRange):
                    plt.vlines(xPlot[ind], 0, colorRange[i], color=colors[i], zorder=-1*i, linewidth=7)
            if val < 0:
                colorRange = np.arange(0, -1*val, 0.01)
                colors = plt.cm.OrRd(np.linspace(0,1, len(colorRange)))
                for i, j in enumerate(colorRange):
                    plt.vlines(xPlot[ind], -1*colorRange[i], 0, color=colors[i], zorder=-1*i, linewidth=7)
        ax.axhline(y=0, linestyle='--', color='gray', alpha=0.5, zorder=0)

        ax.set_xlabel(r'r ($\AA$)')
        ax.set_ylabel(r'$\Delta G_{P.E.}(r)$ - $\Delta G_{T}(r)$')
        ax.text(0.005, 0.90, 'd)', transform=ax.transAxes, size=12, weight='bold', **hfont)

    # Do the plot stuff.
    plotPDF(size, ax1)
    plotPDFdiffs(size, ax2, 'b)')
    plotPDFThermalComparsion(size, ax3, 'c)')
    plotDifferences(size, ax4)

    # plt.tight_layout(pad=0.1, w_pad=0.05, h_pad=0.05)

    plt.tight_layout()
    fig.align_ylabels()
    fig.savefig('E:\\Ediff Samples\\PbS Paper - Final Folder\\Final Push\\Changed Figures\\Figure6_%s_NewTesting.pdf' % size)
    plt.show()

figure5v4('Medium')
#SizeDepIndependent()
#FigureAllDotsTogether2ElectricBoogaloo()