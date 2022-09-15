
import numpy as np
from scipy.special import erf
from scipy.special import erfc
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import cm

def Exp_Fit(timedata, peakdata, a_fit, tau_fit, t0_fit, c_fit, sigma = 1.0, plotlabel=''):
    '''
    Fitting equation:
        erf1 = 1 + erf((x - t0) / (np.sqrt(2) * sigma))
        exp1 = np.exp((sigma ** 2 - 2 * (x - t0) * tau1) / (2 * tau1 ** 2))
        erfc1 = erfc((-sigma + ((x - t0) * tau1 / sigma)) / (np.sqrt(2) * tau1))
        return (1 / 2) * ((A1 * erf1) + (A1 * exp1) * (-2 + erfc1)) + c

    :param timedata: timepoint array for x-axis
    :param peakdata: peak intensity array for y-axis
    :param a_fit: scaling amplitude [-1,1]
    :param tau_fit: time constant of decay, [0,100 ps]
    :param t0_fit: time zero, [-10, 10 ps]
    :param c_fit: offset variable, usually set to 1
    :param plotlabel: optional argument, I don't think it's used anymore
    :return:
        x_out, y_out: best fit curve
        popt, pcov: scipy curve_fit output
    '''

    times = np.array(timedata)   # convert whatever into array
    peaks = np.array(peakdata)

    def ExponentialIntensityDecay(x, A1, tau1, t0, c):
        #sigma = 0.3  # instrument response function ?

        erf1 = 1 + erf((x - t0) / (np.sqrt(2) * sigma))
        exp1 = np.exp((sigma ** 2 - 2 * (x - t0) * tau1) / (2 * tau1 ** 2))
        erfc1 = erfc((-sigma + ((x - t0) * tau1 / sigma)) / (np.sqrt(2) * tau1))

        return (1 / 2) * ((A1 * erf1) + (A1 * exp1) * (-2 + erfc1)) + c  # output of ExponentialIntensityDecay

    popt, pcov = curve_fit(ExponentialIntensityDecay, times, peaks, p0 =(a_fit, tau_fit, t0_fit, c_fit))

    x_out = np.arange(min(times), max(times), 0.01)
    y_out = ExponentialIntensityDecay(x_out, popt[0],popt[1],popt[2],popt[3])

    return x_out, y_out, popt, pcov  # output of Exp_Fit

#testing the shape of the curve_fit function
if __name__ == '__main__':

    import pandas as pd

    # loading the data
    csv_direc = 'D:\\Bismuth Project\\Bismuth-Data-Analysis-github\\peak_sum_sql_export.csv'
    peak_sum_df = pd.read_csv(csv_direc, index_col=[0])

    # initializing the variables for the loop
    # peak_choice = 0, 1, 2, 3, 4, 5
    peak_choice = 0, 1, 4, 5
    peak_colors = cm.turbo(np.linspace(0, 1, len(peak_choice)))
    cols = ['peak1', 'peak2', 'peak3', 'peak4', 'peak5', 'peak6']
    exp_fit_df = pd.DataFrame(columns = ['scan_id', 'peak_id', 'a', 'tau', 't0', 'c', 'a_err', 'tau_err', 't0_err', 'c_err'])


    for scan_index in np.arange(1, 7, 1):  # loop over scans
        print('scan_index: ' + str(scan_index))
        fig1, ax1 = plt.subplots()

        for p, peak_index in enumerate(peak_choice):  # loop over peaks
            print(f'peak_index: {str(peak_index)}')
            # extract the data I want from the dataframe
            scandf = peak_sum_df.loc[[scan_index]]
            tp = scandf['timepoint']
            peak = scandf[cols[peak_index]]
            peak_norm = peak / np.mean(peak[:7])

            ax1.plot(tp, peak_norm, '-o', label=cols[peak_index], color=peak_colors[p])

            a, tau, t0, c = -0.2, 2.0, 0, 1

            pltlabel = f'scan{str(scan_index)}-{cols[peak_index]}'
            x_fit, y_fit, fit_var, pcov = Exp_Fit(tp, peak_norm, a, tau, t0, c)
            perr = np.sqrt(np.diag(pcov))  # calculates sqrt of the diagonals of pcov

            expfit_new_row = {'scan_id': str(scan_index), 'peak_id': str(peak_index + 1),
                              'a': fit_var[0], 'tau': fit_var[1], 't0': fit_var[2], 'c': fit_var[3],
                              'a_err':perr[0], 'tau_err':perr[1], 't0_err':perr[2], 'c_err':perr[3]}

            exp_fit_df = exp_fit_df.append(expfit_new_row, ignore_index=True)


            ax1.plot(x_fit, y_fit, '--', color=peak_colors[p])

        fig1.suptitle('scan_id: {0}'.format(str(scan_index)))
        plt.legend()
        plt.grid(True)

    peak2bragg_map = {'1':'(011)', '2':'(01-1)', '5':'(002)', '6':'(022)'}
    exp_fit_df['bragg_peak'] = exp_fit_df.peak_id.map(peak2bragg_map)
    scan2fluence_map = {'1':'0.78', '2':'1.3', '3':'2.6', '4':'5.2', '5':'7.8', '6':'10.4'}
    exp_fit_df['fluence'] = exp_fit_df.scan_id.map(scan2fluence_map)

    fit_var_sliced = exp_fit_df[['fluence', 'peak_id', 'bragg_peak', 'a', 'a_err', 'c', 'c_err', 'tau', 'tau_err', 't0', 't0_err']]
    print(exp_fit_df)
    print(fit_var_sliced)

    # fit_var_sliced.to_csv('D:\\Bismuth Project\\Bismuth-Data-Analysis-github\\All_fit_constants_w_err.csv', index=False)

    # plt.show()

