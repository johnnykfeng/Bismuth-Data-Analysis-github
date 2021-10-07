
import numpy as np
from scipy.special import erf
from scipy.special import erfc
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def Exp_Fit(timedata, peakdata, a_fit, tau_fit, t0_fit, c_fit, plotlabel=''):
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

    #region OLD CODE
    # def piecewise_exp_function(t, a, c, tau, t0):
    #     if t < t0:
    #         return a + c
    #     else:
    #         return a*np.exp(-1.0*(t - t0)/tau) + c
    #
    # fitfunc_vec = np.vectorize(piecewise_exp_function)
    # def fitfunc_vec_self(t,a,c,tau,t0):
    #     y = np.zeros(t.shape)
    #     for i in range(len(y)):
    #         y[i]=piecewise_exp_function(t[i],a,c,tau,t0)
    #     return y

    # popt, pcov = curve_fit(fitfunc_vec_self, times, peaks, p0 =(a_fit,c_fit,tau_fit,t0_fit))
    #endregion

    def ExponentialIntensityDecay(x, A1, tau1, t0, c):
        sigma = 0.3  # instrument response function ?

        erf1 = 1 + erf((x - t0) / (np.sqrt(2) * sigma))
        exp1 = np.exp((sigma ** 2 - 2 * (x - t0) * tau1) / (2 * tau1 ** 2))
        erfc1 = erfc((-sigma + ((x - t0) * tau1 / sigma)) / (np.sqrt(2) * tau1))

        return (1 / 2) * ((A1 * erf1) + (A1 * exp1) * (-2 + erfc1)) + c

    popt, pcov = curve_fit(ExponentialIntensityDecay, times, peaks, p0 =(a_fit, tau_fit, t0_fit, c_fit))

    x_out = np.arange(min(times),max(times), 0.01)
    y_out = ExponentialIntensityDecay(x_out, popt[0],popt[1],popt[2],popt[3])

    return x_out, y_out, popt, pcov

#testing the shape of the curve_fit function
if __name__ == '__main__':

    import pandas as pd
    csv_direc = 'D:\\Bismuth Project\\Bismuth-Data-Analysis-github\\peak_sum_sql_export.csv'
    peak_sum_df = pd.read_csv(csv_direc, index_col=[0])

    peak_choice = 0, 1, 2, 3
    peak_colors = cm.turbo(np.linspace(0, 1, len(peak_choice)))

    for scan_index in np.arange(1, 4, 1):  # loop over scans
        print('scan_index: ' + str(scan_index))
        fig1, ax1 = plt.subplots()

        for peak_index in peak_choice:  # loop over peaks
            print('peak_index: ' + str(peak_index))
            # extract the data I want from the dataframe
            scandf = peak_sum_df.loc[[scan_index]]
            tp = scandf['timepoint']
            peak = scandf[cols[peak_index]]
            peak_norm = peak / np.mean(peak[:7])


            ax1.plot(tp, peak_norm, '-o', label=cols[peak_index], color=peak_colors[peak_index])

            a, tau, t0, c = -0.2, 2.0, 0, 1

            pltlabel = 'scan' + str(scan_index) + '-' + cols[peak_index]
            x_fit, y_fit, fit_var, pcov = Exp_Fit(tp, peak_norm, a, tau, t0, c, pltlabel)

            ax1.plot(x_fit, y_fit, '--', color=peak_colors[peak_index])

        fig1.suptitle('scan_id: ' + str(scan_index))
        plt.legend()
        plt.grid(True)

    plt.show()