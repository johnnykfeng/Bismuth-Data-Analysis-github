
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.special import erfc
from scipy.optimize import curve_fit

# Kam's code


def BiExponentialIntensityDecay(x, A1, A2, tau1, tau2, t0, c):
    sigma = 0.3  # instrument response function ?

    erf1 = 1 + erf((x - t0) / (np.sqrt(2) * sigma))
    erf2 = 1 + erf((x - t0) / (np.sqrt(2) * sigma))
    exp1 = np.exp((sigma ** 2 - 2 * (x - t0) * tau1) / (2 * tau1 ** 2))
    erfc1 = erfc((-sigma + ((x - t0) * tau1 / sigma)) / (np.sqrt(2) * tau1))
    exp2 = np.exp((sigma ** 2 - 2 * (x - t0) * tau2) / (2 * tau2 ** 2))
    erfc2 = erfc((-sigma + ((x - t0) * tau2 / sigma)) / (np.sqrt(2) * tau2))

    return (1 / 2) * ((A1 * erf1 + A2 * erf2) + (A1 * exp1) * (-2 + erfc1) + (A2 * exp2) * (-2 + erfc2)) + c

def biexp_fit(timedata, peakdata, a1_guess, a2_guess, tau1_guess, tau2_guess, t0_guess, c_guess, plotlabel):
    times = np.array(timedata)   # convert whatever into array
    peaks = np.array(peakdata)


    def BiExponentialIntensityDecay(x, A1, A2, tau1, tau2, t0, c):
        sigma = 0.3  # instrument response function ?

        erf1 = 1 + erf((x - t0)/(np.sqrt(2) * sigma))
        erf2 = 1 + erf((x - t0)/(np.sqrt(2) * sigma))
        exp1 = np.exp((sigma**2 - 2*(x-t0)*tau1)/(2*tau1**2))
        erfc1 = erfc((-sigma + ((x - t0)*tau1 / sigma)) /(np.sqrt(2) * tau1))
        exp2 = np.exp((sigma**2 - 2*(x-t0)*tau2)/(2*tau2**2))
        erfc2 = erfc((-sigma + ((x - t0)*tau2 / sigma)) /(np.sqrt(2) * tau2))

        return (1/2)*((A1*erf1 + A2*erf2) + (A1*exp1)*(-2 + erfc1) + (A2*exp2)*(-2 + erfc2)) + c

#region OLD_CODE
    # def BiExponentialIntensityDecay(x, A1, A2, tau1, tau2, t0, c):
    #     sigma = 0.3
    #     t0 = 0
    #     return (1/2)*((A1 * (1 + erf( (x - t0)/(np.sqrt(2) * sigma) )) + A2 * (1 + erf( (x - t0)/(np.sqrt(2) * sigma) ))) + (A1 * np.exp( (sigma**2 - 2*(x-t0)*tau1)/(2*tau1**2)
    #                             ))*( -2 + erfc( (-sigma + ((x - t0) * tau1 / sigma)) / (np.sqrt(2) * tau1) )) +
    #                   (A2 * np.exp((sigma ** 2 - 2 * (x - t0) * tau2) / (2 * tau2 ** 2)
    #                                )) * (-2 + erfc((-sigma + ((x - t0) * tau2 / sigma)) / (np.sqrt(2) * tau2)))
    #                   ) + c

    # def piecewise_biexp_function(x, A1, A2, tau1, tau2, t0):
    #     if t < t0:
    #         return A1 + A2 + c
    #     else:
    #         return BiExponentialIntensityDecay(x, A1, A2, tau1, tau2, t0, c)
    #
    # fitfunc_vec = np.vectorize(piecewise_biexp_function)
    #
    # def fitfunc_vec_self(x, A1, A2, tau1, tau2, t0):
    #     y = np.zeros(x.shape)
    #     for i in range(len(y)):
    #         y[i]=piecewise_exp_function(x[i], A1, A2, tau1, tau2, t0, c)
    #     return y

    # popt, pcov = curve_fit(fitfunc_vec_self, times, peaks, p0 =(a1_guess, a2_guess, tau1_guess, tau2_guess, t0_guess, c_guess))
#endregion

    popt, pcov = curve_fit(BiExponentialIntensityDecay, times, peaks,
                           p0=(a1_guess, a2_guess, tau1_guess, tau2_guess, t0_guess, c_guess),
                           bounds=([-100., -100., 0, 0, -20., 0.9], [100., 100., 10., 1000., 20., 1.1]))

    # popt, pcov = curve_fit(BiExponentialIntensityDecay, times, peaks,
    #                        p0 =(a1_guess, a2_guess, tau1_guess, tau2_guess, t0_guess, c_guess),
    #                        bounds =(-100,100),(-100,100),(0,10),(0,1000),(-20,20),(0.5,1.5) )

    x_fit = np.arange(min(times),max(times), 0.01)
    # y_out = fitfunc_vec_self(x_out, popt[0],popt[1],popt[2],popt[3])
    y_fit = BiExponentialIntensityDecay(x_fit, popt[0],popt[1],popt[2],popt[3], popt[4], popt[5])

    return x_fit, y_fit, popt, pcov


if __name__ == '__main__':

    import pandas as pd
    csv_direc = 'D:\\Bismuth Project\\Bismuth-Data-Analysis-github\\peak_sum_sql_export.csv'
    peak_sum_df = pd.read_csv(csv_direc, index_col=[0])

    scan1 = peak_sum_df.loc[[1]]
    scan2 = peak_sum_df.loc[[2]]
    scan3 = peak_sum_df.loc[[3]]
    scan4 = peak_sum_df.loc[[4]]
    scan5 = peak_sum_df.loc[[5]]
    scan6 = peak_sum_df.loc[[6]]
    scan7 = peak_sum_df.loc[[7]]

    tp = scan5['timepoint']
    peak = scan3['peak6']
    peak_norm = peak/np.mean(peak[:7])

    fig, ax1 = plt.subplots()
    ax1.plot(tp,peak, 'o')
    plt.grid(1)

    fig, ax2 = plt.subplots()
    ax2.plot(tp, peak_norm, 'o')
    plt.grid(1)


    a1, a2, tau1, tau2, t0, c = -0.4, -0.2, 2, 100, 0, 1
    # a1, a2, tau1, tau2, t0, c = [-1.63862973e+04, 10000,  5, 1000,
    #  0,  3.71796228e+04]

    # a1, a2, tau1, tau2, t0, c = -0.2, 0.1, 2, 100, 0, 1
    x_fit, y_fit, fit, pcov = biexp_fit(tp, peak_norm, a1, a2, tau1, tau2, t0, c, 'scan3-peak2')
    print(fit)
    ax2.plot(x_fit, y_fit)
    # x_fit, y_fit, popt, pcov= biexp_fit(tp, peak, fit[0], fit[1], tau1, tau2, c, 'scan3-peak2')

    # y_biexp = BiExponentialIntensityDecay(tp, a1, a2, tau1, tau2, t0, c)
    # ax2.plot(tp,y_biexp)

    plt.show()

    # biexp_fit