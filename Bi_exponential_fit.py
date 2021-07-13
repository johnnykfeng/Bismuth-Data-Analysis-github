
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

# Kam's code


def biexp_fit(timedata, peakdata, a1_guess, a2_guess, tau1_guess, tau2_guess, t0_guess, c_guess, plotlabel):
    times = np.array(timedata)   # convert whatever into array
    peaks = np.array(peakdata)

    def ExponentialIntensityDecay(x, tau, A):
        sigma = 0.3
        t0 = 0
        return ((A / 2) * (1 + erf((x - t0) / (np.sqrt(2) * sigma)) - (np.exp((-1*(x - t0)) / tau)) * (np.exp((1 / 2) * (sigma / tau) ** 2)) * (1 + erf(
            ((x - t0) - sigma**2 / tau) / (np.sqrt(2) * sigma)))))

    def BiExponentialIntensityDecay(x, A1, A2, tau1, tau2, t0, c):
        sigma = 0.3
        t0 = 0
        return (1/2)*((A1 * (1 + erf( (x - t0)/(np.sqrt(2) * sigma) )) + A2 * (1 + erf( (x - t0)/(np.sqrt(2) * sigma) ))) + (A1 * np.exp( (sigma**2 - 2*(x-t0)*tau1)/(2*tau1**2)
                                ))*( -2 + erfc( (-sigma + ((x - t0) * tau1 / sigma)) / (np.sqrt(2) * tau1) )) +
                      (A2 * np.exp((sigma ** 2 - 2 * (x - t0) * tau2) / (2 * tau2 ** 2)
                                   )) * (-2 + erfc((-sigma + ((x - t0) * tau2 / sigma)) / (np.sqrt(2) * tau2)))
                      ) + c

    def BiExponentialIntensityDecay(x, A1, A2, tau1, tau2, t0, c):
        sigma = 0.3  # instrument response function ?
        f1 = 1 + erf((x - t0)/(np.sqrt(2) * sigma))
        f2 = 1 + erf((x - t0)/(np.sqrt(2) * sigma))
        f3 = np.exp((sigma**2 - 2*(x-t0)*tau1)/(2*tau1**2))
        f4 = erfc((-sigma + ((x - t0)*tau1 / sigma)) /(np.sqrt(2) * tau1))

        return (1/2)*((A1*f1 + A2*f2) + (A1*f3)* (-2 + f4) +
                      (A2*np.exp((sigma ** 2 - 2 * (x - t0) * tau2) / (2 * tau2 ** 2)
                                   )) * (-2 + erfc((-sigma + ((x - t0) * tau2 / sigma)) / (np.sqrt(2) * tau2)))
                      ) + c

    def piecewise_biexp_function(x, A1, A2, tau1, tau2, t0):
        if t < t0:
            return A1 + A2 + c
        else:
            return BiExponentialIntensityDecay(x, A1, A2, tau1, tau2, t0, c)

    fitfunc_vec = np.vectorize(piecewise_biexp_function)

    def fitfunc_vec_self(x, A1, A2, tau1, tau2, t0):
        y = np.zeros(x.shape)
        for i in range(len(y)):
            y[i]=piecewise_exp_function(x[i], A1, A2, tau1, tau2, t0, c)
        return y

    popt, pcov = curve_fit(fitfunc_vec_self, times, peaks, p0 =(a1_guess, a2_guess, tau1_guess, tau2_guess, t0_guess, c_guess))

    x_out = np.arange(min(times),max(times), 0.01)
    y_out = fitfunc_vec_self(x_out, popt[0],popt[1],popt[2],popt[3])

    return x_out, y_out, popt, pcov
