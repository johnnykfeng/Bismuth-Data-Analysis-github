
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

# Kam's code
# def ExponentialIntensityDecay(x, tau, A):
#
#     sigma = 0.3
#     t0 = 0
#
#     return ((A / 2) * (1 + erf((x - t0) / (np.sqrt(2) * sigma)) - (np.exp((-1*(x - t0)) / tau)) * (np.exp((1 / 2) * (sigma / tau) ** 2)) * (1 + erf(
#         ((x - t0) - sigma**2 / tau) / (np.sqrt(2) * sigma)))))
#
# def BiExponentialIntensityDecay(x, tau1, tau2, A1, A2):
#
#     sigma = 0.3
#     t0 = 0
#
#     return (1/2)*((A1 * (1 + erf( (x - t0)/(np.sqrt(2) * sigma) )) + A2 * (1 + erf( (x - t0)/(np.sqrt(2) * sigma) ))) + (A1 * np.exp( (sigma**2 - 2*(x-t0)*tau1)/(2*tau1**2)
#                             ))*( -2 + erfc( (-sigma + ((x - t0) * tau1 / sigma)) / (np.sqrt(2) * tau1) )) +
#                   (A2 * np.exp((sigma ** 2 - 2 * (x - t0) * tau2) / (2 * tau2 ** 2)
#                                )) * (-2 + erfc((-sigma + ((x - t0) * tau2 / sigma)) / (np.sqrt(2) * tau2)))
#                   )

def mainfitting(timedata, peakdata, a_fit, c_fit, tau_fit, t0_fit, plotlabel):
    times = np.array(timedata)   # convert whatever into array
    peaks = np.array(peakdata)

    def piecewise_exp_function(t, a, c, tau, t0):
        if t < t0:
            return a + c
        else:
            return a*np.exp(-1.0*(t - t0)/tau) + c

    fitfunc_vec = np.vectorize(piecewise_exp_function)
    def fitfunc_vec_self(t,a,c,tau,t0):
        y = np.zeros(t.shape)
        for i in range(len(y)):
            y[i]=piecewise_exp_function(t[i],a,c,tau,t0)
        return y

    popt, pcov = curve_fit(fitfunc_vec_self, times, peaks, p0 =(a_fit,c_fit,tau_fit,t0_fit))

    x_out = np.arange(min(times),max(times), 0.01)
    y_out = fitfunc_vec_self(x_out, popt[0],popt[1],popt[2],popt[3])

    return x_out,y_out,popt,pcov
