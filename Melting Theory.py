import math as mt
import numpy as np

k = 1.38065e-23  # Boltzmann constant in SI units
n_s = 10.49e3  # kg/m^3 density at STP
T_m = 1234.78 # melting temperature in Kelvin
q_mass = 105e3  # latent heat of silver, J/kg
q= q_mass * n_s
sigma_sl = 0.126  # J/m^2
c_p = 0.235e3  # J/(kg*K) specific heat at STP
v_m = 343  # m/s speed of sound

A = 16*sigma_sl**3 * mt.pi * k**2 * T_m **2 / (3*q**2)
print(A)
A_factor = A/((k*T_m)**3)
print(A_factor)

N = (3.0/(64*mt.pi)) * ((16*mt.pi/3.0)**(1/6)) * k**(-2/3)
print(N)


M_den = n_s**2 * T_m**(5/3) * c_p * sigma_sl**3 * v_m * (A_factor)**(1/6)
M = q**(14/3) * M_den**-1
# print(q_mass**14/3)
print(M_den)
print(M)
print('N*M = ', str(N*M))

def superheating_func(theta):
    a = ((theta-1)**3) / (theta**0.5)
    b = (A*(k*T_m)**-3)
    c = theta*(theta-1)**2
    return a * mt.exp(b/c)

t_m = N*M*superheating_func(1.4)
print(t_m)
