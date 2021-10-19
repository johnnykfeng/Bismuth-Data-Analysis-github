import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit

#region EXCITED CARRIERS FRACTION
# temp2fluence = 184.85  # taken from the Sciaini (2009) Fig 1e
c = 2.998e8   # speed of light constant
h = 6.626e-34 # planck's constant
Cl = 25.52 # J/(mol*K)  Specific Heat of Bismuth
mol = 6.022e23 # Avogradro's number
laser_wavelength = 800e-9  # excitation photon wavelength
T_room = 300  # Kelvin room temperature


def excited_carriers_fraction(deltaT, wavelength = 800e-9):
    photon_energy_mol= h*c*mol/(wavelength)
    fraction = Cl*deltaT/(5*photon_energy_mol)
    return fraction

# print(excited_carriers_fraction(99.2))
# print(excited_carriers_fraction(221.7))
# print(excited_carriers_fraction(378.6))

#endregion



def find_nearest(array, value):
    ''' 
    Simple function that finds the element in an array closest to a value
    :param array: any array or list
    :param value: single value
    :return: 
    index = index of the array element that's closest
    array[index] = the element in the array closest to the value
    '''
    array = np.asarray(array)
    index = (np.abs(array - value)).argmin()
    return index, array[index]


a0,a1,a2,a3,a4 = 0.03610, 0.3580e-2, 0.1111e-5, -0.1855e-8, 0.1140e-11
a_high = [a0,a1,a2,a3,a4]
a0,a1,a2,a3,a4 = 0.11585, -0.1975e-3, 0.7561e-4, -0.705e-6, 0.2596e-8
a_low = [a0,a1,a2,a3,a4]

print('a coefficients (T<80) = ', a_low)
print('a coefficients (T>80) = ', a_high)

def DWF_polynomial(T, *a):
    return a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4

B_r = DWF_polynomial(T_room, *a_high)

def dw_intensity(s, T_eq):
    B_eq = DWF_polynomial(T_eq, *a_high)
    NI = np.exp(-0.5*(B_eq - B_r) *s**2)
    return NI

print('B @ 300K:', B_r)


# input slope --> output temperature and delta T
def DB_temperature(slope, room_temperature = 300, verbose=False):
    B_r = DWF_polynomial(room_temperature, *a_high)  # DWF at room temp in Kelvin
    BT_eq = 2 * slope + B_r  # DWF from slope of ln(Intensity) vs s^2 curve
    T_high = np.arange(80, 1000, 0.1)  # creates and array of temperature
    B_highT = DWF_polynomial(T_high, *a_high)   # creates and array of DWF
    i, B_nearest = find_nearest(B_highT, BT_eq)   # finds the nearest temperature that matches BT_eq
    delta_T = T_high[i] - room_temperature   # finds the temperature difference

    if verbose:
        print("==========================================")
        print("B(T_eq) = ", str(BT_eq))
        print("Closest B: ", B_nearest)
        print("Closest temperature: ", T_high[i])
        print("delta T: ", delta_T)
        print(" ")

    return B_nearest, T_high[i], delta_T

DB_temperature(0.192, 300, True)
DB_temperature(0.429, 300, True)
DB_temperature(0.735, 300, True)


outputs = 0
if outputs:

    slope1 = 0.400 # A^2 units
    T_room = 300 # Kelvin

    T_low = np.arange(1,80,0.1)
    T_high = np.arange(80,1000,0.1)
    B_highT = DWF_polynomial(T_high, *a_high)
    B_lowT = DWF_polynomial(T_low, *a_low)

    slope_array = np.arange(0.05, 1.0, 0.05)
    deltaT_list = []
    for i in slope_array:
        deltaT_list.append(DB_temperature(i, T_room, 0)[2])


    plt.figure()
    plt.plot(T_high, B_highT, linestyle='-',  color= 'red', label= 'Debye-waller factor Bismuth')
    plt.plot(T_low, B_lowT, linestyle='--', color= 'red')

    plt.axhline(DB_temperature(slope1, T_room, 0)[0])
    plt.axhline(DWF_polynomial(T_room, *a_high), ls='--', color ='orange', label=str(T_room)+' K')

    plt.legend()
    plt.xlabel('Temperature (K)')
    plt.ylabel('Debye Waller Factor B (Angstrom^2)')
    plt.grid(True)

    plt.figure()
    plt.plot(slope_array, deltaT_list, '-o')
    plt.grid(True)
    plt.ylabel('delta T (K)')
    plt.xlabel('DW slope')

    plt.show()

