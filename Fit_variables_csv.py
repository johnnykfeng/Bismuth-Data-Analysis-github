import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# filename = 'D:\\Bismuth Project\\Python by John\\fit_var.csv'
filename = 'D:\\Bismuth Project\\Python by John\\fit_var_14nm.csv'


testlist=[]
fluence = []
tzero1=[]; tzero2=[]; tzero5=[]; tzero6=[]
tau1=[]; tau2=[]; tau5=[]; tau6=[]
# tzero =np.zeros((N_peaks, N_fluence))
tzero_std= []
tzero_avg=[]

with open(filename, 'r') as read_obj:
    # reader = csv.reader(read_obj)
    reader = csv.reader(read_obj, delimiter = ',', quoting=csv.QUOTE_NONNUMERIC)
    print('reader = ')
    print(reader)
    for row in reader:
        # print(row)
        testlist.append(row)
        fluence.append(row[0])
        tau1.append(row[4]); tau2.append(row[9]); tau5.append(row[14]); tau6.append(row[19])
        tzero1.append(row[5]); tzero2.append(row[10]); tzero5.append(row[15]); tzero6.append(row[20])
        a = np.array([row[5], row[10], row[15], row[20]])
        tzero_std.append(np.std(a))
        tzero_avg.append(np.average(a))

fig, ax1 = plt.subplots()
plt.plot(fluence, tzero1, '-+', label= 'Peak 1')
plt.plot(fluence, tzero2, '-^', label= 'Peak 2')
plt.plot(fluence, tzero5, '-x', label= 'Peak 5')
plt.plot(fluence, tzero6, '-*', label= 'Peak 6')
plt.errorbar(fluence, tzero_avg, yerr = tzero_std, marker='s', label= 'Average + Std')

ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
plt.title('Curve fit results of Time-zero ')
plt.xlabel('Fluence  ' + '($mJ/cm^2$)')
plt.ylabel('time-zero (ps)')
plt.legend()
# plt.grid(1, which='both')
plt.grid(1)

fig, ax = plt.subplots()
plt.plot(fluence, tau1, '-o', label= 'Peak 1')
plt.plot(fluence, tau2, '-^', label= 'Peak 2')
plt.plot(fluence, tau5, '-x', label= 'Peak 5')
plt.plot(fluence, tau6, '->', label= 'Peak 6')
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
plt.title('Curve fit results of ' + '$tau$')
plt.ylabel('decay time constant (ps)')
plt.xlabel('Fluence  ' + '($mJ/cm^2$)')
plt.legend()
# plt.grid(True, which='both')
plt.grid(1)

plt.show()
