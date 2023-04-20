import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# ----------------------------------------------------------------------------------------------------------------------
# TASK 1
traces_original = pd.read_csv('traces.csv', sep=',', index_col=0, header=0)
df = traces_original.drop(traces_original.iloc[:, 10:1288], axis=1)
print(df)

df.plot(title='Fluorescence traces of neurons', sharey=True, legend=False, subplots=True, figsize=(7, 7))
plt.text(-10, 800, 'Fluorescence (a.u.)', ha="center", va="center", rotation=90)
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# TASK 2
neuron_4 = df.iloc[:, 4]
print(neuron_4)

neuron_4.plot(title='Fluorescence trace neuron 4', legend=False, subplots=True, figsize=(10, 5))
plt.text(-8, 50, 'Fluorescence (a.u.)', ha="center", va="center", rotation=90)
plt.show()

# Zoom in on first event
neuron_4.plot(title='First event neuron 4', legend=False, subplots=True, figsize=(7, 7), xlim=[4, 8])
plt.text(3.6, 50, 'Fluorescence (a.u.)', ha="center", va="center", rotation=90)
plt.show()

# Fit area
rise = neuron_4[4.5:5.05]
fall = neuron_4[5.05:6.5]

# Rise and fall data, x is time, y fluorescence
xrise = rise.index.values
yrise = rise.values
xfall = fall.index.values
yfall = fall.values

# Recast xdata and ydata into numpy arrays
xrise = np.asarray(xrise)
yrise = np.asarray(yrise)
xfall = np.asarray(xfall)
yfall = np.asarray(yfall)

delta_ff1 = [None] * len(yrise)
delta_ff2 = [None] * len(yfall)

for i in range(len(yrise)):
    delta_ff1[i] = (yrise[i]-yrise[0])/yrise[0]

for i in range(len(yfall)):
    delta_ff2[i] = (yfall[i]-yrise[0])/yrise[0]

delta_ff1 = np.array(delta_ff1)
delta_ff2 = np.array(delta_ff2)

print(xrise.shape)
print(delta_ff1.shape)


# Define the rise function
def risef(t1, A, tau_b):
    y1 = A*np.exp(t1/tau_b)
    return y1


# Define the fall function
def fallf(t2, B, tau_u):
    y2 = B*np.exp(-(t2/tau_u))
    return y2

# Fit
param1, covar1 = curve_fit(risef, xrise, delta_ff1, maxfev=5000)
param2, covar2 = curve_fit(fallf, xfall, delta_ff2, maxfev=5000)

fit_A = param1[0]
fit_tau_b = param1[1]
print('A = ', fit_A)
print('Tau_b = ', fit_tau_b)

fit_B = param2[0]
fit_tau_u = param2[1]
print('B = ', fit_B)
print('Tau_u = ', fit_tau_u)

fit_rise = risef(xrise, fit_A, fit_tau_b)
plt.plot(xrise, delta_ff1, 'o', label='Original data')
plt.plot(xrise, fit_rise, '-', label='Fit')
plt.xlabel('time (s)')
plt.ylabel('$\Delta FF$')
plt.legend()
plt.show()

fit_fall = fallf(xfall, fit_B, fit_tau_u)
plt.plot(xfall, delta_ff2, 'o', label='Original data')
plt.plot(xfall, fit_fall, '-', label='Fit')
plt.xlabel('time (s)')
plt.ylabel('$\Delta FF$')
plt.legend()
plt.show()
