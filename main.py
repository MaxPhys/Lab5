import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

# ----------------------------------------------------------------------------------------------------------------------
# TASK 1
traces_original = pd.read_csv('traces.csv', sep=',', index_col=0, header=0)
df = traces_original.drop(traces_original.iloc[:, 10:1288], axis=1)
print(df)

df.plot(title='Fluorescence traces of neurons', legend=False, subplots=True, figsize=(7, 7))
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# TASK 2
neuron_4 = df.iloc[:, 4]
print(neuron_4)

neuron_4.plot(title='Fluorescence trace neuron 4', legend=False, subplots=True, figsize=(10, 5))
plt.show()

# Zoom in on first event
neuron_4.plot(title='First event neuron 4', legend=False, subplots=True, figsize=(7, 7), xlim=[4, 8])
plt.show()

# Fit
rise = neuron_4[4.5:5.05]
fall = neuron_4[5.05:6.4]
print(rise)
print(fall)

