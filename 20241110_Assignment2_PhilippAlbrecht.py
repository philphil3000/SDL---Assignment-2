#!/usr/bin/env python
# coding: utf-8

# # Space detector laboratory - Assignment 2

# # Import modules

# In[1]:


#main function and entry point not defined due to time constraints 

#import modules 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from numpy import inf as INF
from scipy.stats import linregress
from itertools import islice 


# # User input: detector, source, angle, ROI

# In[2]:


#regions of interest for all peaks at 0° angle; these values were visually identified in plots of the count rate and hard coded 
all_peaks_D2_ROIs = {"S1": [(4, 10), (10, 20), (18, 80)], 
    "S2": [(10, 22), (30, 50), (55, 105), (150, 190)], 
    "S3": [(10, 20), (22, 150), (180, 220), (265, 315)], 
    "S4": [(5, 20), (25, 35)]}

all_peaks_D3_ROIs = {"S1": [(14, 43)], 
    "S2": [(8, 22), (24, 45), (160, 215)], 
    "S3": [(8, 25), (60, 150), (280, 390)], 
    "S4": [(14, 43)]} 

#regions of interest for relevant photopeaks #these values were visually identified in plots of the count rate and hard coded 
D2_ROIs = {"S1": [(18, 80)], 
    "S2": [(30, 50)], 
    "S3": [(265, 315)], 
    "S4": [(25, 35)]}

D3_ROIs = {"S1": [(14, 43)], 
    "S2": [(160, 215)], 
    "S3": [(280, 390)], 
    "S4": [(14, 43)]}

#generate filenames of measurment files based on user input 
def user_input_filename(): 
    filename = "" 
    valid_detectors = ["D1", "D2", "D3", "D4"] 
    valid_sources = ["S1", "S2", "S3", "S4"] 
    valid_angles = ["A00", "A45"]

#each entry except for ROI is compared to valid inputs. user is asked to retry until a valid input is provided. 
    while True:
        detector = input("Select a detector type (D2=NaI, D3=BGO, D1=D4=CdTe): ") #D1 and D4 are the same detector; the difference is the day the measurements were taken 
        if detector in valid_detectors:
            break
        print("Your input is invalid. Please try again.")
    while True: 
        source = input("Select a source (S1=Cobalt60, S2=Barium133, S3=Caesium137, S4=Americium241): ") 
        if source in valid_sources:
            break
        print("Your input is invalid. Please try again.")
    while True: 
        angle = input("Select an angle (A00 = 0°, A45 = 45°): ") 
        if angle in valid_angles:
            break
        print("Your input is invalid. Please try again.")

    ROI_string = input("Select a region of interest (Format: '(XXX, XXX)'): ")
    ROI = eval(ROI_string)

#add the detector-specific filetype 
    if detector == "D3" or detector == "D2": 
        filename = detector + source + angle + ".Spe" 
    elif detector == "D4": 
        filename = detector + source + angle + ".mca" 
    elif detector == "D1": 
        filename = detector + source + angle + ".mca" 
    return filename, detector, source, ROI

filename, detector, source, ROI = user_input_filename()

#CHECK: 
'''
print(f"The detector is {detector}")
print(f"The source is {source}")
print(f"The name of the file is {filename}")
print(f"The ROI is {ROI}")
'''


# # Store measurment values in array

# In[3]:


counts_list = []
channel_list = []

#this section was further developed from assignment 1; ifference: no specific line numbers, but specific start and end word
def create_list(detector, filename):  
    with open(filename, 'r') as file:
        measurement_value = False  
        channel = 0  
#specific words before and after measurement values depending on detector/filetype
        if detector == "D2" or detector == "D3": 
            start_word = "0 1023"
            end_word = "$ROI:"
        elif detector == "D1" or detector == "D4": 
            start_word = "<<DATA>>"
            end_word = "<<END>>"
#add all measurement values between start/end word in list 
        for line in file:
            if start_word in line:
                measurement_value = True
                continue 
            if end_word in line:
                break  
            if measurement_value:
                counts = int(line.strip())
                counts_list.append(counts)
                channel_list.append(channel)
                channel += 1 

    return counts_list 

create_list(detector, filename) 
channel_array = np.array(channel_list) #(x-axis) 
counts_array = np.array(counts_list) #(y-axis) 


#CHECK: 
'''
print(f"The detector is {detector}")
print(f"The source is {source}")
print(f"The name of the file is {filename}")
print(f"The first 20 values of the list are: {counts_array[0:19]}")

plt.figure(figsize=(10, 4))
plt.plot(channel_array, counts_array, label='countrate')  
plt.xlabel('channel') 
plt.ylabel('counts') 
plt.title(f"Total counts by source {source} on detector {detector}") 
plt.ylim(0, None)
plt.xlim(0, None)
plt.grid(True)
plt.show()
'''


# # Extract measurement duration

# In[4]:


#extract measurement duration 

def get_meas_time(detector, filename): 
    meas_time_value = float 
    with open(filename, 'r') as file:
        if detector == "D2" or detector == "D3": 
            time_word = "$MEAS_TIM:"  #for D3
            found_time = False 
            for line in file:
                if time_word in line:
                    found_time = True
                    continue 
                elif found_time: 
                    meas_time_value = float(line.strip().split()[0])  #two integers in the line. only the first one is read for the measurment time 
                    break 
            return meas_time_value

        elif detector == "D1" or detector == "D4": 
            time_word = "REAL_TIME" 
            found_time = False 
            for line in file: 
                if time_word in line: 
                    found_time = True 
                    meas_time_value = float(line.strip().split()[2])
                    break 
    return meas_time_value
            
meas_time_value = get_meas_time(detector, filename)
meas_time_value = int(meas_time_value) 


#CHECK: 
'''
print(f"The detector is {detector}")
print(f"The source is {source}")
print(f"The name of the file is {filename}")
print(f"The measurement duration is {meas_time_value} seconds")
'''


# # Calculate background countrate 

# In[5]:


#using the same code as the measurement extraction value; different filename 

counts_list = []
channel_list = []

def calculate_background_countrate(): 
    if detector == "D1" or detector == "D4": 
        filetype = ".mca" 
    elif detector == "D2" or detector == "D3": 
        filetype = ".Spe" 
    temp_filename = detector + "S1_Background" + filetype #the labeling of the background detection files is not very elegant, but should not be changed in order for all file-names to be consistent 

    create_list(detector, temp_filename) 
    background_counts_array = np.array(counts_list) #(y-axis) 
    countrate_array_entire = []
    countrate_array_entire = background_counts_array / meas_time_value
    background_countrate_array = np.round(countrate_array_entire, 2) 
    
    return background_countrate_array


background_countrate_array = calculate_background_countrate()


#CHECK: 
'''
print(f"The detector is {detector}")
print(f"The source is {source}")
print(f"The name of the file is {filename}")
print(f"The first 20 values of the background count rate are: {background_countrate_array[0:19]}")

plt.figure(figsize=(10, 4))
plt.plot(channel_array, background_countrate_array, label='Measurement data without background') 
plt.xlabel('channel') 
plt.ylabel('counts/s') 
plt.title(f"Countrate of background in detector {detector}") 
#plt.legend()
plt.ylim(0, None)         #change this so that the countrate of the background automatically uses the same scale as the countrate of the detector
plt.xlim(0, None)
plt.grid(True)
plt.show()
'''


# # Calculate total count rate 

# In[6]:


#counts_array divided by measurement time = count rate of entire spectrum
def countrate (counts_array, meas_time_value): 
    countrate_array_entire = []
    countrate_array_entire = counts_array / meas_time_value
    countrate_array = np.round(countrate_array_entire, 4) 
    return countrate_array 

countrate_array = countrate(counts_array, meas_time_value) 


#CHECK: 
'''
print(f"The detector is {detector}")
print(f"The name of the file is {filename}")
print(f"The first 20 values of the count rate are: {countrate_array[0:19]}")

plt.figure(figsize=(10, 4))
plt.plot(channel_array, countrate_array, label='Measurement data without background') 
plt.xlabel('channel') 
plt.ylabel('counts/s') 
plt.title(f"Countrate of source {source} in detector {detector}") 
#plt.legend()
plt.ylim(0, None)         #change this so that the countrate of the background automatically uses the same scale as the countrate of the detector
plt.xlim(0, None)
plt.grid(True)
plt.show()
'''


# # Calculate clean count rate without background

# In[7]:


#subtract the background countrate from the countrate of the measurement 
def subtract_background (countrate_array, background_countrate_array): 
    countrate_array_clean = np.clip(countrate_array - background_countrate_array, a_min=0, a_max=None) #miniumum value is 0 to prevent negative values 
    return countrate_array_clean

countrate_array_clean = subtract_background(countrate_array, background_countrate_array)


#CHECK: 
'''
print(f"The detector is {detector}")
print(f"The name of the file is {filename}")
print(f"The first 20 values of the clean count rate are: {countrate_array_clean[0:19]}")

plt.figure(figsize=(10, 4))
plt.plot(channel_array, countrate_array_clean, label='Measurement data without background') 
plt.xlabel('channel') 
plt.ylabel('counts/s') 
plt.title(f"Countrate array of source {source} in detector {detector} without background") 
#plt.legend()
plt.ylim(0, None)         #change this so that the countrate of the background automatically uses the same scale as the countrate of the detector
plt.xlim(0, None)
plt.grid(True)
plt.show()
'''


# # Plot measurement spectra 

# In[8]:


#plot spectra of total counts, total count rate, background count rate and clean count rate 

def plot_data(ax, channel_array, data_array, xlabel, ylabel, title, xlim=None, ylim=None):
    ax.plot(channel_array, data_array)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, ylim)
    ax.set_xlim(0, xlim)
    ax.grid(True)

max_countrate = max(max(countrate_array), max(background_countrate_array), max(countrate_array_clean))
max_channel= len(channel_array) 

fig, axs = plt.subplots(4, 1, figsize=(10, 16), constrained_layout=True)
plot_data(axs[0], channel_array, counts_array, 'channel', 'counts', f"Total counts by source {source} on detector {detector}", xlim=max_channel)
plot_data(axs[1], channel_array, countrate_array, 'channel', 'counts/s', f"Count rate of source {source} in detector {detector}", xlim=max_channel, ylim=max_countrate)
plot_data(axs[2], channel_array, background_countrate_array, 'channel', 'counts/s', f"Count rate of background in detector {detector}", xlim=max_channel, ylim=max_countrate)
plot_data(axs[3], channel_array, countrate_array_clean, 'channel', 'counts/s', f"Count rate array of source {source} in detector {detector} excluding background", xlim=max_channel, ylim=max_countrate)
plt.show()


# # Filter values from ROI 

# In[9]:


#extract values within a defined range. the two functions were copied from the Brightspace document 'fitting-to-spectra-2102'
#Boolean mask with value True for x in [xmin, xmax) 
def in_interval(x, xmin=-INF, xmax=INF):     
    _x = np.asarray(x) 
    return np.logical_and(xmin <= _x, _x < xmax) 

def filter_in_interval(x, y, xmin, xmax):
    _mask = in_interval(x, xmin, xmax)     #Selects only elements of x and y where xmin <= x < xmax.
    return [np.asarray(x)[_mask] for x in (x, y)]   

_channel_array, _countrate_array_clean = filter_in_interval(channel_array, countrate_array_clean, *ROI) #adjusted from the original code to channel_array_clean and counts_array_clean 


#Check 
'''
plt.scatter(_channel_array, _countrate_array_clean, label='Measurement data') 
plt.xlabel('channel') 
plt.ylabel('counts') 
plt.title('Measurement data') 
plt.legend() 
plt.show() 

print(len(channel_array), len(countrate_array_clean), len(_channel_array), len(_countrate_array_clean)) 
print(f"The detector is {detector}")
print(f"The source is {source}")
print(f"The name of the file is {filename}")
'''


# # Fit gaussian curve to ROI and output parameters and covariance 

# In[10]:


#fit gaussian curve to region of interest 

def gaussian_function (x, mu, sig, a): 
    TWO_PI = np.pi * 2
    return a * np.exp(-0.5 * (x-mu)**2 / sig**2) / np.sqrt(TWO_PI * sig**2)

a1 = np.max(_channel_array)      
a2 = np.std(_channel_array)     
a3 = area = np.trapz(_countrate_array_clean, _channel_array) 
peak_channel = np.argmax(_channel_array)   # this outputs the channel of the peak 

initial_guess = [a1, a2, a3] 
popt, pcov = curve_fit(gaussian_function, _channel_array, _countrate_array_clean, p0=initial_guess) 


fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(channel_array, countrate_array_clean, s=1) #, label='Measurement data'
plt.title(f"Photopeak of source {source} in detector {detector}") 
plt.plot(channel_array, gaussian_function(channel_array, *popt), color='red') #, label='Fitting function'
plt.xlabel('channel') 
plt.ylabel('counts/s') 
plt.xlim(0, len(channel_array))
ax.grid(True)


parameters_text2 = ( f"Mean value: {popt[0]:.2f} \n" 
                    f"Standard deviation: {popt[1]:.2f} \n" 
                    f"Area: {popt[2]:.2f}\n" ) 
ax.text( 0.95, 0.95, parameters_text2, transform=ax.transAxes, 
        fontsize=10, verticalalignment='top', horizontalalignment='right', 
        bbox=dict(facecolor='white', alpha=1))
plt.show()


#CHECK 
print(f"popt are as follows: {popt}")
print(f"pcov are as follows: {pcov}")
print(filename, detector, source)


# # PART B: This section relies on hardcoded values and was used manually to gain the plots relevant for the report 

# # Output peak channels for all sources of one detector 

# In[11]:


#these dictionaries should contain one peak energy channel for each ROI defined in the ROI dictionaries 
D1_Peaks = {}
D2_Peaks = {}
D3_Peaks = {}
D4_Peaks = {}

if detector == "D2": 
    sourcedict_name = D2_ROIs
    peakdict_name = D2_Peaks
elif detector == "D3": 
    sourcedict_name = D3_ROIs
    peakdict_name = D3_Peaks
elif detector == "D1": 
    sourcedict_name = D1_ROIs
    peakdict_name = D1_Peaks
elif detector == "D4": 
    sourcedict_name = D4_ROIs
    peakdict_name = D4_Peaks

#return the channel of the photopeak maximum for each predefined ROI of each source 
#for this section ChatGPT was consulted multiple times; no entire block or line was copied directly 
for source in sourcedict_name: 
    peak_list=[] 
    for ROI in sourcedict_name[source]: 
        _channel_array, _countrate_array_clean = filter_in_interval(channel_array, countrate_array_clean, *ROI)
        local_peak_index = np.argmax(_countrate_array_clean)  
        peak_channel = _channel_array[local_peak_index] 
        peak_list.append(peak_channel) 
    peakdict_name[source] = peak_list    #add the peak channel to the source-specific list in the dictionary 
    peak_list = [] 
    
#CHECK: 
'''
print(peakdict_name) 
'''


# # Plot peak emission energy to channel and fit linear function 

# In[12]:


# peak emission energy values from documentation 
energy_values = {
    'S1': 1332.492,  # Energy of Cobalt 
    'S2': 356.0129,  # Energy of Barium  
    'S3': 661.657,   # Energy of Cesium  
    'S4': 59.5409    # Energy of Americium  
}

only_fitted_peaks = {'S2': [160], 'S3': [280], 'S4': [26]} #manual input. should be derived from function in the code above

x_values = []
y_values = []

for source, x_vals in only_fitted_peaks.items():
    x_values.extend(x_vals)
    y_values.extend([energy_values[source]] * len(x_vals))

#ChatGPT was consulted for the linear fit function 
slope, intercept, r_value, p_value, std_err = linregress(x_values, y_values)

fit_x = range(min(x_values), max(x_values) + 1)  
fit_y = [slope * x + intercept for x in fit_x]   

plt.figure(figsize=(10, 6))

# Scatter plot of original data points
for source, x_vals in only_fitted_peaks.items():
    y = energy_values[source]
    plt.scatter(x_vals, [y] * len(x_vals), label=source)  

plt.plot(fit_x, fit_y, color='red', label=f'Linear Fit: y = {slope:.2f}x + {intercept:.2f}') #This line was copied from Microsoft Copilot 

# Adding labels and legend
plt.xlabel('channel')
plt.ylabel('energy (keV)')
plt.title(f"Calibration curve of photopeaks for detector {detector}")
plt.legend(title="Sources")
plt.xlim(0, None) 
plt.show()


# # Output values for report 

# In[13]:


counts_total = np.sum(counts_array)
countrate_total = counts_total / meas_time_value

print (f"Measurment duration: {meas_time_value}") 
print (f"Counts: {counts_total}") 
print (f"Countrate: {countrate_total}") 
print (f"Filename: {filename}") 


# # Plot and curve-fit the resolution 

# In[14]:


#values were hard-coded due to time constraints. they should be obtained from function further up 

#energy values of peaks S1 to S4 
energy_values = np.array([1173.228, 356.0129, 661.657, 59.5409])

#energy resolution of D3 S1 to S4 
energy_resolution_D3 = np.array([0.00960, 0.08710, 0.05961, 0.21491])
energy_resolution_D2 = np.array([0.04143, 0.09097, 0.05851, 0.15334])
source_labels = ['S1', 'S2', 'S3', 'S4']
colors = ['red', 'blue', 'orange', 'green']  

energy_resolution = energy_resolution_D3    # manual input due to time constraints. should be based on the detector selection by user  

# function from script 
def fitting_function(energy, a, b, c):
    return a * energy**-2 + b * energy**-1 + c

#the code for fitting the function was copied from ChatGPT 
resolution_squared = energy_resolution**2
params, covariance = curve_fit(fitting_function, energy_values, resolution_squared)
a, b, c = params
print(f"Model fit function: resolution^2 = {a:.5e} * energy^-2 + {b:.5e} * energy^-1 + {c:.5e}")
energy_fit_values = np.linspace(min(energy_values), max(energy_values), 100)
resolution_squared_fit_values = fitting_function(energy_fit_values, a, b, c)
resolution_fit_values = np.sqrt(resolution_squared_fit_values)  


plt.figure(figsize=(10, 6))
for i in range(len(energy_values)):
    plt.scatter(energy_values[i], energy_resolution[i], color=colors[i], label=source_labels[i], s=100)
plt.plot(energy_fit_values, resolution_fit_values, label='Model Fit', color='red')


plt.xlabel('photopeak energy (keV)')
plt.ylabel('resolution')
plt.title(f"Photopeak energy with corresponding resolution for detector {detector}")
plt.grid(True)
plt.legend()
plt.show() 

print(f"Model fit function: resolution^2 = {a:.5e} * energy^-2 + {b:.5e} * energy^-1 + {c:.5e}")
print(f"Fitting parameters:\n a = {a:.5e}\n b = {b:.5e}\n c = {c:.5e}")


# # Plot efficency 

# In[15]:


#values were hard-coded due to time constraints. they should be obtained from function further up 

energy_values = np.array([1173.228, 356.0129, 661.657, 59.5409]) 
intrinsic_efficiency_D3 = np.array([57.144, 0.900, 0.301, 0.144]) 
intrinsic_efficiency_D2 = ([2.658, 0.637, 0.204, 0.081]) 

intrinsic_efficiency = intrinsic_efficiency_D3    # manual input due to time constraints. should be based on the detector selection by user

source_labels = ['S1', 'S2', 'S3', 'S4']
colors = ['red', 'blue', 'orange', 'green']  

plt.figure(figsize=(10, 6))
for i in range(len(energy_values)):
    plt.scatter(energy_values[i], intrinsic_efficiency[i], color=colors[i], label=source_labels[i])

plt.xlabel("photopeak energy (keV)")
plt.ylabel("intrinsic efficiency")
plt.title(f"Intrinsic efficiency in detector {detector} with corresponding photopeak energy")
plt.legend(title="Sources")
plt.grid(True)
plt.show()

