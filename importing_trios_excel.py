# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 09:56:38 2021

@author: rmcgorty
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import sys
from scipy import ndimage
from skimage import measure

import os
import glob #glob is helpful for searching for filenames or directories
import scipy #scientific python
import pickle #for saving data
from pathlib import Path

import openpyxl #can be installed with 'conda install -c anaconda openpyxl'
from openpyxl import load_workbook

import xarray as xr

from scipy.interpolate import interp1d, splprep, splev
import scipy.interpolate as spi


def read_timeandtempseries_data(workbook, temperature, geom, date, last_row_nums = [64,259,478,259,1425,1425]):
    time_swp_cols=['A','B','C','D','E','F','G','H','I','J']
    temp_swp_cols=['A','B','C','D','E','F','G','H','I','J','K']
    
    variables_time = ['angular_freq','storage_modulus','loss_modulus','complex_visc',
                      'tan_delta','step_time', 'temperature','raw_phase',
                      'osc_displacement','time']
    variables_temp = ['storage_modulus','loss_modulus','tan_delta','angular_freq',
                      'osc_torque','step_time', 'temperature','raw_phase',
                      'osc_displacement','osc_strain','time']
    
    ###### Names of sheets to read:
    timesweep_1 = 'Time sweep - 5'
    tempramp_2 = 'Temperature ramp - 6'
    timesweep_3 = 'Time sweep - 7'
    tempramp_4 = 'Temperature ramp - 8'
    timesweep_5 = 'Time sweep - 9'
    timesweep_6 = 'Time sweep - 15'
    
    ###### Rows
    rows_timesweep_1 = np.arange(4,last_row_nums[0],dtype=int)
    rows_tempramp_2 = np.arange(4,last_row_nums[1],dtype=int)
    rows_timesweep_3 = np.arange(4,last_row_nums[2],dtype=int)
    rows_tempramp_4 = np.arange(4,last_row_nums[3],dtype=int)
    rows_timesweep_5 = np.arange(4,last_row_nums[4],dtype=int)
    rows_timesweep_6 = np.arange(4,last_row_nums[5],dtype=int)
    
    ################# Read the first time sweep ###################
    data_nparray_1 = np.zeros((len(time_swp_cols),len(rows_timesweep_1)))
    for i,row_num in enumerate(rows_timesweep_1):
        for j,col_lett in enumerate(time_swp_cols):
            data_nparray_1[j,i] = workbook[timesweep_1]['%s%i' % (col_lett, row_num)].value
            
    ################# Read the first temp up-ramp ###################
    data_nparray_2 = np.zeros((len(temp_swp_cols),len(rows_tempramp_2)))
    for i,row_num in enumerate(rows_tempramp_2):
        for j,col_lett in enumerate(temp_swp_cols):
            data_nparray_2[j,i] = workbook[tempramp_2]['%s%i' % (col_lett, row_num)].value
            
    ################# Read the second time sweep ###################
    data_nparray_3 = np.zeros((len(time_swp_cols),len(rows_timesweep_3)))
    for i,row_num in enumerate(rows_timesweep_3):
        for j,col_lett in enumerate(time_swp_cols):
            data_nparray_3[j,i] = workbook[timesweep_3]['%s%i' % (col_lett, row_num)].value  
            
    ################# Read the  temp down-ramp   ###################
    data_nparray_4 = np.zeros((len(temp_swp_cols),len(rows_tempramp_4)))
    for i,row_num in enumerate(rows_tempramp_4):
        for j,col_lett in enumerate(temp_swp_cols):
            data_nparray_4[j,i] = workbook[tempramp_4]['%s%i' % (col_lett, row_num)].value
            
    ################# Read the third time sweep ###################
    data_nparray_5 = np.zeros((len(time_swp_cols),len(rows_timesweep_5)))
    for i,row_num in enumerate(rows_timesweep_5):
        for j,col_lett in enumerate(time_swp_cols):
            data_nparray_5[j,i] = workbook[timesweep_5]['%s%i' % (col_lett, row_num)].value  

    ################# Read the final time sweep ###################
    data_nparray_6 = np.zeros((len(time_swp_cols),len(rows_timesweep_6)))
    for i,row_num in enumerate(rows_timesweep_6):
        for j,col_lett in enumerate(time_swp_cols):
            data_nparray_6[j,i] = workbook[timesweep_6]['%s%i' % (col_lett, row_num)].value  
            
            
    ################# Rearranging arrays ##########################
    new_data_2 = np.zeros((len(time_swp_cols), data_nparray_2.shape[1]))
    new_data_2[0,:] = data_nparray_2[3,:]
    new_data_2[1,:] = data_nparray_2[0,:]
    new_data_2[2,:] = data_nparray_2[1,:]
    new_data_2[3,:] = 0*data_nparray_2[0,:]
    new_data_2[4,:] = data_nparray_2[2,:]
    new_data_2[5,:] = data_nparray_2[5,:]
    new_data_2[6,:] = data_nparray_2[6,:]
    new_data_2[7,:] = data_nparray_2[7,:]
    new_data_2[8,:] = data_nparray_2[8,:]
    new_data_2[9,:] = data_nparray_2[10,:]
    
    new_data_4 = np.zeros((len(time_swp_cols), data_nparray_4.shape[1]))
    new_data_4[0,:] = data_nparray_4[3,:]
    new_data_4[1,:] = data_nparray_4[0,:]
    new_data_4[2,:] = data_nparray_4[1,:]
    new_data_4[3,:] = 0*data_nparray_4[0,:]
    new_data_4[4,:] = data_nparray_4[2,:]
    new_data_4[5,:] = data_nparray_4[5,:]
    new_data_4[6,:] = data_nparray_4[6,:]
    new_data_4[7,:] = data_nparray_4[7,:]
    new_data_4[8,:] = data_nparray_4[8,:]
    new_data_4[9,:] = data_nparray_4[10,:]
    
    
    ################ Combine arrays ###############################
    combined_data = np.hstack((data_nparray_1, new_data_2, data_nparray_3, new_data_4, data_nparray_5, data_nparray_6))
    combined_rows = np.arange(0,combined_data.shape[1])
    
    da = xr.DataArray(combined_data, coords=[variables_time, combined_rows], dims=['variable','index'])
    da.attrs['temperature'] = temperature
    da.attrs['geometry'] = geom
    da.attrs['date'] = date
    
    return da
    


def read_timeswp_data(sheet, temperature, geom, date, cols=['A','B','C','D','E','F','G','H','I','J'], rows=np.arange(4,64,dtype=int)):
    data_nparray = np.zeros((len(cols),len(rows)))
    for i,row_num in enumerate(rows):
        for j,col_lett in enumerate(cols):
            data_nparray[j,i] = sheet['%s%i' % (col_lett, row_num)].value
    variables = ['angular_freq','storage_modulus','loss_modulus','complex_visc','tan_delta','step_time',
                'temperature','raw_phase','osc_displacement','time']
    da = xr.DataArray(data_nparray, coords=[variables, rows-rows[0]], dims=['variable','index'])
    da.attrs['temperature'] = temperature
    da.attrs['geometry'] = geom
    da.attrs['date'] = date
    return da


def read_up_ramp_data(sheet, shear_rate, date, cols=['A','B','C','D','E','F'], rows=np.arange(4,1324,dtype=int)):
    data_nparray = np.zeros((len(cols),len(rows)))
    for i,row_num in enumerate(rows):
        for j,col_lett in enumerate(cols):
            data_nparray[j,i] = sheet['%s%i' % (col_lett, row_num)].value
    variables = ['stress','shear_rate','viscosity','step_time','temperature','normal_stress']
    da = xr.DataArray(data_nparray, coords=[variables, rows-rows[0]], dims=['variable','index'])
    da.attrs['shear_rate'] = shear_rate
    da.attrs['true_shear_rate'] = np.mean(data_nparray[1,:])
    da.attrs['date'] = date
    return da

def read_down_ramp_data(sheet, shear_rate, date, cols=['A','B','C','D','E','F'], rows=np.arange(4,514,dtype=int)):
    data_nparray = np.zeros((len(cols),len(rows)))
    for i,row_num in enumerate(rows):
        for j,col_lett in enumerate(cols):
            data_nparray[j,i] = sheet['%s%i' % (col_lett, row_num)].value
    variables = ['stress','shear_rate','viscosity','step_time','temperature','normal_stress']
    da = xr.DataArray(data_nparray, coords=[variables, rows-rows[0]], dims=['variable','index'])
    da.attrs['shear_rate'] = shear_rate
    da.attrs['true_shear_rate'] = np.mean(data_nparray[1,:])
    da.attrs['date'] = date
    return da

def read_peakhold_data(sheet, shear_rate, date, cols=['A','B','C','D','E','F'], rows=np.arange(4,1804,dtype=int)):
    data_nparray = np.zeros((len(cols),len(rows)))
    for i,row_num in enumerate(rows):
        for j,col_lett in enumerate(cols):
            data_nparray[j,i] = sheet['%s%i' % (col_lett, row_num)].value
    variables = ['stress','shear_rate','viscosity','step_time','temperature','normal_stress']
    da = xr.DataArray(data_nparray, coords=[variables, rows-rows[0]], dims=['variable','index'])
    da.attrs['shear_rate'] = shear_rate
    da.attrs['true_shear_rate'] = np.mean(data_nparray[1,:])
    da.attrs['date'] = date
    return da

def read_freqswp_data(sheet, temperature, geom, date, cols=['A','B','C','D','E','F','G','H','I','J','K'], rows=np.arange(4,45,dtype=int)):
    data_nparray = np.zeros((len(cols),len(rows)))
    for i,row_num in enumerate(rows):
        for j,col_lett in enumerate(cols):
            data_nparray[j,i] = sheet['%s%i' % (col_lett, row_num)].value
    variables = ['storage_modulus','loss_modulus','tan_delta','angular_freq','osc_torque','step_time',
                'temperature','raw_phase','osc_displacement','complex_visc','time']
    da = xr.DataArray(data_nparray, coords=[variables, rows-rows[0]], dims=['variable','index'])
    da.attrs['temperature'] = temperature
    da.attrs['geometry'] = geom
    da.attrs['date'] = date
    return da

def read_ampswp_data(sheet, shear_rate, date, cols=['A','B','C','D','E','F','G','H','I','J'], rows=np.arange(4,25,dtype=int)):
    data_nparray = np.zeros((len(cols),len(rows)))
    for i,row_num in enumerate(rows):
        for j,col_lett in enumerate(cols):
            data_nparray[j,i] = sheet['%s%i' % (col_lett, row_num)].value
    variables = ['storage_modulus','loss_modulus','tan_delta','angular_freq','osc_torque','step_time',
                'temperature','raw_phase','osc_displacement','osc_strain']
    da = xr.DataArray(data_nparray, coords=[variables, rows-rows[0]], dims=['variable','index'])
    da.attrs['prior_shear_rate'] = shear_rate
    da.attrs['date'] = date
    return da

def area_bewteen_temp_ramps(upramp, downramp, shearrate, logvar=True, lower_temp=23, upper_temp = 33.9,
                            up_use_letter_in_key=None):
    if up_use_letter_in_key is not None:
        shearrate_up = '%s%s' % (shearrate, up_use_letter_in_key)
    else:
        shearrate_up = shearrate
    temperatures_up = upramp[shearrate_up].loc['temperature'].values
    x1 = np.arange(0,len(temperatures_up))
    bspline_coeffs_up, u_params1 = splprep([x1, temperatures_up], k = 3)
    interpolated_temp_upramp = splev(u_params1, bspline_coeffs_up)
    
    temperatures_down = downramp[shearrate].loc['temperature'].values
    x2 = np.arange(0,len(temperatures_down))
    bspline_coeffs_down, u_params2 = splprep([x2, temperatures_down], k = 5)
    interpolated_temp_downramp = splev(u_params2, bspline_coeffs_down)
    
    if logvar:
        var_up = np.log(upramp[shearrate_up].loc['stress'])
        var_down = np.log(downramp[shearrate].loc['stress'])
    else:
        var_up = upramp[shearrate_up].loc['stress']
        var_down = downramp[shearrate].loc['stress']
    
    interp1d_upramp_func = interp1d(interpolated_temp_upramp[1], var_up)
    interp1d_downramp_func = interp1d(interpolated_temp_downramp[1], var_down)
    
    new_temperature_spacing = 0.05 #set increment of temperature
    new_temps = np.arange(lower_temp,upper_temp,new_temperature_spacing)
    
    
    area_between_curves = np.sum((interp1d_downramp_func(new_temps) - interp1d_upramp_func(new_temps)) * new_temperature_spacing)
    return area_between_curves