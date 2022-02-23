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

def read_freqswp_data(sheet, shear_rate, date, cols=['A','B','C','D','E','F','G','H','I'], rows=np.arange(4,20,dtype=int)):
    data_nparray = np.zeros((len(cols),len(rows)))
    for i,row_num in enumerate(rows):
        for j,col_lett in enumerate(cols):
            data_nparray[j,i] = sheet['%s%i' % (col_lett, row_num)].value
    variables = ['storage_modulus','loss_modulus','tan_delta','angular_freq','osc_torque','step_time',
                'temperature','raw_phase','osc_displacement']
    da = xr.DataArray(data_nparray, coords=[variables, rows-rows[0]], dims=['variable','index'])
    da.attrs['prior_shear_rate'] = shear_rate
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