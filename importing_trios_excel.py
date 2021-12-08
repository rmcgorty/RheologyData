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