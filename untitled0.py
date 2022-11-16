# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 09:28:28 2022

@author: bmohan
"""

import pandas as pd
import numpy as np
from dtaidistance import dtw, dtw_ndim, dtw_visualisation


df1 = pd.read_excel('overall_index.xlsx').set_index('Item').drop([53,54 ,'Total'], axis=1).sort_index(axis=1).reset_index()
df2 = pd.read_excel('season patterns.xlsx','item').set_index('Item code').drop('Season Pattern', axis=1).sort_index(axis=1).reset_index()


x = df2[df2['Item code'].isin(df1['Item'])]
x = x.drop_duplicates('Item code').sort_values('Item code').drop('Item code', axis=1)
x = x.dropna(axis=0, how='all')

y = df1[df1['Item'].isin(df2['Item code'])].sort_values('Item')
copy = y.copy(deep=True)
y = y.drop('Item', axis=1)


arr1 = x.to_numpy()
arr2 = y.to_numpy()


res_arr  = []
for i in range(len(arr1)):
    d = dtw.distance(arr1[i, :], arr2[i, :])
    res_arr.append(d)
    

copy['distance'] = res_arr




def calculate_similarity():

    
    item_pattern = pd.read_excel('rol_index.xlsx')
    seasonal_pattern =  pd.read_excel('season patterns.xlsx','item rest')
    
    item_pattern = item_pattern.set_index('Item').drop([53,54, 'Total'], axis=1).sort_index(axis=1).reset_index()
    seasonal_pattern = seasonal_pattern.set_index('Item code').drop('Season Pattern', axis=1).sort_index(axis=1).reset_index()
    
    in_seasonal_pattern = seasonal_pattern[seasonal_pattern['Item code'].isin(item_pattern['Item'])]
    in_seasonal_pattern = in_seasonal_pattern.drop_duplicates('Item code').sort_values('Item code').drop('Item code', axis=1)

    out_seasonal_pattern = item_pattern[item_pattern['Item'].isin(seasonal_pattern['Item code'])].sort_values('Item')
    out_seasonal_pattern_copy = out_seasonal_pattern.copy(deep=True)
    out_seasonal_pattern = out_seasonal_pattern.drop('Item', axis=1)
    
    arr1 = in_seasonal_pattern.to_numpy()
    arr2 = out_seasonal_pattern.to_numpy()


    res_arr  = []
    for i in range(len(arr1)):
        d = dtw.distance(arr1[i, :], arr2[i, :])
        res_arr.append(d)
        

    out_seasonal_pattern_copy['distance'] = res_arr
    
    return out_seasonal_pattern_copy



df = calculate_similarity()



df1 = pd.read_excel('rol_index.xlsx').set_index('Item').drop([53,54 ,'Total'], axis=1).sort_index(axis=1).reset_index()
df2 = pd.read_excel('season patterns.xlsx','item rest').set_index('Item code').drop('Season Pattern', axis=1).sort_index(axis=1).reset_index()


res = calculate_similarity(df1, df2)
