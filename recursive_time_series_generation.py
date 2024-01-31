# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 21:43:32 2023

@author: tushar
"""

import pandas as pd
import numpy as np


def data_split(data):
    
    l_d=len(data)/12
    split=(int((l_d)/2)-1)
    In_sample=data.iloc[1:split*12+1,:]
    Out_sample=data.iloc[split*12+1:,:]
    In_sample=In_sample[['Snp_excess_rets','BI_excess_rets']]
    Out_sample=Out_sample[['Snp_excess_rets','BI_excess_rets']]
    
    return In_sample, Out_sample

def recursive_approach(data1,data2,txt):
    roll_windw=60
    if txt=='rolling':
        l_in_smpl=len(data1)
        df=pd.concat([data1,data2])
        time_series = (df.rolling(window=roll_windw)).mean()
        time_series_forecast=time_series[l_in_smpl:]
        time_series_forecast.rename(columns={'Snp_excess_rets':'Snp_Benchmark','BI_excess_rets':'BI_Benchmark'},inplace=True)

    else:
        l_in_smpl=len(data1)
        df=pd.concat([data1,data2])
        time_series=(df.expanding(min_periods=l_in_smpl)).mean()
        time_series_forecast=time_series[l_in_smpl:]
        time_series_forecast.rename(columns={'Snp_excess_rets':'Snp_Benchmark','BI_excess_rets':'BI_Benchmark'},inplace=True)
        
    return time_series_forecast

    
