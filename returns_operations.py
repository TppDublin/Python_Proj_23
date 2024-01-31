# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 06:21:28 2023

@author: tushar
"""

import pandas as pd 
import numpy as np
from scipy.stats import skew, kurtosis as kur


def Returns_ops(data):
    
    
    SnP_rets = ((data['SnP']/data['SnP'].shift(1))-1)*12
    data['Snp_excess_rets'] = SnP_rets- data['Rf']*12
    BI_rets = ((data['BI']/data['BI'].shift(1))-1)*12
    data['BI_excess_rets'] = BI_rets - data['Rf']*12
    
    return data

def stats_calcs(data):
    
    ann_mean=data.mean()
    vol = data.std()
    S_R= ann_mean/vol
    skewness=data.skew()
    kurtosis=data.kurt()
    
    summary_table = pd.DataFrame({
        'Statistic': ['Annualized Mean', 'Annualized Volatility', 'Annualized Sharpe Ratio', 'Skewness', 'Kurtosis'],
        'Value': [ann_mean, vol, S_R, skewness, kurtosis]
    })
    
    return summary_table
        
