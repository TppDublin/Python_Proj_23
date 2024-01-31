# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 02:45:08 2023

@author: tushar
"""
import pandas as pd
import numpy as np


def generate_data_wts(data1,data2,var_covar_matrix):
    excess_Pflio_rets = pd.DataFrame()
    wts=[]
    global data_index
    data_index=data1.index
    print("shape of the data is :\n",data1.shape[1])
    
    for i in range(0,data1.shape[1]):
        sig = pd.concat([data1.iloc[:,i],data2.iloc[:,i]],axis=1)
        #print("after concat Sigma is:\n",sig)
        means,weights= tangency_pfolio(sig,var_covar_matrix,i)
        if data1.columns[i] == 'Combined' or data1.columns[i] == 'Snp_Benchmark':
            wts.append(weights)
        arr = means
        if i == 0:
            excess_Pflio_rets = arr
        else:
            excess_Pflio_rets = pd.concat([excess_Pflio_rets,arr],axis=1)
        
    return excess_Pflio_rets,wts
    
def tangency_pfolio(avg_pred,variance, count = -1):
    column_pred = ['b/m','tbl','ntis','infl','Dfy','Combined','Benchmark']
    mean = []
    var = []
    sharpe_ratio = []
    mu = np.array(avg_pred)
    sigma = np.array(variance)
    e = np.ones(mu.shape[1]).reshape(mu.shape[1],1)
    weight = []
    for i in range(0, mu.shape[0]):
        Sigma_inv = np.linalg.inv(sigma[i])
        mu_s = mu[i].reshape(mu.shape[1],1) 
        num = np.dot(Sigma_inv,mu_s)
        dino = np.dot(np.dot(e.T,Sigma_inv), mu_s)
        weight.append((num/dino))
        
    # Finding stats
    for i in range(0,mu.shape[0]):
        a = np.dot(weight[i].T,mu[i].reshape(mu.shape[1],1))
        mean.append(a[0])

        b = np.dot(np.dot(weight[i].T,sigma[i]),weight[i])
        var.append(b[0])

        c = a[0]/(np.sqrt(b[0]))
        sharpe_ratio.append(c)
    
    if count >=0:
          mean = pd.DataFrame(mean, columns= [column_pred[count]+'_Excess_Return'],index=data_index)
   
    return mean,weight

def portfolio_summary_stat(data):
    #cols = []
    stats = pd.DataFrame(index=['Mean','Volatilty','Sharpe_Ratio'])
    for i in data.columns:
        mean = data[i].mean()
        vol = (data[i].std(ddof=1))**2
        S_R = mean/(data[i].std(ddof=1))

        stats.loc['Mean',i] = mean
        stats.loc['Volatilty',i] = vol
        stats.loc['Sharpe_Ratio',i] = S_R

    return stats

def time_series_data(array):
    # Weight_benchmark = array.iloc[1]
    # Weight_combined =  array.iloc[0]
    Wts = pd.DataFrame(columns = ['Snp500_Wts','Bond_Wts'],index=data_index) 
    for i in range(0,len(array)):
        #l_arr=len(array[i])
        for j in range(0,len(array[i])):
            arr_wts = np.array(array[i][j])
            Wts.iloc[i,j] = arr_wts[0]
    return Wts

def graph_plots_wts(wts,stats_df):
    time_series_wts_rets=pd.DataFrame()
    time_series_wts_rets_benchmark=wts
    time_series_wts_rets_combined=wts
    time_series_wts_rets_benchmark['Benchmark_rets']=stats_df['Benchmark_Excess_Return']
    #print("series for benchmark is:\n",time_series_wts_rets_benchmark)
    time_series_wts_rets_combined['Combined_rets']=stats_df['Combined_Excess_Return']
    #print("series for combined is:\n",time_series_wts_rets_combined)
    time_series_wts_rets_benchmark.plot()
    time_series_wts_rets_combined.plot()
    
    return 0



