# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 07:13:53 2023

@author: tushar
"""
import pandas as pd
import numpy as np
import scipy
import statsmodels.api as sm
from scipy.stats import ttest_rel
from statsmodels.regression.rolling import RollingOLS

def data_cleaning(data1,data2):
    pred_data=data1.dropna()
    l_pd=len(pred_data)
    date=data2.index
    cnt=0
    for i in pred_data['yyyymm']:
        if i ==197912.0 :
            break
        else:
            cnt=cnt+1
    pred_data=pred_data.iloc[cnt:l_pd -12,:]
    pred_data['Dates']=date
    pred_data.reset_index(drop=True,inplace=True)
    pred_data.set_index('Dates', inplace=True)
    pred_data['Dfy'] = pred_data['BAA'] - pred_data['AAA']
    pred_data = pred_data.drop(['AAA','BAA'],axis = 1)

    return pred_data

def pred_data_split(data):
    
    l_d=len(data)/12
    split=(int((l_d)/2)-1)
    In_pred=(data.iloc[1:split*12+1,:])*12
    Out_pred=(data.iloc[split*12+1:,:])*12
    
    return In_pred,Out_pred

def ols_regression(in_predictor,out_predictor,in_sample,out_sample,col,txt):
    roll_windw=60
    forecast = []
    if txt=='rolling':
        y_roll = pd.concat([in_sample,out_sample])
        X_roll = pd.concat([in_predictor,out_predictor])
        X_roll = sm.add_constant(X_roll)
        model_roll = RollingOLS(y_roll,X_roll,window=roll_windw).fit()
        #params gives intercept and Beta value (is an inbuilt func)
        parameters = model_roll.params.copy()
        coeff_beta = parameters[len(in_sample):]
        for date in out_sample.index:
            a = out_predictor[date:date]
            predict_roll = coeff_beta.loc[date,'const'] + a*coeff_beta.loc[date,col]
            forecast.append(predict_roll.item())
            # pred_roll = model_roll.predict(X_roll[date:date])[0]
            # forecast.append(pred_roll)
    else:
         for date in out_sample.index:
             y = pd.concat([in_sample,out_sample[:date]])
             X = pd.concat([in_predictor,out_predictor[:date]])
             X = sm.add_constant(X)
             model = sm.OLS(y,X).fit()
     
             pred = model.predict(X[date:date])[0]
             forecast.append(pred)
    return forecast

def forecast_ols(in_pred,out_pred,in_smp,out_smp,txt):
    
    text = ['BOND','SNP500']
    bond_df = pd.DataFrame()
    SnP500_df = pd.DataFrame()
    date = out_smp.index
    
    for cols in in_pred.columns:
        forecast_snp= ols_regression(in_pred[cols],out_pred[cols],in_smp['Snp_excess_rets'],out_smp['Snp_excess_rets'],cols,txt)
        SnP500_df[cols] = forecast_snp
    SnP500_df.drop(columns=['yyyymm'],inplace=True)
    SnP500_df['Combined'] = SnP500_df.mean(axis=1)
    SnP500_df['Dates'] = date
    SnP500_df.reset_index(drop = True, inplace= True)
    SnP500_df.set_index('Dates', inplace= True)
    
    
    for cols in in_pred.columns:
        forecast_bond= ols_regression(in_pred[cols],out_pred[cols], in_smp['BI_excess_rets'],out_smp['BI_excess_rets'],cols,txt)
        bond_df[cols] = forecast_bond
    bond_df.drop(columns=['yyyymm'],inplace=True)
    bond_df['Combined'] = bond_df.mean(axis=1)
    bond_df['Dates'] = date
    bond_df.reset_index(drop = True, inplace= True)
    bond_df.set_index('Dates', inplace= True)
    
    return SnP500_df,bond_df

def Msfe(pred, out, asset):
    M_sq= []
    ratio=[]
    global col 
    col_err= []
    col_ratio= []
    if asset == 'Sp500':
        for cols in pred.columns:
            col_err.append(cols)
            M_sq.append(((pred[cols] - out['Snp_excess_rets'])**2).mean())
    else:
        for cols in pred.columns:
            col_err.append(cols)
            M_sq.append(((pred[cols] - out['BI_excess_rets'])**2).mean())
    Mean_sq_fcst_err = pd.Series(M_sq,index=col_err)

    return Mean_sq_fcst_err

def MSFE_Ratio(data,asset):
    ratio = []
    col = []
    if asset=='Sp500':
        for cols in data.index:
            col.append(cols)
            x=data[cols] / data['Snp_Benchmark']
            ratio.append(x)
    else :
        for cols in data.index:
            col.append(cols)
            x=data[cols] / data['BI_Benchmark']
            ratio.append(x)
        
    Ratio = pd.DataFrame(ratio,index=col)
    return Ratio

# Diebold and Mariano Test
def Diebold_Mariano(pred,data,text='Sp500'):
    p_val= []
    col = []
    if text == 'Sp500':
        for cols in pred.columns:
            col.append(cols)
            tstat,p_value = ttest_rel((pred[cols] - data['Snp_excess_rets']),(pred['Snp_Benchmark'] - data['Snp_excess_rets']))
            p_val.append(p_value)
    else:
        for cols in pred.columns:
            col.append(cols)
            tstat,p_value = ttest_rel((pred[cols] - data['BI_excess_rets']),(pred['BI_Benchmark'] - data['BI_excess_rets']))
            p_val.append(p_value)
    prob_val = pd.Series(p_val,index=col) 

    return prob_val