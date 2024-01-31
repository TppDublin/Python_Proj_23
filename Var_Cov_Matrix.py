# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 00:54:19 2023

@author: tushar
"""

import pandas as pd
import numpy as np


def var_covar_forecast(in_data,out_data,txt):
    var_covar_matrix=pd.DataFrame(index=out_data.index,columns=['SnP500_Var','Bond_Index_Var','Covariance'])
    cols=['SnP500_Var','Bond_Index_Var','Covariance']
    matrix=[]
    in_out_data=pd.concat([in_data,out_data])
    if txt=='rolling':
        matrix_roll = (in_out_data.rolling(window=60)).cov(ddof=1)
        for date in out_data.index:
            cov_matrix = matrix_roll.loc[date]
            matrix.append(cov_matrix)

            snp_500_roll_var=cov_matrix.loc['Snp_excess_rets','Snp_excess_rets']
            bond_index_roll_var=cov_matrix.loc['BI_excess_rets','BI_excess_rets']
            SnP_BI_roll_Cov=cov_matrix.loc['Snp_excess_rets','BI_excess_rets']
            
            var_covar_matrix.loc[date] = [snp_500_roll_var,bond_index_roll_var,SnP_BI_roll_Cov]
    else:
        for date in out_data.index:
            in_out_date=in_out_data[:date]
            cov_matrix=in_out_date.cov(ddof=1)
            matrix.append(cov_matrix)
            snp_500_var=cov_matrix.loc['Snp_excess_rets','Snp_excess_rets']
            bond_index_var=cov_matrix.loc['BI_excess_rets','BI_excess_rets']
            SnP_BI_Cov=cov_matrix.loc['Snp_excess_rets','BI_excess_rets']
            
            var_covar_matrix.loc[date] = [snp_500_var,bond_index_var,SnP_BI_Cov]
            
    return var_covar_matrix,matrix
