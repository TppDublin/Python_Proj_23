# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 04:08:26 2023

@author: tushar
"""

import numpy as np
import pandas as pd
from Data_Load_Creation import Data_Creation
from returns_operations import Returns_ops,stats_calcs
from recursive_time_series_generation import data_split,recursive_approach
from OLS import data_cleaning,pred_data_split,ols_regression,forecast_ols,Msfe,MSFE_Ratio,Diebold_Mariano
from Var_Cov_Matrix import var_covar_forecast
from tangency_pfolio_calc_stats import tangency_pfolio,generate_data_wts,portfolio_summary_stat,time_series_data,graph_plots_wts
pd.options.mode.chained_assignment = None  # default='warn'




SP_500 = pd.read_csv('S&P_500_stock_index.csv')
bond_index = pd.read_csv('US_Aggregate_Bond_index.csv')
risk_free_rtn = pd.read_csv('Risk-free_rate_of_return.csv')
Predictor_data= pd.read_csv('Predictor_data.csv')
print("S&P 500 Index is :\n",SP_500)
print("Bond index is :\n",bond_index)
print("Risk Free Return is :\n",risk_free_rtn)

Merged_df=Data_Creation(risk_free_rtn,SP_500, bond_index)
print("Final Dataframe is:\n",Merged_df)

Rets_df=Returns_ops(Merged_df)
print("Returns Dataframe is:\n",Rets_df)

Snp_Statistics = stats_calcs(Rets_df['Snp_excess_rets'])
print("Statistics for SnP 500 :\n",Snp_Statistics)

BI_Statistics = stats_calcs(Rets_df['BI_excess_rets'])
print("Statistics for Bond Index :\n",BI_Statistics)

In_sample,Out_sample= data_split(Rets_df)
print("In Sample is :\n",In_sample,"Out sample is:\n",Out_sample)

# =============================================================================
#                                  RECURSIVE APPROACH
# =============================================================================


time_series_rec=recursive_approach(In_sample, Out_sample,'rec')
print("Time Series for recursive approach is:\n",time_series_rec)

data_prep=data_cleaning(Predictor_data,Rets_df)
print("The predictor data is :\n",data_prep)

In_Predictor,Out_Predictor=pred_data_split(data_prep)
print("In predictor,:\n",In_Predictor,"Out Predictor,:\n",Out_Predictor)

Snp_ols,Bond_ols=forecast_ols(In_Predictor,Out_Predictor,In_sample,Out_sample,'rec')
print("Snp OlS forecast is :\n ",Snp_ols.head(5),"\nBond OLS forecast is:\n",Bond_ols.head(5))

Snp_forecast_data=pd.merge(Snp_ols,time_series_rec['Snp_Benchmark'],left_index=True,right_index=True,how='left')
Bond_forecast_data=pd.merge(Bond_ols,time_series_rec['BI_Benchmark'],left_index=True,right_index=True,how='left')

SnP500_Msfe= Msfe(Snp_forecast_data,Out_sample,'Sp500')
MSFE_BOND=Msfe(Bond_forecast_data,Out_sample,'Bond')

print("***************************")
print("***************************")
print("MSFE for SNP500 is :\n",SnP500_Msfe)

print("***************************")
print("***************************")
print("MSFE for Bond Index is :\n",MSFE_BOND)


SnP500_Msfe_Ratios= MSFE_Ratio(SnP500_Msfe,'Sp500')
MSFE_BOND_Ratios=MSFE_Ratio(MSFE_BOND,'Bond')
print("***************************")
print("***************************")
print("MSFE Ratios for SNP500 is :\n",SnP500_Msfe_Ratios)

print("***************************")
print("***************************")
print("MSFE Ratios for Bond Index is :\n",MSFE_BOND_Ratios)

SnP_DnB=Diebold_Mariano(Snp_forecast_data,Out_sample,'Sp500')
BI_DnB=Diebold_Mariano(Bond_forecast_data,Out_sample,'Bond')
print("***************************")
print("Diebold and Mariano test for Snp :\n",SnP_DnB)
print("***************************")
print("Diebold and Mariano test for Bond Index :\n",BI_DnB)

# Plot of Predictior S&P500
Snp_forecast_data['Snp_Benchmark'].plot()
Bond_forecast_data['BI_Benchmark'].plot()

print("\n SNP_forecast data:\n",Snp_forecast_data)
print("\n Bond index forecast data:\n",Bond_forecast_data)

Var_Covar_Matrix,array_matrix=var_covar_forecast(In_sample,Out_sample,'rec')
#print("Variance Covariance matrix:\n",Var_Covar_Matrix,"\n array of Matrix is:\n ",array_matrix)

#,combined_wts,benchmark_wts
returns,all_wts=generate_data_wts(Snp_forecast_data,Bond_forecast_data,array_matrix)
#Out_sample_wts=tangency_pfolio(Out_sample,,Var_Covar_Matrix)
print("The excess returns for Out sample are:\n",returns)
#print("\n the weights fore recursive approach are:\n",all_wts)

#Optimal portfolio summary stats
Sum_stats_optimal_pflio=portfolio_summary_stat(returns)
print("Summary stats for the optimal pflio are:\n",Sum_stats_optimal_pflio['Benchmark_Excess_Return'])


#Summary stats for 6 predictors and comibination
Sum_stats_6preds__pflio=portfolio_summary_stat(returns)
print("Summary stats for 6 predictors:\n",Sum_stats_6preds__pflio)

Snp_forecast_data.plot()
Bond_forecast_data.plot()

##Plot for combined and benchmark forecast
Snp_forecast_data.plot()
Weights_Pflio_cmbn=time_series_data(all_wts[0])
Weights_Pflio_bench=time_series_data(all_wts[1])
print("wts for Combined for the Pfolio:\n",Weights_Pflio_cmbn)
print("wts for benchmark for the Pfolio:\n",Weights_Pflio_bench)

Weights_Pflio_cmbn.plot(title='Weights for ccombined preds for recursive approach',label='Combined')
Weights_Pflio_bench.plot(title='Weights for benchmark preds for recursive approach',label='benchmark')

#Graphs for weights and combined and benchmark excess returns
graph_plots_wts(Weights_Pflio_cmbn,returns)
#graph_plots_wts(Weights_Pflio_bench,returns)

print("*******************************************************")
print("*******************************************************")

# =============================================================================
#                                  ROLLING WINDOW
# =============================================================================

print("*******************************************************")
print("*****ROLLING WINDOW**********")
print("*******************************************************")
time_series_roll=recursive_approach(In_sample, Out_sample,'rolling')
print("Time Series for rolling window is:\n",time_series_roll)

Snp_ols_Roll,Bond_ols_Roll=forecast_ols(In_Predictor,Out_Predictor,In_sample,Out_sample,'rolling')
print("Snp OlS forecast for Rolling is :\n ",Snp_ols_Roll.head(5),"\nBond OLS forecast for Rolling is:\n",Bond_ols_Roll.head(5))

Snp_forecast_Roll_data=pd.merge(Snp_ols_Roll,time_series_roll['Snp_Benchmark'],left_index=True,right_index=True,how='left')
Bond_forecast_Roll_data=pd.merge(Bond_ols_Roll,time_series_roll['BI_Benchmark'],left_index=True,right_index=True,how='left')

SnP500_Roll_Msfe= Msfe(Snp_forecast_Roll_data,Out_sample,'Sp500')
MSFE_Roll_BOND=Msfe(Bond_forecast_Roll_data,Out_sample,'Bond')

print("***************************")
print("***************************")
print("MSFE for SNP500 is :\n",SnP500_Roll_Msfe)

print("***************************")
print("***************************")
print("MSFE for Bond Index is :\n",MSFE_Roll_BOND)


SnP500_Msfe_Roll_Ratios= MSFE_Ratio(SnP500_Roll_Msfe,'Sp500')
MSFE_BOND_Roll_Ratios=MSFE_Ratio(MSFE_Roll_BOND,'Bond')
print("***************************")
print("***************************")
print("MSFE Ratios for SNP500 is :\n",SnP500_Msfe_Roll_Ratios)

print("***************************")
print("***************************")
print("MSFE Ratios for Bond Index is :\n",MSFE_BOND_Roll_Ratios)

SnP_Roll_DnB=Diebold_Mariano(Snp_forecast_Roll_data,Out_sample,'Sp500')
BI_Roll_DnB=Diebold_Mariano(Bond_forecast_Roll_data,Out_sample,'Bond')
print("***************************")
print("Diebold and Mariano test for Snp :\n",SnP_Roll_DnB)
print("***************************")
print("Diebold and Mariano test for Bond Index :\n",BI_Roll_DnB)

# Plot of Predictior S&P500
Snp_forecast_Roll_data['Snp_Benchmark'].plot()
Bond_forecast_Roll_data['BI_Benchmark'].plot()

Snp_forecast_Roll_data.plot()
Bond_forecast_Roll_data.plot()
# print("\n SNP_forecast data:\n",Snp_forecast_Roll_data)
# print("\n Bond index forecast data:\n",Bond_forecast_Roll_data)

Var_Covar_Matrix_Roll,array_matrix_Roll=var_covar_forecast(In_sample,Out_sample,'rolling')
#print("Variance Covariance matrix:\n",Var_Covar_Matrix_Roll,"\n array of Matrix is:\n ",array_matrix_Roll)

#,combined_wts,benchmark_wts
returns_Roll,all_wts_Roll=generate_data_wts(Snp_forecast_Roll_data,Bond_forecast_Roll_data,array_matrix_Roll)
#Out_sample_wts=tangency_pfolio(Out_sample,,Var_Covar_Matrix_Roll)
print("The returns for Out sample are:\n",returns_Roll)
#print("\n the weights are:\n",all_wts)

#Optimal portfolio summary stats
Sum_stats_optimal_pflio_Roll=portfolio_summary_stat(returns_Roll)
print("Summary stats for the optimal pflio are:\n",Sum_stats_optimal_pflio_Roll['Benchmark_Excess_Return'])


#Summary stats for 6 predictors and comibination
Sum_stats_6preds__pflio_Roll=portfolio_summary_stat(returns_Roll)
print("Summary stats for 6 predictors:\n",Sum_stats_6preds__pflio_Roll)

##Plot for combined and benchmark forecast
Snp_forecast_Roll_data.plot()
Weights_Pflio_cmbn_Roll=time_series_data(all_wts_Roll[0])
Weights_Pflio_bench_Roll=time_series_data(all_wts_Roll[1])
print("wts for Combined of the Rolling  Pfolio:\n",Weights_Pflio_cmbn_Roll)
print("wts for benchmark of the Rolling Pfolio:\n",Weights_Pflio_bench_Roll)

Weights_Pflio_cmbn_Roll.plot()
Weights_Pflio_bench_Roll.plot()

#Graphs for weights and combined excess returns
graph_plots_wts(Weights_Pflio_cmbn_Roll,returns_Roll)

#Graphs for weights and benchmark excess returns
graph_plots_wts(Weights_Pflio_bench_Roll,returns_Roll)
print("Th gihub test demo")










