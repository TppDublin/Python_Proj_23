# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 04:27:52 2023

@author: tushar
"""

import pandas as pd
import numpy as np


def Data_Creation(df1,df2,df3):
   
    ldf1=len(df1)
    ldf2=len(df2)
    ldf3=len(df3)
  
    df1.rename(columns={'Date':'Dates'},inplace=True)
    df1.reset_index(drop=True, inplace=True)
    df1.set_index('Dates', inplace=True)
            
    df2.reset_index(drop=True, inplace=True)
    df2.set_index('Dates', inplace=True)
            
    df3.reset_index(drop=True, inplace=True)
    df3.set_index('Dates', inplace=True)   
         
    
    print("RF return is ,\n",df1,"\n snp is:",df2)
    Data_Frame_tmp=pd.merge(df2,df3,left_index=True,right_index=True,how='left')
    print(Data_Frame_tmp)
    Data_Frame=pd.merge(Data_Frame_tmp,df1,left_index=True,right_index=True,how='left')
    print(Data_Frame)
    Data_Frame.rename(columns={'SP500 index price':'SnP','LBUSTRUU Index price':'BI','Risk free rate of return ':'Rf'},inplace=True)
    

    return Data_Frame
