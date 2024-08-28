from pathlib import Path 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from collections import OrderedDict
import sys
import os
import seaborn as sns
import researchpy as rp
import statsmodels.formula.api as smf
import scipy.stats as stats

from helper import rr, var_dict

def get_masked_df(dataframe, column_name, condition, value_for_condition):
    if condition == 'eq': 
        mask = dataframe[column_name] == value_for_condition
        masked_df = dataframe[mask]
        return masked_df
    elif condition == 'gt': 
        mask = dataframe[column_name] > value_for_condition
        masked_df = dataframe[mask]
        return masked_df
    elif condition == 'get': 
        mask = dataframe[column_name] >= value_for_condition
        masked_df = dataframe[mask]
        return masked_df
    elif condition == 'lt': 
        mask = dataframe[column_name] < value_for_condition
        masked_df = dataframe[mask]
        return masked_df
    elif condition == 'let': 
        mask = dataframe[column_name] <= value_for_condition
        masked_df = dataframe[mask]
        return masked_df
    else : raise ValueError('Verify your condition')
    
    
def split_data_from_timepoints(df): 
    try : 
        df['days_baseline'] = df['days_baseline'].astype(int)
    except ValueError as e :
        if str(e) == ("invalid literal for int() with base 10: 'Days since baseline'"):
            df = df.drop(0, axis = 0)
            df['days_baseline'] = df['days_baseline'].astype(int)
        else :
            raise e
    
    df_baseline = get_masked_df(df, 'days_baseline', 'eq' , 0) #baseline 
    df_14 = get_masked_df(df, 'days_baseline' ,'lt', 578) # 14 months 
    df_24 = get_masked_df(df, 'days_baseline', 'lt' , 912) # 24 months 
    df_36 = get_masked_df(df, 'days_baseline', 'lt' ,  1195 ) # 36 months 
    
    return df_baseline, df_14, df_24, df_36

def print_nonNaN_shapes(df, contains = None):
    if contains is not None : 
        for i, col in zip(range(df.shape[1]), df.columns): 
            if df.iloc[:,i].dropna().shape[0] > 1 and df.iloc[0, i].rfind(contains) != -1 :
                print(i,col, df.iloc[0, i], df.iloc[:,i].dropna().shape)
    else: 
        for i, col in zip(range(df.shape[1]), df.columns): 
            if df.iloc[:,i].dropna().shape[0] > 1 :#and (file.iloc[0, i].rfind('anx') != -1 or  file.iloc[0, i].rfind('cond') != -1 or file.iloc[0, i].rfind('opp') != -1 or file.iloc[0, i].rfind('pho') != -1 ) : 
                print(i,col, df.iloc[0, i], df.iloc[:,i].dropna().shape)
                
def get_nonNaN_cols(df):
    cols = []
    for i, col in zip(range(df.shape[1]), df.columns): 
        if df.iloc[:,i].dropna().shape[0] > 1 :#and (file.iloc[0, i].rfind('anx') != -1 or  file.iloc[0, i].rfind('cond') != -1 or file.iloc[0, i].rfind('opp') != -1 or file.iloc[0, i].rfind('pho') != -1 ) : 
            cols.append(col)    
    return df[cols]
    

def set_baseline_dtypes(df, dropna = False):

    try:
        if df.shape != df.dropna().shape and dropna:
            print('Dropping rows containing NaN. Old shape:  {}, new shape : {}'.format(df.shape, df.dropna().shape))
            df = df.dropna()
            
        df['src_subject_id'] = df['src_subject_id'].astype('str')#df['src_subject_id'].str.strip()
        df['sex'] = df['sex'].astype('category')
        df['site'] = df['sex'].astype('category')
        df['interview_date'] = pd.to_datetime(df['interview_date'], format='%m/%d/%Y')
        df[['interview_age', 'relationship',  'days_baseline']] = df[['interview_age', 'relationship', 'days_baseline']].astype(int)
        
        if 'trtname' in df.columns:
            df['trtname'] = df['sex'].astype('category')
        print('Success')
        return df
    
    except(ValueError):

        print('Conversion encountered a problem. Attempt to drop description line.')
        if df['src_subject_id'].iloc[0] == 'Subject ID how it\'s defined in lab/project': #check whether description line exists 
            df = df.drop(0, axis = 0)
            set_baseline_dtypes(df)
        else:
            print('Could not identify problem. Exiting... ')
            raise(ValueError)
    return df


def set_dtypes(df,  dtypes_dict , set_baseline=True):
    if set_baseline:
        df = set_baseline_dtypes(df)
    for var, dtype in zip(dtypes_dict.keys(), dtypes_dict.values()):
        if var in df.columns:
            df[var] = df[var].astype(dtype)
        else: 
            pass
        
    return df
    
        
# def drop_description():
#     pass