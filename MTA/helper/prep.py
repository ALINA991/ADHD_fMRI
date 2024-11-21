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
import collections

from helper import rr, var_dict

def get_data(path, columns, treat_group, set_dtypes = True, version_form = False, split_timepoints = False):
    ##treat_group = pd.read_csv('/Volumes/Samsung_T5/MIT/mta/output/derived_data/treatment_groups.csv')
    if columns is not None: 
        if any(isinstance(i, collections.abc.Iterable) for i in columns): 
            columns = list(np.concatenate(columns))
            
        test = ['version_form' in col for col in columns]
        
        if version_form and not np.array(test).any():
            columns.append('version_form')
        elif not version_form and np.array(test).any():
            columns.remove('version_form')
        
        try : 
            df = pd.read_csv(path, delimiter="\t", usecols=columns, skiprows=[1] , parse_dates=['interview_date']).dropna(subset='days_baseline').drop_duplicates()
        except Exception as e:
            if str(e) ==  "Missing column provided to 'parse_dates': 'interview_date'":
                df = pd.read_csv(path, delimiter="\t", usecols=columns, skiprows=[1]).dropna(subset='days_baseline').drop_duplicates()
            else:
                raise e
    else: 
        print("all columns")
        try : 
            df = pd.read_csv(path, delimiter="\t",  skiprows=[1] , parse_dates=['interview_date']).dropna(subset='days_baseline').drop_duplicates()
  
        except Exception as e:
            if str(e) ==  "Missing column provided to 'parse_dates': 'interview_date'":
                df = pd.read_csv(path, delimiter="\t", skiprows=[1]).dropna(subset='days_baseline').drop_duplicates()
            else:
                raise e
    if set_dtypes:
        set_baseline_dtypes(df)
    df = pd.merge(df, treat_group, how='inner', on = 'src_subject_id')#.dropna()#.table with relevant snap vales, rater, and treatment group 
    print(df.shape)

        
    if version_form:

        df.loc[df['version_form'].str.startswith('Teacher'), 'version_form'] = 'Teacher'
        df.loc[df['version_form'].str.startswith('Parent'), 'version_form'] = 'Parent'
        
    if split_timepoints:
        return split_data_from_timepoints(df)
    else: 
        return df


    
    

def get_masked_df(dataframe, column_name, condition, value_for_condition, delta = None):
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
    elif condition == 'range' and delta is not None:
        mask = (dataframe[column_name] <= value_for_condition + delta) & (dataframe[column_name] >= value_for_condition - delta)
        masked_df = dataframe[mask]
    elif condition == 'range' and value_for_condition.isinstance(tuple):
        mask = (dataframe[column_name] <= value_for_condition[1]) & (dataframe[column_name] >= value_for_condition[0])
        masked_df = dataframe[mask]
        
    else : raise ValueError('Verify the condition')
    return masked_df
    
    
def split_data_from_timepoints(df, timepoints = None): 
    df = df.dropna(subset='days_baseline')
    try : 
        df['days_baseline'] = df['days_baseline'].astype(int)
    except ValueError as e :
        if str(e) == ("invalid literal for int() with base 10: 'Days since baseline'"):
            df = df.drop(0, axis = 0)
            df['days_baseline'] = df['days_baseline'].astype(int)
        else :
            raise e
    if timepoints is not None: 
        
        if isinstance(timepoints, (list, np.ndarray, tuple)) and isinstance(timepoints[0], int): 
            dfs = [get_masked_df(df,  'days_baseline', 'lt' , timepoint) for timepoint in timepoints]
            dictt = dict(zip(timepoints, dfs))
        
        elif isinstance(timepoints, str) and timepoints == "standard_range" or (isinstance(timepoints, (list, np.ndarray, tuple)) and isinstance(timepoints[0], tuple)):
            dfs = [get_masked_df(df,  'days_baseline', 'range' , timepoint) for timepoint in timepoints]
            dictt = dict(zip(timepoints, dfs))

            dictt = {'b' : df_baseline, '14': df_14, '24': df_24, '36' : df_36}
        
        else: 
            print('Timepoints condition not recognized. Please specify a list, tuple, or nd array.')

    else:
        print('No timepoints specifed. Using (46,168,319,500) by default.')
        df_baseline = get_masked_df(df, 'days_baseline', 'lt' , 46).copy() #baseline 
        df_14 = get_masked_df(df, 'days_baseline' ,'lt', 168).copy() # 14 months 
        df_24 = get_masked_df(df, 'days_baseline', 'lt' , 319).copy() # 24 months 
        df_36 = get_masked_df(df, 'days_baseline', 'lt' ,  500 ).copy() # 36 months 
        
        dictt = {'b' : df_baseline, '14': df_14, '24': df_24, '36' : df_36}

        

    return dictt

def split_data_from_timepoints_custom(df, timepoints, how = 'lt', delta = None): 
    try : 
        df['days_baseline'] = df['days_baseline'].astype(int)
    except ValueError as e :
        if str(e) == ("invalid literal for int() with base 10: 'Days since baseline'"):
            df = df.drop(0, axis = 0)
            df['days_baseline'] = df['days_baseline'].astype(int)
        else :
            raise e
        
    if timepoints[0] == 0:
        df0 = [get_masked_df(df, 'days_baseline', 'eq', timepoints[0]).copy() ]
        dfs_ = [get_masked_df(df, 'days_baseline', how , timepoint, delta).copy() for timepoint in timepoints[1:]]
        dfs =  dfs = df0 + dfs_
        
    else:
        dfs = [get_masked_df(df, 'days_baseline', how , timepoint, delta).copy() for timepoint in timepoints]
    
    
    dictt = dict(zip(timepoints, dfs))

    return dictt
def print_nonNaN_shapes(df, contains = None):
    if contains is not None : 
        for i, col in zip(range(df.shape[1]), df.columns): 
            if df.iloc[:,i].dropna().shape[0] > 1 and df.iloc[0, i].lower().rfind(contains.lower()) != -1 :
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
        df['sex'] = df['sex'].astype('str')
        df['site'] = df['site'].astype('str')
        df['interview_date'] = pd.to_datetime(df['interview_date'], format='%m/%d/%Y')
        df[['interview_age', 'days_baseline']] = df[['interview_age',  'days_baseline']].astype(int)
        
        if 'trtname' in df.columns:
            df['trtname'] = df['trtname'].astype('str')
        print('Success')
        return df
    
    except ValueError as e :

        print('Conversion encountered a problem. Attempt to drop description line.')
        if df['src_subject_id'].iloc[0] == 'Subject ID how it\'s defined in lab/project': #check whether description line exists 
            df = df.drop(0, axis = 0)
            set_baseline_dtypes(df)
        else:
            print('Could not identify problem. Exiting... ')
            raise(ValueError(e))
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
    
        
########## format results ########

def find_first_index(df, column, value, condition = None):
    if condition is not None:
        condition_column, condition_value = condition.split(' == ')
        return df[(df[column] == value) & (df[condition_column] == condition_value)].index[0]
    else:
        return (df[column] == value).idxmax()
    
    
def split_on_occurrence(s, char, occurrence=1):
    if occurrence == 1:
        # Find the index of the first occurrence of the character
        index = s.find(char)
    elif occurrence == 2:
        # Find the index of the second occurrence of the character
        first_occurrence = s.find(char)
        index = s.find(char, first_occurrence + 1)
    else:
        raise ValueError("Occurrence must be 1 or 2")

    # If the occurrence is found, split the string
    if index != -1:
        return s[:index], s[index+1:]
    else:
        # If the occurrence isn't found, return the original string
        return s, ''