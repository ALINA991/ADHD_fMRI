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


################## STATS #######################

def get_summary_mixed_lm(result): #mixedlm result output 
    
    summary_df = pd.DataFrame({
    'Coef.': result.params,
    'Std.Err.': result.bse,
    'z': result.tvalues,
    'P>|z|': result.pvalues,
    '0.025': result.conf_int()[0],
    '0.975': result.conf_int()[1]
})
    return summary_df

def get_RR_stats(formula, data, groups, alpha):
    result = smf.mixedlm(formula, data, groups = groups).fit()
    summary_df = get_summary_mixed_lm(result)
    highlighted_summary = summary_df.style.apply(
    lambda x: ['background-color: blue' if v < alpha else '' for v in x], 
    subset=['P>|z|'])
  
    return result, summary_df , highlighted_summary


    
def f_test_interactions(mixedlm_result, hyps, alpha):

    results = []

    for desc, hyp in zip(hyps.keys(), hyps.values()):
        f_test_result = mixedlm_result.f_test(hyp)
        p_value = f_test_result.pvalue 
        f_value = f_test_result.fvalue 

        significance = "*Significant*" if p_value < alpha else "Not Significant"

        results.append({
            "Description": desc,
            "Significance": significance,
            "F-Value": f_value,  
            "P-Value": p_value,
        })

    return pd.DataFrame(results)



def get_sig_vars(df_interaction_results, alpha):

    mask = df_interaction_results['P-Value'] < alpha
    masked_interact = df_interaction_results[mask]
    
    vars_ =   list(masked_interact['Description'])
    return vars_

def get_formula(outcome_var, predictor_list,  include_all = True):
    hyps_to_vars_dict = get_hyps_to_vars()
    
    predictor_list_form = [hyps_to_vars_dict[ predictor_list[i] ] for i in range(len(predictor_list)) ]
    if include_all : 
        predictors = ' * '.join(predictor_list_form)
        formula = ' ~ '.join((outcome_var, predictors))
    else : 
        form = [' ~ '.join((outcome_var, predictor_list_form[0]))]
        for vr in predictor_list_form[1:]:
            form.append(vr)
        formula = ' + '.join(form)
    return formula 


def highlight_significant_p_values(val, alpha):
    color = 'background-color: blue' if val < alpha else ''
    return color

#################### DATA PREPARE ####################

def get_hyps_to_vars():
    hyps_to_vars= {
    'site' : 'C(site)', 
    'time' : 'days_baseline', 
    'treat' : 'C(trtname, Treatment(reference="L"))',
    'site_treat' : 'C(site) * C(trtname, Treatment(reference="L"))',
    'time_treat': 'days_baseline * C(trtname, Treatment(reference="L"))',
    'site_time_treat' : 'days_baseline * C(trtname, Treatment(reference="L")) * C(site))'
}
    return hyps_to_vars

def get_masked_df(dataframe, column_name, value_for_condition):
    mask = dataframe[column_name] = value_for_condition
    masked_df = dataframe[mask]
    return masked_df
    
def get_hyps_interactions():
    hyps_interactions = {
        'site' : (
            'C(site)[T.2] = '
            'C(site)[T.3] = '
            'C(site)[T.4] = '
            'C(site)[T.5] = '
            'C(site)[T.6] = 0'),
        
        # 'hypothesis_sex' : (
        #     'C(sex)[T.M] = 0'), 
        
        'time' : "days_baseline = 0",
        
        'treat' : (
            'C(trtname, Treatment(reference="L"))[T.M] = '
            'C(trtname, Treatment(reference="L"))[T.P] = '
            'C(trtname, Treatment(reference="L"))[T.C] = 0'),
        
        'site_treat' :  (
            'C(trtname, Treatment(reference="L"))[T.M]:C(site)[T.2] = '
            'C(trtname, Treatment(reference="L"))[T.P]:C(site)[T.2] = '
            'C(trtname, Treatment(reference="L"))[T.C]:C(site)[T.2] = '
            
            'C(trtname, Treatment(reference="L"))[T.M]:C(site)[T.3] = '
            'C(trtname, Treatment(reference="L"))[T.P]:C(site)[T.3] = '
            'C(trtname, Treatment(reference="L"))[T.C]:C(site)[T.3] = '
            
            'C(trtname, Treatment(reference="L"))[T.M]:C(site)[T.4] = '
            'C(trtname, Treatment(reference="L"))[T.P]:C(site)[T.4] = '
            'C(trtname, Treatment(reference="L"))[T.C]:C(site)[T.4] = '
            
            'C(trtname, Treatment(reference="L"))[T.M]:C(site)[T.5] = '
            'C(trtname, Treatment(reference="L"))[T.P]:C(site)[T.5] = '
            'C(trtname, Treatment(reference="L"))[T.C]:C(site)[T.5] = '
            
            'C(trtname, Treatment(reference="L"))[T.M]:C(site)[T.6] = '
            'C(trtname, Treatment(reference="L"))[T.P]:C(site)[T.6] = '
            'C(trtname, Treatment(reference="L"))[T.C]:C(site)[T.6] = 0'),
        
        'time_treat' :  ('C(trtname, Treatment(reference="L"))[T.M]:days_baseline = C(trtname, Treatment(reference="L"))[T.P]:days_baseline  = C(trtname, Treatment(reference="L"))[T.C]:days_baseline = 0'),
        
        'site_time_treat' : (   


            'C(trtname, Treatment(reference="L"))[T.M]:days_baseline:C(site)[T.2] = '
            'C(trtname, Treatment(reference="L"))[T.P]:days_baseline:C(site)[T.2] = '
            'C(trtname, Treatment(reference="L"))[T.C]:days_baseline:C(site)[T.2] = '

            'C(trtname, Treatment(reference="L"))[T.M]:days_baseline:C(site)[T.3] = '
            'C(trtname, Treatment(reference="L"))[T.P]:days_baseline:C(site)[T.3] = '
            'C(trtname, Treatment(reference="L"))[T.C]:days_baseline:C(site)[T.2] = '

            'C(trtname, Treatment(reference="L"))[T.M]:days_baseline:C(site)[T.4] = '
            'C(trtname, Treatment(reference="L"))[T.P]:days_baseline:C(site)[T.4] = '
            'C(trtname, Treatment(reference="L"))[T.C]:days_baseline:C(site)[T.2] = '

            'C(trtname, Treatment(reference="L"))[T.M]:days_baseline:C(site)[T.5] = '
            'C(trtname, Treatment(reference="L"))[T.P]:days_baseline:C(site)[T.5] = '
            'C(trtname, Treatment(reference="L"))[T.C]:days_baseline:C(site)[T.2] = '

            'C(trtname, Treatment(reference="L"))[T.M]:days_baseline:C(site)[T.6] = '
            'C(trtname, Treatment(reference="L"))[T.P]:days_baseline:C(site)[T.6] = '
            'C(trtname, Treatment(reference="L"))[T.C]:days_baseline:C(site)[T.2] = 0')}

    return hyps_interactions

#################### PLOT ########################

