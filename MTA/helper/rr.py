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

def extract_mean_for_contrast(mean_dict, qst, var, time ):
    mean = []
    for group in ['C', 'M', 'P', 'L']:
        mean.append(mean_dict[qst][time][group][var])
    return mean
    
    

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

def get_contrast_matrix(paper = 'original', new_order = None):
    if paper == 'original':
        
        contrast_matrix = np.array([ # add 0 column for intercept generated from RR, ignored bc coefficients for contrasts = 0
            [0, 1, -1, 0, 0],  # M vs P
            [0, -1, 0, 1, 0],  # C vs M
            [0, 0, -1, 1, 0],  # C vs P
            [0, -1, 0, 0, 1],  # L vs M
            [0, 0, -1, 0, 1],  # L vs P
            [0, 0, 0, -1, 1]   # L vs C
        ])

    elif paper == 'molina':
        contrast_matrix = np.array([[ 0, 1, -1	,1,-1],   # (M,C) vs (B,CC)
                                    [ 0, 1, 0,	-1,	0],   # (M vs C)
                                    [ 0, 0, 1	,0,-1]])  # (B vs CC)
  
    contrast_m = pd.DataFrame(contrast_matrix, columns=['intercept', 'M', 'P', 'C', 'L'])
    if new_order is not None:
        contrast_m = contrast_m[new_order]
    return contrast_m

def ortho_contrasts_molina(outcome, predictor, data):

    formula =' ~ '.join((outcome, predictor))
    # change when get access to CC data, careful with interpretations !
    contrasts_matrix = np.array([[  0, 1,	-1	,1,	-1],
                                [	0, 1,	0,	-1,	0],
                                [ 	0, 0	,1	,0	,-1]])
    
    trtname = ['intercept', 'M', 'P', 'C', 'L']
    
    contrasts_matrix = pd.DataFrame(contrasts_matrix, columns= trtname)


    model = smf.mixedlm(formula, data = data, groups='src_subject_id')
    result = model.fit()

    new_order = [item.split('[')[0].lower() if 'Intercept' in item else item.split('[')[1][2] for item in result.model.exog_names]
    contrasts_matrix = contrasts_matrix[new_order]


    contrasts = result.t_test(contrasts_matrix)
    # Print the results of the contrast tests
    print(contrasts)

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

def get_hyps_interaction_mediators():
    pass



def perform_contrasts(data, formula, type_contrast, alpha, version_form, bonferroni):
    if version_form is not None:
        data = data[data['version_form'] == version_form]
        
    if bonferroni : 
        alpha = alpha / data.shape[0]


    # change when get access to CC data, careful with interpretations !
    contrasts_matrix = get_contrast_matrix(type_contrast)

    model = smf.mixedlm(formula, data = data, groups='src_subject_id')
    result = model.fit()

    new_order = [item.split('[')[0].lower() if 'Intercept' in item else item.split('[')[1][2] for item in result.model.exog_names]
    contrasts_matrix = contrasts_matrix[new_order]


    contrasts = result.t_test(contrasts_matrix)

    coef = contrasts.effect
    stderr = contrasts.sd
    z_scores = contrasts.tvalue
    p_values = contrasts.pvalue
    conf_int = contrasts.conf_int()

    significant =  p_values < alpha

    comp_results_str = []

    for i, val in enumerate(significant): 
        print(val)
        if val and coef[i] > 0 :
            results_sign = ' < ' # this would be inverted if we were comparing the means, but we want to know which treatment DECREASE the outcome variable
        elif val and coef[i] < 0 :
            results_sign = ' > '
        else: 
            results_sign = ' ~ '
        comp = np.array(contrasts_matrix.columns[contrasts_matrix.iloc[i] != 0])
        comp_results_str.append(results_sign.join(comp))
    
    return p_values, comp_results_str