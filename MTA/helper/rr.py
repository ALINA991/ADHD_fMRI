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

def get_rolling_av(subset, pred_col_name, window):
    subset_sorted = subset.sort_values('days_baseline')
# Apply rolling mean across the 'days_baseline'
    smoothed = subset_sorted[pred_col_name].rolling(window=window, min_periods=1).mean()

    # Ensure the rolling mean values are aligned with the sorted 'days_baseline'
    smoothed_df = pd.DataFrame({
        'days_baseline': subset_sorted['days_baseline'],
        'smoothed_value': smoothed
    })
    return smoothed_df

def extract_prediction(results, data_dict, to_plot):
    # extract predicted datapoints for all questionnaires
    for qst in to_plot:
        if qst == 'snap' or qst == 'ssrs':
            print(qst.upper())
            for rater in raters:
                for var_result in results[qst][rater].keys():    
                    result = results[qst][rater][var_result]
                    #print(result)
                    data = data_dict[qst]
                    data_dict[qst]['predicted_' +  str(var_result) +'_' + str(rater[0])] = result.predict(data) #
                    
        elif qst == 'masc' or qst == 'pc':    
            print(qst.upper())       
            for var_result in results[qst].keys():
                data = data_dict[qst]
                result = results[qst][var_result]
                data_dict[qst]['predicted_' + str(var_result)] = result.predict(data) 
                
                
def get_point_av(df, pred_col_name, timepoints_range):
    df.sort_values('days_baseline')
    subset = df[['days_baseline', pred_col_name]]
    point_av = np.array([ subset[subset['days_baseline'].isin(range(range_t[0], range_t[1]))].mean(axis = 0) for range_t in timepoints_range])
    return point_av 


def get_timepoints_range(timepoints, delta):
    return  [[time - delta, time + delta] for time in timepoints]


def extract_line_plot(df, pred_col_name, type_plot, pnt_av = None,  window = None):
    
    timepoints_range = get_timepoints_range(timepoints, delta)
    ptn_av = get_point_av(df,pred_col_name, timepoints_range )
    
    subset = df.dropna(subset=pred_col_name)
    subset_sorted = subset.sort_values('days_baseline')
    print(subset_sorted[pred_col_name].isna().sum())
    print("SHAPE subset:", subset[pred_col_name].dropna().shape, "SHAPE subset sorted", subset_sorted[pred_col_name].dropna().shape, )
    #lowess smoothing function 
    if type_plot == "smooth":
        print('type smoothed')
        print(subset_sorted[pred_col_name].shape, subset_sorted['days_baseline'].shape)
        smoothed = lowess(subset_sorted[pred_col_name], subset_sorted['days_baseline'], frac=0.1)
        print(smoothed.shape)
        smoothed_df = pd.DataFrame({
        'days_baseline': smoothed[:,0],
        'smoothed_value': smoothed[:,1]
    })

        
    elif type_plot == "mov_av":
        print('type moving average')
        #moving average with size window
        smoothed_df = get_rolling_av(subset_sorted, pred_col_name, window)

    
    elif type_plot == 'poly_fit':
        print('type four point polynomial fit')
        # polynomial fit to average datapoints 
        
        if len(pnt_av) > 2 and len(pnt_av) > 2:
            
            x_fit_min = min(pnt_av[:, 0].min(), pnt_av[:, 0].min())
            x_fit_max = max(pnt_av[:, 0].max(), pnt_av[:, 0].max())
            x_fit = np.linspace(x_fit_min, x_fit_max, 100)
            
            try:
                # Polynomial fit 
                poly_fit = np.polyfit(pnt_av[:, 0], pnt_av[:, 1], 2)
                poly_function_yes = np.poly1d(poly_fit)
                smoothed = poly_function_yes(x_fit)
                print(smoothed.shape, pnt_av.shape)
                smoothed_df = pd.DataFrame({
                    'days_baseline': x_fit,
                    'smoothed_value': smoothed
            })
                
                
            except np.linalg.LinAlgError:
                print("Polynomial fit for 'no' failed due to singular matrix")
                return
                
           
        else:
            print("Not enough points for polynomial fit in treatment group:")   
            return
    else:
        print("Plot type not recognized.")
        return
    print("SMOOTHED shape", smoothed_df.shape)
    return smoothed_df
        

            
            

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

'''
def plot_4_point_av_fit(data, var_mod, var_out, med_values, window,  trt_dict , rater = None, show = True, save_path= None):
    print(show, save_path)
    ylim = [0,3]
    xlim = [-10, 450]
    pred_col_name= 'predicted_' + var_mod + '_' + var_out
    var_result = pred_col_name
    trtnames = data['trtname'].unique()
    pred_col_name= 'predicted_' + var_mod + '_' + var_out

    title = ['14-Month Outcomes \n {} \n {}'.format(med_dict_fig[var_mod][0], outcomes_dict_fig[var_out]), 
                '14-Month Outcomes \n {} \n {}'.format(med_dict_fig[var_mod][1], outcomes_dict_fig[var_out])]

    values = med_values[var_mod]
    print(values)
    # Create subplots: 2 rows (1st row for polynomial fit, 2nd row for bar plot)
    if var_mod == 'd2dresp':
        no_mod = data[(data[var_mod].isin(values[:2])) & (data['days_baseline'] >= -window) & (data['days_baseline'] <= 450)]
        yes_mod = data[(data[var_mod].isin(values[2:])) & (data['days_baseline'] >= -window) & (data['days_baseline'] <= 450)]
    else:
        no_mod = data[(data[var_mod] == values[0]) & (data['days_baseline'] >= -window) & (data['days_baseline'] <= 450)]
        yes_mod = data[(data[var_mod] == values[1]) & (data['days_baseline'] >=-window) & (data['days_baseline'] <= 450)]
        
        
    print(yes_mod.shape, no_mod.shape)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True, sharey='row')
    if rater is not None: 
        plt.suptitle('Rater: ' + rater)

    # Loop over each treatment name and plot both the 'yes' and 'no' condition in separate subplots
    for trt in trtnames:


        df_yes = yes_mod[yes_mod['trtname'] == trt].dropna(subset=[pred_col_name])
        df_no = no_mod[no_mod['trtname'] == trt].dropna(subset=[pred_col_name])
        
        smoothed_yes = get_rolling_av(df_yes, pred_col_name, window)
        pnt_av = get_point_av(smoothed_yes, timepoints_range)
        
        smoothed_no = get_rolling_av(df_no, pred_col_name, window)
        pnt_av = get_point_av(smoothed_no, timepoints_range)
        x_fit_min = min(point_av_yes[:, 0].min(), point_av_yes[:, 0].min())
        x_fit_max = max(point_av_yes[:, 0].max(), point_av_no[:, 0].max())

        # Generate 100 evenly spaced points between the min and max values
        x_fit = np.linspace(x_fit_min, x_fit_max, 100)

        # Remove NaN and Infinite values
        point_av_no = point_av_no[~np.isnan(point_av_no).any(axis=1)]
        point_av_no = point_av_no[~np.isinf(point_av_no).any(axis=1)]
        point_av_yes = point_av_yes[~np.isnan(point_av_yes).any(axis=1)]
        point_av_yes = point_av_yes[~np.isinf(point_av_yes).any(axis=1)]

        # Ensure enough points for polynomial fitting
        if len(point_av_no) > 2 and len(point_av_yes) > 2:
            try:
                # Polynomial fit for 'yes'
                poly_fit_yes = np.polyfit(point_av_yes[:, 0], point_av_yes[:, 1], 2)
                poly_function_yes = np.poly1d(poly_fit_yes)
                y_fit_yes = poly_function_yes(x_fit)
                axes[0, 0].scatter(point_av_yes[:, 0], point_av_yes[:, 1], label=trt_dict[trt])
                axes[0, 0].plot(x_fit, y_fit_yes, linestyle='--', label=trt_dict[trt])
            except np.linalg.LinAlgError:
                print("Polynomial fit for 'yes' failed due to singular matrix")

            try:
                # Polynomial fit for 'no'
                poly_fit_no = np.polyfit(point_av_no[:, 0], point_av_no[:, 1], 2)
                poly_function_no = np.poly1d(poly_fit_no)
                y_fit_no = poly_function_no(x_fit)
                axes[0, 1].scatter(point_av_no[:, 0], point_av_no[:, 1], label=trt_dict[trt])
                axes[0, 1].plot(x_fit, y_fit_no, linestyle='--', label=trt_dict[trt])
            except np.linalg.LinAlgError:
                print("Polynomial fit for 'no' failed due to singular matrix")
        else:
            print("Not enough points for polynomial fit in treatment group:", trt)

            # Plot bar plot for the number of data points in the 'yes' condition (bottom left)
            #axes[1, 0].bar(time, n_yes, width=40, align='center', label=trt_dict[trt])
            
            # Plot bar plot for the number of data points in the 'no' condition (bottom right)
            #axes[1, 1].bar(time, n_no, width=40, align='center', label=trt_dict[trt])


    axes[0, 0].set_xlabel('Assessment point (d)')
    axes[0, 0].set_ylabel('Fitted Predicted Score')
    axes[0, 0].legend(title='Treatment Arm')
    axes[0, 0].set_xlim(xlim)  # Set x-axis limits
    axes[0, 0].set_ylim(ylim)  # Set y-axis limits
    axes[0, 0].set_title(title[0])
    # Add labels and titles to the subplots
    #axes[1, 0].set_title(title[1])
    #axes[0, 1].set_xlim(xlim)  # Set x-axis from 0 to 450
    #axes[0, 1].set_ylim(ylim)  # Ensure y-axis is from 0 to 3

    axes[0, 1].set_title(title[1])
    axes[0, 1].set_xlabel('Assessment point (d)')
    axes[0, 1].set_ylabel('Fitted Predicted Score')
    axes[0, 1].legend()




    # Set xlim and ylim
    axes[0, 0].set_xlim(x_lim)
    axes[0, 1].set_xlim(x_lim)
    axes[1, 0].set_xlim(x_lim)
    axes[1, 1].set_xlim(x_lim)

    axes[0, 0].set_ylim(y_lim)
    axes[0, 1].set_ylim(y_lim)


    bins = timepoints  # Adjust the number of bins as needed

    #Bar plot for "No Anxiety"
    trt_groups_no = no_mod['trtname'].unique()
    counts_by_trt_no = {trt: np.histogram(no_mod[no_mod['trtname'] == trt]['days_baseline'], bins=bins)[0] for trt in trt_groups_no}

    # Initialize bottom for stacked bars
    bottom_no = np.zeros(len(bins)-1)

    # Plot the stacked bars for "No  mediator"
    for trt in trt_groups_no:
        counts = counts_by_trt_no[trt]
        axes[1, 0].bar(bins[:-1], counts, width=45, align='edge', bottom=bottom_no, label=trt_dict[trt], alpha=0.7)
        bottom_no += counts  # Update bottom for the next treatment group

    # Bar plot for "Anxiety"
    trt_groups_yes = yes_mod['trtname'].unique()
    counts_by_trt_yes = {trt: np.histogram(yes_mod[yes_mod['trtname'] == trt]['days_baseline'], bins=bins)[0] for trt in trt_groups_yes}

    # Initialize bottom for stacked bars
    bottom_yes = np.zeros(len(bins)-1)


    # Plot the stacked bars for "Anxiety"
    for trt in trt_groups_yes:
        counts = counts_by_trt_yes[trt]
        axes[1, 1].bar(bins[:-1], counts, width=45, align='edge', bottom=bottom_yes, label=trt_dict[trt], alpha=0.7)
        bottom_yes += counts  # Update bottom for the next treatment group

    axes[1, 1].set_xlabel('Assessment point (d)')
    axes[1, 0].set_xlabel('Assessment point (d)')
    axes[1, 1].set_xlim(xlim)  # Match x-axis to the upper plot
    axes[1, 1].set_ylim([0,800])  # Match x-axis to the upper plot
    axes[1, 1].legend(title='Treatment Arm')

    if save_path is not None:
        folder_path = Path(save_path) / 'four_pt_fit'
        folder_path.mkdir(exist_ok=True)
        fig_name = var_result + '_' + str(rater[0]) + '.jpg' if rater is not None else var_result + '.jpg'
        print(folder_path)
        plt.savefig(folder_path / fig_name, dpi=300)

    if show:
        plt.tight_layout()
        plt.show()
        
        
    def plot_moving_av(data, var_mod, var_out, med_values, window,  trt_dict , rater =None, show = True, save_path= None):
    ylim = [0,3]
    xlim = [-10, 450]
    pred_col_name= 'predicted_' + var_mod + '_' + var_out
    values = med_values[var_mod]    
    
    title = ['14-Month Outcomes \n {} \n {}'.format(med_dict_fig[var_mod][0], outcomes_dict_fig[var_out]), 
                '14-Month Outcomes \n {} \n {}'.format(med_dict_fig[var_mod][1], outcomes_dict_fig[var_out])]
    
    type_plot_dict ={"poly_fit": "Polyniomial fit on four point average, degree 2", 
                 "mov_av": "Moving average on predicted score, window = {}".format(window),
                 "smooth": "Smoothed Predicted Score"}

    # get data for yes and no mediator condition 
    if var_mod == 'd2dresp':
        no_mod = data[(data[var_mod].isin(values[:2])) & (data['days_baseline'] >= -window) & (data['days_baseline'] <= 450)]
        yes_mod = data[(data[var_mod].isin(values[2:])) & (data['days_baseline'] >= -window) & (data['days_baseline'] <= 450)]
    else:
        no_mod = data[(data[var_mod] == values[0]) & (data['days_baseline'] >= -window) & (data['days_baseline'] <= 450)]
        yes_mod = data[(data[var_mod] == values[1]) & (data['days_baseline'] >=-window) & (data['days_baseline'] <= 450)]

    # Create subplots (2 rows, 2 columns) with shared x-axis
    fig, axes = plt.subplots(2, 2, figsize=(20, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    if rater is not None: 
        plt.suptitle('Rater: ' + rater)
        
    # Plot for "No Anxiety" subgroup
    for trt in no_mod['trtname'].unique():
        print(trt)
        # Subset data by treatment group and sort by 'days_baseline'
        subset = no_mod[no_mod['trtname'] == trt]
        n = subset.shape[0]

        smoothed_df = get_rolling_av(subset, pred_col_name, window)
        point_av = get_point_av(smoothed_df, timepoints_range)

        axes[ 0, 0 ].scatter(point_av[:,0], point_av[:,1])
        axes[0, 0].plot(smoothed_df['days_baseline'], smoothed_df['smoothed_value'], label=trt_dict[trt] + ' (n = {})'.format(n))

    # Set labels, title, and limits
    axes[0, 0].set_xlabel('Days Baseline')
    axes[0, 0].set_ylabel(type_plot_dict[type_plot])
    axes[0, 0].legend(title='Treatment Arm')
    axes[0, 0].set_xlim(xlim)  # Set x-axis limits
    axes[0, 0].set_ylim(ylim)  # Set y-axis limits
    axes[0, 0].set_title(title[0])


    for trt in yes_mod['trtname'].unique():
        print(trt)
        # Subset data by treatment group and sort by 'days_baseline'
        subset = yes_mod[yes_mod['trtname'] == trt]
        n = subset.shape[0]
        
        smoothed_df = get_rolling_av(subset, pred_col_name, window)
        point_av = get_point_av(smoothed_df, timepoints_range)
        
        axes[ 0, 1].scatter(point_av[:,0], point_av[:,1])
            # Plot the line using the sorted 'days_baseline' and corresponding rolling mean
        axes[0, 1].plot(smoothed_df['days_baseline'], smoothed_df['smoothed_value'], label=trt_dict[trt] + ' (n = {})'.format(n))

    axes[0, 1].set_title(title[1])
    axes[0, 1].set_xlim(xlim)  # Set x-axis from 0 to 450
    axes[0, 1].set_ylim(ylim)  # Ensure y-axis is from 0 to 3

    ### Second row: Stacked bar plots for the number of data points colored by treatment group ###

    # Define bins for the time points (based on 'days_baseline')
    bins = np.linspace(0, 450, num=10)  # Adjust the number of bins as needed

    #Bar plot for "No Anxiety"
    trt_groups_no = no_mod['trtname'].unique()
    counts_by_trt_no = {trt: np.histogram(no_mod[no_mod['trtname'] == trt]['days_baseline'], bins=bins)[0] for trt in trt_groups_no}

    # Initialize bottom for stacked bars
    bottom_no = np.zeros(len(bins)-1)

    # Plot the stacked bars for "No  mediator"
    for trt in trt_groups_no:
        counts = counts_by_trt_no[trt]
        axes[1, 0].bar(bins[:-1], counts, width=45, align='edge', bottom=bottom_no, label=trt_dict[trt], alpha=0.7)
        bottom_no += counts  # Update bottom for the next treatment group


    # Bar plot for "Anxiety"
    trt_groups_yes = yes_mod['trtname'].unique()
    counts_by_trt_yes = {trt: np.histogram(yes_mod[yes_mod['trtname'] == trt]['days_baseline'], bins=bins)[0] for trt in trt_groups_yes}

    # Initialize bottom for stacked bars
    bottom_yes = np.zeros(len(bins)-1)
    # Plot the stacked bars for "Anxiety"
    for trt in trt_groups_yes:
        counts = counts_by_trt_yes[trt]
        axes[1, 1].bar(bins[:-1], counts, width=45, align='edge', bottom=bottom_yes, label=trt_dict[trt], alpha=0.7)
        bottom_yes += counts  # Update bottom for the next treatment group

    axes[1, 0].set_ylabel('Number of Data Points')
    axes[1, 0].set_xlabel('Assessment point (d)')
    axes[1, 0].set_xlim(xlim)  # Match x-axis to the upper plot
    axes[1, 0].set_ylim([0,800])
    axes[1, 0].legend(title='Treatment Arm')

    axes[1, 1].set_xlabel('Assessment point (d)')
    axes[1, 1].set_xlim(xlim)  # Match x-axis to the upper plot
    axes[1, 1].set_ylim([0,800])  # Match x-axis to the upper plot
    axes[1, 1].legend(title='Treatment Arm')

    ## Save or Show the plot ###
    if save_path is not None:
        folder_path = Path(save_path) / 'mov_av'
        folder_path.mkdir(parents=True, exist_ok=True)
        fig_name = pred_col_name + '_' + str(rater[0]) + '.jpg' if rater is not None else var_result + '.jpg'
        plt.savefig(folder_path / fig_name, dpi=300)

    if show:
        plt.tight_layout()
        plt.show()
        
        
        
def plot_RR_curves_med_mod_smooth(data, var_mod, var_out, results, outcomes_dict_fig, med_dict_fig, med_values, xlim, ylim, rater=None, show=True, save_path=None):
    
    var_result = var_mod + '_' + var_out
    title = ['14-Month Outcomes \n {} \n {}'.format(med_dict_fig[var_mod][0], outcomes_dict_fig[var_out]), 
             '14-Month Outcomes \n {} \n {}'.format(med_dict_fig[var_mod][1], outcomes_dict_fig[var_out])]
    values = med_values[var_mod]
    result = results[var_result]
    data = data_dict[qst]
    data['predicted'] = result.predict(data)

    # Filter data for different conditions
    if var_mod == 'd2dresp':
        no_mod = data[(data[var_mod].isin(values[:2])) & (data['days_baseline'] >= 0) & (data['days_baseline'] <= 450)]
        yes_mod = data[(data[var_mod].isin(values[2:])) & (data['days_baseline'] >= 0) & (data['days_baseline'] <= 450)]
    else:
        no_mod = data[(data[var_mod] == values[0]) & (data['days_baseline'] >= 0) & (data['days_baseline'] <= 450)]
        yes_mod = data[(data[var_mod] == values[1]) & (data['days_baseline'] >= 0) & (data['days_baseline'] <= 450)]

    # Create subplots (2 rows, 2 columns) with shared x-axis
    fig, axes = plt.subplots(2, 2, figsize=(20, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    if rater is not None: 
        plt.suptitle('Rater: ' + rater)

    ### First row: Line plots (No Anxiety and Anxiety) ###
    
    # Plot for "No Anxiety" subgroup
    for trt in no_mod['trtname'].unique():
        subset = no_mod[no_mod['trtname'] == trt]
        n = subset.shape[0]
        smoothed = lowess(subset['predicted'], subset['days_baseline'], frac=0.1)
        axes[0, 0].plot(smoothed[:, 0], smoothed[:, 1], label=trt_dict[trt] + ' (n = {})'.format(n))

    axes[0, 0].set_title(title[0])
    axes[0, 0].set_ylabel('Smoothed Predicted Score')
    axes[0, 0].legend(title='Treatment Arm')
    axes[0, 0].set_xlim(xlim)  # Set x-axis from 0 to 450
    axes[0, 0].set_ylim(ylim)  # Set y-axis

    # Plot for "Anxiety" subgroup
    for trt in yes_mod['trtname'].unique():
        subset = yes_mod[yes_mod['trtname'] == trt]
        smoothed = lowess(subset['predicted'], subset['days_baseline'], frac=0.1)
        axes[0, 1].plot(smoothed[:, 0], smoothed[:, 1], label=trt_dict[trt] + ' (n = {})'.format(n))

    axes[0, 1].set_title(title[1])
    axes[0, 1].set_xlim(xlim)  # Set x-axis from 0 to 450
    axes[0, 1].set_ylim(ylim)  # Ensure y-axis is from 0 to 3

    ### Second row: Stacked bar plots for the number of data points colored by treatment group ###
    
    # Define bins for the time points (based on 'days_baseline')
    bins = np.linspace(0, 450, num=10)  # Adjust the number of bins as needed
    
    # Bar plot for "No Anxiety"
    trt_groups_no = no_mod['trtname'].unique()
    counts_by_trt_no = {trt: np.histogram(no_mod[no_mod['trtname'] == trt]['days_baseline'], bins=bins)[0] for trt in trt_groups_no}
    
    # Initialize bottom for stacked bars
    bottom_no = np.zeros(len(bins)-1)

    # Plot the stacked bars for "No Anxiety"
    for trt in trt_groups_no:
        counts = counts_by_trt_no[trt]
        axes[1, 0].bar(bins[:-1], counts, width=45, align='edge', bottom=bottom_no, label=trt_dict[trt], alpha=0.7)
        bottom_no += counts  # Update bottom for the next treatment group

    axes[1, 0].set_ylabel('Number of Data Points')
    axes[1, 0].set_xlabel('Assessment point (d)')
    axes[1, 0].set_xlim(xlim)  # Match x-axis to the upper plot
    axes[1, 0].set_ylim([0,800])
    axes[1, 0].legend(title='Treatment Arm')

    # Bar plot for "Anxiety"
    trt_groups_yes = yes_mod['trtname'].unique()
    counts_by_trt_yes = {trt: np.histogram(yes_mod[yes_mod['trtname'] == trt]['days_baseline'], bins=bins)[0] for trt in trt_groups_yes}
    
    # Initialize bottom for stacked bars
    bottom_yes = np.zeros(len(bins)-1)

    # Plot the stacked bars for "Anxiety"
    for trt in trt_groups_yes:
        counts = counts_by_trt_yes[trt]
        axes[1, 1].bar(bins[:-1], counts, width=45, align='edge', bottom=bottom_yes, label=trt_dict[trt], alpha=0.7)
        bottom_yes += counts  # Update bottom for the next treatment group

    axes[1, 1].set_xlabel('Assessment point (d)')
    axes[1, 1].set_xlim(xlim)  # Match x-axis to the upper plot
    axes[1, 1].set_ylim([0,800])  # Match x-axis to the upper plot
    axes[1, 1].legend(title='Treatment Arm')

    ### Save or Show the plot ###
    if save_path is not None:
        folder_path = Path(save_path) / 'smooth'
        folder_path.mkdir(parents=True, exist_ok=True)
        fig_name = var_result + '_' + str(rater[0]) + '.jpg' if rater is not None else var_result + '.jpg'
        plt.savefig(folder_path / fig_name, dpi=300)

    if show:
        plt.tight_layout()
        plt.show()
'''