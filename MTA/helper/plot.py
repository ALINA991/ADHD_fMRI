import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from pathlib import Path

def plot_RR_curves_med_mod_smooth(data, var_mod, var_out, results, outcomes_dict_fig, med_dict_fig , med_values, xlim, ylim, rater = None, show = True, save_path = None):
    
    var_result = var_mod + '_' + var_out
    title = ['14-Month Outcomes \n {} \n {}'.format(med_dict_fig[var_mod][0], outcomes_dict_fig[var_out]) , '14-Month Outcomes \n {} \n {}'.format(med_dict_fig[var_mod][1], outcomes_dict_fig[var_out])]
    values = med_values[var_mod]
    result = results[var_result]
    #data = data_dict[qst]
    data['predicted'] = result.predict(data)

    if var_mod == 'd2dresp':
        no_mod = data[(data[var_mod].isin(values[:2])) & (data['days_baseline'] >= 0) & (data['days_baseline'] <= 450)]
        yes_mod = data[(data[var_mod].isin(values[2:])) & (data['days_baseline'] >= 0) & (data['days_baseline'] <= 450)]
    else:
    # Apply LOESS smoothing for "No Anxiety" and "Anxiety" subgroups, and filter for days_baseline >= 0
        no_mod = data[(data[var_mod] == values[0]) & (data['days_baseline'] >= 0) & (data['days_baseline'] <= 450)]
        yes_mod = data[(data[var_mod] == values[1]) & (data['days_baseline'] >= 0) & (data['days_baseline'] <= 450)]

    # Create subplots (1 row, 2 columns) with shared y-axis
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=True)

    if rater is not None: 
        plt.suptitle('Rater: ' + rater)
    # Plot for "No Anxiety" subgroup
    for trt in no_mod['trtname'].unique():
        subset = no_mod[no_mod['trtname'] == trt]
        n = subset.shape[0]

        smoothed = lowess(subset['predicted'], subset['days_baseline'], frac=0.1)
        axes[0].plot(smoothed[:, 0], smoothed[:, 1], label=trt_dict[trt]+ ' (n = {})'.format(n))

    axes[0].set_title(title[0])
    axes[0].set_xlabel('Assessment point (d)')
    axes[0].set_ylabel('Smoothed Predicted Score')
    axes[0].legend(title='Treatment Arm')
    axes[0].set_xlim(xlim)  # Set x-axis from 0 to 450
    axes[0].set_ylim(ylim)    # Set y-axis from 0 to 3

    # Plot for "Anxiety" subgroup
    for trt in yes_mod['trtname'].unique():
        subset = yes_mod[yes_mod['trtname'] == trt]
        smoothed = lowess(subset['predicted'], subset['days_baseline'], frac=0.1)
        axes[1].plot(smoothed[:, 0], smoothed[:, 1],label=trt_dict[trt]+ ' (n = {})'.format(n))

    axes[1].set_title(title[1])
    axes[1].set_xlabel('Assessment point (d)')
    axes[1].set_xlim(xlim)  # Set x-axis from 0 to 450
    axes[1].set_ylim(ylim)   # Ensure y-axis is from 0 to 3

    if save_path is not None:
        fig_name = var_result + '_' + str(rater[0]) + '.jpg' if rater is not None else var_result + '.jpg'
        plt.savefig(Path(save_path, fig_name), dpi=300)
    # Show the plot
    if show: 
        plt.tight_layout()
        plt.show()
