import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

###################### INSPECT ################################
def scatter_per_subject(df, var, days_baseline = None, outcomes_dict_fig = None, groupby = 'src_subject_id', x_axis_type = 'days', n_subjects = None, cutoffs = None, plot_L = False, save_path = None):
    
    if days_baseline is not None : 
        if isinstance(days_baseline, int):
            df = df[df['days_baseline'] < days_baseline]
        elif isinstance(days_baseline, (list, tuple, np.ndarray)):
            df = df[ (df['days_baseline'] > days_baseline[0]) & (df['days_baseline']< days_baseline[1])]
            
    if plot_L == False:
        df = df[df['trtname'] != 'L']
    
    groups = df.groupby(groupby)
        # Optionally limit the number of subjects to plot
    if n_subjects is not None:
        # Select only the first n_subjects groups
        groups = list(groups)[:n_subjects]

    plt.figure(figsize=(10, 7))
    
    trtname_c_dict = {"P": 'purple', "M": 'red', "C": 'orange', "A": "green"}

    # Plot each subject's data in a different color
    for name, group in groups:
        if name != 'L':
            if x_axis_type == 'months':
                # Convert days to months by dividing by 30
                plt.scatter(group['days_baseline'] / 30, group[var], color= trtname_c_dict[name], label=name + ', n = ' + str(group['src_subject_id'].unique().shape[0]))
            else:
                # Use days as is
                plt.scatter(group['days_baseline'], group[var], color=  trtname_c_dict[name],  label=name+ ', n = ' +str(group['src_subject_id'].unique().shape[0]))

    # Set axis labels
    plt.ylabel(var.upper())

    if x_axis_type == 'days':
        plt.xlabel('Days Since Baseline')
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(50))  # Set major ticks every 200 days
    elif x_axis_type == 'months':
        plt.xlabel('Months Since Baseline')
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))  # Set major ticks every 1 month
  
  
    if cutoffs is not None :
        for cutoff in cutoffs:
            plt.axvline(x=cutoff[1], color='r', linestyle='--')
    # Add labels and a legend

    plt.ylabel(var.upper())
    plt.tight_layout()
    
    if groupby == 'trtname':
        plt.legend(title='Treatment arm', bbox_to_anchor=(1.05, 1), loc='upper left')
    # Show the plot
 
    
    plt.title(outcomes_dict_fig[var] + ' Assessements Over Time')
    
    if save_path is not None:
        plt.savefig(Path(save_path ,  'scattered_' + var + '_gb_' + groupby + '.png'), bbox_inches='tight')
    plt.show()
    

########################## RR ##################################
def get_point_av(df, pred_col_name, timepoints_range):
    df.sort_values('days_baseline')
    subset = df[['days_baseline', pred_col_name]]
    point_av = np.array([ subset[subset['days_baseline'].isin(range(range_t[0], range_t[1]))].mean(axis = 0) for range_t in timepoints_range])
    return point_av 


def get_timepoints_range(timepoints, delta):
    return  [[time - delta, time + delta] for time in timepoints]

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

def extract_line_plot(df, pred_col_name, type_plot, timepoints, delta = None, pnt_av = None,  window = None):
    
    if timepoints is not None : 
        timepoints_range = get_timepoints_range(timepoints, delta)
        ptn_av = get_point_av(df,pred_col_name, timepoints_range )
    
    subset = df.dropna(subset = 'days_baseline')
    subset = df.dropna(subset=pred_col_name)
    subset = df[df[pred_col_name] != 999.0]
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
    elif type_plot == "exp":
        print('type negative exponential fit')
        if pnt_av is not None and len(pnt_av) >= 4:
            # Define a negative exponential function
            def negative_exponential(x, a, b, c):
                return a * np.exp(-b * x) + c
            
            try:
                # Fit the negative exponential to the timepoints
                params, _ = curve_fit(negative_exponential, pnt_av[:, 0], pnt_av[:, 1], p0=(1, 0.1, 1))
                x_fit = np.linspace(min(pnt_av[:, 0]), max(pnt_av[:, 0]), 100)
                smoothed = negative_exponential(x_fit, *params)
                
                smoothed_df = pd.DataFrame({
                    'days_baseline': x_fit,
                    'smoothed_value': smoothed
                })
                print(f"Exponential fit parameters: a={params[0]}, b={params[1]}, c={params[2]}")
            except RuntimeError:
                print("Exponential fit failed to converge.")
                return
        else:
            print("Not enough points for exponential fit.")
            return
    else:
        print("Plot type not recognized.")
        return
    print("SMOOTHED shape", smoothed_df.shape)
    return smoothed_df 

def plot_RR_results(data, var_mod, var_out, med_values, type_plot, window, timepoints,delta,  trt_dict , med_dict_fig, outcomes_dict_fig, rater = None, x_lim= [0,3], y_lim= [-10,450], show = True, save_path= None):
    
    timepoints_range = get_timepoints_range(timepoints, delta)
    
    pred_col_name= 'predicted_' + var_mod + '_' + var_out

    trtnames = data['trtname'].unique()
    title = ['14-Month Outcomes \n {} \n {}'.format(med_dict_fig[var_mod][0], outcomes_dict_fig[var_out]), 
                '14-Month Outcomes \n {} \n {}'.format(med_dict_fig[var_mod][1], outcomes_dict_fig[var_out])]
    
    type_plot_dict ={"poly_fit": "Polyniomial fit on four point average, degree 2", 
                 "mov_av": "Moving average on predicted score, window = {}".format(window),
                 "smooth": "Smoothed Predicted Score"}
    
    
    values = med_values[var_mod]
    print(values)
    # Create subplots: 2 rows (1st row for polynomial fit, 2nd row for bar plot)
    if var_mod == 'd2dresp':
        no_mod = data[(data[var_mod].isin(values[:2])) & (data['days_baseline'] >= -window) & (data['days_baseline'] <= 450)]
        yes_mod = data[(data[var_mod].isin(values[2:])) & (data['days_baseline'] >= -window) & (data['days_baseline'] <= 450)]
    else:
        no_mod = data[(data[var_mod] == values[0]) & (data['days_baseline'] >= -window) & (data['days_baseline'] <= 450)]
        yes_mod = data[(data[var_mod] == values[1]) & (data['days_baseline'] >=-window) & (data['days_baseline'] <= 450)]
        
    print("YES MOD: ", yes_mod.shape)
    print("NO MOD :", no_mod.shape)
        
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True, sharey='row')
    
    if rater is not None: 
        plt.suptitle('Rater: ' + rater)
        pred_col_name = pred_col_name +'_' +str(rater[0])

    # Loop over each treatment name and plot both the 'yes' and 'no' condition in separate subplots
    for trt in trtnames:
        
        df_yes = yes_mod[yes_mod['trtname'] == trt].dropna(subset=[pred_col_name])
        df_no = no_mod[no_mod['trtname'] == trt].dropna(subset=[pred_col_name])
        print("DF per TRT: ",trt, df_yes.shape, df_no.shape)
        
        point_av_yes = get_point_av(df_yes, pred_col_name, timepoints_range)
        point_av_no = get_point_av(df_no, pred_col_name, timepoints_range)
        print("PTN AV: ", point_av_yes, point_av_no)
        
        line_yes = extract_line_plot(df_yes, pred_col_name, type_plot,point_av_yes, window)
        line_no = extract_line_plot(df_no, pred_col_name, type_plot,point_av_no, window)

        print("LINE YES, LINE NO smoothed value")
        print(line_yes['smoothed_value'].shape, line_no['smoothed_value'].shape)
        print("LINE YES, LINE NO days baseline ")
        print(line_yes['days_baseline'].shape, line_no['days_baseline'].shape)
        if line_yes is None or line_no is None :
            print('Computation has failed for {} : '.format(type_plot_dict[type_plot]) + str(pred_col_name))
            return 
        
        axes[0,0].plot(line_yes['days_baseline'],line_yes['smoothed_value'],  label= trt_dict[trt])
        axes[0,0].scatter(point_av_yes[:,0], point_av_yes[:,1])
        
        axes[0,1].plot(line_no['days_baseline'],line_no['smoothed_value'],  label= trt_dict[trt])
        axes[0,1].scatter(point_av_no[:,0], point_av_no[:,1])
        
        
    axes[0, 0].set_xlabel('Assessment point (d)')
    axes[0, 0].set_ylabel(type_plot_dict[type_plot])
    axes[0, 0].legend(title='Treatment Arm')
    axes[0, 0].set_xlim(x_lim)  # Set x-axis limits
    axes[0, 0].set_ylim(y_lim)  # Set y-axis limits
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


    if type_plot == "ploy_fit":
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
            
    else: 
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

    axes[1, 1].set_xlabel('Assessment point (d)')
    axes[1, 0].set_xlabel('Assessment point (d)')
    axes[1, 1].set_xlim(x_lim)  # Match x-axis to the upper plot
    axes[1, 1].set_ylim([0,800])  # Match x-axis to the upper plot
    axes[1, 1].legend(title='Treatment Arm')

    if save_path is not None:
        save_path.mkdir(exist_ok=True)
        folder_path = Path(save_path) / str(type_plot)
        folder_path.mkdir(exist_ok=True)
        fig_name = pred_col_name + '_' + str(rater[0]) + '.jpg' if rater is not None else pred_col_name + '.jpg'
        print(folder_path)
        plt.savefig(folder_path / fig_name, dpi=300)

    if show:
        print(show)
        plt.tight_layout()
        plt.show()
        
    plt.close(fig)
    
    
    
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
