from pathlib import Path 
import ast 
import pandas as pd 
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from decimal import Decimal, ROUND_DOWN
from helper import audit
import os 
import re 
from collections import defaultdict

def value_matches(table_value,input_value):

    rater_lower = input_value.lower()
    value_lower = table_value.lower()
    return rater_lower in  value_lower

def feature_selection_matches(table_value, corr_select):
    """
    Check if the table's Feature Selection Method text matches the expected flag.
    If corr_select is True, the text must indicate correlation selection.
    If False, it must indicate no/insufficient feature selection.
    """
    value_lower = table_value.lower()
    if corr_select:
        return "correlation selector" in value_lower
    else:
        # Here we assume that if not using correlation selector, 
        # the text indicates no feature selection (or "enough feature selection")
        return ("no feature selection" in value_lower)


######### this function ode snot work properly 
def find_result_in_file(file_path_save, model_type, corr_select, thr_drop_row, rater_pred, rater_out):
    
    rater_dict = {
        "m": "Mother", 
        "f": "Father", 
        "t": "Teacher", 
        "all": "All Raters"
    }

    if not os.path.exists(file_path_save):
        return False

    print("Reading file... ")
    df = pd.read_csv(file_path_save)

    required_columns = [
        'Model Name', 'Feature Selection Method', 
        'Outcome Variable', 'Number of Features',  # Number of Features contains string that specifies input data
        'Threshold Drop Row'
    ]

    if not all(col in df.columns for col in required_columns):
        return False

    # Determine feature selection method
    feature_select_meth_to_find = "correlation" if corr_select else "No"

    # Ensure rater values are in the correct format for lookup
    rater_pred_ = "all" if rater_pred is None else rater_pred
    rater_pred_to_find = rater_dict[rater_pred_]
    rater_out_to_find = rater_dict[rater_out]
    thr_drop_row_to_find = thr_drop_row

    # Extract Outcome Variable for verification
    outcome_var_col = df['Outcome Variable'].iloc[0]
    print("Extracted Outcome Variable:", outcome_var_col.split("\n")[-1].split(" ")[-1])
    
    pred_rater = df["Number of Features"].iloc[0]
    print("Pred rater ", pred_rater.split(":")[1].split("\n")[0].strip())

    # Ensure the extraction and comparison is applied correctly
    match = df[
        (df['Model Name'] == model_type) &
        (df['Feature Selection Method'].apply(lambda x: value_matches(x, feature_select_meth_to_find))) &
        (df["Threshold Drop Row"] == thr_drop_row_to_find) &
        (df['Outcome Variable'].apply(lambda x: value_matches(x.split("\n")[-1].split(" ")[-1].strip(), rater_out_to_find))) &
        (df["Number of Features"].apply(lambda x: value_matches(x.split(":")[1].split("\n")[0].strip(), rater_pred_to_find))) 
    ]


    if not match.empty:
        print("\nResults already exists in file.. ")
        print("Skipping ...\n")

    return not match.empty

def get_support_data(pipeline):
    # Define the extensions to check
    extensions = ['m', #mother 
                #'_p'# proffesionals
                'f', # father,
                'c',# child,
                't'] # teacher 
    processed_feature_names = pipeline["preprocessor"].get_feature_names_out()
# Count extensions (assumes extensions is defined, e.g. extensions = ["m", "f", "t", "c", "n"])
    extension_counts = {ext: sum(col.endswith(ext) for col in processed_feature_names) for ext in extensions}
    print(extension_counts)

    # Build a support text using plain text with aligned keys.
    support_data = {
        "Total": len(processed_feature_names),
        "Mother": extension_counts.get("m", 0),
        "Father": extension_counts.get("f", 0),
        "Teacher": extension_counts.get("t", 0),
        "Child": extension_counts.get("c", 0)
    }
    return support_data

#support_data_dict = get_support_data(pipeline)


def format_dict_to_text(data_dict): # get support data as formated text for results table 

    
    max_key_len = max(len(key) for key in data_dict.keys())
    lines = []
    for key, value in data_dict.items():
        # Right-align the keys so the colon lines up.
        line = "{:>{width}}: {}".format(key, value, width=max_key_len)
        lines.append(line.strip())
    formatted_text = "\n".join(lines)
    return formatted_text

def get_results_from_random_search(random_search, outcome_short, rater_out, rater_pred, thr_drop_missing, get_dist=False):
    
    cols_res = ['Model Name', 'Cross Validation Type', 'Hyperparameters',
                'Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error',
                'R² Score', 'Outcome Variable', 'Number of Features',
                'Feature Selection Method', 'Threshold Drop Row']

    cols_res_dist = cols_res + [  # store additional data 
        'Std RMSE', 'Std MAE', 'Std R²',  
        'Train RMSE', 'Train MAE', 'Train R²',  
        'Mean Fit Time', 'Mean Score Time'
    ]
    
    outcome_result_dict = {
        "ODD": "SNAP ODD Symptoms T Score", "HYP": "SNAP Hyperactivity Symptoms T Score", 
        "INATT": "SNAP Inattention Symptoms T Score", "INTERN": "SSRS Internalizing Symptoms", 
        "SS": "SSRS Social Skills T Score", "DOM": "Parental Dominance Mean Score", 
        "INTIM": "Parent-Child Intimacy Mean Score"
    }

    rater_dict = {
        "m": "Mother", 
        "f": "Father", 
        "t": "Teacher", 
        "all": "All Raters"
    }
    
    outcome_table = outcome_result_dict[outcome_short] + "\n- rated by {}".format(rater_dict[rater_out])
    
    cv_results = random_search.cv_results_
    results_df = pd.DataFrame(cv_results)
    cross_val_strategy = str(random_search.cv)

    best_index = random_search.best_index_
    best_results = results_df.iloc[best_index]  # Best results relating to best hyperparam configuration 
    best_pipeline = random_search.best_estimator_
    
    support_data = get_support_data(best_pipeline)
    formatted_support_data_short = format_dict_to_text(support_data)
    
    input_rater_short = rater_pred if rater_pred is not None else "all"
    formatted_support_data = "Selection : {} \n {}".format(rater_dict[input_rater_short], formatted_support_data_short)

    print("Metrics for Best Parameters:")
    print(best_results[['mean_test_rmse', 'mean_test_mae', 'mean_test_r2']])
    
    if get_dist:
        n_splits = max(int(key.split('_')[0].replace("split", "")) 
                for key in random_search.cv_results_.keys() 
                if key.startswith("split")) + 1
        
        r2 = [cv_results[f"split{i}_test_r2"][best_index] for i in range(n_splits)]
        mae = [cv_results[f"split{i}_test_mae"][best_index] for i in range(n_splits)]
        mse = [cv_results[f"split{i}_test_rmse"][best_index] ** 2 for i in range(n_splits)]
        rmse = [cv_results[f"split{i}_test_rmse"][best_index] for i in range(n_splits)]

        # Extract additional metrics for later analysis
        std_rmse = truncate(best_results['std_test_rmse'], 4)
        std_mae = truncate(best_results['std_test_mae'], 4)
        std_r2 = truncate(best_results['std_test_r2'], 4)
        train_rmse = truncate(best_results['mean_train_rmse'], 4)
        train_mae = truncate(best_results['mean_train_mae'], 4)
        train_r2 = truncate(best_results['mean_train_r2'], 4)
        mean_fit_time = truncate(best_results['mean_fit_time'], 4)
        mean_score_time = truncate(best_results['mean_score_time'], 4)

    else: 
        rmse = - truncate(best_results['mean_test_rmse'], 4)
        mse = truncate(best_results['mean_test_rmse'] ** 2, 4)
        mae = - truncate(best_results['mean_test_mae'], 4)
        r2 = truncate(best_results['mean_test_r2'], 4)

    model_name = best_pipeline.named_steps['regressor'].__class__.__name__

    params_values = dict(zip([key.replace('regressor__', '') for key in random_search.best_params_.keys()], 
                             random_search.best_params_.values()))
    
    formatted_params = format_dict_to_text(params_values)

    if 'correlation_selector' in random_search.best_estimator_.named_steps.keys():
        n_estimators = best_pipeline.named_steps['correlation_selector'].get_params()["model"].get_params()["n_estimators"]
        name = type(best_pipeline.named_steps['correlation_selector'].get_params()["model"]).__name__
        threshold = best_pipeline.named_steps['correlation_selector'].get_params()['threshold']
        
        corr_params_dict = {"Model Type": name, "N Estimators": n_estimators, "Threshold": threshold}
        corr_params_text = format_dict_to_text(corr_params_dict)

        feature_select_meth = 'Correlation Selector (Model-Based) \n{}'.format(corr_params_text)
        print(feature_select_meth)
        
    else: 
        feature_select_meth = 'No feature selection'

    # Choose the appropriate column list based on `get_dist`
    new_row = dict(zip(
        cols_res_dist if get_dist else cols_res,
        [
            model_name, cross_val_strategy, formatted_params, mse, rmse, mae, r2,  
            outcome_table, formatted_support_data, feature_select_meth, thr_drop_missing
        ] + ([std_rmse, std_mae, std_r2, train_rmse, train_mae, train_r2, mean_fit_time, mean_score_time] if get_dist else [])
    ))

    return new_row


def save_cv_result_to_table(file_path_save, new_row, nrows2drop = 0 , save= False, reduced= False, verify_before_save = True):
    
    cols_res = list(new_row.keys())
    
    if os.path.exists(file_path_save):
        df_result  = pd.read_csv(file_path_save)

        if 'Unnamed: 0' in df_result.columns:
            df_result = df_result.drop(columns= 'Unnamed: 0')
        print("Column names from table : ", list(df_result.columns))
        # will throw an error is the column naames in file ar enot the same as cols res. ß
    else:
        df_result = pd.DataFrame(columns= cols_res)
    
    if df_result.empty:
        result_df = pd.DataFrame([new_row]) # if dataframe doesnt exists yet or is empty, create it 
    else: 
        if not ((df_result == new_row).all(axis=1)).any() : # check if row already exists in saved file 
            result_df = pd.concat([df_result, pd.DataFrame([new_row])], ignore_index=True)
        else: 
            print('\n ROW ALREADY EXISTS')
            result_df = df_result

    print("\n NEW RESULTS TABLE (not saved):") 
    print(result_df[:10].to_string())
    
    
    if verify_before_save:
        reduce_check= input("Do you wish to remove a row / N rows? (y,n)")
        if reduce_check.upper() == "Y":
            nrows2drop = int(input("How many rows should be dropped? (enter a number)"))
            df_result_reduced = result_df.iloc[:df_result.shape[0]-nrows2drop, :] # if desired rwmove some rows (if mistake)        
            print("\n REDUCED RESULTS TABLE :") 
            print(df_result_reduced[:10].to_string())
        
            reduced_  = input("Save reduced table? (y/n)...")
            reduced = True if reduced_.upper() == "Y" else False
        
        if not reduce_check or not reduced:
            print(" \n NEW RESULTS TABLE (not saved):") 
            print(result_df[:10].to_string())
            save_ = input("Save full table? (y,n)" )

            save = True if save_.upper() == "Y" else False
    
    else: 
        if reduced : 
            df_result_reduced = result_df.iloc[:df_result.shape[0]-nrows2drop, :] # if desired rwmove some rows (if mistake)        
            print("REDUCED RESULTS TABLE :") 
            print(df_result_reduced[:10].to_string())
    

    if save :
        print('... Saving')
        if reduced:
            print('... Reduced table')
            df_result_reduced.to_csv(file_path_save)
        else:
            print('... Full')
            result_df.to_csv(file_path_save)
        verify   = pd.read_csv(file_path_save)
    
        print("\n TABLE SAVED TO FILE : ")
        print(verify[:20].drop(columns="Unnamed: 0").to_string())
    else: 
        print("Save set to False. Set save to True to save the new Dataframe.")

def truncate(value, decimals):
    value = Decimal(value)
    return float(value.quantize(Decimal(f"1.{'0' * decimals}"), rounding=ROUND_DOWN))


def rater_count(data):
    # Define the extensions to check ## keep them with underscore 
    extensions = ['_m', #mother 
                #'_p'# proffesionals
                '_f', # father,
                '_c',# child,
                '_t'] # teacher 

    # Count columns for each extension
    extension_counts = {ext: sum(col.endswith(ext) for col in data.columns) for ext in extensions}
    clean_extension_counts= { (k[1:] if k.startswith('_') else k): v for k, v in extension_counts.items() }

    for ext, count in clean_extension_counts.items():
        print(f"Columns ending with '{ext}': {count}")
    return clean_extension_counts
    
def parse_formatted_text_to_dict(formatted_text):
    """
    Parses the formatted text (e.g., created by format_dict_to_text) back into a dictionary.
    
    Expects lines of the form:
        Key: Value
    where Value is assumed to be an integer. Adjust parsing if the values can be non-integer.
    """
    data_dict = {}
    lines = formatted_text.strip().split('\n')
    for line in lines:
        # Split the line into key and value on the first ': '
        parts = line.split(': ', 1)
        if len(parts) == 2:
            key, value_str = parts
            key = key.strip()
            # Convert value to integer; adjust if you need floats or strings
            try:
                value = int(value_str)
            except ValueError:
                # Handle non-integer values if necessary
                value = value_str
            data_dict[key] = value

    return data_dict

def get_params_from_result(file_path_save, how= "best", index = None): 
 ######### change arams t fit new table columns 
 
    file_name_save = str(file_path_save).split("/")[-1]
    rater_dict_rev = {"Mother": "m", 
                     "Father": "f", 
                     "Teacher": "t", 
                     "All Raters": None}

    outcome_dict = {'ODD':  "snap_snaoddt", "HYP": "snap_snahypat", "INATT" :"snap_snainatt" , "INTERN": "ssrs_sspintt", "SS": "ssrs_ssptosst", "DOM": "pcrc_pcrcpax", "INTIM": "pcrc_pcrcprx"}

    df_result  = pd.read_csv(file_path_save)

    if how == "best":
        print("Extracting best result...")
        result = df_result.loc[df_result['R² Score'] == df_result['R² Score'].max(), :].drop(columns='Unnamed: 0')
        
    elif how == 'index':
        print("Extracting result at index {} ...".format(index))
        result = df_result.iloc[[index], :].drop(columns='Unnamed: 0')

    else: 
        raise ValueError("Result to extract not specified..")
    
        
    params = parse_formatted_text_to_dict(result['Hyperparameters'].iloc[0])
    model_type =result['Model Name'].iloc[0]
    out_str  = result["Outcome Variable"].iloc[0]
    out_str.split("\n")[1].split(" ")[-1]
    outcome_var = outcome_dict[file_name_save.split(".")[0].split("_")[-1]]
    rater_out = rater_dict_rev[out_str.split("\n")[-1].split(" ")[-1]]
    
    rater_pred = rater_dict_rev[parse_formatted_text_to_dict(result['Number of Features'].iloc[0])["Selection"].strip()]

    if 'correlation_selector__threshold' in params:
        corr_select = True 
        thr_corr= params['correlation_selector__threshold']
        params.pop('correlation_selector__threshold' )
    else: 
        thr_corr = None
        corr_select = False
        
    if rater_pred == 'all':
        rater_pred = None

    thr_drop_missing = result['Threshold Drop Row'].iloc[0]
    original_r2 = result['R² Score']


    print("original r2 result : ", original_r2.iloc[0])
    print('model_type', model_type)
    print('corr_select :', corr_select)
    print("thr_corr: ", thr_corr )
    print('outcome : ',outcome_var)
    print("rater out", rater_out)
    print('rater input data : ', rater_pred)
    print("thr drop missing: ", thr_drop_missing)
    print("params model : ", params)


    # thr_corr = params['subsample']
    # params.pop('subsample', None)
    return  model_type,  corr_select,   thr_corr, params,  outcome_var,rater_out, rater_pred, thr_drop_missing, original_r2.iloc[0]



def check_overlap(ord_vars, num_vars,  cat_vars_str, cat_vars_num):
    print("\n Checking for overlaps.. ")
    feature_assignments = defaultdict(list)

    # Track which transformer each column is assigned to
    for feature in num_vars:
        feature_assignments[feature].append('num_pipe')
    for feature in cat_vars_str:
        feature_assignments[feature].append('cat_str_pipe')
    for feature in cat_vars_num:
        feature_assignments[feature].append('cat_num_pipe')
    for feature in ord_vars:
        feature_assignments[feature].append('ord_pipe')

    # Identify duplicates
    duplicates = {feat: pipes for feat, pipes in feature_assignments.items() if len(pipes) > 1}

    if duplicates:
        print("❌ Duplicate features assigned to multiple transformers:")
        for feature, pipes in duplicates.items():
            print(f"{feature}: {pipes}")
        raise ValueError("Fix duplicate column assignments before running the pipeline.")
        
def prepare_data(pred, out, rater_pred, rater_out, thr_drop_missing, outcome_var):
    print("\nPreparing data...")

    if rater_pred is not None:
        col_pred = [col for col in pred.columns if col.endswith(rater_pred) or col.endswith("c")]
        col_pred.append("src_subject_id")
        col_pred.append("trtname")
        pred = pred[col_pred]

    # Process outcome columns
    col_out_rater = [col for col in out.columns if col.endswith(rater_out) ]
    out_rater = out[np.concatenate((['src_subject_id'], col_out_rater))]

    out_rater = out_rater.rename(
        columns={col: f"{col}_out" for col in out_rater.columns if col != 'src_subject_id'}
    )
    

    y_col = "_".join([outcome_var, rater_out, "out"])


    # Merge predictions and outcomes on 'src_subject_id'
    try : 
        data = pd.merge(out_rater[[y_col, 'src_subject_id']], pred, how='left', on='src_subject_id')
        data = audit.remove_cols(data, thr_drop_missing=thr_drop_missing)
        if y_col not in data.columns: 
            print("{} has been removed from dataframe during audit, as there is not enough data available.".format(y_col))
            return None, None, None, None 
    except KeyError as e:
        print(str(e))
        return None, None, None, None 

    # Prepare feature matrix and target array
    X_cols = [col for col in data.columns if col not in [y_col]]
    data = data.dropna(subset=[y_col])
    y = np.array(data[y_col])
    print("\nOutcome to predict (Y) : ", y_col)
    print("\nY shape: ", y.shape)
    df_X = data[X_cols].drop(columns='src_subject_id')
    print("X shape ", df_X.shape)
    rater_count_df_X = rater_count(df_X)

    return data, df_X, y, rater_count_df_X

def get_var_types(df_X, types_file_path ):
    print("\nExtracting variable types from file...")
    
    col_names_data = list(df_X.columns)

    ord_vars, num_vars, cat_vars, rest = [], [], [], []

    types_df = pd.read_excel(types_file_path, sheet_name='Sheet1')

    for _, row in types_df.iterrows():
        var_name = row.iloc[1]  # e.g. variable name in the spreadsheet
        var_type = row.iloc[4]  # e.g. "ord" / "num" / "cat"

        # Collect all columns in `data` that contain `var_name`
        var_in_data = [col for col in col_names_data if var_name in col]

        if var_type == "ord":
            ord_vars.append(var_in_data)
        elif var_type == "num":
            num_vars.append(var_in_data)
        elif var_type == "cat":
            cat_vars.append(var_in_data)
        else: 
            rest.append(var_in_data)

    # Example: manually add a column named 'trtname' to cat_vars
    cat_vars.append(['trtname'])

    # Flatten each list-of-lists into a single array
    ord_vars = np.concatenate(ord_vars)
    cat_vars = np.concatenate(cat_vars)
    num_vars = np.concatenate(num_vars)

    # Convert them to plain Python strings
    ord_vars = [str(col) for col in ord_vars]
    cat_vars = [str(col) for col in cat_vars]
    num_vars = [str(col) for col in num_vars]
    if 'masc_ma22acx_c' in ord_vars: 
        ord_vars.remove('masc_ma22acx_c')
    if 'masc_ma31hfx_c' in ord_vars:
        ord_vars.remove('masc_ma31hfx_c')
    
    cat_vars_str, cat_vars_num =  [], []
    
    for col in cat_vars:
        val = df_X[col].dropna().unique()[0]
        
        # Check if it's a (Python or NumPy) string
        if isinstance(val, (str, np.str_)):
            cat_vars_str.append(str(col))  # ensure column name is a Python str
        # Check if it's a (Python or NumPy) float
        elif isinstance(val, (float, np.floating)):
            cat_vars_num.append(str(col))  # ensure column name is a Python str
        else:
            rest.append(str(col))          # store in `rest` for debugging


    ord_vars = list(set(ord_vars))
    ord_vars = list(set(ord_vars) - set(num_vars) - set(cat_vars_str) - set(cat_vars_num))
    cat_vars_str = list(set(cat_vars_str) - set(num_vars) - set(ord_vars))
    cat_vars_num = list(set(cat_vars_num) - set(num_vars) - set(ord_vars))
    
    
    print("\nOrdinal variables:", len(ord_vars))
    print("Numeric variables:", len(num_vars))
    print("Categorical numeric variables:", len(cat_vars_num))
    print("Categorical string variables:", len(cat_vars_str))
    
    check_overlap(num_vars= num_vars, ord_vars= ord_vars, cat_vars_str=cat_vars_str, cat_vars_num = cat_vars_num)
    
    return ord_vars, num_vars,  cat_vars_str, cat_vars_num

def check_duplicates(df_X):
    print("\nChecking for duplicates...")
    dup_cols = df_X.columns[df_X.columns.duplicated()].tolist()
    print("Duplicated column names:", dup_cols)
    return dup_cols