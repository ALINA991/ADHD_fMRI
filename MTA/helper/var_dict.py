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

def get_baseline_vars():
    return ['src_subject_id', 'interview_date', 'interview_age', 'sex', 'site', 'days_baseline', 'relationship']

def get_baseline_types():
    baseline_types = {
        'src_subject_id': 'str',
        'interview_date': 'date', 
        'interview_age': 'int',
        'sex': 'categorical',
        'site': 'categorical',
        'days_baseline' : 'int'
    }
    return baseline_types
    

    
    
def get_hyps_to_vars():
    hyps_to_vars= {
    'site' : 'C(site)', 
    'time' : 'days_baseline', 
    'treat' : 'C(trtname, Treatment(reference="L"))',
    'site_treat' : 'C(site) * C(trtname, Treatment(reference="L"))',
    'time_treat': 'days_baseline * C(trtname, Treatment(reference="L"))',
    'site_time_treat' : 'days_baseline * C(trtname, Treatment(reference="L")) * C(site))',
}
    return hyps_to_vars
    
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

def get_relationship(): 
    relationship = {
        'Biological mom': 1,
        'Biological dad': 2,
        'Grandparent': 3,
        'Special education (sped) teacher': 4,
        'General education teacher': 5,
        'Occupational therapist': 6,
        'Speech and language therapist': 7,
        'Behavioral therapist': 8,
        'Paraprofessional': 9,
        'Aide': 10,
        'Principal': 11,
        'Administrator': 12,
        'Content teacher': 14,
        'Parent center director': 15,
        'Self': 16,
        'Adoptive mother': 17,
        'Adoptive father': 18,
        'Foster mother': 19,
        'Foster father': 20,
        'Grandmother': 21,
        'Grandfather': 22,
        'Step-mother': 23,
        'Step-father': 24,
        'Aunt': 25,
        'Uncle': 26,
        'Both parents': 28,
        'Grandmother from mother side': 31,
        'Grandfather from mother side': 32,
        'Grandmother from father side': 33,
        'Grandfather from father side': 34,
        'Brother': 36,
        'Sister': 37,
        'Cousin': 38,
        'Female caregiver': 39,
        'Male caregiver': 40,
        'Female child': 41,
        'Male child': 42,
        'Spouse/Mate': 43,
        'Friend': 44,
        'Parent': 45,
        'Significant other': 46,
        'Sibling': 47,
        'Son/Daughter': 48,
        'Son-in-law/Daughter-in-law': 49,
        'Other Relative': 50,
        'Paid caregiver': 51,
        'Friends': 52,
        'Roommate': 53,
        'Supervisor': 54,
        "Mother's boyfriend": 55,
        'Other parental figure': 56,
        'Summary': 57,
        'Counselor': 58,
        'Other female relative': 59,
        'Other male relative': 60,
        'Non-relative': 61,
        'Maternal Aunt': 62,
        'Maternal Uncle': 63,
        'Maternal Cousin': 64,
        'Paternal Aunt': 65,
        'Paternal Uncle': 66,
        'Paternal Cousin': 67,
        'Biological/Adoptive Mother and Grandmother': 68,
        'Biological/Adoptive Mother and Stepmother and Grandmother': 69,
        'Biological/Adoptive Mother and Grandmother and Foster Father': 70,
        'Biological/Adoptive Mother and Stepmother and Foster Mother': 71,
        'Biological/Adoptive Mother and Foster Mother': 72,
        'Biological/Adoptive Mother and Biological/Adoptive Father': 73,
        'Biological/Adoptive Mother and Stepmother and Biological/Adoptive Father': 74,
        'Biological/Adoptive Mother and Other': 75,
        'Biological/Adoptive Mother and Stepmother and Stepfather': 76,
        'Biological/Adoptive Mother and Stepfather': 77,
        'Biological/Adoptive Mother and Grandfather': 78,
        'Biological/Adoptive Mother and Stepmother and Foster Father': 79,
        'Biological/Adoptive Mother and Stepmother': 80,
        'Guardian, female': 81,
        'Other female': 82,
        'Guardian, male': 83,
        'Other male': 84,
        'Other/Grandparent/Nanny': 85,
        'Mother, Father, Guardian': 86,
        'Daughter, son, grandchild': 87,
        'Professional (e.g., social worker, nurse, therapist, psychiatrist, or group home staff)': 88,
        'Missing': -999,
        'Biological parent': 89,
        'Other': 90,
        'Stepparent': 91,
        'Adoptive parent': 92,
        'Foster parent': 93,
        'Co-worker': 94,
        'Independent Evaluator': 95
    }
    return {v: k for k, v in relationship.items()} # return revers dict 

def get_teacher():
    teacher_dict = {
    'Special education (sped) teacher': 4,
    'General education teacher': 5,
    'Content teacher': 14,
    }
    return {v: k for k, v in teacher_dict.items()}


def get_parents():
    parent_dict = {    'Biological mom': 1,
    'Biological dad': 2,
    'Grandparent': 3,
    'Adoptive mother': 17,
    'Adoptive father': 18,
    'Foster mother': 19,
    'Foster father': 20,
    'Grandmother': 21,
    'Grandfather': 22,
    'Step-mother': 23,
    'Step-father': 24,
    'Both parents': 28,
    "Mother's boyfriend": 55,
    'Other parental figure': 56,
    'Biological/Adoptive Mother and Grandmother': 68,
    'Biological/Adoptive Mother and Stepmother and Grandmother': 69,
    'Biological/Adoptive Mother and Grandmother and Foster Father': 70,
    'Biological/Adoptive Mother and Stepmother and Foster Mother': 71,
    'Biological/Adoptive Mother and Foster Mother': 72,
    'Biological/Adoptive Mother and Biological/Adoptive Father': 73,
    'Biological/Adoptive Mother and Stepmother and Biological/Adoptive Father': 74,
    'Biological/Adoptive Mother and Other': 75,
    'Biological/Adoptive Mother and Stepmother and Stepfather': 76,
    'Biological/Adoptive Mother and Stepfather': 77,
    'Biological/Adoptive Mother and Grandfather': 78,
    'Biological/Adoptive Mother and Stepmother and Foster Father': 79,
    'Biological/Adoptive Mother and Stepmother': 80,
    'Guardian, female': 81,    
    'Guardian, male': 83,
    'Mother, Father, Guardian': 86,
    'Biological parent': 89,
    'Stepparent': 91,
    'Adoptive parent': 92,
    'Foster parent': 93,
    }
    return  {v: k for k, v in parent_dict.items()}


def get_replication_vars(questionnaire): # vars for replication 
    
    if questionnaire == 'ssrs': 
        
        return  {
        'relationship' : 'relationship', # just use relationship directly
        'ssrs_ss_mean' :'ssptossx', #social skilla mean 
        'ssrs_ss_std' : 'ssptosst', #std
        'ssrs_int_mean' : 'sspintx', #internalizing 
        'ssrs_int_std' : 'sspintt'
    }
    

    elif questionnaire == 'snap': 
        
        return {
    'relationship' : 'relationship',
    'snap_inatt_mean' : 'snainatx', #inattentuin 
    'snap_inatt_tot' :'snainatt', # hyperactie 
    'snap_hyp_mean' : 'snahypax',
    'snap_hyp_tot' : 'snahypat',
    'snap_imp_mean' : 'snaimpux', #impusive 
    'snap_imp_tot': 'snaimput',
    'snap_odd_mean' :'snaoddx', #oppositional defiant 
    'snap_odd_tot' :  'snaoddt'
    }
        
    elif questionnaire == 'masc': 
        
        return {
    'relationship' : 'relationship', 
    'masc_tot_T_score' : 'masc_masctotalt'
    }  
        
    elif questionnaire == 'pc': #parent child 
        
        return {
    'pc_relationship' : 'relationship', 
    'pc_dominance_mean' : 'pcrcpax', #power assertion
    'pc_pro_social_mean' : 'pcrcprx' #personal closeness 
    }   
        
    elif questionnaire == 'wechsler': 
        
        return {
    'relationship' : 'relationship',
    'wiat_reading_sc': 'w1readb', # scaled scores 
    'wiat_math_sc' : 'w2math', 
    'wiat_read_sc' : 'w3spell'
    }   
        
    else:
        raise ValueError('Questionnaire name is not recognized. Questionnaires supported : ssrs snap, masc, pc, wechsler')   