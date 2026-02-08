import pandas as pd
import numpy as np
from scipy import stats

def check_for_missing_data(df):
    return df.isna().any().any()

def identify_which_cols_have_missing_data(df):
    return df.columns[df.isna().any()].tolist()

def identify_data_types_of_each_column(df):
    list_dtypes = list(df.dtypes.astype(str))
    list_feature_names = list(df.columns)
    dict_dtypes = dict(zip(list_feature_names, list_dtypes))
    return dict_dtypes

def isolate_present_values_for_cols_with_missing_data(df):
    #check if missing data an issue
    if check_for_missing_data(df):
        cols_missing_data = identify_which_cols_have_missing_data(df)
        dict_dtypes = identify_data_types_of_each_column(df)
        dict_missing_features_data = {}
        for col in cols_missing_data:
            data_type = dict_dtypes[col]
            arr = df[col].dropna()
            temp_dict = {}
            temp_dict['Data Type'] = data_type
            temp_dict['values'] = arr
            dict_missing_features_data[col] = temp_dict

        
        list_float_int_features = []
        list_string_features = []
        for col in dict_missing_features_data:
            data_type = dict_missing_features_data[col]['Data Type']
            if data_type == 'float64' or data_type == 'int64':
                list_float_int_features.append(col)
            elif data_type == 'object':
                list_string_features.append(col)

        return dict_missing_features_data, list_float_int_features, list_string_features
    else:
        return None, None, None


def check_distributon_int_float_features(dict_missing_features_data, list_float_int_features):
    dict_distribution_results = {}
    for feature in list_float_int_features:
        arr = dict_missing_features_data[feature]['values']
        
        # Shapiro-Wilk test for normality
        # H0: the sample is drawn from a normal distribution
        stat, p_value = stats.shapiro(arr)
        
        # If p > 0.05, we fail to reject H0 (assume normal) -> Use Mean
        # If p <= 0.05, we reject H0 (not normal) -> Use Median
        is_normal = p_value > 0.05
        
        dict_distribution_results[feature] = {
            'p_value': p_value,
            'is_normal': is_normal,
            'recommended_imputation': 'mean' if is_normal else 'median'
        }

    return dict_distribution_results

def check_distributon_string_features(dict_missing_features_data, list_string_features):
    dict_distribution_results = {}
    for feature in list_string_features:
        #The word missing will be used
        dict_distribution_results[feature] = {
            'recommended_imputation': 'missing'
        }
    return dict_distribution_results

def impute_missing_values(df, dict_distribution_results, feature_type = "int_float"):
    if feature_type == "int_float":
        for feature in dict_distribution_results.keys():
            if dict_distribution_results[feature]['recommended_imputation'] == 'mean':
                df[feature] = df[feature].fillna(df[feature].mean())
            elif dict_distribution_results[feature]['recommended_imputation'] == 'median':
                df[feature] = df[feature].fillna(df[feature].median())

    elif feature_type == "string":
        for feature in dict_distribution_results.keys():
            if dict_distribution_results[feature]['recommended_imputation'] == 'missing':
                df[feature] = df[feature].fillna('missing')
    return df



def check_and_impute_missing_values(df):
    #main function for the above
    if check_for_missing_data(df):
        dict_missing_features_data, list_float_int_features, list_string_features = isolate_present_values_for_cols_with_missing_data(df)

        if list_float_int_features or list_string_features:
            
            if list_float_int_features is not None:
                dict_distribution_results_int_float = check_distributon_int_float_features(dict_missing_features_data, list_float_int_features)
                df = impute_missing_values(df, dict_distribution_results_int_float, feature_type = "int_float")

            if list_string_features is not None:
                dict_distribution_results_string = check_distributon_string_features(dict_missing_features_data, list_string_features)
                df = impute_missing_values(df, dict_distribution_results_string, feature_type = "string")

        return df
    else:
        return df
