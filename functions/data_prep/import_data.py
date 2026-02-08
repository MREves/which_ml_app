from sklearn.datasets import load_diabetes, load_breast_cancer
import pandas as pd

def import_classification_data():
    """
    This function imports the breast cancer dataset from sklearn
    This is a classification toy dataset.
    """
    data_dict = load_breast_cancer()

    X = data_dict['data']
    y = data_dict['target']

    feat_names = data_dict["feature_names"]

    df = pd.DataFrame(X, columns=feat_names )

    df_full  = pd.concat([df, pd.Series(y, name='target')], axis=1)

    return df_full, X, y


def import_regression_data():
    """
    This function imports the diabetes dataset from sklearn
    This is a regression toy dataset.
    """
    data_dict = load_diabetes()
    X = data_dict['data']
    y = data_dict['target']

    feat_names = data_dict["feature_names"]

    df = pd.DataFrame(X, columns=feat_names )

    df_full  = pd.concat([df, pd.Series(y, name='target')], axis=1)

    return df_full, X, y

def data_loader(model_type):
    if model_type == "Unsupervised (clustering)" or model_type == "Label (classification)":
        df_full, X, y = import_classification_data()

    elif model_type ==  "Number (regression)":
        df_full, X, y = import_regression_data()
    
    return df_full, X, y
