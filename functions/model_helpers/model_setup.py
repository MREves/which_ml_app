from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error

def create_train_test_split(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def get_model_preds(model, model_type, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    if model_type == "Classification":
        y_test_pred_proba = model.predict_proba(X_test)
        y_train_pred_proba = model.predict_proba(X_train)

        return y_train_pred, y_train_pred_proba, y_test_pred, y_test_pred_proba
    else:
        return y_train_pred, y_test_pred, 


def evaluate_model(model, model_type, X_train, y_train, X_test, y_test):
    '''
    Docstring for evaluate_model
    
    :param model: The fitted model that is to be evaluated
    :param model_type: "Classification or Regression"
    '''
    dict_train_results = {}
    dict_test_results = {}
    

    if model_type == "Classification":
        y_train_pred, y_train_pred_proba, y_test_pred, y_test_pred_proba = get_model_preds(model, model_type, X_train, y_train, X_test, y_test)
        
        dict_train_results['accuracy'] = accuracy_score(y_train, y_train_pred)
        dict_train_results['precision'] = precision_score(y_train, y_train_pred)
        dict_train_results['recall'] = recall_score(y_train, y_train_pred)
        dict_train_results['f1'] = f1_score(y_train, y_train_pred)
        dict_train_results['roc_auc'] = roc_auc_score(y_train, y_train_pred_proba[:, 1])

        dict_test_results['accuracy'] = accuracy_score(y_test, y_test_pred)
        dict_test_results['precision'] = precision_score(y_test, y_test_pred)
        dict_test_results['recall'] = recall_score(y_test, y_test_pred)
        dict_test_results['f1'] = f1_score(y_test, y_test_pred)
        dict_test_results['roc_auc'] = roc_auc_score(y_test, y_test_pred_proba[:, 1])    
    
    elif model_type == "Regression":
        y_train_pred, y_test_pred = get_model_preds(model, model_type, X_train, y_train, X_test, y_test)
        
        dict_train_results['mse'] = mean_squared_error(y_train, y_train_pred)
        dict_train_results['mae'] = mean_absolute_error(y_train, y_train_pred)
        dict_train_results['rmse'] = root_mean_squared_error(y_train, y_train_pred)
        dict_train_results['r2'] = r2_score(y_train, y_train_pred)

        dict_test_results['mse'] = mean_squared_error(y_test, y_test_pred)
        dict_test_results['mae'] = mean_absolute_error(y_test, y_test_pred)
        dict_test_results['rmse'] = root_mean_squared_error(y_test, y_test_pred)
        dict_test_results['r2'] = r2_score(y_test, y_test_pred) 

    return dict_train_results, dict_test_results
    

