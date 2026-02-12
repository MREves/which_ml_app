import numpy as np
from sklearn.utils import resample
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor

def calculate_bootstrap_intervals(model, X_train, y_train, X_test, n_bootstraps=50):
    """
    Calculates prediction intervals using bootstrapping.
    Returns lower (2.5th percentile) and upper (97.5th percentile) bounds.
    """
    preds = []
    for _ in range(n_bootstraps):
        model_clone = clone(model)
        # Resample with replacement
        X_boot, y_boot = resample(X_train, y_train, random_state=None) # random_state None for random resampling
        model_clone.fit(X_boot, y_boot)
        preds.append(model_clone.predict(X_test))
    
    preds = np.array(preds)
    lower = np.percentile(preds, 2.5, axis=0)
    upper = np.percentile(preds, 97.5, axis=0)
    return lower, upper

def calculate_rf_variance(model, X):
    """
    Calculates the standard deviation of predictions across all trees in a Random Forest.
    """
    if not hasattr(model, "estimators_"):
        return None
    
    # Collect predictions from all trees
    preds = np.stack([tree.predict(X) for tree in model.estimators_])
    return np.std(preds, axis=0)

def calculate_quantile_intervals(X_train, y_train, X_test, n_estimators=100, max_depth=3, learning_rate=0.1):
    """
    Trains two Gradient Boosting models to predict the 5th and 95th quantiles.
    """
    lower_model = GradientBoostingRegressor(loss='quantile', alpha=0.05, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
    upper_model = GradientBoostingRegressor(loss='quantile', alpha=0.95, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
    
    lower_model.fit(X_train, y_train)
    upper_model.fit(X_train, y_train)
    
    return lower_model.predict(X_test), upper_model.predict(X_test)

def calculate_entropy(probs):
    """Calculates the entropy of predicted probabilities."""
    return -np.sum(probs * np.log(probs + 1e-9), axis=1)