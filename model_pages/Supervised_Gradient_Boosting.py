import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

from functions.data_prep.import_data import data_loader
from functions.data_prep.handle_missing_values import check_and_impute_missing_values
from functions.model_helpers.model_setup import evaluate_model, create_train_test_split, train_model
from functions.visualisation.charts import histogram
from functions.model_helpers.uncertainty import calculate_quantile_intervals, calculate_entropy

from sidebar import render_sidebar

def app():
    # Determine problem type early to condition the content
    problem_type = st.session_state.get("problem_type", "Number (regression)")
    is_regression = "regression" in problem_type.lower()
    task_title = "Regression" if is_regression else "Classification"

    st.title("Gradient Boosting")
    st.caption(f"Family: Supervised | Task: {task_title}")

    tab_overview, tab_deep_dive, tab_demo, tab_model = st.tabs(["Overview", "Deep Dive", "Interactive Demo", "ML Model"])

    with tab_overview:
        st.header("What is Gradient Boosting?")
        st.write("""
        Gradient Boosting is an ensemble technique that builds models sequentially. 
        Unlike Random Forest, which builds trees independently (Bagging), Gradient Boosting builds trees one at a time, 
        where each new tree helps to correct errors made by the previously trained tree.
        """)
        
        st.subheader("Plain English Intuition")
        st.write("""
        Imagine you are playing golf. 
        1.  You hit the ball towards the hole (Target), but it lands short (Error/Residual).
        2.  You don't aim for the hole anymore; you aim to fix the mistake of the first shot. You hit a second shot from where the first landed towards the hole.
        3.  You repeat this, taking small shots (Learning Rate) to inch closer and closer to the hole.
        
        The final prediction is the sum of all these shots.
        """)
        
        st.subheader("When to use")
        col1, col2 = st.columns(2)
        with col1:
            st.success("Use when:")
            st.write("""
            *   You need state-of-the-art accuracy on tabular data.
            *   You want to win Kaggle competitions (XGBoost/LightGBM are variants of this).
            *   Data has complex non-linear patterns.
            """)
        with col2:
            st.error("Avoid when:")
            st.write("""
            *   Training time is a bottleneck (sequential training is slower than parallel).
            *   Data is very noisy (boosting can overfit to noise).
            *   Interpretability is critical (harder to interpret than a single tree).
            """)

        st.subheader("Pros & Cons")
        col_pros, col_cons = st.columns(2)
        with col_pros:
            st.markdown("**Pros**")
            st.markdown("*   Often provides higher accuracy than Random Forest.")
            st.markdown("*   Flexible - can optimize different loss functions.")
            st.markdown("*   Handles mixed data types well.")
        with col_cons:
            st.markdown("**Cons**")
            st.markdown("*   Sensitive to outliers (tries to fix them aggressively).")
            st.markdown("*   Harder to tune (more hyperparameters).")
            st.markdown("*   Computationally expensive to train.")

    with tab_deep_dive:
        st.header("Mechanics")
        st.write("""
        **Boosting vs Bagging:**
        *   **Random Forest (Bagging)**: Trains heavy trees in parallel and averages them to reduce variance.
        *   **Gradient Boosting**: Trains weak trees (shallow) sequentially to reduce bias.
        
        **The Process:**
        1.  Fit a model to the data ($F_0(x)$).
        2.  Calculate residuals (Actual - Predicted).
        3.  Fit a new model ($h_1(x)$) to the *residuals*.
        4.  Update model: $F_1(x) = F_0(x) + \text{learning\_rate} \times h_1(x)$.
        5.  Repeat.
        """)
        
        st.subheader("Hyperparameters")
        st.write(f"Key parameters in `sklearn.ensemble.GradientBoosting{'Regressor' if is_regression else 'Classifier'}`:")
        
        st.markdown("""
        *   **n_estimators** (int): The number of boosting stages (trees) to perform.
        *   **learning_rate** (float): Shrinks the contribution of each tree. Lower values require more trees but generalize better.
        *   **max_depth** (int): Maximum depth of the individual regression estimators. Usually kept low (3-5).
        *   **subsample** (float): The fraction of samples to be used for fitting the individual base learners (Stochastic Gradient Boosting).
        """)

    with tab_demo:
        st.header("Interactive Demo")
        
        col_settings, col_plot = st.columns([1, 2])
        
        with col_settings:
            st.subheader("Data Generation")
            n_samples = st.slider("Number of samples", 10, 500, 100, step=10, help="Number of data points.")
            noise_level = st.slider("Noise level", 0.0, 1.0, 0.1, step=0.05, help="Noise added to the data.")
            
            st.subheader("Model Settings")
            n_estimators = st.slider("Number of Trees", 1, 200, 50, step=10, help="Number of boosting stages.")
            learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1, step=0.01, help="Step size shrinkage used in update to prevents overfitting.")
            max_depth = st.slider("Max Depth", 1, 10, 3, help="Maximum depth of the individual regression estimators.")
        
        with col_plot:
            if is_regression:
                st.write("Fitting Gradient Boosting Regressor to a Sine wave.")
                np.random.seed(42)
                X = np.sort(5 * np.random.rand(n_samples, 1), axis=0)
                y = np.sin(X).ravel()
                y[::5] += 3 * (0.5 - np.random.rand(int(n_samples/5))) * noise_level
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
                model.fit(X_train, y_train)
                
                train_mse = mean_squared_error(y_train, model.predict(X_train))
                test_mse = mean_squared_error(y_test, model.predict(X_test))
                
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(X_train, y_train, color="orange", alpha=0.6, label="Train Data")
                ax.scatter(X_test, y_test, color="green", alpha=0.6, label="Test Data")
                
                X_plot = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
                y_plot = model.predict(X_plot)
                ax.plot(X_plot, y_plot, color="blue", linewidth=2, label="Prediction")

                # Uncertainty: Quantile Intervals
                lower, upper = calculate_quantile_intervals(X_train, y_train, X_plot, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
                ax.fill_between(X_plot.flatten(), lower, upper, color='blue', alpha=0.15, label="90% Prediction Interval")
                
                ax.set_xlabel("Feature X")
                ax.set_ylabel("Target y")
                ax.legend()
                ax.grid(True, linestyle="--", alpha=0.5)
                st.pyplot(fig)
                
                st.info(f"""
                **Performance:**
                *   Train MSE: {train_mse:.4f}
                *   Test MSE: {test_mse:.4f}
                """)
            else:
                st.write("Fitting Gradient Boosting Classifier to 'Moons' dataset.")
                X, y = make_moons(n_samples=n_samples, noise=noise_level, random_state=42)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
                model.fit(X_train, y_train)
                
                train_acc = accuracy_score(y_train, model.predict(X_train))
                test_acc = accuracy_score(y_test, model.predict(X_test))
                
                fig, ax = plt.subplots(figsize=(8, 5))
                x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
                y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
                ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdBu_r, edgecolors='k', alpha=0.8, label="Train")
                ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdBu_r, marker='x', s=80, alpha=0.8, label="Test")
                
                # Uncertainty Metric
                probs = model.predict_proba(X_test)
                avg_entropy = np.mean(calculate_entropy(probs))

                ax.set_xlabel("Feature 1")
                ax.set_ylabel("Feature 2")
                ax.legend()
                st.pyplot(fig)
                
                st.info(f"""
                **Performance:**
                *   Train Accuracy: {train_acc:.2f}
                *   Test Accuracy: {test_acc:.2f}
                *   Avg Test Entropy (Uncertainty): {avg_entropy:.3f}
                """)

    with tab_model:
        df_full, X, y = data_loader(problem_type)
        st.title("ML Model Overview")
        
        with st.expander(label="Set preferences:"):
            missing_data_percentage = st.slider("Percentage of missing data", 0, 50, 5)
            features_missing_data = st.multiselect("Which features to remove data from?", options=df_full.columns[:-1])
            np.random.seed(42)
            indices_missing_data = np.random.choice(df_full.shape[0], size=int(df_full.shape[0] * missing_data_percentage / 100), replace=False)
            for feature in features_missing_data:
                df_full.loc[indices_missing_data, feature] = np.nan

        st.subheader("1. Data Preparation")
        st.write(f"The data has {df_full.shape[0]} rows and {df_full.shape[1]} columns (including the target).")
        
        df_missing_values = df_full.isnull().sum()
        feat_names = df_missing_values.index.tolist()
        missing_values = df_missing_values.values
        df_missing_values = pd.DataFrame({'Feature': feat_names, 'Missing Values': missing_values}).T
        st.write("The table below shows any missing values for this data.")
        st.write(df_missing_values.to_html(header=False, index=False), unsafe_allow_html=True)

        if missing_data_percentage > 0:
            df_dtypes = pd.DataFrame(df_full.dtypes.astype(str)).rename(columns={0: 'Data Type'})
            df_dtypes_float_int = df_dtypes[df_dtypes['Data Type'].isin(['float64', 'int64'])]
            df_dtypes_object = df_dtypes[df_dtypes['Data Type'] == 'object']
            list_float_int_features = df_dtypes_float_int.index.tolist()
            list_float_int_features = [item for item in features_missing_data if item in list_float_int_features]
            list_object_features = df_dtypes_object.index.tolist()
            list_object_features = [item for item in features_missing_data if item in list_object_features]

            if len(list_float_int_features) > 0:
                selected_int_float_feature = st.selectbox("Select which numeric missing data to visualise", options=list_float_int_features)
                st.subheader("Controlling for missing data")
                st.write("From the histograms we can see in this case the distribution of data is mostly normal about the mean. As such, the mean value can be used here for imputation. If the data was skewed, then either consider transformation and then mean value, or, use the median for missing values. Alternatively, look for correlation between features, as it may be possible to impute a missing value for one feature from the present values of another (potentially using anoter lineanr regression or other regression model). If the total count of missing values is low, consider removing the rows. Finally, consider using the Shapiro-Wilk test to quantitatively test for normality (as is the method used here)")
                fig = histogram(df_full, selected_int_float_feature)
                st.pyplot(fig)
                df_full_imputed = check_and_impute_missing_values(df_full)
                df_full = df_full_imputed

        st.subheader("2. Set up model")
        X_train, X_test, y_train, y_test = create_train_test_split(X, y)
        
        if is_regression:
            model = GradientBoostingRegressor(random_state=42)
            model_type_str = "Regression"
        else:
            model = GradientBoostingClassifier(random_state=42)
            model_type_str = "Classification"

        model = train_model(model, X_train, y_train)
        dict_train_results, dict_test_results = evaluate_model(model, model_type_str, X_train, y_train, X_test, y_test)
        
        st.subheader("3. Evaluate the model")
        if is_regression:
            mae_train = dict_train_results['mae']
            mae_test = dict_test_results['mae']
            st.write(f"Training MAE: {mae_train:.2f} | Test MAE: {mae_test:.2f}")
            try: 
                percent_diff = ((mae_test - mae_train) / mae_train) * 100
                st.metric("Generalization Gap", f"{percent_diff:.1f}%", delta_color="inverse")
                st.write(f"Target range {y.min()} to {y.max()}")
            except:
                pass
        else:
            acc_train = dict_train_results['accuracy']
            acc_test = dict_test_results['accuracy']
            st.write(f"Training Accuracy: {acc_train:.2f} | Test Accuracy: {acc_test:.2f}")
            percent_diff = ((acc_train - acc_test) / acc_train) * 100
            st.metric("Accuracy Drop (Overfitting Risk)", f"{percent_diff:.1f}%", delta_color="inverse")
            st.write(f"Classes: {np.unique(y)}")
