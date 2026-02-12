import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

from functions.data_prep.import_data import data_loader
from functions.data_prep.handle_missing_values import check_and_impute_missing_values
from functions.model_helpers.model_setup import evaluate_model, create_train_test_split, train_model
from functions.visualisation.charts import histogram
from functions.model_helpers.uncertainty import calculate_rf_variance, calculate_entropy

from sidebar import render_sidebar

def app():
    # Determine problem type early to condition the content
    problem_type = st.session_state.get("problem_type", "Number (regression)")
    is_regression = "regression" in problem_type.lower()
    task_title = "Regression" if is_regression else "Classification"

    st.title("Random Forest")
    st.caption(f"Family: Supervised | Task: {task_title}")

    tab_overview, tab_deep_dive, tab_demo, tab_model = st.tabs(["Overview", "Deep Dive", "Interactive Demo", "ML Model"])

    with tab_overview:
        st.header("What is a Random Forest?")
        st.write("""
        Random Forest is an ensemble learning method that constructs a multitude of decision trees at training time.
        For regression tasks, it outputs the mean prediction of the individual trees. 
        For classification tasks, it outputs the class selected by the most trees (mode).
        """)
        
        st.subheader("Plain English Intuition")
        st.write("""
        Imagine you want to predict the value of a stock. You could ask a single analyst (a Decision Tree), but they might be biased or make a mistake based on specific noise they've seen.
        
        Instead, you ask 100 analysts. Each analyst looks at a slightly different subset of the data and uses slightly different criteria. 
        Finally, you average all their predictions. The "Wisdom of the Crowds" suggests the average of many independent errors will be closer to the truth than any single prediction.
        """)
        
        st.subheader("When to use")
        col1, col2 = st.columns(2)
        with col1:
            st.success("Use when:")
            st.write("""
            *   You need high accuracy and robustness.
            *   Data has complex, non-linear relationships.
            *   You have many features (high dimensionality).
            *   You want to avoid overfitting common in single decision trees.
            """)
        with col2:
            st.error("Avoid when:")
            st.write("""
            *   Interpretability is the highest priority (it's a "black box").
            *   Training time and memory usage are constrained (can be slow/heavy).
            *   Predicting on very sparse data (like text) where linear models might be faster/better.
            """)

        st.subheader("Pros & Cons")
        col_pros, col_cons = st.columns(2)
        with col_pros:
            st.markdown("**Pros**")
            st.markdown("*   Very accurate and robust to overfitting.")
            st.markdown("*   Handles missing values and outliers well.")
            st.markdown("*   Provides feature importance estimates.")
        with col_cons:
            st.markdown("**Cons**")
            st.markdown("*   Slow to train and predict on large datasets.")
            st.markdown("*   Difficult to interpret (hundreds of trees).")
            st.markdown("*   Model file size can be large.")

    with tab_deep_dive:
        st.header("Mechanics")
        st.write("""
        Random Forest relies on **Bagging** (Bootstrap Aggregating) and **Feature Randomness**.
        
        1.  **Bootstrapping**: Each tree is trained on a random sample of the data drawn with replacement.
        2.  **Feature Randomness**: When splitting a node, the tree considers only a random subset of features, not all of them.
        
        This diversity ensures the trees are not correlated, which reduces the variance of the final model.
        """)
        
        st.subheader("Hyperparameters")
        st.write(f"Key parameters in `sklearn.ensemble.RandomForest{'Regressor' if is_regression else 'Classifier'}`:")
        
        st.markdown("""
        *   **n_estimators** (int): The number of trees in the forest. More is usually better but slower.
        *   **max_depth** (int): The maximum depth of the tree.
        *   **min_samples_split** (int): The minimum number of samples required to split an internal node.
        *   **max_features** (int/float/str): The number of features to consider when looking for the best split.
        """)

    with tab_demo:
        st.header("Interactive Demo")
        
        col_settings, col_plot = st.columns([1, 2])
        
        with col_settings:
            st.subheader("Data Generation")
            n_samples = st.slider("Number of samples", 10, 500, 100, step=10, help="The number of data points to generate.")
            noise_level = st.slider("Noise level", 0.0, 1.0, 0.1, step=0.05, help="Amount of noise added to the data.")
            
            st.subheader("Model Settings")
            n_estimators = st.slider("Number of Trees (n_estimators)", 1, 100, 10, step=1, help="The number of trees in the forest.")
            max_depth = st.slider("Max Depth", 1, 20, 5, help="Maximum depth of each tree.")
            min_samples_leaf = st.slider("Min Samples Leaf", 1, 20, 1, help="Minimum samples required at a leaf node.")
        
        with col_plot:
            if is_regression:
                st.write("Fitting Random Forest Regressor to a Sine wave.")
                np.random.seed(42)
                X = np.sort(5 * np.random.rand(n_samples, 1), axis=0)
                y = np.sin(X).ravel()
                y[::5] += 3 * (0.5 - np.random.rand(int(n_samples/5))) * noise_level
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
                model.fit(X_train, y_train)
                
                train_mse = mean_squared_error(y_train, model.predict(X_train))
                test_mse = mean_squared_error(y_test, model.predict(X_test))
                
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(X_train, y_train, color="orange", alpha=0.6, label="Train Data")
                ax.scatter(X_test, y_test, color="green", alpha=0.6, label="Test Data")
                
                X_plot = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
                y_plot = model.predict(X_plot)
                ax.plot(X_plot, y_plot, color="blue", linewidth=2, label="Prediction")

                # Uncertainty: Tree Variance
                std_dev = calculate_rf_variance(model, X_plot)
                if std_dev is not None:
                    ax.fill_between(X_plot.flatten(), y_plot - 1.96*std_dev, y_plot + 1.96*std_dev, color='blue', alpha=0.15, label="95% CI (Tree Variance)")
                
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
                st.write("Fitting Random Forest Classifier to 'Moons' dataset.")
                X, y = make_moons(n_samples=n_samples, noise=noise_level, random_state=42)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
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
            model = RandomForestRegressor(random_state=42)
            model_type_str = "Regression"
        else:
            model = RandomForestClassifier(random_state=42)
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
