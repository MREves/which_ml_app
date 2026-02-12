import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

from functions.data_prep.import_data import data_loader
from functions.data_prep.handle_missing_values import check_and_impute_missing_values
from functions.model_helpers.model_setup import evaluate_model, create_train_test_split, train_model
from functions.visualisation.charts import histogram
from functions.model_helpers.uncertainty import calculate_entropy

from sidebar import render_sidebar

def app():
    st.title("Logistic Regression")
    st.caption("Family: Supervised | Task: Classification")

    tab_overview, tab_deep_dive, tab_demo, tab_model = st.tabs(["Overview", "Deep Dive", "Interactive Demo", "ML Model"])

    with tab_overview:
        st.header("What is Logistic Regression?")
        st.write("""
        Despite its name, Logistic Regression is a **classification** algorithm used to predict the probability of a target variable belonging to a certain class. 
        It fits an "S" shaped curve (Sigmoid function) to the data, rather than a straight line.
        """)
        
        st.subheader("Plain English Intuition")
        st.write("""
        Imagine you want to predict if a student will **Pass (1)** or **Fail (0)** an exam based on the number of hours they studied.
        
        *   A Linear Regression line might predict a "score" of 1.5 or -0.2, which doesn't make sense for a Yes/No outcome.
        *   Logistic Regression squashes the output between 0 and 1. 
        *   If the output is 0.8, it means there is an 80% probability of passing. You can set a threshold (e.g., 0.5) to say "Yes, they will pass".
        """)
        
        st.subheader("When to use")
        col1, col2 = st.columns(2)
        with col1:
            st.success("Use when:")
            st.write("""
            *   Predicting a binary outcome (Yes/No, True/False).
            *   You need probabilistic results (e.g., "70% chance of rain").
            *   The data is roughly linearly separable.
            *   You need a simple, interpretable baseline.
            """)
        with col2:
            st.error("Avoid when:")
            st.write("""
            *   The relationship between features and target is highly non-linear.
            *   There are many outliers (can skew the decision boundary).
            *   Missing data is present (needs imputation).
            """)

        st.subheader("Pros & Cons")
        col_pros, col_cons = st.columns(2)
        with col_pros:
            st.markdown("**Pros**")
            st.markdown("*   Highly interpretable (weights indicate feature importance).")
            st.markdown("*   Fast to train and very efficient.")
            st.markdown("*   Outputs probabilities, not just classes.")
        with col_cons:
            st.markdown("**Cons**")
            st.markdown("*   Assumes linearity between input features and log-odds.")
            st.markdown("*   Can't solve non-linear problems (like XOR) without feature engineering.")
            st.markdown("*   Prone to overfitting if number of features > number of observations.")

    with tab_deep_dive:
        st.header("Mechanics")
        st.write("""
        Logistic Regression applies the **Sigmoid** (or Logistic) function to the output of a linear equation.
        """)
        st.latex(r"P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}}")
        st.write("""
        *   The linear part $(\beta_0 + \beta_1 x)$ is the "log-odds".
        *   The Sigmoid function maps any real number to the range $(0, 1)$.
        """)
        
        st.subheader("Hyperparameters")
        st.write("Key parameters in `sklearn.linear_model.LogisticRegression`:")
        st.markdown("""
        *   **C** (float): Inverse of regularization strength. Smaller values specify stronger regularization (like SVM).
        *   **penalty** (str): 'l1', 'l2', 'elasticnet', 'none'. Used to prevent overfitting.
        *   **solver** (str): Algorithm to use in the optimization problem (e.g., 'liblinear', 'lbfgs').
        """)

    with tab_demo:
        st.header("Interactive Demo")
        
        col_settings, col_plot = st.columns([1, 2])
        
        with col_settings:
            st.subheader("Data Generation")
            n_samples = st.slider("Number of samples", 10, 500, 100, step=10, help="Number of data points.")
            cluster_std = st.slider("Cluster Spread (Noise)", 0.1, 5.0, 1.0, step=0.1, help="Standard deviation of the clusters. Higher means more overlap.")
            
            st.subheader("Model Settings")
            C = st.slider("C (Regularization)", 0.01, 10.0, 1.0, step=0.01, help="Inverse of regularization strength. Smaller values = stronger regularization (simpler boundary).")
        
        with col_plot:
            st.write("Fitting Logistic Regression to two blobs.")
            # Generate synthetic data
            X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=cluster_std, random_state=42)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LogisticRegression(C=C)
            model.fit(X_train, y_train)
            
            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc = accuracy_score(y_test, model.predict(X_test))
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Decision Boundary
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdBu_r, edgecolors='k', alpha=0.8, label="Train")
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdBu_r, marker='x', s=80, alpha=0.8, label="Test")
            
            # Uncertainty Metric
            probs = model.predict_proba(X_test)
            avg_entropy = np.mean(calculate_entropy(probs))

            ax.set_title("Decision Boundary")
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
        # Logistic Regression is only for classification in this app context
        problem_type = "Label (classification)"
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
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model = train_model(model, X_train, y_train)
        
        dict_train_results, dict_test_results = evaluate_model(model, "Classification", X_train, y_train, X_test, y_test)
        
        st.subheader("3. Evaluate the model")
        acc_train = dict_train_results['accuracy']
        acc_test = dict_test_results['accuracy']
        st.write(f"Training Accuracy: {acc_train:.2f} | Test Accuracy: {acc_test:.2f}")
        percent_diff = ((acc_train - acc_test) / acc_train) * 100
        st.metric("Accuracy Drop (Overfitting Risk)", f"{percent_diff:.1f}%", delta_color="inverse")
        st.write(f"Classes: {np.unique(y)}")
