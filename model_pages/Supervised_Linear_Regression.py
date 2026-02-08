import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score

from functions.data_prep.import_data import data_loader
from functions.data_prep.handle_missing_values import check_and_impute_missing_values
from functions.model_helpers.model_setup import evaluate_model, create_train_test_split, train_model

from functions.visualisation.charts import histogram

from sklearn.model_selection import train_test_split

from sidebar import render_sidebar

def app():
    st.title("Linear Regression")
    st.caption("Family: Supervised | Task: Regression")

    tab_overview, tab_deep_dive, tab_demo, tab_model = st.tabs(["Overview", "Deep Dive", "Interactive Demo", "ML Model"])

    with tab_overview:
        st.header("What is Linear Regression?")
        st.write("""
        Linear Regression is a fundamental supervised learning algorithm used to predict a continuous target variable 
        based on one or more input features. It assumes a linear relationship between the inputs and the target.
        """)
        
        st.subheader("Plain English Intuition")
        st.write("""
        Imagine you want to predict the price of a house based on its size. You have a list of houses with their 
        sizes and prices. If you plot these on a graph, you'll likely see that as size increases, price increases.
        
        Linear Regression tries to draw the "best-fitting" straight line through these data points. Once this line 
        is drawn, you can use it to predict the price of any house if you know its size, simply by finding the 
        corresponding point on the line.
        """)
        
        st.subheader("When to use")
        col1, col2 = st.columns(2)
        with col1:
            st.success("Use when:")
            st.write("""
            *   Predicting a continuous number (e.g., price, temperature).
            *   The relationship between variables is roughly linear.
            *   You need a simple, interpretable baseline.
            """)
        with col2:
            st.error("Avoid when:")
            st.write("""
            *   The relationship is highly non-linear (curved).
            *   Predicting a category (use Classification instead).
            *   Data has many outliers that might skew the line.
            """)

    with tab_deep_dive:
        st.header("Mechanics")
        st.latex(r"y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \epsilon")
        st.write("""
        Where:
        *   $y$ is the predicted value.
        *   $\\beta_0$ is the y-intercept (bias).
        *   $\\beta_1, \\beta_2$ are the coefficients (weights) for each feature.
        *   $x_1, x_2$ are the feature values.
        *   $\\epsilon$ is the error term.
        """)
        
        st.subheader("Pros & Cons")
        col_pros, col_cons = st.columns(2)
        with col_pros:
            st.markdown("**Pros**")
            st.markdown("*   Simple and easy to interpret.")
            st.markdown("*   Fast to train and predict.")
            st.markdown("*   Works well on small datasets.")
        with col_cons:
            st.markdown("**Cons**")
            st.markdown("*   Assumes a linear relationship.")
            st.markdown("*   Sensitive to outliers.")
            st.markdown("*   Prone to underfitting complex data.")

        st.subheader("Hyperparameters")
        st.write("Key parameters in `sklearn.linear_model.LinearRegression`:")
        st.markdown("""
        *   **fit_intercept** (bool): Whether to calculate the intercept for this model. If set to False, no intercept will be used (data is expected to be centered).
        *   **positive** (bool): When set to True, forces the coefficients to be positive.
        *   **n_jobs** (int): The number of jobs to use for the computation (for large datasets).
        """)

    with tab_demo:
        st.header("Interactive Demo")
        st.write("Tune the parameters to see how the model fits the data.")
        
        col_settings, col_plot = st.columns([1, 2])
        
        with col_settings:
            st.subheader("Data Generation")
            n_samples = st.slider("Number of samples", 10, 200, 50, step=10)
            noise_level = st.slider("Noise level", 0.0, 50.0, 10.0, step=1.0)
            true_slope = st.slider("True Slope (Beta 1)", -10.0, 10.0, 2.0, step=0.5)
            true_intercept = st.slider("True Intercept (Beta 0)", -50.0, 50.0, 0.0, step=5.0)
            
            st.subheader("Model Settings")
            fit_intercept = st.checkbox("Fit Intercept", value=True)
        
        with col_plot:
            # Generate synthetic data
            np.random.seed(42)
            X = np.random.rand(n_samples, 1) * 10  # Feature range 0-10
            y = true_slope * X.flatten() + true_intercept + np.random.randn(n_samples) * noise_level
            
            # Train model
            model = LinearRegression(fit_intercept=fit_intercept)
            model.fit(X, y)
            y_pred = model.predict(X)
            
            # Metrics
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            
            # Plotting
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(X, y, color="blue", alpha=0.6, label="Data Points")
            ax.plot(X, y_pred, color="red", linewidth=2, label=f"Prediction (R2={r2:.2f})")
            ax.set_xlabel("Feature X")
            ax.set_ylabel("Target y")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            
            st.pyplot(fig)
            
            st.info(f"""
            **Model Coefficients:**
            *   Slope: {model.coef_[0]:.2f} (True: {true_slope})
            *   Intercept: {model.intercept_:.2f} (True: {true_intercept})
            *   MSE: {mse:.2f}
            """)

        with tab_model:
            #load the data
            df_full, X, y = data_loader(st.session_state["problem_type"])

            st.title("ML Model Overview")

            with st.expander(label="Set preferences:"):
                
                missing_data_percentage = st.slider("Percentage of missing data", 0, 50, 5)
                features_missing_data = st.multiselect("Which features to remove data from?", options=df_full.columns[:-1])

                np.random.seed(42)
                indices_missing_data = np.random.choice(df_full.shape[0], size=int(df_full.shape[0] * missing_data_percentage / 100), replace=False)
                for feature in features_missing_data:
                    df_full.loc[indices_missing_data, feature] = np.nan
            

            #----------------------------------
            # Section 1
            #----------------------------------
            st.subheader("1. Data Preparation")

            st.write(f"The data has {df_full.shape[0]} rows and {df_full.shape[1]} columns (including the target).")
            

            df_missing_values = df_full.isnull().sum()
            feat_names = df_missing_values.index.tolist()
            missing_values = df_missing_values.values
            df_missing_values = pd.DataFrame({'Feature': feat_names, 'Missing Values': missing_values}).T
            st.write("The table below shows any missing values for this data.")
            #st.dataframe(df_missing_values, hide_index=True)
            st.write(df_missing_values.to_html(header=False, index=False), unsafe_allow_html=True)

            if missing_data_percentage >  0:
                #get df for each dtype we will visualise
                df_dtypes = pd.DataFrame(df_full.dtypes.astype(str)).rename(columns={0: 'Data Type'})
                df_dtypes_float_int = df_dtypes[df_dtypes['Data Type'].isin(['float64', 'int64'])]
                df_dtypes_object = df_dtypes[df_dtypes['Data Type'] == 'object']

                #create lists for features for int/float and string/object
                list_float_int_features = df_dtypes_float_int.index.tolist()
                list_float_int_features = [item for item in features_missing_data if item in list_float_int_features]

                list_object_features = df_dtypes_object.index.tolist()
                list_object_features = [item for item in features_missing_data if item in list_object_features]

                #if missing features are present, allow user to inspect distribution
                if len(list_float_int_features) > 0:
                    selected_int_float_feature = st.selectbox(
                        "Select which numeric missing data to visualise",
                        options=list_float_int_features)

                    st.subheader("Controlling for missing data")
                    st.write("From the histograms we can see in this case the distribution of data is mostly normal about the mean." \
                    "As such, the mean value can be used here for imputation. If the data was skewed, then either consider transformation and then mean value, or, use the median for missing values." \
                    "Alternatively, look for correlation between features, as it may be possible to impute a missing value for one feature from the present values of another (potentially using anoter lineanr regression or other regression model). " \
                    "If the total count of missing values is low, consider removing the rows. Finally, consider using the Shapiro-Wilk test to quantitatively test for normality (as is the method used here)")
                    
                    fig = histogram(df_full, selected_int_float_feature)
                    st.pyplot(fig)

                    #-----------------------------------------
                    #update the data to impute missing values
                    #-----------------------------------------
                    df_full_imputed = check_and_impute_missing_values(df_full)
                    df_full = df_full_imputed

            #----------------------------------
            # Section 2 - set up model
            #----------------------------------
            st.subheader("2. Set up model")
            
            #get train test split
            X_train, X_test, y_train, y_test = create_train_test_split(X, y)
            #instantiate model
            model = LinearRegression()
            #fit model
            model = train_model(model, X_train, y_train)
            #evaluate the fitted model
            dict_train_results, dict_test_results = evaluate_model(model, "Regression", X_train, y_train, X_test, y_test)
            #extract specific metrics
            mae_train = dict_train_results['mae']
            mae_test = dict_test_results['mae']

            
            #----------------------------------
            # Section 3 - evaluate the model
            #----------------------------------
            st.subheader("3. Evaluate the model")
            st.write(f"Training MAE: {mae_train:.2f} | Test MAE: {mae_test:.2f}")

            percent_diff = ((mae_test - mae_train) / mae_train) * 100

            st.metric("Generalization Gap", f"{percent_diff:.1f}%", delta_color="inverse")
            # A large positive delta warns the user of potential overfitting.

            st.write(f"Target range {y.min()} to {y.max()}")

