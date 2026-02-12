import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

from functions.data_prep.import_data import data_loader
from functions.data_prep.handle_missing_values import check_and_impute_missing_values
from functions.model_helpers.model_setup import evaluate_model, create_train_test_split, train_model
from functions.model_helpers.uncertainty import calculate_entropy

from functions.visualisation.charts import histogram

from sidebar import render_sidebar

def app():
    # Determine problem type early to condition the content
    problem_type = st.session_state.get("problem_type", "Number (regression)")
    is_regression = "regression" in problem_type.lower()
    task_title = "Regression" if is_regression else "Classification"

    st.title("Decision Trees")
    st.caption(f"Family: Supervised | Task: {task_title}")

    tab_overview, tab_deep_dive, tab_demo, tab_model = st.tabs(["Overview", "Deep Dive", "Interactive Demo", "ML Model"])

    with tab_overview:
        st.header("What is a Decision Tree?")
        if is_regression:
            st.write("""
            A Decision Tree Regressor learns simple decision rules inferred from the data features to predict a **continuous target variable**.
            It splits the data into smaller groups and predicts the average value for each group.
            """)
        else:
            st.write("""
            A Decision Tree Classifier learns simple decision rules inferred from the data features to predict a **class label**.
            It splits the data into smaller groups and predicts the most frequent class for each group.
            """)
        
        st.subheader("Plain English Intuition")
        if is_regression:
            st.write("""
            Imagine you want to predict the **price of a used car**.
            
            1.  Is the car less than 3 years old? 
                *   (Yes) -> Is the mileage under 20k? 
                    *   (Yes) -> Predict **$25,000**
                    *   (No) -> Predict **$20,000**
                *   (No) -> Is it a luxury brand?
                    *   (Yes) -> Predict **$18,000**
                    *   (No) -> Predict **$10,000**
            
            The tree splits the cars into groups based on age, mileage, and brand, and assigns a price (the average of similar cars) to each leaf.
            """)
        else:
            st.write("""
            Imagine playing a game of "20 Questions" to decide **whether to wear a jacket**.
            
            1.  Is it raining? 
                *   (Yes) -> **Wear a jacket.**
            2.  Is it raining? (No) -> Is it cold? 
                *   (Yes) -> **Wear a jacket.**
            3.  Is it cold? (No) -> **Don't wear a jacket.**
            
            A Decision Tree builds this exact structure: a flowchart where each internal node represents a "test" on an attribute, 
            and each leaf node represents the final class decision.
            """)
        
        st.subheader("When to use")
        col1, col2 = st.columns(2)
        with col1:
            st.success("Use when:")
            if is_regression:
                st.write("""
                *   Predicting continuous values with non-linear patterns.
                *   Interpretability is key (white-box model).
                *   Data has mixed types (numerical and categorical).
                """)
            else:
                st.write("""
                *   Predicting distinct categories (labels).
                *   Decision rules need to be transparent.
                *   Data has non-linear boundaries between classes.
                """)
        with col2:
            st.error("Avoid when:")
            st.write("""
            *   The model needs to extrapolate outside the training data range.
            *   You need a very smooth prediction surface (trees are step-wise).
            *   Data is very small and sensitive to noise (trees can be unstable).
            """)

        st.subheader("Pros & Cons")
        col_pros, col_cons = st.columns(2)
        with col_pros:
            st.markdown("**Pros**")
            st.markdown("*   Easy to understand and visualize.")
            st.markdown("*   Handles non-linear data well.")
            st.markdown("*   Robust to outliers compared to linear models.")
        with col_cons:
            st.markdown("**Cons**")
            st.markdown("*   Prone to overfitting (learning noise).")
            st.markdown("*   High variance (small data changes = different tree).")
            st.markdown("*   Biased towards dominant classes if unbalanced.")

    with tab_deep_dive:
        st.header("Mechanics")
        st.write("""
        The tree is built using a recursive splitting process (often CART - Classification and Regression Trees).
        
        *   **Root Node**: The top node containing the entire dataset.
        *   **Splitting**: The process of dividing a node into two or more sub-nodes based on a feature threshold.
        """)
        
        if is_regression:
            st.markdown("""
            *   **Objective**: Minimize variance (MSE) within each split.
            *   **Leaf Node**: Predicts the **mean** value of samples in that leaf.
            """)
        else:
            st.markdown("""
            *   **Objective**: Maximize purity (Gini Impurity or Entropy).
            *   **Leaf Node**: Predicts the **mode** (most frequent) class of samples in that leaf.
            """)
        
        st.subheader("Hyperparameters")
        st.write(f"Key parameters in `sklearn.tree.DecisionTree{'Regressor' if is_regression else 'Classifier'}`:")
        
        if is_regression:
            st.markdown("""
            *   **criterion** (str): 'squared_error' (MSE), 'friedman_mse', 'absolute_error'. Measures split quality.
            """)
        else:
            st.markdown("""
            *   **criterion** (str): 'gini', 'entropy', 'log_loss'. Measures split quality (impurity).
            """)
            
        st.markdown("""
        *   **max_depth** (int): The maximum depth of the tree. Limits how deep the tree can grow to prevent overfitting.
        *   **min_samples_split** (int): The minimum number of samples required to split an internal node.
        *   **min_samples_leaf** (int): The minimum number of samples required to be at a leaf node.
        """)

    with tab_demo:
        st.header("Interactive Demo")
        
        col_settings, col_plot = st.columns([1, 2])
        
        with col_settings:
            st.subheader("Data Generation")
            n_samples = st.slider("Number of samples", 10, 300, 100, step=10, help="The number of data points to generate for this simulation.")
            noise_level = st.slider("Noise level", 0.0, 1.0, 0.1, step=0.05, help="The amount of random noise added to the target variable. Higher noise makes the pattern harder to learn.")
            
            st.subheader("Model Settings")
            max_depth = st.slider("Max Depth", 1, 20, 3, help="The maximum depth of the tree. Deeper trees capture more details but are more likely to overfit (memorize noise).")
            min_samples_leaf = st.slider("Min Samples Leaf", 1, 20, 1, help="The minimum number of samples required to be at a leaf node. Increasing this number smoothes the model, reducing variance and overfitting.")
        
        with col_plot:
            if is_regression:
                st.write("Fitting a Decision Tree Regressor to a Sine wave.")
                # Generate synthetic data (Sine wave)
                np.random.seed(42)
                X = np.sort(5 * np.random.rand(n_samples, 1), axis=0)
                y = np.sin(X).ravel()
                y[::5] += 3 * (0.5 - np.random.rand(int(n_samples/5))) * noise_level # Add noise
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train model
                model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
                model.fit(X_train, y_train)
                
                # Metrics
                train_mse = mean_squared_error(y_train, model.predict(X_train))
                test_mse = mean_squared_error(y_test, model.predict(X_test))
                
                # Plotting
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(X_train, y_train, color="orange", alpha=0.6, label="Train Data")
                ax.scatter(X_test, y_test, color="green", alpha=0.6, label="Test Data")
                
                # Plot the step function prediction
                X_plot = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
                y_plot = model.predict(X_plot)
                ax.plot(X_plot, y_plot, color="blue", linewidth=2, label="Prediction")
                
                ax.set_xlabel("Feature X")
                ax.set_ylabel("Target y")
                ax.legend()
                ax.grid(True, linestyle="--", alpha=0.5)
                
                st.pyplot(fig)
                
                st.info(f"""
                **Model Complexity:**
                *   Depth: {model.get_depth()}
                *   Number of Leaves: {model.get_n_leaves()}
                
                **Performance:**
                *   Train MSE: {train_mse:.4f}
                *   Test MSE: {test_mse:.4f}
                """)
            else:
                st.write("Fitting a Decision Tree Classifier to the 'Moons' dataset.")
                # Generate synthetic data (Moons)
                X, y = make_moons(n_samples=n_samples, noise=noise_level, random_state=42)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train model
                model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
                model.fit(X_train, y_train)
                
                # Metrics
                train_acc = accuracy_score(y_train, model.predict(X_train))
                test_acc = accuracy_score(y_test, model.predict(X_test))
                
                # Plotting
                fig, ax = plt.subplots(figsize=(8, 5))
                
                # Plot decision boundary
                x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
                y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                     np.arange(y_min, y_max, 0.02))
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
                
                # Plot data points
                ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdBu_r, edgecolors='k', alpha=0.8, label="Train")
                ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdBu_r, marker='x', s=80, alpha=0.8, label="Test")
                
                # Uncertainty Metric
                probs = model.predict_proba(X_test)
                avg_entropy = np.mean(calculate_entropy(probs))

                ax.set_xlabel("Feature 1")
                ax.set_ylabel("Feature 2")
                ax.set_title(f"Decision Boundary")
                ax.legend()
                
                st.pyplot(fig)
                
                st.info(f"""
                **Model Complexity:**
                *   Depth: {model.get_depth()}
                *   Number of Leaves: {model.get_n_leaves()}
                
                **Performance:**
                *   Train Accuracy: {train_acc:.2f}
                *   Test Accuracy: {test_acc:.2f}
                *   Avg Test Entropy (Uncertainty): {avg_entropy:.3f}
                """)

    with tab_model:
        #load the data
        df_full, X, y = data_loader(problem_type)

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
        if is_regression:
            model = DecisionTreeRegressor(random_state=42)
            model_type_str = "Regression"
        else:
            model = DecisionTreeClassifier(random_state=42)
            model_type_str = "Classification"

        #fit model
        model = train_model(model, X_train, y_train)
        
        #evaluate the fitted model
        dict_train_results, dict_test_results = evaluate_model(model, model_type_str, X_train, y_train, X_test, y_test)
        
        #----------------------------------
        # Section 3 - evaluate the model
        #----------------------------------
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
            
            # For accuracy, a drop is bad (positive diff if train > test)
            percent_diff = ((acc_train - acc_test) / acc_train) * 100
            st.metric("Accuracy Drop (Overfitting Risk)", f"{percent_diff:.1f}%", delta_color="inverse")
            
            st.write(f"Classes: {np.unique(y)}")
