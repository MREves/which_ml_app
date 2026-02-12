import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

from functions.data_prep.import_data import data_loader
from functions.data_prep.handle_missing_values import check_and_impute_missing_values
from functions.model_helpers.model_setup import evaluate_model, create_train_test_split, train_model
from functions.visualisation.charts import histogram
from functions.model_helpers.uncertainty import calculate_entropy

from sidebar import render_sidebar

def app():
    # Determine problem type early to condition the content
    problem_type = st.session_state.get("problem_type", "Number (regression)")
    is_regression = "regression" in problem_type.lower()
    task_title = "Regression" if is_regression else "Classification"

    st.title("Support Vector Machines")
    st.caption(f"Family: Supervised | Task: {task_title}")

    tab_overview, tab_deep_dive, tab_demo, tab_model = st.tabs(["Overview", "Deep Dive", "Interactive Demo", "ML Model"])

    with tab_overview:
        st.header("What is a Support Vector Machine?")
        if is_regression:
            st.write("""
            Support Vector Regression (SVR) is a supervised algorithm that tries to fit the best line (or hyperplane) within a threshold value (distance $\epsilon$). 
            Unlike standard linear regression that minimizes the squared error, SVR tries to fit the error within a certain "tube" while keeping the model as "flat" (simple) as possible.
            """)
        else:
            st.write("""
            Support Vector Classification (SVC) is a supervised algorithm that finds a hyperplane in an N-dimensional space that distinctly classifies the data points. 
            It aims to maximize the margin (distance) between the data points of the two classes.
            """)
        
        st.subheader("Plain English Intuition")
        if is_regression:
            st.write("""
            Imagine you are trying to lay a **tube** through a cloud of data points. 
            
            *   Your goal is to include as many points as possible inside the tube.
            *   The width of the tube is determined by a parameter $\epsilon$ (epsilon).
            *   Points inside the tube are considered "correctly predicted" and don't add to the error.
            *   Points outside the tube are the "Support Vectors" that dictate the position of the tube.
            """)
        else:
            st.write("""
            Imagine you have red balls and blue balls on a table, and you want to separate them with a stick.
            
            *   You could put the stick anywhere between them, but the best stick is the one that has the **widest gap** (margin) on either side before it touches a ball.
            *   The balls that are closest to the stick and "support" the margin are called **Support Vectors**.
            *   If the balls are mixed up in a way a straight stick can't separate them, you lift the balls into the air (higher dimension) where you might be able to slide a flat sheet between them. This is the **Kernel Trick**.
            """)
        
        st.subheader("When to use")
        col1, col2 = st.columns(2)
        with col1:
            st.success("Use when:")
            st.write("""
            *   High-dimensional data (many features).
            *   Data is not linearly separable (using Kernels).
            *   Accuracy is more important than speed.
            """)
        with col2:
            st.error("Avoid when:")
            st.write("""
            *   Dataset is very large (slow training time).
            *   Data has a lot of noise or overlapping target classes.
            *   You need probability estimates (SVC doesn't provide these directly).
            """)

        st.subheader("Pros & Cons")
        col_pros, col_cons = st.columns(2)
        with col_pros:
            st.markdown("**Pros**")
            st.markdown("*   Effective in high dimensional spaces.")
            st.markdown("*   Memory efficient (uses a subset of training points).")
            st.markdown("*   Versatile (different Kernel functions).")
        with col_cons:
            st.markdown("**Cons**")
            st.markdown("*   Sensitive to noise and outliers.")
            st.markdown("*   Requires feature scaling (normalization).")
            st.markdown("*   Hard to interpret (Black box).")

    with tab_deep_dive:
        st.header("Mechanics")
        st.write("""
        SVMs rely on the concept of a **Hyperplane** (decision boundary) and **Support Vectors** (data points closest to the boundary).
        
        **The Kernel Trick:**
        SVMs are powerful because they can project data into higher dimensions to make it linearly separable.
        *   **Linear**: Good for simple, linearly separable data.
        *   **RBF (Radial Basis Function)**: The default. Projects data into infinite dimensions; creates non-linear boundaries.
        *   **Polynomial**: Creates curved boundaries based on polynomial degrees.
        """)
        
        st.subheader("Hyperparameters")
        st.write(f"Key parameters in `sklearn.svm.{'SVR' if is_regression else 'SVC'}`:")
        
        st.markdown("""
        *   **C** (float): Regularization parameter. 
            *   Low C = Smoother boundary, more misclassifications allowed (High Bias).
            *   High C = Strict boundary, tries to classify all training points correctly (High Variance).
        *   **kernel** (str): 'linear', 'poly', 'rbf', 'sigmoid'.
        *   **gamma** (str/float): Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. Defines how far the influence of a single training example reaches.
        """)
        if is_regression:
            st.markdown("""
            *   **epsilon** (float): The epsilon-tube within which no penalty is associated in the training loss function.
            """)

    with tab_demo:
        st.header("Interactive Demo")
        
        col_settings, col_plot = st.columns([1, 2])
        
        with col_settings:
            st.subheader("Data Generation")
            n_samples = st.slider("Number of samples", 10, 300, 100, step=10, help="The number of data points to generate for this simulation.")
            noise_level = st.slider("Noise level", 0.0, 1.0, 0.1, step=0.05, help="The amount of random noise added to the target variable. Higher noise makes the pattern harder to learn.")
            
            st.subheader("Model Settings")
            kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], help="The kernel function used to map the data into a higher-dimensional space. 'rbf' is versatile, 'linear' is simple, 'poly' can model polynomial relationships.")
            C = st.slider("C (Regularization)", 0.1, 10.0, 1.0, step=0.1, help="Regularization parameter. Lower values create a smoother decision boundary (preventing overfitting), while higher values try to classify all training points correctly (risking overfitting).")
            gamma = st.select_slider("Gamma", options=["scale", "auto", 0.1, 1, 10], value="scale", help="Kernel coefficient. Defines how far the influence of a single training example reaches. Low values mean 'far' (smoother), high values mean 'close' (more complex/wiggly).")
            
            epsilon = 0.1
            if is_regression:
                epsilon = st.slider("Epsilon (Tube width)", 0.01, 1.0, 0.1, step=0.05, help="The epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.")
        
        with col_plot:
            if is_regression:
                st.write("Fitting SVR to a Sine wave.")
                # Generate synthetic data (Sine wave)
                np.random.seed(42)
                X = np.sort(5 * np.random.rand(n_samples, 1), axis=0)
                y = np.sin(X).ravel()
                y[::5] += 3 * (0.5 - np.random.rand(int(n_samples/5))) * noise_level # Add noise
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train model
                model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
                model.fit(X_train, y_train)
                
                # Metrics
                train_mse = mean_squared_error(y_train, model.predict(X_train))
                test_mse = mean_squared_error(y_test, model.predict(X_test))
                
                # Plotting
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(X_train, y_train, color="orange", alpha=0.6, label="Train Data")
                ax.scatter(X_test, y_test, color="green", alpha=0.6, label="Test Data")
                
                # Plot the prediction
                X_plot = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
                y_plot = model.predict(X_plot)
                ax.plot(X_plot, y_plot, color="blue", linewidth=2, label="Prediction")
                
                # Visualize the tube (approximate)
                ax.fill_between(X_plot.ravel(), y_plot - epsilon, y_plot + epsilon, color='blue', alpha=0.1, label="Epsilon Tube")

                ax.set_xlabel("Feature X")
                ax.set_ylabel("Target y")
                ax.legend()
                ax.grid(True, linestyle="--", alpha=0.5)
                
                st.pyplot(fig)
                
                st.info(f"""
                **Model Metrics:**
                *   Train MSE: {train_mse:.4f}
                *   Test MSE: {test_mse:.4f}
                *   Support Vectors: {len(model.support_)}
                """)
            else:
                st.write("Fitting SVC to the 'Moons' dataset.")
                # Generate synthetic data (Moons)
                X, y = make_moons(n_samples=n_samples, noise=noise_level, random_state=42)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train model
                model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
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
                **Model Metrics:**
                *   Train Accuracy: {train_acc:.2f}
                *   Test Accuracy: {test_acc:.2f}
                *   Support Vectors: {len(model.support_)}
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
            model = SVR()
            model_type_str = "Regression"
        else:
            model = SVC(probability=True) # probability=True needed for ROC/AUC in evaluate_model
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
