import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

from functions.data_prep.import_data import data_loader

from sidebar import render_sidebar

def app():
    st.title("Principal Component Analysis (PCA)")
    st.caption("Family: Unsupervised | Task: Dimensionality Reduction")

    tab_overview, tab_deep_dive, tab_demo, tab_model = st.tabs(["Overview", "Deep Dive", "Interactive Demo", "ML Model"])

    with tab_overview:
        st.header("What is PCA?")
        st.write("""
        Principal Component Analysis (PCA) is a technique used to reduce the dimensionality of a dataset while preserving as much variability (information) as possible.
        It transforms correlated features into a new set of uncorrelated features called **Principal Components**.
        """)
        
        st.subheader("Plain English Intuition")
        st.write("""
        Imagine you are taking a photo of a teapot. 
        *   If you take the photo from the top, it looks like a circle (you lose information about its height).
        *   If you take it from the side, you see the spout, handle, and body (maximum information).
        
        PCA finds the "best angle" to view your data so that the differences between data points are most visible, allowing you to ignore the "angles" that don't show much detail.
        """)
        
        st.subheader("When to use")
        st.success("Use when:")
        st.write("""
        *   You have high-dimensional data (many features) and suffer from the "Curse of Dimensionality".
        *   You want to visualize complex data in 2D or 3D.
        *   Features are highly correlated (multicollinearity).
        """)

        st.subheader("Pros & Cons")
        col_pros, col_cons = st.columns(2)
        with col_pros:
            st.markdown("**Pros**")
            st.markdown("*   Reduces overfitting by removing noise.")
            st.markdown("*   Improves algorithm speed.")
            st.markdown("*   Removes correlated features.")
        with col_cons:
            st.markdown("**Cons**")
            st.markdown("*   Principal Components are less interpretable than original features.")
            st.markdown("*   Data must be standardized first.")
            st.markdown("*   Loss of information is inevitable.")

    with tab_deep_dive:
        st.header("Mechanics")
        st.write("""
        1.  **Standardize** the data (mean=0, variance=1).
        2.  Compute the **Covariance Matrix** to understand how variables vary together.
        3.  Compute **Eigenvectors** (directions) and **Eigenvalues** (magnitude of variance) of the covariance matrix.
        4.  Sort Eigenvectors by Eigenvalues (highest to lowest).
        5.  Select top $k$ Eigenvectors to form the new feature space.
        """)

    with tab_demo:
        st.header("Interactive Demo")
        st.write("Visualizing the Principal Components of a 2D dataset.")
        
        col_settings, col_plot = st.columns([1, 2])
        with col_settings:
            n_samples = st.slider("Number of samples", 50, 500, 200)
            correlation = st.slider("Stretch Data (Correlation)", 1.0, 10.0, 3.0, help="Stretches the blobs to create correlation between x and y.")
            
        with col_plot:
            # Generate data
            rng = np.random.RandomState(42)
            X = np.dot(rng.rand(2, 2), rng.randn(2, n_samples)).T
            X[:, 0] = X[:, 0] * correlation # Stretch
            
            pca = PCA(n_components=2)
            pca.fit(X)
            
            # Uncertainty: Reconstruction Error (Loss)
            X_reconstructed = pca.inverse_transform(pca.transform(X))
            loss = np.mean(np.sum((X - X_reconstructed) ** 2, axis=1))

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(X[:, 0], X[:, 1], alpha=0.5)
            
            # Plot vectors
            mean = pca.mean_
            for i, (length, vector) in enumerate(zip(pca.explained_variance_, pca.components_)):
                v = vector * 3 * np.sqrt(length)
                ax.arrow(mean[0], mean[1], v[0], v[1], head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2)
                ax.text(mean[0]+v[0], mean[1]+v[1], f"PC{i+1}", color='red', fontsize=12)
            
            ax.set_title("Data with Principal Component Vectors")
            ax.axis('equal')
            st.pyplot(fig)
            st.info(f"**Information Loss (Uncertainty):** Reconstruction Error = {loss:.4f}")

    with tab_model:
        st.header("Model Application")
        df_full, X, y = data_loader("Unsupervised (clustering)")
        
        # Preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        n_components = st.slider("Number of Components to Keep", 1, X.shape[1], 5)
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        st.write(f"Original Shape: {X.shape} -> Transformed Shape: {X_pca.shape}")
        
        # Explained Variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        st.metric("Total Variance Explained", f"{cumulative_variance[-1]*100:.2f}%")
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(1, n_components + 1), explained_variance, alpha=0.5, align='center', label='Individual explained variance')
        ax.step(range(1, n_components + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
        ax.set_ylabel('Explained variance ratio')
        ax.set_xlabel('Principal component index')
        ax.legend(loc='best')
        st.pyplot(fig)
