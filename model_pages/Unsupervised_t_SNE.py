import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from functions.data_prep.import_data import data_loader

from sidebar import render_sidebar

def app():
    st.title("t-SNE (t-Distributed Stochastic Neighbor Embedding)")
    st.caption("Family: Unsupervised | Task: Dimensionality Reduction / Visualization")

    tab_overview, tab_deep_dive, tab_demo, tab_model = st.tabs(["Overview", "Deep Dive", "Interactive Demo", "ML Model"])

    with tab_overview:
        st.header("What is t-SNE?")
        st.write("""
        t-SNE is a statistical method for visualizing high-dimensional data by giving each datapoint a location in a two or three-dimensional map.
        Unlike PCA (which is linear), t-SNE is non-linear and is particularly good at preserving local structure (keeping similar points close together).
        """)
        
        st.subheader("Plain English Intuition")
        st.write("""
        Imagine you have a crumpled ball of paper (the high-dimensional manifold) with points drawn on it.
        *   **PCA** is like shining a light and looking at the shadow. It works well if the paper is mostly flat, but if it's crumpled, points far apart might look close in the shadow.
        *   **t-SNE** is like carefully unfolding the paper and pinning it flat on a board, trying to make sure that points that were close on the crumpled ball stay close on the board.
        """)
        
        st.subheader("When to use")
        st.success("Use when:")
        st.write("""
        *   Visualizing complex, high-dimensional datasets (e.g., images, genomic data).
        *   You want to see clusters that are not linearly separable.
        *   PCA fails to show clear structure.
        """)

        st.subheader("Pros & Cons")
        col_pros, col_cons = st.columns(2)
        with col_pros:
            st.markdown("**Pros**")
            st.markdown("*   Excellent for visualization (2D/3D).")
            st.markdown("*   Captures non-linear structure.")
            st.markdown("*   Preserves local neighborhoods well.")
        with col_cons:
            st.markdown("**Cons**")
            st.markdown("*   Computationally expensive ($O(n^2)$).")
            st.markdown("*   Stochastic (results change with different seeds).")
            st.markdown("*   Global structure (distances between far clusters) is not always preserved.")
            st.markdown("*   Cannot be applied to new data (must re-run on full set).")

    with tab_deep_dive:
        st.header("Mechanics")
        st.write("""
        1.  **High-Dimensional Space**: Measure similarities between points using a Gaussian distribution. Close points have high probability, far points have low.
        2.  **Low-Dimensional Space**: Define a similar probability distribution (using a Student's t-distribution) for points in the low-dimensional map.
        3.  **Optimization**: Minimize the **Kullback-Leibler (KL) Divergence** between the two distributions using Gradient Descent.
        """)
        
        st.subheader("Hyperparameters")
        st.write("Key parameters in `sklearn.manifold.TSNE`:")
        st.markdown("""
        *   **perplexity** (float): Related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50.
        *   **n_iter** (int): Maximum number of iterations for the optimization. Should be at least 250.
        *   **learning_rate** (float): The learning rate for the t-SNE is usually in the range [10.0, 1000.0].
        """)

    with tab_demo:
        st.header("Interactive Demo")
        st.write("Effect of Perplexity on t-SNE.")
        
        col_settings, col_plot = st.columns([1, 2])
        
        with col_settings:
            n_samples = st.slider("Number of samples", 50, 500, 200, step=50, help="Number of data points.")
            n_centers = st.slider("Clusters", 2, 5, 3, help="Number of clusters in high-dimensional space.")
            cluster_std = st.slider("Cluster Spread", 0.1, 2.0, 1.0, help="Spread of clusters.")
            
            st.divider()
            perplexity = st.slider("Perplexity", 2, 100, 30, help="The balance between local and global aspects of your data. Roughly the number of neighbors.")
        
        with col_plot:
            # Generate high-dim data (e.g., 10 features)
            X, y = make_blobs(n_samples=n_samples, centers=n_centers, n_features=10, cluster_std=cluster_std, random_state=42)
            
            # Run t-SNE
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='random', learning_rate='auto')
            X_embedded = tsne.fit_transform(X)
            
            # Metric: KL Divergence
            kl_divergence = tsne.kl_divergence_
            
            fig, ax = plt.subplots(figsize=(8, 5))
            scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', alpha=0.7)
            ax.set_title(f"t-SNE Visualization")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            st.pyplot(fig)
            
            st.info(f"**Model Quality (Uncertainty):** Final KL Divergence = {kl_divergence:.4f} (Lower is better)")
            with st.expander("Interpreting Perplexity"):
                st.write("""
                *   **Low Perplexity**: Focuses heavily on local structure. Can break clusters into artificial sub-clusters.
                *   **High Perplexity**: Considers more global structure. Can merge distinct clusters if set too high relative to sample size.
                """)

    with tab_model:
        st.header("Model Application")
        df_full, X, y = data_loader("Unsupervised (clustering)")
        
        # Preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        col_params, col_viz = st.columns([1, 3])
        
        with col_params:
            perplexity_real = st.slider("Perplexity", 5, 50, 30, key="perp_real")
            
        with col_viz:
            with st.spinner("Running t-SNE..."):
                tsne = TSNE(n_components=2, perplexity=perplexity_real, random_state=42, init='random', learning_rate='auto')
                X_embedded = tsne.fit_transform(X_scaled)
                kl_divergence = tsne.kl_divergence_
            
            fig, ax = plt.subplots(figsize=(8, 5))
            # We don't have labels for unsupervised usually, but data_loader returns y (target) which we can use for coloring to verify
            scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', alpha=0.6)
            ax.set_title("t-SNE Projection of Breast Cancer Data")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            st.pyplot(fig)
            
            st.info(f"**Model Quality (Uncertainty):** Final KL Divergence = {kl_divergence:.4f}")
