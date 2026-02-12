import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from functions.data_prep.import_data import data_loader
from functions.data_prep.handle_missing_values import check_and_impute_missing_values

from sidebar import render_sidebar

def app():
    st.title("k-Means Clustering")
    st.caption("Family: Unsupervised | Task: Clustering")

    tab_overview, tab_deep_dive, tab_demo, tab_model = st.tabs(["Overview", "Deep Dive", "Interactive Demo", "ML Model"])

    with tab_overview:
        st.header("What is k-Means?")
        st.write("""
        k-Means is a popular centroid-based clustering algorithm. It partitions the dataset into $k$ distinct, non-overlapping subgroups (clusters) 
        where each data point belongs to the cluster with the nearest mean (centroid).
        """)
        
        st.subheader("Plain English Intuition")
        st.write("""
        Imagine you have a pile of unlabelled photos on a table and you want to organize them into 3 piles ($k=3$).
        
        1.  You randomly place 3 markers on the table.
        2.  You assign every photo to the marker closest to it.
        3.  You move each marker to the exact center of its assigned pile.
        4.  You repeat steps 2 and 3 until the markers stop moving.
        """)
        
        st.subheader("When to use")
        st.success("Use when:")
        st.write("""
        *   You have a general idea of how many clusters ($k$) exist.
        *   Clusters are expected to be roughly spherical and of similar size.
        *   You need a fast and scalable algorithm.
        """)

        st.subheader("Pros & Cons")
        col_pros, col_cons = st.columns(2)
        with col_pros:
            st.markdown("**Pros**")
            st.markdown("*   Simple to implement and interpret.")
            st.markdown("*   Scales well to large datasets.")
            st.markdown("*   Guarantees convergence.")
        with col_cons:
            st.markdown("**Cons**")
            st.markdown("*   Must specify $k$ manually.")
            st.markdown("*   Sensitive to initial centroid placement.")
            st.markdown("*   Struggles with non-spherical shapes or varying densities.")

    with tab_deep_dive:
        st.header("Mechanics")
        st.write("The algorithm minimizes the **Within-Cluster Sum of Squares (WCSS)**, also known as inertia.")
        st.latex(r"\sum_{i=0}^{n} \min_{\mu_j \in C} (||x_i - \mu_j||^2)")
        st.write("It iterates between two steps: **Assignment** (assign points to nearest centroid) and **Update** (recalculate centroids).")
        
        st.subheader("Hyperparameters")
        st.markdown("""
        *   **n_clusters** (int): The number of clusters to form as well as the number of centroids to generate.
        *   **init** (str): Method for initialization (e.g., 'k-means++' selects initial cluster centers smartly to speed up convergence).
        *   **n_init** (int): Number of time the k-means algorithm will be run with different centroid seeds.
        """)

    with tab_demo:
        st.header("Interactive Demo")
        col_settings, col_plot = st.columns([1, 2])
        
        with col_settings:
            n_samples = st.slider("Number of samples", 50, 500, 200, step=10)
            n_centers_gen = st.slider("True Clusters (Data)", 2, 6, 3)
            cluster_std = st.slider("Cluster Spread", 0.1, 2.0, 0.8)
            
            st.divider()
            k_clusters = st.slider("k (Model Clusters)", 1, 8, 3, help="The number of clusters the model will try to find.")

        with col_plot:
            X, y_true = make_blobs(n_samples=n_samples, centers=n_centers_gen, cluster_std=cluster_std, random_state=42)
            
            model = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
            y_pred = model.fit_predict(X)
            
            # Uncertainty: Silhouette Score
            sil_score = silhouette_score(X, y_pred)

            fig, ax = plt.subplots(figsize=(8, 5))
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
            
            # Plot centroids
            centers = model.cluster_centers_
            ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.9, marker='X', label="Centroids")
            
            ax.set_title(f"k-Means Clustering (k={k_clusters})")
            ax.legend()
            st.pyplot(fig)
            st.info(f"**Cluster Quality (Uncertainty):** Silhouette Score = {sil_score:.3f} (Closer to 1 is better)")

    with tab_model:
        st.header("Model Application")
        st.write("Using the Breast Cancer dataset (ignoring labels) to find natural groupings.")
        
        df_full, X, y = data_loader("Unsupervised (clustering)")
        
        # Preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        col_params, col_viz = st.columns([1, 3])
        
        with col_params:
            k_model = st.slider("Select k", 2, 10, 2, key="k_model_real")
            
        with col_viz:
            model = KMeans(n_clusters=k_model, random_state=42, n_init=10)
            clusters = model.fit_predict(X_scaled)
            
            # Metrics
            sil_score = silhouette_score(X_scaled, clusters)
            st.metric("Silhouette Score", f"{sil_score:.3f}", help="Ranges from -1 to 1. High values indicate that the object is well matched to its own cluster and poorly matched to neighboring clusters.")
            
            # Visualization using PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")
            ax.set_title("Clusters Visualized in 2D (PCA)")
            st.pyplot(fig)
