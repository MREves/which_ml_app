import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from functions.data_prep.import_data import data_loader

from sidebar import render_sidebar

def app():
    st.title("DBSCAN")
    st.caption("Family: Unsupervised | Task: Clustering")

    tab_overview, tab_deep_dive, tab_demo, tab_model = st.tabs(["Overview", "Deep Dive", "Interactive Demo", "ML Model"])

    with tab_overview:
        st.header("What is DBSCAN?")
        st.write("""
        **D**ensity-**B**ased **S**patial **C**lustering of **A**pplications with **N**oise.
        Unlike k-Means, DBSCAN groups points that are closely packed together (points with many nearby neighbors) and marks points that lie alone in low-density regions as outliers (noise).
        """)
        
        st.subheader("Plain English Intuition")
        st.write("""
        Imagine a crowded party.
        *   **Core Person**: Someone standing in a group of at least 4 people.
        *   **Border Person**: Someone standing next to a Core Person, but not surrounded by enough people to be a Core themselves.
        *   **Noise (Loner)**: Someone standing alone, far away from any groups.
        
        DBSCAN finds the groups (clusters) and identifies the loners (noise).
        """)
        
        st.subheader("When to use")
        st.success("Use when:")
        st.write("""
        *   Data has irregular shapes (e.g., crescents, rings) where k-Means fails.
        *   The data contains noise/outliers you want to isolate.
        *   You do not know the number of clusters beforehand.
        """)

        st.subheader("Pros & Cons")
        col_pros, col_cons = st.columns(2)
        with col_pros:
            st.markdown("**Pros**")
            st.markdown("*   Does not require specifying $k$ (number of clusters).")
            st.markdown("*   Robust to outliers.")
            st.markdown("*   Can find arbitrarily shaped clusters.")
        with col_cons:
            st.markdown("**Cons**")
            st.markdown("*   Sensitive to hyperparameters (`eps`, `min_samples`).")
            st.markdown("*   Struggles with clusters of varying densities.")
            st.markdown("*   Not deterministic for border points.")

    with tab_deep_dive:
        st.header("Mechanics")
        st.write("DBSCAN relies on two key parameters:")
        st.markdown("""
        1.  **eps ($\epsilon$)**: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        2.  **min_samples**: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        """)

    with tab_demo:
        st.header("Interactive Demo")
        st.write("DBSCAN vs k-Means on irregular shapes.")
        
        col_settings, col_plot = st.columns([1, 2])
        with col_settings:
            dataset_type = st.selectbox("Dataset Shape", ["Moons", "Circles"])
            noise = st.slider("Noise", 0.0, 0.2, 0.05)
            
            st.divider()
            eps = st.slider("Epsilon (eps)", 0.1, 1.0, 0.3, step=0.05)
            min_samples = st.slider("Min Samples", 2, 20, 5)
            
        with col_plot:
            if dataset_type == "Moons":
                X, _ = make_moons(n_samples=300, noise=noise, random_state=42)
            else:
                X, _ = make_circles(n_samples=300, factor=0.5, noise=noise, random_state=42)
            
            # Standardize for DBSCAN (important for eps)
            X = StandardScaler().fit_transform(X)
            
            model = DBSCAN(eps=eps, min_samples=min_samples)
            y_pred = model.fit_predict(X)
            
            # Uncertainty: Noise Ratio
            n_noise = np.sum(y_pred == -1)
            noise_ratio = n_noise / len(y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            # Plot noise as black
            unique_labels = set(y_pred)
            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
            
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    col = [0, 0, 0, 1] # Black for noise
                    label = "Noise"
                else:
                    label = f"Cluster {k}"

                class_member_mask = (y_pred == k)
                xy = X[class_member_mask]
                ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=8, label=label)
            
            ax.set_title(f"DBSCAN Clustering (Clusters: {len(unique_labels)-1 if -1 in unique_labels else len(unique_labels)})")
            st.pyplot(fig)
            st.info(f"**Outlier Detection (Uncertainty):** {n_noise} points classified as noise ({noise_ratio:.1%} of data).")
            
            with st.expander("How to interpret noise levels?"):
                st.markdown("""
                *   **< 5% (Low):** `eps` might be too large; risk of merging distinct clusters.
                *   **5-20% (Moderate):** Typical for real-world data; background noise is being filtered.
                *   **> 25% (High):** `eps` might be too small or `min_samples` too high; valid data is being dropped.
                """)

    with tab_model:
        st.header("Model Application")
        df_full, X, y = data_loader("Unsupervised (clustering)")
        X_scaled = StandardScaler().fit_transform(X)
        
        model = DBSCAN(eps=3.0, min_samples=5) # Relaxed params for high-dim data
        clusters = model.fit_predict(X_scaled)
        
        # Uncertainty: Noise Ratio
        n_noise = np.sum(clusters == -1)
        noise_ratio = n_noise / len(clusters)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        st.pyplot(fig)
        st.info(f"**Outlier Detection (Uncertainty):** {n_noise} points classified as noise ({noise_ratio:.1%} of data).")
        
        with st.expander("How to interpret noise levels?"):
            st.markdown("""
            *   **< 5% (Low):** `eps` might be too large; risk of merging distinct clusters.
            *   **5-20% (Moderate):** Typical for real-world data; background noise is being filtered.
            *   **> 25% (High):** `eps` might be too small or `min_samples` too high; valid data is being dropped.
            """)
