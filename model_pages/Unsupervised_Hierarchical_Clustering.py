import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from functions.data_prep.import_data import data_loader

from sidebar import render_sidebar

def app():
    st.title("Hierarchical Clustering")
    st.caption("Family: Unsupervised | Task: Clustering")

    tab_overview, tab_deep_dive, tab_demo, tab_model = st.tabs(["Overview", "Deep Dive", "Interactive Demo", "ML Model"])

    with tab_overview:
        st.header("What is Hierarchical Clustering?")
        st.write("""
        Hierarchical Clustering is a method of cluster analysis which seeks to build a hierarchy of clusters. 
        The most common type is **Agglomerative** (bottom-up): each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy.
        """)
        
        st.subheader("Plain English Intuition")
        st.write("""
        Imagine a family tree.
        *   At the bottom, you have individual people (data points).
        *   Brothers and sisters are merged into a family unit.
        *   Cousins are merged into a larger extended family.
        *   Eventually, everyone is merged into one giant "Humanity" cluster at the top.
        
        You can choose where to "cut" the tree to decide how many families (clusters) you want to look at.
        """)
        
        st.subheader("When to use")
        st.success("Use when:")
        st.write("""
        *   You want to visualize the relationship between clusters (Dendrogram).
        *   You don't know the number of clusters $k$ in advance (though you still need to pick a cut-off).
        *   The dataset is small to medium size (it's computationally expensive).
        """)

        st.subheader("Pros & Cons")
        col_pros, col_cons = st.columns(2)
        with col_pros:
            st.markdown("**Pros**")
            st.markdown("*   Produces a Dendrogram (great visualization).")
            st.markdown("*   Captures hierarchical relationships.")
            st.markdown("*   No need to pre-specify $k$ (can choose after viewing dendrogram).")
        with col_cons:
            st.markdown("**Cons**")
            st.markdown("*   Computationally expensive ($O(n^3)$ or $O(n^2)$).")
            st.markdown("*   Sensitive to noise and outliers.")
            st.markdown("*   Once a step (merge) is done, it cannot be undone.")

    with tab_deep_dive:
        st.header("Mechanics")
        st.write("""
        **Agglomerative Clustering Steps:**
        1.  Compute the proximity matrix (distance between every pair of points).
        2.  Merge the two closest clusters.
        3.  Update the proximity matrix.
        4.  Repeat until only one cluster remains.
        """)
        
        st.subheader("Linkage Criteria")
        st.write("How do we measure the distance between two *clusters*?")
        st.markdown("""
        *   **Ward**: Minimizes the variance of the clusters being merged. (Most common, tends to produce spherical clusters).
        *   **Complete (Maximum)**: Distance between the two *farthest* points in the clusters.
        *   **Average**: Average distance between all points in cluster A and all points in cluster B.
        *   **Single (Minimum)**: Distance between the two *closest* points. (Can produce "chaining" effect).
        """)

    with tab_demo:
        st.header("Interactive Demo")
        
        col_settings, col_plot = st.columns([1, 2])
        
        with col_settings:
            n_samples = st.slider("Number of samples", 10, 100, 50, step=10, help="The number of data points to generate.")
            n_centers = st.slider("True Clusters", 2, 5, 3, help="The actual number of clusters in the generated data.")
            cluster_std = st.slider("Cluster Spread", 0.1, 2.0, 1.0, help="Standard deviation of the clusters. Higher means more overlap.")
            
            st.divider()
            linkage_method = st.selectbox("Linkage Method", ["ward", "complete", "average", "single"], help="""The criterion used to determine the distance between two clusters:
- Ward: Minimizes variance (spherical clusters).
- Complete: Max distance between points (compact clusters).
- Average: Average distance between points.
- Single: Min distance between points (can chain).""")
            n_clusters_model = st.slider("Number of Clusters (Cut)", 1, 10, 3, help="The number of clusters to find (where to cut the dendrogram).")
        
        with col_plot:
            X, y = make_blobs(n_samples=n_samples, centers=n_centers, cluster_std=cluster_std, random_state=42)
            
            # 1. Dendrogram
            st.subheader("Dendrogram")
            fig_dendro, ax_dendro = plt.subplots(figsize=(8, 4))
            Z = linkage(X, method=linkage_method)
            dendrogram(Z, ax=ax_dendro)
            ax_dendro.set_title(f"Hierarchical Clustering Dendrogram ({linkage_method})")
            ax_dendro.set_xlabel("Sample index")
            ax_dendro.set_ylabel("Distance")
            st.pyplot(fig_dendro)
            
            # 2. Scatter Plot
            st.subheader("Cluster Visualization")
            model = AgglomerativeClustering(n_clusters=n_clusters_model, linkage=linkage_method)
            y_pred = model.fit_predict(X)
            
            # Uncertainty
            if n_clusters_model > 1:
                sil_score = silhouette_score(X, y_pred)
                st.info(f"**Cluster Quality (Uncertainty):** Silhouette Score = {sil_score:.3f}")
                with st.expander("How to interpret Silhouette Score?"):
                    st.markdown("""
                    *   **Close to +1:** Samples are far away from neighboring clusters (Good).
                    *   **0:** Samples are on or very close to the decision boundary between two neighboring clusters.
                    *   **Close to -1:** Samples might have been assigned to the wrong cluster.
                    """)
            else:
                st.warning("Silhouette Score requires at least 2 clusters.")
            
            fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
            ax_scatter.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50, alpha=0.8)
            ax_scatter.set_title(f"Clusters (k={n_clusters_model})")
            st.pyplot(fig_scatter)

    with tab_model:
        st.header("Model Application")
        df_full, X, y = data_loader("Unsupervised (clustering)")
        X_scaled = StandardScaler().fit_transform(X)
        
        col_params, col_viz = st.columns([1, 3])
        
        with col_params:
            k_model = st.slider("Select k", 2, 10, 2, key="k_model_real")
            linkage_real = st.selectbox("Linkage", ["ward", "complete", "average", "single"], key="linkage_real")
            
        with col_viz:
            model = AgglomerativeClustering(n_clusters=k_model, linkage=linkage_real)
            clusters = model.fit_predict(X_scaled)
            
            # Metrics
            sil_score = silhouette_score(X_scaled, clusters)
            st.metric("Silhouette Score", f"{sil_score:.3f}", help="Ranges from -1 to 1. High values indicate that the object is well matched to its own cluster and poorly matched to neighboring clusters.")
            
            with st.expander("How to interpret Silhouette Score?"):
                st.markdown("""
                *   **Close to +1:** Samples are far away from neighboring clusters (Good).
                *   **0:** Samples are on or very close to the decision boundary between two neighboring clusters.
                *   **Close to -1:** Samples might have been assigned to the wrong cluster.
                """)
            
            # Visualization using PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")
            ax.set_title("Clusters Visualized in 2D (PCA)")
            st.pyplot(fig)
