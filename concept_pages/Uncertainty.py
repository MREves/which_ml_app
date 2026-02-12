import streamlit as st
import pandas as pd

def app():
    st.title("Uncertainty Quantification")
    st.markdown("### Metrics for Quantifying Uncertainty")
    st.write("Incorporating uncertainty quantification transforms a model from a 'black box' that outputs a single number into a system that expresses what it *doesn't* know. This is crucial for decision-making.")
    
    data = [
        {"Model Family": "Linear Regression", "Uncertainty Source": "Data noise & Parameter variance", "Key Metric": "Prediction Interval Width (via Bootstrap)"},
        {"Model Family": "Random Forest (Reg)", "Uncertainty Source": "Disagreement among trees", "Key Metric": "Standard Deviation of tree predictions"},
        {"Model Family": "Gradient Boosting", "Uncertainty Source": "Distributional tails", "Key Metric": "Quantile Interval Width"},
        {"Model Family": "Classifiers (LogReg/SVM)", "Uncertainty Source": "Class overlap", "Key Metric": "Prediction Entropy / Probability"},
        {"Model Family": "k-Means", "Uncertainty Source": "Cluster cohesion", "Key Metric": "Silhouette Score (per sample)"},
        {"Model Family": "PCA", "Uncertainty Source": "Information loss", "Key Metric": "Reconstruction Error"},
        {"Model Family": "DBSCAN", "Uncertainty Source": "Outlier detection", "Key Metric": "Classification as Noise (-1)"}
    ]
    
    df = pd.DataFrame(data)
    # Reorder columns to match the requested table format
    st.table(df[["Model Family", "Uncertainty Source", "Key Metric"]])