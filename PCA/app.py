import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Breast Cancer PCA Visualizer",
    layout="wide"
)

st.title("Breast Cancer PCA Visualizer")
st.markdown(
    """
    This application performs Principal Component Analysis (PCA)
    on the Breast Cancer dataset and visualizes the reduced
    dimensions.
    """
)

# -----------------------------
# Load Models
# -----------------------------

from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"

scaler = joblib.load(MODELS_DIR / "scaler.pkl")
pca = joblib.load(MODELS_DIR / "pca.pkl")

# -----------------------------
# Upload Dataset
# -----------------------------
DATA_PATH = BASE_DIR / "data" / "data.csv"

df = pd.read_csv(DATA_PATH)
if not df.empty:

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    rows, cols = df.shape

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", rows)

    with col2:
        st.metric("Columns", cols)

    # -----------------------------
    # Save Diagnosis Labels
    # -----------------------------
    diagnosis = None

    if "diagnosis" in df.columns:
        diagnosis = df["diagnosis"]

    # -----------------------------
    # Drop Unnecessary Columns
    # -----------------------------
    drop_cols = []

    for col in ["id", "diagnosis", "Unnamed: 32"]:
        if col in df.columns:
            drop_cols.append(col)

    X = df.drop(columns=drop_cols)

    # -----------------------------
    # Scaling
    # -----------------------------
    X_scaled = scaler.transform(X)

    # -----------------------------
    # PCA Transformation
    # -----------------------------
    X_pca = pca.transform(X_scaled)

    pca_df = pd.DataFrame(
        X_pca,
        columns=["PC1", "PC2"]
    )

    st.subheader("PCA Transformed Data")

    st.dataframe(pca_df.head())

    # -----------------------------
    # Explained Variance
    # -----------------------------
    st.subheader("Explained Variance")

    variance = pca.explained_variance_ratio_

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "PC1 Variance",
            f"{variance[0]*100:.2f}%"
        )

    with c2:
        st.metric(
            "PC2 Variance",
            f"{variance[1]*100:.2f}%"
        )

    with c3:
        st.metric(
            "Total Variance Retained",
            f"{variance.sum()*100:.2f}%"
        )

    # -----------------------------
    # PCA Scatter Plot
    # -----------------------------
    st.subheader("PCA Scatter Plot")

    fig1, ax1 = plt.subplots(figsize=(8, 6))

    if diagnosis is not None:

        colors = {
            "M": "red",
            "B": "blue"
        }

        for label in diagnosis.unique():

            mask = diagnosis == label

            ax1.scatter(
                pca_df.loc[mask, "PC1"],
                pca_df.loc[mask, "PC2"],
                color=colors.get(label, "gray"),
                label=label,
                alpha=0.7
            )

        ax1.legend(
            title="Diagnosis"
        )

    else:

        ax1.scatter(
            pca_df["PC1"],
            pca_df["PC2"],
            alpha=0.7
        )

    ax1.set_xlabel("Principal Component 1")
    ax1.set_ylabel("Principal Component 2")
    ax1.set_title("2D PCA Projection")

    st.pyplot(fig1)

    # -----------------------------
    # Cumulative Variance Plot
    # -----------------------------
    st.subheader("Cumulative Variance Plot")

    pca_full = PCA()
    pca_full.fit(X_scaled)

    cumulative_variance = np.cumsum(
        pca_full.explained_variance_ratio_
    )

    fig2, ax2 = plt.subplots(figsize=(8, 6))

    ax2.plot(
        range(
            1,
            len(cumulative_variance) + 1
        ),
        cumulative_variance,
        marker="o"
    )

    ax2.axhline(
        y=0.95,
        linestyle="--",
        label="95% Variance"
    )

    ax2.set_xlabel(
        "Number of Components"
    )

    ax2.set_ylabel(
        "Cumulative Variance"
    )

    ax2.set_title(
        "Variance Explained by Components"
    )

    ax2.legend()

    ax2.grid(True)

    st.pyplot(fig2)

    # -----------------------------
    # Components Needed for 95%
    # -----------------------------
    st.subheader("95% Variance Analysis")

    pca_95 = PCA(n_components=0.95)

    pca_95.fit(X_scaled)

    st.success(
        f"{pca_95.n_components_} principal components are required to retain 95% of the variance."
    )

    # -----------------------------
    # Download PCA Output
    # -----------------------------
    st.subheader("Download PCA Results")

    csv = pca_df.to_csv(
        index=False
    )

    st.download_button(
        label="Download PCA Dataset",
        data=csv,
        file_name="pca_output.csv",
        mime="text/csv"
    )