import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="t-SNE Digits Visualizer",
    layout="wide"
)

st.title("t-SNE Digits Visualizer")

st.markdown("""
This application demonstrates **t-SNE (t-Distributed Stochastic Neighbor Embedding)**
on the Digits Dataset.

Adjust the perplexity value and observe how the clusters change.
""")

# -----------------------------
# Load Dataset
# -----------------------------
digits = load_digits()

X = digits.data
y = digits.target

# -----------------------------
# Dataset Info
# -----------------------------
st.subheader("Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Samples", X.shape[0])

with col2:
    st.metric("Features", X.shape[1])

with col3:
    st.metric("Classes", len(set(y)))

# -----------------------------
# Sample Images
# -----------------------------
st.subheader("Sample Digit Images")

fig, axes = plt.subplots(1, 5, figsize=(10, 3))

for i in range(5):
    axes[i].imshow(
        digits.images[i],
        cmap="gray"
    )
    axes[i].set_title(f"Digit {y[i]}")
    axes[i].axis("off")

st.pyplot(fig)

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("t-SNE Parameters")

perplexity = st.sidebar.slider(
    "Perplexity",
    min_value=5,
    max_value=50,
    value=30,
    step=5
)

# -----------------------------
# Scale Data
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Run t-SNE
# -----------------------------
with st.spinner("Running t-SNE..."):

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42
    )

    X_tsne = tsne.fit_transform(X_scaled)

# -----------------------------
# Create DataFrame
# -----------------------------
tsne_df = pd.DataFrame({
    "TSNE1": X_tsne[:, 0],
    "TSNE2": X_tsne[:, 1],
    "Digit": y
})

# -----------------------------
# t-SNE Plot
# -----------------------------
st.subheader("t-SNE Visualization")

fig2, ax = plt.subplots(figsize=(10, 7))

scatter = ax.scatter(
    tsne_df["TSNE1"],
    tsne_df["TSNE2"],
    c=tsne_df["Digit"],
    alpha=0.7
)

legend = ax.legend(
    *scatter.legend_elements(),
    title="Digits"
)

ax.add_artist(legend)

ax.set_xlabel("t-SNE Component 1")
ax.set_ylabel("t-SNE Component 2")

ax.set_title(
    f"t-SNE Projection (Perplexity = {perplexity})"
)

st.pyplot(fig2)

# -----------------------------
# Explanation
# -----------------------------
st.subheader("Interpretation")

st.info(
    """
    t-SNE reduces 64-dimensional digit data into 2 dimensions.
    
    Points that appear close together represent similar digits.
    
    Different clusters correspond to different handwritten digit classes.
    
    Changing the perplexity alters how local and global relationships
    are balanced during visualization.
    """
)