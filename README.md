# Dimensionality Reduction Techniques: PCA and t-SNE

## Project Overview

This project demonstrates two widely used dimensionality reduction techniques in Machine Learning:

1. **Principal Component Analysis (PCA)**
2. **t-Distributed Stochastic Neighbor Embedding (t-SNE)**

The goal is to reduce high-dimensional data into lower dimensions while preserving important information for visualization and analysis.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Streamlit
* Joblib

---

# Project 1: Principal Component Analysis (PCA)

## Dataset

Breast Cancer Wisconsin Dataset

### Dataset Characteristics

* Samples: 569
* Features: 30 numerical features
* Target:

  * M = Malignant
  * B = Benign

## Objective

Reduce 30-dimensional breast cancer data into 2 principal components while preserving maximum variance.

## Workflow

Dataset
→ Data Cleaning
→ Feature Scaling
→ PCA
→ Explained Variance Analysis
→ Visualization
→ Streamlit Dashboard

## Preprocessing

* Removed `id` column
* Removed `Unnamed: 32` column
* Encoded diagnosis labels
* Applied StandardScaler

## PCA Implementation

* Reduced dimensions from 30 to 2
* Generated:

  * PCA Scatter Plot
  * Explained Variance Analysis
  * Cumulative Variance Plot
  * 95% Variance Analysis

## Features of PCA Dashboard

* Dataset Preview
* PCA Transformation
* Explained Variance Metrics
* PCA Scatter Plot
* Cumulative Variance Visualization
* Download PCA Output

## Key Concepts

### Principal Components

Principal Components are new features created by combining the original features in a way that captures maximum variance.

### Explained Variance Ratio

Measures how much information is retained by each principal component.

### Advantages of PCA

* Reduces dimensionality
* Removes redundancy
* Speeds up computation
* Helps visualization
* Reduces noise

---

# Project 2: t-SNE

## Dataset

Digits Dataset (Scikit-learn)

### Dataset Characteristics

* Samples: 1797
* Features: 64
* Classes: 10 (Digits 0–9)

## Objective

Visualize high-dimensional handwritten digit data in two dimensions while preserving neighborhood relationships.

## Workflow

Dataset
→ Feature Scaling
→ t-SNE
→ Cluster Visualization
→ Streamlit Dashboard

## Preprocessing

* Loaded Digits Dataset
* Applied StandardScaler
* Performed t-SNE transformation

## t-SNE Implementation

* Reduced 64 dimensions to 2 dimensions
* Visualized digit clusters
* Added interactive perplexity control

## Features of t-SNE Dashboard

* Dataset Statistics
* Sample Digit Images
* Interactive Perplexity Slider
* t-SNE Cluster Visualization
* Cluster Interpretation

## Key Concepts

### Neighborhood Preservation

t-SNE attempts to keep similar data points close together in lower dimensions.

### Perplexity

Perplexity determines how many neighboring points influence each data point during visualization.

Typical range:

* 5 to 50

Default value:

* 30

### Advantages of t-SNE

* Excellent visualization quality
* Reveals hidden clusters
* Preserves local relationships
* Useful for exploratory data analysis

---

# PCA vs t-SNE

| Feature               | PCA                      | t-SNE                  |
| --------------------- | ------------------------ | ---------------------- |
| Type                  | Linear                   | Non-Linear             |
| Purpose               | Dimensionality Reduction | Visualization          |
| Preserves             | Variance                 | Neighborhood Structure |
| Speed                 | Fast                     | Slow                   |
| Interpretability      | High                     | Low                    |
| Suitable for New Data | Yes                      | Limited                |
| Common Output         | Feature Reduction        | Cluster Visualization  |

---

# How to Run the Applications

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run PCA Application

```bash
streamlit run pca_app.py
```

## Run t-SNE Application

```bash
streamlit run tsne_app.py
```

---

# Project Structure

```text
Dimensionality_Reduction/
│
├── PCA/
│   ├── notebook.ipynb
│   ├── app.py
│   ├── pca.pkl
│   ├── scaler.pkl
│   └── requirements.txt
│
├── TSNE/
│   ├── notebook.ipynb
│   ├── app.py
│   └── requirements.txt
│
└── README.md
```

---

# Conclusion

This project demonstrates the practical implementation of two important dimensionality reduction techniques.

PCA focuses on preserving maximum variance and is commonly used for feature reduction, while t-SNE focuses on preserving local relationships and is widely used for data visualization.

Together, these techniques help simplify complex high-dimensional datasets, making them easier to analyze, visualize, and understand.
