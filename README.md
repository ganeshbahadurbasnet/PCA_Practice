# Wine Dataset: PCA + KNN Model

This project applies **Principal Component Analysis (PCA)** to the Wine dataset to reduce dimensionality and visualize data in 3D, followed by using a **K-Nearest Neighbors (KNN)** classifier to assess model performance.

---

## 📊 Dataset

The dataset includes the following numerical features:
- fixed acidity  
- volatile acidity  
- citric acid  
- residual sugar  
- chlorides  
- free sulfur dioxide  
- total sulfur dioxide  
- density  
- pH  
- sulphates  
- alcohol  
- quality

✅ No missing values detected.

---

## 🔍 Approach

- Applied **PCA** with `n_components=None` to extract all eigenvalues and understand variance explained by each component.
- Used cumulative variance (`np.cumsum`) to determine how many components retain ~90% of the dataset’s variance.
- Reduced data to **3D principal components** for visualization with:
  - **Matplotlib**
  - **Plotly**
- Trained a **KNN** model on the PCA-transformed data.
- Achieved ~56% accuracy on the Wine dataset after dimensionality reduction.

---

## 💡 Key Insights

- **PCA reduces dimensionality** and enables meaningful 3D visualizations of high-dimensional data.
- Applying PCA did **not improve model accuracy** significantly — the KNN model achieved similar performance (~56%) as with the original features.
- PCA effectively compresses data while retaining key patterns, making it useful for:
  - Visualization
  - Reducing computational cost
  - Handling multicollinearity

---

## ⚙️ Main Code Snippet

```python
from sklearn.decomposition import PCA

# Fit PCA to extract all eigenvalues
pca_Cmp = PCA(n_components=None)
x_train_cmp = pca_Cmp.fit_transform(x_train_scaled)

# Explained variance ratios
cmp = pca_Cmp.explained_variance_ratio_
np.cumsum(cmp)
