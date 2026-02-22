## SVD for recommender systems

```

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load Data
data_ratings = pd.read_csv('/content/drive/MyDrive/ratings.csv')
data_movies = pd.read_csv('/content/drive/MyDrive/movies.csv')

# 2. Setup ID Mappings
movies_uniques = np.unique(data_ratings.movieId.values)
users_uniques = np.unique(data_ratings.userId.values)

movies_dict_real_ref = {res: idx for idx, res in enumerate(movies_uniques)}
movies_dict_ref_real = {idx: res for idx, res in enumerate(movies_uniques)}
users_dict_real_ref = {res: idx for idx, res in enumerate(users_uniques)}

# 3. Create Pivot Matrix (Float32 for decimals/negative numbers)
# Rows = Movies, Cols = Users
ratings_mat = np.zeros(shape=(len(movies_uniques), len(users_uniques)), dtype=np.float32)

for row in data_ratings.itertuples():
    ref_u = users_dict_real_ref[row.userId]
    ref_m = movies_dict_real_ref[row.movieId]
    ratings_mat[ref_m, ref_u] = row.rating

# 4. MEAN CENTERING (Fixes the "0" rating issue)
# Subtract the average rating of each movie from its row.
# We only average non-zero ratings to avoid dragging the mean down by unrated movies.
movie_means = np.array([np.mean(row[row > 0]) if any(row > 0) else 0 for row in ratings_mat])
ratings_mat_centered = ratings_mat - movie_means.reshape(-1, 1)
# Keep unrated movies at 0 (now representing 'average')
ratings_mat_centered[ratings_mat == 0] = 0

# 5. SVD
U, S, V = np.linalg.svd(ratings_mat_centered, full_matrices=False)

# 6. Recommendation Function
def get_recommendations(latent_matrix, movie_real_id, top_n=10):
    ref_id = movies_dict_real_ref[movie_real_id]
    movie_vector = latent_matrix[ref_id, :].reshape(1, -1)
    
    # Calculate true Cosine Similarity
    scores = cosine_similarity(movie_vector, latent_matrix).flatten()
    
    # Sort DESCENDING (Highest score first)
    top_indices = np.argsort(scores)[::-1]
    
    # Skip index 0 (the search movie itself)
    return top_indices[1:top_n+1]

# 7. Run Test (Example: Inception ID 79132)
k = 50  # Latent factors (50-100 is ideal for MovieLens)
search_movie_id = 3948. ##79132
reduced_U = U[:, :k]

top_indices = get_recommendations(reduced_U, search_movie_id, top_n=10)

search_title = data_movies[data_movies.movieId == search_movie_id].title.values[0]
print(f"Movies similar to: {search_title}\n" + "-"*30)

for idx in top_indices:
    real_id = movies_dict_ref_real[idx]
    title = data_movies[data_movies.movieId == real_id].title.values[0]
    print(f"ID {real_id}: {title}")




```




# Tiny SVD Example — README.md Format

This README shows **two contrasting ways to compute singular values**:

1. **Direct algorithmic method** (using eigenvalues of \(X^T X\))  
2. **NumPy's built‑in SVD** (which includes the automatic sorting step)

The contrast helps make it clear **where singular values come from** and **why SVD implementations sort them in descending order**.

---

## 📌 Matrix Used in This Example
```python
import numpy as np

X = np.array([[3.0, 1.0],
              [2.0, 2.0]])
print("Matrix X:", X)
```

---

# 1. Algorithmic SVD (Manual Computation)

This section computes singular values via the mathematical pipeline:

\[
X^T X 
ightarrow 	ext{eigenvalues} 
ightarrow 	ext{singular values}
\]

### **Step A — Compute \(X^T X\)**
```python
XtX = X.T @ X
print("X^T X:", XtX)
```

### **Step B — Compute eigenvalues of \(X^T X\)**
Eigenvalues may come out **unordered** depending on the algorithm.
```python
vals, vecs = np.linalg.eig(XtX)
print("Eigenvalues (unordered):", vals)
print("Eigenvectors:", vecs)
```

### **Step C — Singular values are the square roots of eigenvalues**
```python
singular_values_algo = np.sqrt(np.abs(vals))
print("Singular values (unordered):", singular_values_algo)
```

### **Step D — Sort singular values in descending order**
Every real SVD implementation performs this step.
```python
idx = np.argsort(-singular_values_algo)
singular_values_algo_sorted = singular_values_algo[idx]
print("Singular values (sorted descending):", singular_values_algo_sorted)
```

---

# 2. NumPy SVD (LAPACK Implementation)
NumPy computes the SVD using a stable algorithm (bidiagonalization + QR/divide‑and‑conquer) and **always sorts singular values**.

```python
U, S, Vt = np.linalg.svd(X, full_matrices=False)
print("NumPy SVD singular values (always sorted):", S)
print("U matrix:", U)
print("V^T matrix:", Vt)
```

---

# 3. Comparison
```python
print("Comparison (algorithm vs NumPy):")
print("Manual (sorted):", singular_values_algo_sorted)
print("NumPy:          ", S)
```

---

# ✔️ Summary
- Singular values come from **eigenvalues of \(X^T X\)**.  
- The eigenvalue solver **does not guarantee ordering**.  
- SVD libraries (LAPACK, NumPy, MATLAB, PyTorch) explicitly **sort the singular values** and reorder \(U\) and \(V\) accordingly.  


