# Least Squares via SVD 

This document is formatted for perfect rendering in **GitHub**, **Jupyter Notebook**, and **VS Code**. All equations use standard LaTeX math blocks.

---

## 📘 Problem Setup

We assume a linear relationship between an input feature **x**, a true weight **w**, and an output **y**:

$$
y = w x + \epsilon
$$

where:

* **x** is the input data (column vector)
* **w** is the unknown scalar parameter
* **y** is the observed output
* **ε** is noise

Our goal is to estimate the weight 

$$ \tilde{w} $$

from data.

---

## 📐 Matrix Form

We rewrite the system as:

$$
Y = X w
$$

Where:

$$ X \in \mathbb{R}^{n \times 1} $$

$$  Y \in \mathbb{R}^{n \times 1}  $$

(w) is a scalar

The least-squares estimate solves:

$$
\tilde{w} = \arg\min_w |Xw - Y|^2
$$

---

## 🧮 Classical Normal-Equation Solution

The analytical formula is:

$$
\tilde{w} = (X^T X)^{-1} X^T Y
$$

However, this formula fails if 

$$ (X^T X) $$

is singular or ill-conditioned.

---

# 🔢 SVD Decomposition

We compute the singular value decomposition of the input matrix:

$$
X = U S V^T
$$

Where:

$$ U \in \mathbb{R}^{n \times 1} $$

$$ S = [\sigma] $$ 

$$ V \in \mathbb{R}^{1 \times 1} $$

This is a **rank‑1** matrix, making the math extremely clean.

---

# 🟦 Moore–Penrose Pseudoinverse

The pseudoinverse of (X) is:

$$
X^{+} = V S^{-1} U^T
$$

This always exists as long as 

$$ \sigma \neq 0 $$

---

# 🟩 Final Least-Squares Estimate

The optimal estimate of the weight is:

$$
\tilde{w} = X^{+} Y
$$

Substituting the SVD form:

$$
\tilde{w} = V S^{-1} U^T Y
$$

This exactly matches your code:

```python
wtilde = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ y
```

If noise is small:

$$
\tilde{w} \approx 3
$$

If noise = 0:

$$
\tilde{w} = 3
$$

---

## 🔁 Equivalent NumPy Solution

NumPy's pseudoinverse implementation uses the same SVD math internally:

```python
wtilde2 = np.linalg.pinv(x) @ y
```

Therefore:

$$
\tilde{w} = \tilde{w}_1 = \tilde{w}_2
$$

---

## 🤖 Interpretation in Machine Learning Terms

This is the simplest possible supervised learning model:

$$
y = w x
$$

Training means recovering the parameter:

$$
\tilde{w} = \arg\min_w |Xw - Y|^2
$$

Here, you solved it **analytically**, not with gradient descent.

---

## 📊 Visualization Summary

When you plot your results, you show:

1. The **true line**: (y = 3x)
2. The **noisy sampled data**
3. The **regression line** using
   $$ \tilde{w} $$

The regression line should be close to the true line.

---

