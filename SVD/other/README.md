# Least Squares via SVD — Explanation & Equations

## Problem Setup

We assume a simple linear relationship between input data **x**, true weight **w**, and output **y**:

[
y = w x + \epsilon
\]

where:

* **x** is the input data (column vector),
* **w** is the true unknown weight (scalar),
* **y** is the observed output (with noise),
* **ε** is random noise.

Our goal is to estimate the weight \( \tilde{w} \) from data.

---

## Matrix Form

We rewrite the system as:

[
Y = X w
\]

Where:

* \( X \) is an \( n \times 1 \) design matrix,
* \( Y \) is an \( n \times 1 \) output vector,
* \( w \) is a scalar parameter.

We solve the least-squares problem:

[
\tilde{w} = \arg\min_w \|Xw - Y\|^2
\]

---

## Classical Analytical Solution

The normal-equation solution is:

[
\tilde{w} = (X^T X)^{-1} X^T Y
\]

However, this only works when \(X^T X\) is invertible.
To obtain a stable and general solution, we use **SVD**.

---

# SVD Decomposition

We compute the singular value decomposition:

[
X = U S V^T
\]

Where:

* **U** is an orthonormal \( n \times 1 \) matrix,
* **S** is a \( 1 \times 1 \) diagonal matrix (one singular value),
* **V** is a \( 1 \times 1 \) orthonormal matrix.

---

# Pseudoinverse of X

The Moore–Penrose pseudoinverse of \(X\) is:

[
X^{+} = V S^{-1} U^T
\]

This always exists as long as \(S \neq 0\), and it provides the exact optimal least-squares estimator.

---

# Final Analytical Solution for the Weight

We compute:

[
\tilde{w} = X^{+} Y = V S^{-1} U^T Y
\]

This matches the code:

```python
wtilde = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ y
```

With noise, \( \tilde{w} \approx 3 \).
Without noise, \( \tilde{w} = 3 \) exactly.

---

## Visual Interpretation

You typically plot:

1. **True Line** \( y = 3x \)
2. **Noisy observations**
3. **Regression line** using \( \tilde{w} \)

The regression line should closely match the true line.

---

## Equivalent Alternative Solution

NumPy's pseudoinverse uses the same SVD logic internally:

```python
wtilde2 = np.linalg.pinv(x) @ y
```

Thus:

[
\tilde{w} = \tilde{w}_1 = \tilde{w}_2
\]

---

## Interpretation in Machine Learning Terms

* **x** = input feature
* **w** = model weight
* **y** = output
* **Learning** = estimating the weight \(w\) that minimizes prediction error

This is the simplest supervised learning model, solved **analytically** rather than with gradient descent.

---

This text is suitable for `README.md` and will render properly on GitHub or Jupyter Markdown.
