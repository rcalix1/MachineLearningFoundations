## Naive Bayes


# Naive Bayes: Multinomial vs Gaussian

* link

  
## Understanding Why Gaussian NB Replaces Word Counts

This expands the intuition behind why Gaussian NB is used for continuous features.

### Case 1: Text Classification (Multinomial NB)

With word counts, we estimate probabilities using frequencies:

P(w_i | c) = count(w_i in class c) / total words in class c

Simple counting works because features are discrete.

### Case 2: Continuous Features

Example feature vector:
[5.4, 2.1, 1.3, 0.2]

You cannot count how often 5.4 appears—continuous values almost never repeat.
So instead of frequencies, you must use a probability density model.

Naive Bayes still assumes:


$$
P(x \mid c) = \prod_{j=1}^{d} P(x_j \mid c)
$$


But each 



$$
P(x_j \mid c)
$$

must now handle real numbers.

### Gaussian = Continuous Equivalent of Counting

We assume:


$$
x_j \mid c \sim \mathcal{N}(\mu_{c,j},\, \sigma_{c,j}^2)
$$


So instead of counting, we compute mean & variance and evaluate:





$$
P(x_j \mid c)=
\frac{1}{\sqrt{2\pi\sigma_{c,j}^2}}
\exp\left(
-\frac{(x_j - \mu_{c,j})^2}{2\sigma_{c,j}^2}
\right)
$$



Same purpose as word likelihoods, just adapted for continuous data.

### Key Insight

YES — Gaussian NB is the continuous analogue of Multinomial NB.

* Words → counts → categorical probabilities
* Continuous → mean/variance → Gaussian densities

Both compute:
P(c) * Π_j P(x_j | c)




$$
P(c) \times \prod_{j} P(x_j \mid c)
$$



Only the form of  

$P(x_j \mid c)$

changes.


### Final Translation Table

| Discrete Words      | Continuos features       |       |                  |
| ----------------------------- | ---------------------------- | ----- | ---------------- |
| Count word frequencies        | Compute mean & variance      |       |                  |
| Categorical probability table | Gaussian probability density |       |                  |
| Multinomial Naive Bayes       | Gaussian Naive Bayes         |       |                  |
| P(w_i / c) from counts        |  P(x_j / c) from Gaussian    |

They are structurally identical; only the likelihood model changes.


---

# 1. Multinomial Naive Bayes (Text / Word Counts)

Multinomial NB is used when features are **counts** (e.g., word counts).

### Key idea

* Estimate **P(word | class)** from word frequencies.
* Multiply all word likelihoods with the class prior.

### Minimal Example (Toy)

```python
from collections import Counter, defaultdict
import numpy as np

def train_multinomial(docs, labels):
    classes = set(labels)
    priors = {}
    word_counts = {c: Counter() for c in classes}
    total_words = {c: 0 for c in classes}

    for doc, c in zip(docs, labels):
        priors[c] = priors.get(c, 0) + 1
        for w in doc:
            word_counts[c][w] += 1
            total_words[c] += 1

    total_docs = len(docs)
    for c in priors:
        priors[c] /= total_docs

    return classes, priors, word_counts, total_words


def predict_multinomial(doc, classes, priors, word_counts, total_words):
    scores = {}
    for c in classes:
        log_prob = np.log(priors[c])
        for w in doc:
            # Laplace smoothing
            count_w = word_counts[c][w] + 1
            denom = total_words[c] + len(word_counts[c])
            log_prob += np.log(count_w / denom)
        scores[c] = log_prob
    return max(scores, key=scores.get)

# Example usage
X = [["cat", "sat", "mat"], ["dog", "ran"], ["cat", "ran"]]
y = [0, 1, 0]
classes, priors, wc, totals = train_multinomial(X, y)
print(predict_multinomial(["cat"], classes, priors, wc, totals))
```

---

# 2. Gaussian Naive Bayes (Continuous Features)

Used when features are **real-valued numbers**, not counts.

### Key idea

* Each feature is modeled with a **Gaussian distribution per class**.
* Compute:

  * mean per class per feature
  * variance per class per feature
  * prior probability of each class
* Evaluate Gaussian PDF at test time.

### Minimal Example (Toy)

```python
import numpy as np

# Gaussian PDF
def gaussian_pdf(x, mean, var):
    return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(- (x - mean)**2 / (2 * var))


def train_gaussian(X, y):
    classes = np.unique(y)
    n_classes = len(classes)
    n_features = X.shape[1]

    means = np.zeros((n_classes, n_features))
    vars_ = np.zeros((n_classes, n_features))
    priors = np.zeros(n_classes)

    for idx, c in enumerate(classes):
        Xc = X[y == c]
        means[idx] = Xc.mean(axis=0)
        vars_[idx] = Xc.var(axis=0)
        priors[idx] = len(Xc) / len(X)

    return classes, means, vars_, priors


def predict_gaussian(x, classes, means, vars_, priors):
    posteriors = []
    for idx, c in enumerate(classes):
        prior = np.log(priors[idx])
        likelihood = np.sum(np.log(gaussian_pdf(x, means[idx], vars_[idx])))
        posteriors.append(prior + likelihood)
    return classes[np.argmax(posteriors)]

# Example usage
X = np.array([[1.0, 2.0], [1.1, 1.9], [5.0, 6.0], [5.2, 5.9]])
y = np.array([0, 0, 1, 1])
classes, means, vars_, priors = train_gaussian(X, y)
print(predict_gaussian(np.array([1.05, 2.1]), classes, means, vars_, priors))
```

---

# 3. Core Insight: Why They Look Different

| Text NB (Multinomial)                | Continuous NB (Gaussian)           |
| ------------------------------------ | ---------------------------------- |
| Uses **counts**                      | Uses **real numbers**              |
| Likelihood from **frequency tables** | Likelihood from **Gaussian PDF**   |
| (P(w \mid c)) from counting words    | (P(x \mid c)) from mean & variance |
| Works on bag-of-words                | Works on numeric features          |

Both share the **same structure**:

[
P(c \mid x) \propto P(c) \prod_j P(x_j \mid c)
]

Only the **form of the likelihood** changes.

---

# 4. Summary

* If features are **counts** → use Multinomial NB.
* If features are **continuous real values** → use Gaussian NB.
* Both rely on the Naive Bayes independence assumption.
* Both compute **log posteriors** to avoid numerical underflow.

---

Let me know if you want:

* a hybrid NB example (categorical + continuous)
* comparisons with scikit-learn
* or a visual diagram of the two models.

