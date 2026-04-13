## XOR

* https://github.com/rcalix1/MachineLearningFoundations/blob/main/NeuralNets/PyTorch/FoundationsOfNeuralNetworks_gradients_XOR.ipynb
* 

## Derivative

## Squared Error Gradient (Simple)

### Start

$$
L = (wx + b - y)^2
$$

---

### Rule

$$
\frac{d}{dw} \left( (\text{something})^2 \right)
= 2(\text{something}) \cdot \frac{d}{dw}(\text{something})
$$

---

### Apply to $w$

$$
\frac{dL}{dw}
= 2(wx + b - y) \cdot \frac{d}{dw}(wx + b - y)
$$

---

### Inner derivative

$$
\frac{d}{dw}(wx + b - y) = x
$$

---

### Final

$$
\frac{dL}{dw} = 2(wx + b - y)x
$$

---

### Simple idea

$$
\text{error} = (wx + b - y)
$$

$$
\text{gradient} = 2 \cdot \text{error} \cdot x
$$







## Derivative of the Perceptron



### Start

$$
\hat{y} = \sigma(wx + b)
$$

$$
L = \frac{1}{2}(y - \hat{y})^2
$$

---

### Rule

$$
\frac{d}{dw} \left( (\text{something})^2 \right)
= 2(\text{something}) \cdot \frac{d}{dw}(\text{something})
$$

---

### Apply to $w$

$$
\frac{dL}{dw}
= ( \hat{y} - y ) \cdot \frac{d\hat{y}}{dw}
$$

(note: 1/2 cancels the 2)

---

### Now the key step

$$
\frac{d\hat{y}}{dw}
= \sigma(wx + b)(1 - \sigma(wx + b)) \cdot x
$$

---

### Full derivative

$$
\frac{dL}{dw}
= (\hat{y} - y)\,\hat{y}(1 - \hat{y})\,x
$$

---

## 🔥 BUT — here is the important simplification

If we use **cross-entropy loss**, the sigmoid term cancels:

$$
\frac{dL}{dw} = (\hat{y} - y)x
$$

$$
\frac{dL}{db} = (\hat{y} - y)
$$

---

## Final Update Rule

Gradient descent:

$$
w := w - \eta \frac{dL}{dw}
$$

$$
b := b - \eta \frac{dL}{db}
$$

---

## Result

$$
w := w - \eta(\hat{y} - y)x
$$

$$
b := b - \eta(\hat{y} - y)
$$

---

## Same as your form

$$
-\eta(y - \hat{y})x
\quad = \quad
-\eta(\hat{y} - y)x \;\;(\text{just sign flip})
$$

---

## 🔑 Simple idea

$$
\text{gradient} = (\hat{y} - y)\,x
$$

$$
\text{update} = -\eta \cdot \text{gradient}
$$
