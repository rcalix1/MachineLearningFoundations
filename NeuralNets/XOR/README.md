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



## Widrow–Hoff (1959) Perceptron / Delta Rule

### Start

$$
\hat{y} = w x + b
$$

---

### Loss (linear error, no square written explicitly in form)

$$
L = (y - \hat{y})^2
$$

---

### Rule

$$
\frac{d}{dw} ( \text{something} )^2
= 2(\text{something}) \cdot \frac{d}{dw}(\text{something})
$$

---

### Apply to $w$

$$
\frac{dL}{dw}
= 2(y - \hat{y}) \cdot \frac{d}{dw}(y - \hat{y})
$$

---

### Inner derivative

$$
\frac{d}{dw}(y - \hat{y}) = -\frac{d}{dw}(w x + b) = -x
$$

---

### Combine

$$
\frac{dL}{dw}
= 2(y - \hat{y})(-x)
$$

$$
= -2(y - \hat{y})x
$$

---

### Same form (clean)

$$
\frac{dL}{dw} = 2(\hat{y} - y)x
$$

---

## Gradient Descent Update

$$
w := w - \eta \frac{dL}{dw}
$$

---

### Substitute

$$
w := w - \eta \cdot 2(\hat{y} - y)x
$$

---

### Absorb constant into learning rate

Let:

$$
\eta' = 2\eta
$$

---

### Final (Widrow–Hoff rule)

$$
\boxed{
w := w - \eta ( \hat{y} - y ) x
}
$$

$$
\boxed{
b := b - \eta ( \hat{y} - y )
}
$$

---

## 🔑 Simple idea

$$
\text{update} = -\eta \cdot (\hat{y} - y) \cdot x
$$

---

## 🧠 One sentence

> Move weights in proportion to the error and the input.

