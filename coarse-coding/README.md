# Understanding Coarse Coding and Feature Width

## 🌟 Overview

This project explores how the **width of receptive fields**—the overlapping regions used in coarse coding—affects the learning process of a **linear function approximator**.  
By adjusting the size of these receptive fields, we can study how **generalization** and **convergence speed** vary when learning a simple 1‑D function.

---

## 🎯 Objective

The task is to approximate a **square‑wave function** on the interval [0, 2).  
We use a **linear approximator** whose features are overlapping intervals that respond when a given input value falls inside them.

### Target Function

\[
U(x) =
\begin{cases}
1, & 0.5 < x < 1.5 \\
0, & \text{otherwise}
\end{cases}
\]

- Domain: \([0, 2)\)
- Number of features: about **50**, kept constant
- Feature widths tested: **narrow**, **medium**, **broad**
- Samples \((x, U(x))\) drawn uniformly at random
- Step size scaled as \(\alpha = \frac{0.2}{n}\), where *n* is the number of active features

---

## 🧩 Representation and Learning

Each feature corresponds to an **interval** [l, r).  
An input *x* activates every feature whose interval contains it.

### Value Estimate
\[
\hat{v}(x) = \sum_{i \in \text{active}(x)} w_i
\]

### Update Rule
For each sample \((x, y)\):
\[
\delta = y - \hat{v}(x)
\]
\[
w_i \leftarrow w_i + \frac{\alpha}{n}\,\delta
\]
This update allows overlapping features to share information—producing smoother and faster generalization.

---

## 🧪 Experimental Setup

We repeat training for three receptive‑field widths:

| Setting | Overlap | Description |
|----------|----------|-------------|
| **Narrow** | Low | Highly local learning, slower convergence |
| **Medium** | Moderate | Balanced overlap and learning speed |
| **Broad** | High | Strong generalization, smoother estimates |

Each configuration uses identical training samples and updates.

---

## 📊 Results

![Result Figure](generated_images/figure_9_8.png)

- **Broad** features generalize early, giving smooth approximations from few samples.  
- **Narrow** features are slower and initially produce uneven curves.  
- **Medium** features achieve a compromise between both extremes.  
- Ultimately, all settings learn the correct square‑wave shape with enough data.

### Key Insight
Feature width controls **how quickly** generalization happens, while **final accuracy** remains similar once the model has seen sufficient examples.

---

## 📘 Summary

| Feature Width | Early Learning | Generalization | Final Accuracy |
|----------------|----------------|----------------|----------------|
| Narrow | Slow, uneven | Weak | High |
| Medium | Balanced | Moderate | High |
| Broad | Smooth, fast | Strong | High |

---

## ⚙️ Reproduction

To recreate the experiment and figure:
1. Run the provided Colab notebook or Python script.  
2. Generated figures are automatically saved to:
   ```
   generated_images/figure_9_8.png
   ```
3. Upload the image to your GitHub `generated_images` folder for visualization.

---
