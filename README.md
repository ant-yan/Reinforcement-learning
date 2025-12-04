# 📘 Reinforcement Learning – Off-Policy Learning Experiments  
*(Baird’s Example, TDC, and Emphatic TD)*

This project explores several off-policy value-learning methods using a small, intentionally tricky Markov Decision Process.  
The goal is to understand why some reinforcement learning algorithms behave unpredictably when the behavior policy differs from the target policy — and how more advanced methods overcome those issues.

The project contains three standalone experiments:

1. **A simple off-policy TD(0) setup** that often becomes unstable  
2. **A Gradient-TD method (TDC)** that is designed to remain stable  
3. **An Expected Emphatic-TD method**, which handles off-policy correction through emphasis weighting  

Everything is implemented with linear function approximation on a seven-state environment.

---

## 📂 Project Structure

```
/counter-examples
│
├── bairds_counterexample_rewritten.ipynb     # Off-policy TD(0) experiment
├── tdc_baird_rewritten.ipynb                 # TDC / GTD(0) experiment
├── emphatic_baird_rewritten.ipynb            # Expected Emphatic TD experiment
│
└── generated_images/                         # All plots created by the notebooks
```

Each notebook can be executed independently and contains its own code, explanation, and plots.

---

## 🧩 Environment Overview

All experiments use the same small MDP:

- **7 states**  
  - 6 “upper” states  
  - 1 “lower” state  

- **2 actions**
  - *dashed* → jumps to one of the upper states  
  - *solid* → always moves to the lower state  

- **Behavior policy**  
  Chooses *solid* only occasionally.  

- **Target policy**  
  Always chooses *solid*.  

- **Rewards**  
  Every transition gives reward **0** — so stability, not reward, is the focus.

- **Value function**  
  Approximated with an **8-dimensional linear feature vector** for each state.

This environment is known for exposing weaknesses in certain RL algorithms when used off-policy.

---

## 1️⃣ Off-Policy TD(0)

The first notebook runs a basic off-policy TD update.  
This method uses importance sampling but relies on *semi-gradient updates*, which can behave poorly in this environment.

### What this notebook covers:

- Step-by-step TD(0) update  
- Importance sampling ratios  
- RMS Value Error (RMSVE)  
- RMS Projected Bellman Error (RMSPBE)  
- Plots showing how errors evolve through time  

---

## 2️⃣ Gradient-TD: TDC / GTD(0)

The second notebook implements a more stable algorithm known as **TDC** (also called **GTD(0)**).

TDC introduces a secondary weight vector that corrects the gradient direction, allowing the algorithm to converge even in difficult off-policy setups.

### This notebook includes:

- TDC’s two-time-scale update  
- Tracking error measures over time  
- A comparison of how TDC behaves vs. the simpler TD update  
- Smooth and stable learning curves  

---

## 3️⃣ Expected Emphatic TD

The third notebook focuses on **Emphatic TD**, but run in *expectation* instead of sampling.  
The emphasis term is designed to stabilize learning by controlling how updates depend on importance sampling.

### Contents:

- Implementation of the expected Emphatic-TD update  
- Tracking the “emphasis” value over time  
- RMS value error and projected error curves  
- Visualization of smooth, variance-free dynamics  

---

## 🖼 Generated Plots

All figures created by the notebooks are saved automatically to:

```
generated_images/
```

Each run regenerates the plots so the repository always reflects your most recent experiments.

---

## 🎯 Summary of Findings

Across the three experiments, we can see:

- Standard TD(0) struggles in off-policy settings  
- Gradient-TD methods (like TDC) provide stable updates  
- Emphatic TD introduces a unique weighting mechanism that steers learning toward the correct direction  
- Even extremely small MDPs highlight why off-policy learning is challenging  

This project demonstrates how different algorithms react to the same environment and why modern RL relies on more robust methods when learning off-policy.

---

## 🚀 How to Run

1. Install dependencies:

```bash
pip install numpy matplotlib
```

2. Open any notebook in VS Code or Jupyter  
3. Run all cells  
4. Plots will be saved automatically to `generated_images/`

---

## 📖 Reference

This project is inspired by well-known reinforcement learning research problems, but all notebooks, code, explanations, and plots have been fully rewritten and adapted for this repository.
