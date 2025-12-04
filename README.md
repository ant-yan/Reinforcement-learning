# 📘 Reinforcement Learning — Concepts, Algorithms & Experiments

This project is a collection of reinforcement learning (RL) experiments designed to help understand how learning agents behave, why certain algorithms succeed, and where some classical methods fail.  
The goal is to provide **clear explanations, readable code, and practical visualizations** that make RL concepts easy to explore.

Reinforcement learning is a field of machine learning where an agent interacts with an environment, receives feedback (rewards), and gradually learns to make better decisions. These experiments demonstrate both the **strengths and weaknesses** of popular RL methods using small, controlled environments.

---

# 🎯 Purpose of This Project

The main purpose of this project is to:

- Build intuition for how reinforcement learning works  
- Show real examples of **policy evaluation, function approximation, and TD learning**  
- Demonstrate classical problems such as **divergence in off-policy TD**  
- Implement modern, stable algorithms like **TDC (GTD)** and **Emphatic TD**  
- Provide high-quality visualizations for studying algorithm behavior  
- Serve as an educational resource for students, researchers, or developers learning RL  

Every experiment is focused on helping you *understand the behavior* of RL algorithms — not just the final result.

---

# 🧠 Key Reinforcement Learning Concepts Explained

## **1. Temporal-Difference Learning (TD)**

Temporal-Difference learning updates value estimates after each step using:

- the current value estimate  
- observed reward  
- predicted next value  

You’ll explore:

- TD(0)  
- On-policy vs off-policy TD  
- Function approximation using feature vectors  
- Divergence issues in off-policy settings (e.g., Baird’s counterexample)

These experiments show why TD can become unstable when the behavior policy differs from the target policy.

---

## **2. Dynamic Programming (DP)**

Dynamic Programming is used when the environment model is known.  
This project includes conceptual demonstrations of:

- Policy evaluation  
- Policy iteration  
- Value iteration  

DP provides a solid foundation for understanding RL prediction and control.

---

## **3. Gradient-TD Methods (Stable Off-Policy Learning)**

Classical TD algorithms can diverge in off-policy settings.  
Gradient-TD algorithms correct this by following the **true gradient** of the value-error objective.

Included methods:

- **TDC (GTD(0))** — stable off-policy TD  
- Secondary weight vectors  
- Two-time-scale learning  
- Error metrics such as RMSVE & RMSPBE  

These experiments highlight how TDC maintains stability even in environments where TD(0) fails.

---

## **4. Emphatic Temporal-Difference Learning (ETD)**

Emphatic TD introduces **emphasis weighting**, a mechanism that adjusts learning updates to maintain stability in off-policy scenarios.

Key ideas:

- Follow-on traces  
- Emphasis value  
- Expected ETD updates  
- Variance reduction  
- Stability under off-policy training  

These experiments show how ETD overcomes classical instability problems.

---

## **5. Feature Representations**

RL algorithms rely on feature vectors when approximating value functions.  
This project showcases:

- Coarse coding  
- Tile-like binary features  
- Linear feature representations  
- How feature design affects learning stability  

Understanding representation is essential for building scalable RL systems.

---

# 📊 Visualizations

Each experiment includes plots illustrating:

- RMS Value Error (RMSVE)  
- Projected Bellman Error (RMSPBE)  
- Weight vector changes  
- Emphasis term evolution  
- Comparison of algorithm performance over time  

These visualizations make it easier to see how algorithms learn and why they behave differently.

---

# 🚀 How to Run the Experiments

Install dependencies:

```bash
pip install numpy matplotlib
```

Run notebooks using:

- Jupyter Notebook  
- VS Code  
- PyCharm  
- Google Colab  

Each notebook is self-contained and automatically generates its plots.

---

# 📖 What You Will Learn

By exploring this project, you will understand:

- Why off-policy TD learning sometimes diverges  
- How TDC and ETD stabilize learning  
- How importance sampling affects TD updates  
- The role of discount factors, weights, and feature vectors  
- Expected vs sampled updates  
- How RL behaves in controlled MDP environments  

These concepts form the foundation for deeper RL topics such as Q-learning, Actor–Critic methods, and Policy Gradients.

---

# 🧩 Summary

This project provides a clear and practical deep dive into reinforcement learning fundamentals.  
Through small environments, visual learning curves, and clean implementations, you can:

- Experiment with RL algorithms  
- Understand divergence and stability  
- Explore modern solutions to classical RL problems  
- Build intuition that scales to advanced RL techniques  

Whether you're a student, researcher, or developer, this project aims to be a solid starting point in your RL journey.
