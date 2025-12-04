# 🚗 Mountain Car — Semi-Gradient n-Step SARSA (Tile Coding)

This project implements the **semi-gradient n-step SARSA** algorithm with **tile-coding function approximation** to solve the classic **Mountain Car** reinforcement learning task.

The Mountain Car problem highlights an important challenge in RL:  
**sometimes achieving long-term goals requires temporarily moving away from them.**

This implementation recreates the experiments and visualizations from **Figures 10.1–10.4** in Sutton & Barto’s *Reinforcement Learning: An Introduction*.

---

# 🎯 Project Summary

The aim of this project is to show how an agent can learn to solve the Mountain Car task—a continuous control environment where the agent must build momentum by moving *away from the goal* before successfully driving up a steep hill.

This project demonstrates:

- How **tile coding** can represent continuous state spaces.
- How **n-step SARSA** improves learning efficiency.
- How **step size (α)** and **bootstrapping depth (n)** affect performance.
- How the learned **cost-to-go function** evolves over training.

---

# 🏞️ Mountain Car Environment

The agent controls a car moving along a valley between two hills.  
At each step it chooses one of three actions:

1. throttle left (−1)  
2. no throttle (0)  
3. throttle right (+1)

## State Dynamics

Velocity update:

\[
\dot{x}_{t+1} = \text{bound}[\dot{x}_t + 0.001 A_t - 0.0025\cos(3x_t)]
\]

Position update:

\[
x_{t+1} = \text{bound}[x_t + \dot{x}_{t+1}]
\]

Constraints:

\[
-1.2 \le x_t \le 0.5, \qquad -0.07 \le \dot{x}_t \le 0.07
\]

The episode terminates when the car reaches the goal state:

\[
x = 0.5
\]

Reward at every step:

\[
R_t = -1
\]

Initial conditions:

\[
x_0 \in [-0.6, -0.4),\quad \dot{x}_0 = 0
\]

---

# 🔢 Function Approximation via Tile Coding

To handle the continuous state space, tile coding is used with:

- **8** overlapping tilings  
- asymmetric offsets  
- binary feature vectors representing active tiles  

The action-value function is approximated using:

\[
\hat{q}(s, a; \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s, a)
\]

where only a few entries of the feature vector are active (1’s), making the representation sparse and efficient.

---

# 🤖 Learning Algorithm — n-Step SARSA

The algorithm estimates \( q_\pi(s,a) \) using multi-step bootstrapping.  
For a given time \(t\), the n-step return is:

\[
G_{t:t+n} = R_{t+1} + \dots + R_{t+n} + \hat{q}(S_{t+n}, A_{t+n})
\]

Parameter update:

\[
\mathbf{w} \leftarrow 
\mathbf{w} + \alpha\,[G_{t:t+n} - \hat{q}(S_t,A_t)]\,\nabla_{\mathbf{w}} \hat{q}(S_t,A_t)
\]

### Exploration Strategy

No ε-greedy randomness is used:

- **ε = 0**
- exploration is driven entirely by **optimistic initialization**

This simplifies the algorithm and matches the original textbook experiments.

---

# 🧪 Experiments Included

The following experiments reproduce Figures 10.1–10.4 from the Sutton & Barto textbook:

| Figure | Description |
|-------|-------------|
| **10.1** | Evolution of the cost-to-go function at episodes 1, 100, and 9000 |
| **10.2** | Learning curves for step sizes \( \alpha \in \{0.1, 0.2, 0.5\} \) |
| **10.3** | Comparison of 1-step vs 8-step SARSA |
| **10.4** | Combined effect of α and n on early learning performance |

All experiments average results over multiple runs for statistical reliability.

---

# 📊 Summary of Results

### **Figure 10.1 — Cost-to-Go Evolution**
The agent initially oscillates inside the valley; over time the learned value function sharpens, guiding the agent efficiently toward the goal.

### **Figure 10.2 — Step Size Comparison**
Medium step sizes lead to stable and fast learning. Large α values cause divergence, confirming the sensitivity of linear function approximation.

### **Figure 10.3 — Multi-step vs 1-step SARSA**
Using \( n = 8 \) significantly speeds up learning and improves asymptotic performance.

### **Figure 10.4 — Interaction of α and n**
Performance peaks with moderate values of both α and n.  
Too much bootstrapping (large n) or too large α destabilize the update.

---

# 🔍 Key Takeaways

- Tile coding provides an effective, lightweight representation for continuous RL tasks.
- Multi-step bootstrapping (n-step SARSA) dramatically improves sample efficiency.
- Optimistic initialization alone can drive exploration without ε-greedy.
- Hyperparameter tuning (especially α and n) is essential in function approximation settings.
- Mountain Car clearly demonstrates the need for long-term planning to succeed in control tasks.

---

# ✅ Conclusion

This project successfully implements and analyzes **semi-gradient n-step SARSA** with **tile-coding** on the Mountain Car environment.  
The experiments match textbook results and provide insights into how learning dynamics change with different step sizes and planning horizons.
