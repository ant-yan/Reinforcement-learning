## 1. Baird’s Counterexample — Divergence of Semi-Gradient Bootstrapping

**Baird’s Counterexample:**
A classical reinforcement learning example demonstrating that **semi-gradient TD(0)** and **semi-gradient DP** can diverge under **off-policy learning**, even with linear function approximation and a seemingly well-conditioned feature set.

---

## Project Overview

Baird’s counterexample is a 7-state, 2-action episodic MDP specifically constructed to expose instability in off-policy TD learning.
The example shows that even when:

* rewards are zero,
* features are linearly independent,
* the value function is exactly representable, and
* the target policy is deterministic,

the combination of *off-policy bootstrapping* and *function approximation* can cause the weights to **diverge to infinity**, regardless of step-size.

The project reproduces **Figure 11.2** from Sutton & Barto by implementing:

1. **Semi-gradient Off-policy TD(0)**
2. **Semi-gradient DP (expected update)**

and plotting the evolution of the weight vector.

---

## Problem Setup

The MDP consists of **7 states**:

* **States 0–5** → upper states
* **State 6** → lower state

There are **2 actions**:

* **dashed** → transitions to one of the 6 upper states (uniformly)
* **solid** → transitions to the lower state

### Policies

* **Behavior policy**

  $$
  b(\text{solid}) = \frac{1}{7}, \quad b(\text{dashed}) = \frac{6}{7}
  $$
  
  Produces a *uniform next-state distribution*.

* **Target policy**

$$
\pi(\text{solid}) = 1
$$
  
  
  Concentrates the on-policy distribution on the **lower** state.

### Rewards and Discount

* Reward on all transitions:

$$
R(s,a) = 0
$$
  
* Discount rate:

$$
\gamma = 0.99
$$
  

### Feature Representation

Each state is represented by an **8-dimensional** linear feature vector.

For upper states (i = 0, ... , 5):

$$
\mathbf{x}(s=i) = (0,\ldots,2,\ldots,0,\ 1)
$$


For the lower state:

$$
\mathbf{x}(s=6) = (0,\ldots,0,\ 1,\ 2)
$$

The true value function is:

$$
v_\pi(s) = 0
$$

and is exactly representable by:

$$
\mathbf{w} = \mathbf{0}
$$

Even though the features are linearly independent and exact representation is possible, semi-gradient TD diverges.

---

## Learning Algorithm

### Semi-gradient Off-policy TD(0)

Using importance sampling ratio:

$$
\rho =
\begin{cases}
0, & a=\text{dashed} \
\frac{1}{b(\text{solid})} = 7, & a=\text{solid}
\end{cases}
$$

TD error:

$$
\delta = R + \gamma \mathbf{w}^\top \mathbf{x}(S') - \mathbf{w}^\top \mathbf{x}(S)
$$

Update:

$$
\mathbf{w} \leftarrow \mathbf{w} + \alpha \rho , \delta , \mathbf{x}(S)
$$

---

### Semi-gradient DP (Expected Update)

Expected next-state value:

$$
\mathbb{E}_\pi [V(S')] = \gamma \mathbf{w}^\top \mathbf{x}(\text{lower})
$$

Bellman error:

$$
\delta(s) = \gamma \mathbf{w}^\top \mathbf{x}(\text{lower}) - \mathbf{w}^\top \mathbf{x}(s)
$$

Update over all states:

$$
\mathbf{w} \leftarrow \mathbf{w} + \frac{\alpha}{|S|} \sum_{s} \delta(s),\mathbf{x}(s)
$$

Even with **full sweeps**, no randomness, and DP-style expectation updates, the weights diverge.

---

## Experiments

Two experiments were run for **1,000 steps/sweeps**:

| Experiment                         | Description                                                      |
| ---------------------------------- | ---------------------------------------------------------------- |
| **Semi-gradient Off-policy TD(0)** | Update using behavior policy transitions and importance sampling |
| **Semi-gradient DP**               | Update using the full DP expectation sweep                       |

Initial weights are:

$$
\mathbf{w} = (1,1,1,1,1,1,10,1)
$$

The weight vector is recorded at every iteration and plotted component-wise.

---

## Results

**[Figure 11.2:](https://github.com/AlisaSujyan/Reinforcement-Learning/blob/main/counter-examples/generated_images/figure_11_2.png)**
Both experiments show unstable learning:

* All weight components diverge without bound.
* Divergence occurs **for any positive step size**.
* Divergence happens **even when the true value function is representable**.
* Semi-gradient DP diverges despite full-sweep updates and absence of sampling noise.

This reproduces the canonical instability result from the textbook.

---

## Key Insights

* **Off-policy bootstrapping** + **function approximation** is inherently unstable.
* Importance sampling ratios amplify TD updates when the behavior policy differs sharply from the target policy.
* Semi-gradient methods do not minimize any true objective under off-policy sampling.
* The feature vectors form a linearly independent set, yet instability still occurs — demonstrating that linear FA alone is not enough.
* Even DP with exact expectations diverges if updates are not performed using the **on-policy distribution**.

---

## Conclusion

Baird’s counterexample highlights a critical limitation of classic TD learning:

Semi-gradient TD(0) and semi-gradient DP can diverge under off-policy sampling, even in the simplest possible settings.

This example motivates the need for:

* **Gradient-TD methods** (GTD, GTD2, TDC)
* **Emphatic TD**
* **Careful handling of off-policy distributions**

The project provides a faithful reproduction of the original instability plots, offering a clear demonstration of why standard TD cannot be safely used off-policy with linear function approximation.

---

---

## 2. TD(0) with Gradient Correction (TDC / GTD(0)) on Baird’s Counterexample

**Temporal-Difference Learning with Gradient Correction:**
An implementation of the **TDC (also known as GTD(0))** algorithm on **Baird’s counterexample**, including both a sampled run and a deterministic expected version.
The results reproduce **Figure 11.5** from Sutton & Barto.

---

## Project Overview

This project studies the behavior of the **two-time-scale TDC algorithm** on Baird’s off-policy counterexample.
TDC ensures stable value-function learning through a coupled update of:

* a **primary parameter vector** ($$\mathbf{w}$$) for the value function, and
* a **secondary vector** ($$\mathbf{v}$$) solving a correction equation.

Two experiments are performed:

1. A **single sampled run**, showing typical noisy TDC behavior.
2. An **expected update version**, eliminating sampling variance to reveal the theoretical behavior.

TDC successfully drives the **Projected Bellman Error (PBE)** toward zero, but both the VE and the weight components converge **very slowly**, as described in the book.

---

## Problem Setup

The Baird counterexample consists of 7 states with deterministic features and a behavior policy that chooses the “solid” action with low probability.
The target policy always chooses the solid action.

### Objective

Learn the value function:

$$
\hat{v}(s) = \mathbf{w}^\top \mathbf{x}(s)
$$

where the optimal weights are proportional to:

$$
(1,1,1,1,1,1,4,2)^T
$$

TDC aims to minimize the **mean squared projected Bellman error**, measured using:

* RMS-VE
* RMS-PBE

These diagnostics are plotted along with each component of ($$\mathbf{w}$$).

---

## Learning Algorithm

TDC is a **two-time-scale** method:

### Primary update (for ($$\mathbf{w}$$))

$$
  \mathbf{w} \leftarrow \mathbf{w} * \alpha, \rho_t \left( \delta_t \mathbf{x}(S_t) - \gamma, \mathbf{x}(S_{t+1}) , \mathbf{v}^\top \mathbf{x}(S_t) \right)
$$

### Secondary update (for ($$\mathbf{v}$$))

$$
\mathbf{v} \leftarrow \mathbf{v} * \beta, \rho_t \left( \delta_t - \mathbf{v}^\top \mathbf{x}(S_t) \right) \mathbf{x}(S_t)
$$

with:

$$
\delta_t = R_{t+1} + \gamma \mathbf{w}^\top \mathbf{x}(S_{t+1}) * \mathbf{w}^\top \mathbf{x}(S_t)
$$

The step sizes satisfy:

$$
0 < \alpha \ll \beta
$$

to ensure the secondary process converges faster.

### Expected TDC

The second experiment computes:

* the expected TD error
* the expected importance sampling ratio
* the expected update for both ($$\mathbf{w}$$) and ($$\mathbf{v}$$)

This eliminates variance and produces the smooth trajectory shown on the right side of Figure 11.5.

---

## Experiments

### Experimental Setup

| Parameter                           | Value                   |
| ----------------------------------- | ----------------------- |
| Initial ($$\mathbf{w}$$)            | ( (1,1,1,1,1,1,10,1) )  |
| Initial ($$\mathbf{v}$$)            | Zero vector             |
| Steps / sweeps                      | 1000                    |
| ($$\alpha$$) (for ($$\mathbf{w}$$)) | 0.005                   |
| ($$\beta$$) (for ($$\mathbf{v}$$))  | 0.05                    |
| Discount                            | ($$\gamma = 0.99$$)   |
| Behavior policy                     | Solid with prob ( 1/7 ) |
| Target policy                       | Always solid            |

### Two experiments are run:

| Figure           | Description                                                 |
| ---------------- | ----------------------------------------------------------- |
| **11.5 (left)**  | Sampled TDC run: noisy weight trajectories and diagnostics. |
| **11.5 (right)** | Expected TDC: deterministic synchronous update over states. |

Both experiments track:

* Individual weight components ($$w_i$$) 
* ($$\sqrt{\overline{VE}}$$)
* ($$\sqrt{\overline{PBE}}$$)

---

## Results

**[Figure 11.5:](https://github.com/AlisaSujyan/Reinforcement-Learning/blob/main/counter-examples/generated_images/figure_11_5.png)** Temporal-Difference with Gradient Correction on Baird’s Counterexample

The results show:

* The **PBE decreases to 0**, as predicted by theory.
* The weight components do **not** approach zero; instead, they move slowly toward the optimal proportional solution.
* Even after 1000 iterations, the value error stays near 2.
* Expected TDC produces much smoother trajectories, revealing the underlying convergence pattern.
* Convergence is slow because the PBE becomes small early, causing the updates to shrink.

These observations match the discussion in Sutton & Barto and replicate the figure accurately.

---

## Key Insights

* TDC is stable on this difficult off-policy problem, unlike semi-gradient TD.
* The algorithm’s **two-time-scale design** is crucial for its theoretical guarantees.
* The secondary vector ($$\mathbf{v}$$) must learn faster than ($$\mathbf{w}$$).
* Driving PBE to zero does **not** imply fast value-function convergence.
* Expected TDC helps visualize theoretical behavior that is otherwise obscured by sampling noise.

---

## Conclusion

The TDC algorithm successfully reduces the projected Bellman error on Baird’s counterexample, demonstrating the stability properties of Gradient-TD methods.
However, convergence toward the optimal value function is slow, and individual weight components exhibit wide, persistent oscillations.
The expected TDC experiment provides a clear, noise-free visualization of this behavior.

---

---



## 3. Expected Emphatic TD on Baird’s Counterexample

**Emphatic Temporal-Difference Learning in Expectation:**
An implementation of the **Expected Emphatic-TD (ETD)** algorithm on **Baird’s counterexample**.

---

## Project Overview

This project examines the behavior of **Emphatic TD** when run in expectation on Baird’s counterexample.
The expected form eliminates sampling variance, allowing the theoretical dynamics of the algorithm to be observed clearly.

Unlike TDC or TD(0), the direct application of ETD on this example has **extremely high variance**, making empirical runs unstable.
For this reason, only the **expected** trajectories are displayed, as in the original book.

---

## Problem Setup

The task is to estimate the value function:

$$
\hat{v}(s) = \mathbf{w}^\top \mathbf{x}(s)
$$

under the target policy (always choosing the solid action), while data is generated by the behavior policy.

The Emphatic-TD update depends on:

* **TD error**

$$
  \delta_t = R_{t+1} + \gamma \mathbf{w}^\top \mathbf{x}(S_{t+1}) * \mathbf{w}^\top \mathbf{x}(S_t)
$$
  
* **Importance sampling ratio** ( $$\rho_t$$ )
* **Emphasis** ( $$M_t$$ ), evolving as

$$
  M_t = \rho_t , \gamma , M_{t-1} + I_t
$$

All experiments use **interest ($$I_t = 1$$)**, matching Figure 11.6.

---

## Learning Algorithm

Expected Emphatic TD computes the **expected update** of both the weight vector ($$\mathbf{w}$$) and the emphasis ($$M$$) across all states.

### Weight Update

$$
\mathbf{w} \leftarrow \mathbf{w} * \alpha , \mathbb{E}!\left[ M_t , \rho_t , \delta_t , \mathbf{x}(S_t) \right]
$$

### Emphasis Update

$$
M_{t+1} = \mathbb{E}!\left[ \rho_t \gamma M_t + I_t \right]
$$

The implementation computes these expectations by iterating over all states and weighting transitions according to the behavior policy.

This produces smooth, variance-free learning curves that match the theoretical trajectory in the book.

---

## Experiments

### Experimental Configuration

| Parameter                            | Value                                     |
| ------------------------------------ | ----------------------------------------- |
| Initialization of ($$\mathbf{w}$$)   | ((1,1,1,1,1,1,10,1))                      |
| Initial emphasis                     | ($$M_0 = 0$$)                             |
| Step size                            | ($$\alpha = 0.03$$)                       |
| Sweeps                               | 1000                                      |
| Discount                             | ($$\gamma = 0.99$$)                       |
| Interest                             | ($$I_t = 1$$)                             |
| Policy                               | Target: solid; Behavior: solid w.p. (1/7) |

### Outputs Recorded

* Each component of the weight vector ($$w_i$$)
* RMS value error ($$\sqrt{\overline{VE}}$$)

These diagnostics form Figure 11.6.

---

## Results

**[Figure 11.6:](https://github.com/AlisaSujyan/Reinforcement-Learning/blob/main/counter-examples/generated_images/figure_11_6.png)** Expected Emphatic TD on Baird’s Counterexample

The expected ETD trajectories show:

* Some oscillations early in learning.
* Eventual convergence of all weight components.
* RMS-VE gradually decreasing toward zero.
* Smooth, stable curves due to elimination of sampling variance.

This reproduces the behavior described in the text, where convergence occurs **in expectation**, even though empirical ETD runs are too noisy to observe this directly.

---

## Key Insights

* Emphasis is essential for stabilizing off-policy TD learning under function approximation.
* ETD converges in expectation on Baird’s counterexample, unlike semi-gradient TD.
* Direct (sample-based) ETD is impractical here due to extremely high variance.
* Expected ETD provides a clean view of the algorithm’s theoretical dynamics.
* Oscillations occur early due to the emphasis process but diminish over time.

---

## Conclusion

Expected Emphatic TD successfully demonstrates theoretical convergence on Baird’s counterexample, with both the weights and value error stabilizing over time.
However, the algorithm’s variance in practice is so high that these trajectories cannot be reproduced reliably using sampled updates.
This highlights both the strengths and limitations of Emphatic-TD methods and motivates the need for variance-reduction techniques explored in later chapters.

---
