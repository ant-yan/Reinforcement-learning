
# Gridworld MDP Simulation

## Overview
This project features a custom-built simulation of the Gridworld Markov Decision Process (MDP), inspired by Chapter 3 of *Reinforcement Learning: An Introduction* by Sutton and Barto. It highlights essential reinforcement learning principles such as state-value estimation, reward mechanisms, and policy evaluation.

## The Gridworld Challenge
The Gridworld is a simple grid-based environment where an agent moves around attempting to maximize its total reward. The agent can move in one of four directions and learns how to act based on the structure of the grid and rewards.

### Highlights:
* **States:** Each grid cell acts as a unique state the agent can occupy.
* **Actions:** Available movements include up (north), down (south), left (west), and right (east).
* **Rewards:**
  * Hitting a wall results in a -1 penalty.
  * Entering special cells A or B yields rewards of +10 and +5, respectively, and teleports the agent to A′ or B′.
  * All other valid moves give a reward of 0.
* **Value Calculation:** The simulation calculates the state-value function under a policy where all actions are equally likely.

## Project Structure

### 1. Gridworld Logic
* Encapsulates the environment's reward rules and transition model.
* Manages edge cases, including wall collisions and the behavior of special states.

### 2. Value Function Algorithms
* Applies the Bellman equation to compute the state values under a given policy.
* Includes both policy evaluation and value iteration strategies.
* Visualizations show how values evolve depending on policy.

### 3. Policy and Value Insights
* **Random Policy:** The agent selects each of the four actions with equal chance.
* **Value Interpretation:** Cells near walls often have lower values due to penalties from invalid moves.
* **Special States Review:**
  * **A:** The return is slightly less than +10 because teleportation may lead to further penalties.
  * **B:** The value exceeds +5 because the resulting position allows access to high-value areas.

## Key Takeaways
The calculated state-value function helps illustrate:
* Long-term reward potential for each grid cell
* The influence of different policies on state values
* The role of transition probabilities and reward structures in learning

## Further Reading
For more context and theory, refer to:  
**Sutton, R. S., & Barto, A. G.** — [*Reinforcement Learning: An Introduction* (2nd edition)](https://archive.org/details/rlbook2018/mode/2up)
