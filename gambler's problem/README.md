
# Gambler’s Problem – A Simple Value Iteration Demo

Hey there! 👋 This little project is a fun take on the classic Gambler’s Problem from the Sutton & Barto book on Reinforcement Learning. We’ve solved it using value iteration and added some visualizations to show how things change as the algorithm runs.

## What’s the Problem?

Imagine a gambler flipping coins 🎲:

- If it’s heads, they win the amount they bet.
- If it’s tails, they lose that bet.
- The goal? Turn whatever money they start with (1 to 99 dollars) into a clean $100.
- If they hit 0 or reach 100, the game’s over.

### Technically speaking:

- **States** are capital amounts from 0 to 100.
- **Actions** are the possible bets they can make at each state.
- **Reward** is 1 when reaching 100, and 0 otherwise.
- **Goal** is to maximize the chance of hitting $100.

## How It’s Solved

We use a classic technique called **value iteration**. It’s part of dynamic programming and helps us figure out the best choices (a.k.a. policy) at every step to increase the chances of winning.

## What You’ll See

- A plot showing how the value function (aka “how good is each state”) improves with every sweep of the algorithm.
- A second plot showing the final “what should I bet” strategy (the optimal policy).
