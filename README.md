# **Reinforcement Learning Projects**

This repository contains a collection of hands-on **Reinforcement Learning (RL)** projects, each focused on implementing and analyzing different algorithms and environments from the textbook **Reinforcement Learning: An Introduction** by Sutton & Barto.

The goal of this repository is to develop a deeper understanding of:
- Agent learning and policy optimization
- Environment dynamics and reward mechanisms
- Exploration-exploitation tradeoffs
- On-policy vs. off-policy learning
- Planning and learning integration
- Function approximation and generalization

---

## **Projects Overview**

Each project explores a specific RL concept or algorithm, and includes its own `README.md` with:
- **Objectives** – What the project aims to demonstrate
- **Methodology** – Core algorithms and environment setup
- **Key Results** – Visualizations, learning curves, and observations
- **Implementation Details** – Code structure and experimental configurations

---
## **Key RL Concepts Summary**

### **General Concepts**
- **Generalized Policy Iteration (GPI)** refers to the interplay between policy evaluation and policy improvement.
- **Bootstrapping** means updating value estimates based on other learned value estimates.

### **Model-Free Methods**
- **TD methods** bootstrap and learn faster but introduce bias.
- **Monte Carlo** methods are unbiased but have higher variance.
- **Q-learning** learns optimal policy faster but is less stable than **SARSA**.
- **Exploration strategies** critically affect learning performance.

### **Function Approximation**
- The need for **generalization** arises when the state space is too large.
- **Feature representation** is often more important than algorithm choice.
- **Deep Q-Networks (DQN)** use neural networks to approximate the action-value function, enabling scaling to high-dimensional state spaces.
- **Policy Gradient** methods learn the policy directly using function approximation.
- **Gradient MC** converges to better solutions but **Semi-gradient TD** learns faster.
- **Receptive field width** critically affects generalization vs discrimination trade-off.
- **Broad features** enable fast early learning but produce overly smooth approximations.
- **Narrow features** provide high resolution but require more samples and generalize poorly.

### **Visualization Examples**

Projects include comprehensive visualizations:
- **Learning curves** – Performance over time/episodes
- **Value function estimates** – Heat maps and 3D plots
- **Policy evolution** – How decisions change during learning
- **Comparative analysis** – Side-by-side algorithm comparisons
- **Error metrics** – RMSE, convergence rates, sample efficiency

---

## **Reference**

> **Reinforcement Learning: An Introduction**
> Richard S. Sutton & Andrew G. Barto
> *Second Edition, 2018 – MIT Press*
> [Read online](http://incompleteideas.net/book/the-book-2nd.html)
---
## **Author**

**Sona Davtyan**
- GitHub: [@davtyansonaa](https://github.com/davtyansonaa)

---

*Continuously updated with new RL implementations and experiments*