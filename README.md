# **Reinforcement Learning Projects**

This repository contains a collection of hands-on **Reinforcement Learning (RL)** projects, each focused on implementing and analyzing different algorithms and environments from the textbook **Reinforcement Learning: An Introduction** by Sutton & Barto.

The goal of this repository is to develop a deeper understanding of:
- Agent learning and policy optimization
- Environment dynamics and reward mechanisms  
- Exploration-exploitation tradeoffs
- On-policy vs. off-policy learning

---

## **Projects Overview**

Each project explores a specific RL concept or algorithm, and includes its own `README.md` with:
- **Objectives** — What the project aims to demonstrate
- **Methodology** — Core algorithms and environment setup
- **Key Results** — Visualizations, learning curves, and observations

---

## **Current Projects**

- [tic-tac-toe](./tic-tac-toe/) — Simple RL project demonstrating value iteration in game strategy
- [ten-armed-testbed](./ten-armed-testbed/) — Action-value methods and exploration strategies (ε-greedy, UCB, GBA)
- [gridworld_mdp](./gridworld-mdp/) — Grid-based environment modeled as an MDP
- [gridworld-dp](./gridworld-dp/) — Policy evaluation and improvement in gridworld using dynamic programming
- [gambler-problem](./gambler-problem/) — Value iteration for optimal betting strategy
- [blackjack](./blackjack/) — First-visit Monte Carlo methods in card game environment
- [infinite-variance](./infinite-variance/) — Off-policy Monte Carlo with importance sampling variance analysis
- [random-walk](./random-walk/) — TD(0) vs Monte Carlo comparison in 5-state random walk
- [windy-gridworld](./windy-gridworld/) — SARSA implementation in stochastic wind environment
- [cliff-walking](./cliff-walking/) — SARSA vs Q-learning comparison in cliff navigation

Each project may contain **notebooks**, **scripts**, and **configuration files** specific to its implementation.

---

## **Getting Started**

### **Clone the Repository**
```bash
git clone https://github.com/davtyansonaa/Reinforcement-Learning.git  
cd Reinforcement-Learning
```

### **Install Dependencies**
```bash
pip install -r requirements.txt  
```

### **Run a Specific Project**
Navigate to any project directory and follow its individual README:
```bash
cd ten-armed-testbed
python src/bandit_algorithms.py  
```

### **Explore Jupyter Notebooks**
Most projects include interactive notebooks for experimentation:
```bash
jupyter notebook
```

---

## **Key Topics Covered**

### **Core Algorithms**
- Multi-Armed Bandits (ε-greedy, UCB, Gradient Bandits)
- Dynamic Programming (Value Iteration, Policy Iteration)
- Monte Carlo Methods (First-visit, Every-visit, Control)
- Temporal-Difference Learning (TD(0), SARSA, Q-Learning, Expected SARSA)
- n-step Methods and Eligibility Traces

### **Experimental Environments**
- Multi-Armed Bandits Testbed
- Gridworld Variants (Windy, Cliff Walking)
- Gambler's Problem
- Blackjack
- Random Walk
- Tic-Tac-Toe
- Mountain Car
- CartPole
- Atari Games (selected)

---

## **Project Structure**

```
Reinforcement-Learning/
├── tic-tac-toe/
│   ├── src/
│   ├── notebooks/
│   └── README.md
├── ten-armed-testbed/
│   ├── src/
│   ├── notebooks/
│   └── README.md
├── gridworld_mdp/
│   ├── src/
│   ├── notebooks/
│   └── README.md
└── ...
```

---

## **Reference**

> **Reinforcement Learning: An Introduction**  
> Richard S. Sutton & Andrew G. Barto  
> *Second Edition, 2018 — MIT Press*  
> [Read online](http://incompleteideas.net/book/the-book-2nd.html)

Additional references and papers are cited within individual project directories.

---