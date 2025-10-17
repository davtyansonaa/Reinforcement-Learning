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

## **Current Projects**

### **Tabular Methods**
- [tic-tac-toe](./tic-tac-toe/) – Simple RL project demonstrating value iteration in game strategy
- [ten-armed-testbed](./ten-armed-testbed/) – Action-value methods and exploration strategies (ε-greedy, UCB, GBA)
- [gridworld_mdp](./gridworld-mdp/) – Grid-based environment modeled as an MDP
- [gridworld-dp](./gridworld-dp/) – Policy evaluation and improvement in gridworld using dynamic programming
- [gambler-problem](./gambler-problem/) – Value iteration for optimal betting strategy
- [blackjack](./blackjack/) – First-visit Monte Carlo methods in card game environment
- [infinite-variance](./infinite-variance/) – Off-policy Monte Carlo with importance sampling variance analysis
- [random-walk](./random-walk/) – TD(0) vs Monte Carlo comparison in 5-state random walk
- [windy-gridworld](./windy-gridworld/) – SARSA implementation in stochastic wind environment
- [cliff-walking](./cliff-walking/) – SARSA vs Q-learning comparison in cliff navigation

### **Planning and Learning**
- [updates-comparison](./updates-comparison/) – Expected vs sample updates efficiency analysis (Figure 8.7)
- [trajectory-sampling](./trajectory-sampling/) – Uniform vs on-policy sampling distributions comparison (Figure 8.8)

### **Function Approximation**
- [function-approximation](./function-approximation/) – State aggregation, polynomial, Fourier, and tile coding methods for 1000-state random walk (Chapter 9)

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
- **Multi-Armed Bandits** (ε-greedy, UCB, Gradient Bandits)
- **Dynamic Programming** (Value Iteration, Policy Iteration)
- **Monte Carlo Methods** (First-visit, Every-visit, Control)
- **Temporal-Difference Learning** (TD(0), SARSA, Q-Learning, Expected SARSA)
- **n-step Methods** and Eligibility Traces
- **Planning Methods** (Expected updates, Sample updates, Dyna architecture)
- **Function Approximation** (Linear methods, Gradient descent, Semi-gradient methods)

### **Advanced Concepts**
- **Sample Efficiency** – Comparing update mechanisms and their computational costs
- **Distribution Selection** – On-policy vs uniform sampling for planning
- **Generalization** – Function approximation methods for large state spaces
- **Feature Engineering** – Basis functions (Fourier, polynomial, tile coding)
- **Bias-Variance Trade-offs** – Monte Carlo vs Temporal Difference learning

### **Experimental Environments**
- Multi-Armed Bandits Testbed
- Gridworld Variants (Windy, Cliff Walking)
- Gambler's Problem
- Blackjack
- Random Walk (5-state and 1000-state)
- Tic-Tac-Toe
- Randomly Generated MDPs (for planning experiments)

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

## **Textbook Chapter Coverage**

This repository progressively implements concepts from Sutton & Barto's textbook:

- **Chapter 2:** Multi-Armed Bandits → `ten-armed-testbed/`
- **Chapter 3-4:** MDPs & Dynamic Programming → `gridworld-mdp/`, `gridworld-dp/`, `gambler-problem/`
- **Chapter 5:** Monte Carlo Methods → `blackjack/`, `infinite-variance/`
- **Chapter 6:** Temporal-Difference Learning → `random-walk/`, `windy-gridworld/`, `cliff-walking/`
- **Chapter 7:** n-step Bootstrapping → *(Coming soon)*
- **Chapter 8:** Planning and Learning → `updates-comparison/`, `trajectory-sampling/`
- **Chapter 9:** On-Policy Prediction with Approximation → `function-approximation/`
- **Chapter 10:** On-Policy Control with Approximation → *(Coming soon)*
- **Chapter 11-13:** Off-Policy Methods & Policy Gradient → *(Coming soon)*

---

## **Key Findings Across Projects**

### **Planning and Learning Integration**
- **Sample updates** are more efficient than expected updates when computational budget is limited
- **On-policy sampling** dramatically outperforms uniform sampling in large state spaces
- **Trajectory-based updates** focus computation on relevant state-action pairs

### **Function Approximation**
- **Fourier basis** significantly outperforms polynomial basis for smooth value functions
- **Tile coding** provides excellent balance of performance and computational efficiency
- **Feature representation** is often more important than algorithm choice
- **Gradient MC** converges to better solutions but **Semi-gradient TD** learns faster

### **Learning Algorithms**
- **Q-learning** learns optimal policy faster but is less stable than **SARSA**
- **TD methods** bootstrap and learn faster but introduce bias
- **Monte Carlo** methods are unbiased but have higher variance
- **Exploration strategies** critically affect learning performance

---

## **Visualization Examples**

Projects include comprehensive visualizations:
- **Learning curves** – Performance over time/episodes
- **Value function estimates** – Heat maps and 3D plots
- **Policy evolution** – How decisions change during learning
- **Comparative analysis** – Side-by-side algorithm comparisons
- **Error metrics** – RMSE, convergence rates, sample efficiency

All figures are reproducible and compare against textbook results.

---

## **Reference**

> **Reinforcement Learning: An Introduction**  
> Richard S. Sutton & Andrew G. Barto  
> *Second Edition, 2018 – MIT Press*  
> [Read online](http://incompleteideas.net/book/the-book-2nd.html)

Additional references and papers are cited within individual project directories.

---

## **Author**

**Sona Davtyan**  
- GitHub: [@davtyansonaa](https://github.com/davtyansonaa)

---

*Continuously updated with new RL implementations and experiments*