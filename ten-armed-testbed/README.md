# Multi-Armed Bandit Problem - 10-Armed Testbed

This project implements and compares various algorithms for solving the multi-armed bandit problem using a 10-armed testbed. The implementation includes several classical approaches to the exploration-exploitation dilemma in reinforcement learning.

## Overview

The multi-armed bandit problem is a classical problem in probability theory and machine learning that exemplifies the exploration-exploitation dilemma. In this problem, an agent must choose between multiple actions (arms) to maximize cumulative reward, while learning about the reward distributions through trial and error.

## Project Structure

```
├── src/
│   └── bandit.py              # Main Bandit class implementation
├── ten_armed_testbed.ipynb    # Jupyter notebook with experiments
└── generated_images/          # Output plots and figures
```

## Features

The project implements and compares the following algorithms:

### 1. Action Selection Methods
- **ε-greedy**: Balances exploration and exploitation with probability ε
- **Greedy**: Pure exploitation (ε = 0)
- **Upper Confidence Bound (UCB)**: Selects actions based on confidence intervals
- **Gradient Bandit Algorithm (GBA)**: Uses preference-based action selection

### 2. Value Estimation Methods
- **Sample-average method**: Updates action values using running averages
- **Constant step-size**: Uses fixed learning rate for non-stationary problems
- **Optimistic initial values**: Encourages exploration through optimistic initialization

## Key Classes and Methods

### Bandit Class

The main `Bandit` class supports multiple configurations:

```python
Bandit(
    arms_number=10,                    # Number of bandit arms
    use_sample_averages=False,         # Use sample-average method
    epsilon=0.,                        # Exploration probability
    initial_action_value_estimates=0., # Initial Q-values
    confidence_level=None,             # UCB parameter c
    use_gradient=False,                # Enable gradient bandit
    step_size=0.1,                     # Learning rate α
    use_gradient_baseline=False,       # Use baseline in gradient method
    true_expected_reward=0.            # True reward mean for GBA
)
```

#### Key Methods
- `initialize()`: Reset bandit state and generate new action values
- `act()`: Select an action based on the chosen algorithm
- `step(action)`: Execute action and return reward, update estimates

## Experiments and Results

### 1. Reward Distribution (Figure 2.1)
Visualizes the reward distribution for a 10-armed bandit problem using violin plots.

### 2. ε-greedy vs Greedy Comparison (Figure 2.2)
Compares performance of:
- Greedy (ε = 0.00)
- ε-greedy (ε = 0.10)
- ε-greedy (ε = 0.01)

**Results**: ε-greedy methods show better long-term performance due to continued exploration.

### 3. Optimistic Initial Values (Figure 2.3)
Compares:
- Optimistic initialization (ε = 0, Q₁ = 5)
- Realistic initialization (ε = 0.1, Q₁ = 0)

**Results**: Optimistic initialization encourages early exploration even with greedy selection.

### 4. Upper Confidence Bound (Figure 2.4)
Compares:
- UCB with c = 2
- ε-greedy with ε = 0.1

**Results**: UCB shows superior performance by intelligently selecting actions based on uncertainty.

### 5. Gradient Bandit Algorithm (Figure 2.5)
Compares different configurations:
- α = 0.1 with/without baseline
- α = 0.4 with/without baseline

**Results**: Using baseline significantly improves performance, and higher learning rates can be beneficial.

## Installation and Usage

### Requirements
```bash
pip install numpy matplotlib tqdm
```

### Running the Experiments

1. **Using Jupyter Notebook**:
   ```bash
   jupyter notebook ten_armed_testbed.ipynb
   ```

2. **Using the Bandit Class**:
   ```python
   from src.bandit import Bandit
   import numpy as np
   
   # Create a bandit with ε-greedy selection
   bandit = Bandit(epsilon=0.1, use_sample_averages=True)
   bandit.initialize()
   
   # Run for 1000 steps
   rewards = []
   for _ in range(1000):
       action = bandit.act()
       reward = bandit.step(action)
       rewards.append(reward)
   
   print(f"Average reward: {np.mean(rewards):.3f}")
   ```

### Simulation Function

The `simulate()` function runs multiple independent experiments:

```python
def simulate(runs, times, bandits):
    """
    Run multiple bandit experiments and return performance metrics.
    
    Args:
        runs: Number of independent runs
        times: Number of time steps per run
        bandits: List of bandit configurations to test
    
    Returns:
        optimal_action_counts: Fraction of optimal actions selected
        rewards: Average rewards obtained
    """
```

## Key Findings

1. **Exploration vs Exploitation**: Pure greedy methods get stuck in suboptimal actions, while methods with exploration (ε-greedy, UCB) achieve better long-term performance.

2. **UCB Superiority**: UCB outperforms ε-greedy by making smarter exploration decisions based on action uncertainty.

3. **Optimistic Initialization**: Can substitute for systematic exploration in stationary problems.

4. **Gradient Methods**: Benefit significantly from baseline subtraction and appropriate learning rates.

## Algorithm Details

### ε-greedy Action Selection
- With probability ε: select random action (exploration)
- With probability 1-ε: select action with highest estimated value (exploitation)

### UCB Action Selection
```
UCB(a) = Q(a) + c√(ln(t)/N(a))
```
Where c controls exploration level and the square root term represents uncertainty.

### Gradient Bandit Algorithm
Updates action preferences H(a) using:
```
H(a) ← H(a) + α(R - baseline)(𝟙{a=A} - π(a))
```
Where π(a) = e^H(a)/∑e^H(b) is the action probability.

## Performance Metrics

- **Average Reward**: Cumulative reward divided by number of steps
- **% Optimal Action**: Fraction of time the optimal action was selected
- **Learning Curves**: Performance over time to show convergence behavior

## Future Extensions

- Non-stationary bandits with changing reward distributions
- Contextual bandits with state information
- Thompson Sampling (Bayesian approach)
- Comparison with more recent algorithms

## References

This implementation is based on Chapter 2 of "Reinforcement Learning: An Introduction" by Sutton and Barto (2nd Edition).