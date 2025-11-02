# Mountain Car Task

## Problem Description

The Mountain Car task is a classic reinforcement learning problem that demonstrates the challenge of continuous control in environments where immediate progress toward the goal may be counterproductive.

### The Challenge

A car is positioned in a valley between two hills. The goal is at the top of the hill on the right side. However, **the car's engine is not powerful enough to drive directly up the steep slope**, even at full throttle. 

### The Solution Strategy

To reach the goal, the agent must learn a counterintuitive strategy:

1. **Move away from the goal** - Drive up the opposite slope on the left
2. **Build momentum** - Apply full throttle to gain enough inertia
3. **Carry through** - Use the accumulated momentum to climb the steep right slope, even while slowing down

This is a prime example of a task where things must get worse (moving farther from the goal) before they can get better. Many control methods struggle with such problems without explicit human guidance.

## Environment Specifications

### State Space

The environment has two continuous state variables:

- **Position** ($x_t$): The car's horizontal position
  - Range: $[-1.2, 0.5]$
  - Left boundary: $-1.2$
  - Right boundary (goal): $0.5$

- **Velocity** ($\dot{x}_t$): The car's speed
  - Range: $[-0.07, 0.07]$

### Action Space

Three discrete actions are available:

1. **Full throttle forward**: $A_t = +1$
2. **Full throttle reverse**: $A_t = -1$
3. **Zero throttle (coast)**: $A_t = 0$

### Dynamics

The car moves according to simplified physics:

**Position update:**
$$x_{t+1} = \text{bound}[x_t + \dot{x}_{t+1}]$$

**Velocity update:**
$$\dot{x}_{t+1} = \text{bound}[\dot{x}_t + 0.001A_t - 0.0025\cos(3x_t)]$$

The cosine term represents the effect of gravity, which varies with the slope of the hills.

### Boundary Conditions

- When $x_{t+1}$ reaches the **left bound** ($-1.2$): velocity is reset to 0
- When $x_{t+1}$ reaches the **right bound** ($0.5$): **goal is reached** and the episode terminates

### Initial Conditions

Each episode starts with:
- Random position: $x_t \in [-0.6, 0.4)$
- Zero velocity: $\dot{x}_t = 0$

## Reward Structure

- **Reward per timestep**: $-1$ (until goal is reached)
- **Episode termination**: When the car reaches position $\geq 0.5$
- **Objective**: Minimize the number of steps to reach the goal

This sparse reward structure makes the problem challenging, as the agent receives no positive feedback until success.

## Function Approximation

### Tile Coding

To handle the continuous state space, the implementation uses **grid-tiling** (tile coding):

- **Number of tilings**: 8
- **Tile coverage**: Each tile covers $1/8$ of the bounded distance in each dimension
- **Offsets**: Asymmetrical offsets between tilings (as described in RL literature Section 9.5.4)

### Value Function Approximation

The action-value function is approximated using linear combination:

$$\hat{q}(s, a, \mathbf{w}) = \mathbf{w}^T \mathbf{x}(s, a) = \sum_{i=1}^{d} w_i \cdot x_i(s, a)$$

where:
- $\mathbf{w}$ is the parameter vector
- $\mathbf{x}(s, a)$ is the feature vector created by tile coding
- $d$ is the dimensionality of the feature vector

## Learning Behavior

### Optimistic Initialization

Initial action values are set to **0**, which is optimistic (all true values are negative). This encourages exploration even with $\varepsilon = 0$.

### Exploration Pattern

During early learning (e.g., Step 428):
- The car oscillates back and forth in the valley
- Circular trajectories form in state space
- Frequently visited states are valued worse than unexplored states
- This drives the agent to continually explore new states until a solution is found

The agent learns that its initial expectations were unrealistic and is pushed to explore systematically.

## Key Insights

1. **Delayed rewards**: Success requires a sequence of actions that don't immediately improve the situation
2. **Momentum is essential**: The physics of the problem necessitates building kinetic energy
3. **Exploration challenges**: The agent must discover that moving away from the goal is part of the solution
4. **Function approximation**: Continuous states require appropriate feature representations

