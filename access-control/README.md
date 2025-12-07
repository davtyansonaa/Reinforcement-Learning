# Access-Control Queuing Task

## Problem Description

The Access-Control Queuing Task is a classic reinforcement learning problem that demonstrates the challenge of resource allocation under uncertainty. This is a **continuing task** (no episode boundaries) where an agent must make real-time decisions to maximize long-term average reward.

### The Challenge

A server system has **10 servers** available to handle incoming customer requests. Customers arrive one at a time with **4 different priority levels** (0-3), offering rewards of {1, 2, 4, 8} respectively.

**The core difficulty**: The agent must balance:
- **Accepting high-value customers** when servers are scarce
- **Reserving capacity** for potentially valuable future requests
- **Maximizing throughput** without overwhelming the system

### The Solution Strategy

The agent must learn a priority-based acceptance policy:

1. **Reserve capacity for high-priority customers** - Reject low-priority requests when servers are limited
2. **Accept aggressively for high rewards** - Take priority 3 customers even with few free servers
3. **Be selective with low priorities** - Accept priority 0-1 customers only when resources are abundant

This requires learning that **immediate reward isn't always optimal** - sometimes rejecting a customer now preserves capacity for higher-value customers later.

## Environment Specifications

### State Space

The environment state is defined by two variables:

- **Free Servers** ($n$): Number of currently available servers
  - Range: $[0, 10]$ (discrete)
  
- **Customer Priority** ($p$): Priority level of current customer
  - Values: $\{0, 1, 2, 3\}$ (discrete)
  - Corresponding rewards: $\{1, 2, 4, 8\}$

**State representation**: $S_t = (n_{\text{free}}, p_{\text{customer}})$

### Action Space

Two discrete actions are available:

1. **Accept**: $A_t = 1$ - Assign customer to a server (if available)
2. **Reject**: $A_t = 0$ - Turn customer away

**Constraint**: When $n_{\text{free}} = 0$, only rejection is possible.

### Dynamics

The system evolves according to:

**Server availability update:**
- If customer accepted: $n_{\text{free}} \leftarrow n_{\text{free}} - 1$
- Each busy server becomes free with probability $p = 0.06$

**Reward structure:**
$$R_{t+1} = \begin{cases} 
r_p & \text{if customer accepted} \\
0 & \text{if customer rejected}
\end{cases}$$

where $r_p \in \{1, 2, 4, 8\}$ is the reward for priority $p$.

**Customer arrivals:**
- New customer arrives every timestep
- Priority drawn uniformly: $p \sim \text{Uniform}(\{0, 1, 2, 3\})$
- Queue never empties (continuing task)

### Initial Conditions

Each run starts with:
- All servers available: $n_{\text{free}} = 10$
- Random customer priority: $p \sim \text{Uniform}(\{0, 1, 2, 3\})$

## Objective

**Maximize differential return** (average reward formulation):

$$\max \mathbb{E}[R_t - \bar{R}]$$

where $\bar{R}$ is the average reward rate learned during training.

This formulation is appropriate for continuing tasks with no discount factor ($\gamma = 1$).

## Algorithm: Differential Semi-Gradient SARSA

### Tile Coding Function Approximation

To represent the discrete state-action space efficiently:

- **Number of tilings**: 8
- **State features**: Position and priority are scaled appropriately
- **Hash table size**: 2,048 indices
- **Feature vector**: $\mathbf{x}(s, a)$ - binary features indicating active tiles

### Value Function Approximation

The action-value function is approximated as:

$$\hat{q}(s, a, \mathbf{w}) = \mathbf{w}^T \mathbf{x}(s, a) = \sum_{i \in \text{active}} w_i$$

where only tiles active for state-action pair $(s, a)$ contribute to the sum.

### Update Rules

For each transition $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$:

**TD Error:**
$$\delta_t = R_{t+1} - \bar{R} + \hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}) - \hat{q}(S_t, A_t, \mathbf{w})$$

**Average reward update:**
$$\bar{R} \leftarrow \bar{R} + \beta \cdot \delta_t$$

**Weight vector update:**
$$\mathbf{w} \leftarrow \mathbf{w} + \alpha \cdot \delta_t \cdot \nabla \hat{q}(S_t, A_t, \mathbf{w})$$

For tile coding, the gradient is simply the feature vector, so:
$$w_i \leftarrow w_i + \frac{\alpha}{8} \cdot \delta_t \quad \text{for each active tile } i$$

(The step size is divided by 8 to account for 8 tilings)

### Hyperparameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Action-value step size | $\alpha$ | 0.01 |
| Average reward step size | $\beta$ | 0.01 |
| Exploration rate | $\varepsilon$ | 0.1 |
| Server free probability | $p$ | 0.06 |
| Total servers | - | 10 |
| Number of tilings | - | 8 |
| Training steps | - | 1,000,000 |

### Policy

**$\varepsilon$-greedy policy**:
- With probability $\varepsilon$: select random action
- With probability $1-\varepsilon$: select $\arg\max_a \hat{q}(s, a, \mathbf{w})$
- Exception: Always reject when no servers are free

## Learning Behavior

### Value Function Evolution

The differential value function (best action value per state) shows clear patterns:

- **Priority 3 (reward 8)**: High values across all server states
- **Priority 2 (reward 4)**: Moderate values, declining with fewer servers
- **Priority 1 (reward 2)**: Low values, positive only with many free servers
- **Priority 0 (reward 1)**: Negative values in most states

### Policy Characteristics

The learned policy exhibits:

1. **Threshold-based acceptance**: Clear boundaries between accept/reject regions
2. **Priority ordering**: Higher priorities accepted with fewer available servers
3. **Resource reservation**: Low priorities rejected unless abundant capacity exists

Example decision boundaries (approximate):
- Priority 3: Accept with ≥1 free server
- Priority 2: Accept with ≥3 free servers
- Priority 1: Accept with ≥6 free servers
- Priority 0: Accept with ≥8 free servers

### Performance Metrics

After 1,000,000 training steps:
- **Average reward rate**: $\bar{R} \approx 2.31$
- **Convergence**: Stable policy within training period
- **State distribution**: Most time spent with 1-3 free servers

## Key Insights

1. **Continuing task formulation**: Average reward (differential) approach handles infinite horizon naturally
2. **Priority-based thresholds**: Optimal policy reserves capacity strategically based on customer value
3. **Function approximation**: Tile coding provides smooth generalization despite discrete state space
4. **Exploration importance**: $\varepsilon$-greedy ensures discovery of optimal acceptance thresholds
5. **Reward structure**: Exponential priority rewards (1, 2, 4, 8) create clear value distinctions


### Visualization

The implementation reproduces **figure_10_5** from Sutton & Barto, showing:
- **Top plot**: Differential value functions for each priority level
- **Bottom plot**: Learned policy heatmap (0 = Reject, 1 = Accept)

