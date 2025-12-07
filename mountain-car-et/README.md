# Reinforcement Learning: Mountain Car with Eligibility Traces

This project implements the **Mountain Car** problem with eligibility traces from **Chapter 12 — Eligibility Traces** in *Reinforcement Learning: An Introduction* by *Richard S. Sutton & Andrew G. Barto*.

It demonstrates how different **eligibility trace mechanisms** affect learning performance and credit assignment in continuous state spaces requiring long action sequences.

The project includes:

- **Mountain Car environment** (continuous state, discrete actions)
- **SARSA(λ)** with on-policy learning
- **Four eligibility trace types** (accumulating, dutch, replacing, replacing with clearing)
- **Tile coding** for function approximation
- **Performance analysis** across trace types and λ values

---

## Key Features

- Implements **Mountain Car** with continuous state space
- Four **eligibility trace mechanisms** for comparative analysis
- **SARSA(λ)** for on-policy control with function approximation
- **Tile coding** with 8 tilings for efficient state representation
- Episode length tracking and convergence analysis
- Value function visualization across state space

---

## Environment: Mountain Car

The Mountain Car is a classic continuous control problem requiring momentum-based navigation:

**State Space:**
- Position: x ∈ [-1.2, 0.5]
- Velocity: v ∈ [-0.07, 0.07]

**Action Space:**
- Three discrete actions: reverse (-1), neutral (0), forward (+1)

**Dynamics:**
Physics-based velocity update:

$$
v_{t+1} = v_t + 0.001 \times a_t - 0.0025 \times \cos(3 \times x_t)
$$

$$
x_{t+1} = x_t + v_{t+1}
$$

**Initial State:**
- Position: random uniform in [-0.6, -0.4]
- Velocity: 0.0

**Goal:**
Reach position ≥ 0.5 (right hill peak)

**Rewards:**
- -1 on every time step until goal reached
- 0 upon reaching goal

**Episode Termination:**
- Goal reached, or
- 5000 steps exceeded

**Discount Factor:**
γ = 1.0 (undiscounted episodic task)

**Challenge:**
The car's engine is too weak to drive directly up the steep hill. The agent must learn to build momentum by driving back and forth.

---

## Algorithm: SARSA(λ)

SARSA(λ) extends the basic SARSA algorithm with eligibility traces for improved credit assignment:

**On-Policy Learning:**
Follows and improves ε-greedy policy (ε = 0 with optimistic initialization)

**TD Error:**

$$
\delta_t = R_{t+1} + \gamma \hat{Q}(S_{t+1}, A_{t+1}, w) - \hat{Q}(S_t, A_t, w)
$$

**Weight Update:**

$$
w \leftarrow w + \alpha \delta_t z_t
$$

**Characteristics:**
- Updates action-value function Q(s,a) rather than state-value V(s)
- Uses eligibility traces z_t for credit assignment
- On-policy: learns about policy being followed
- λ parameter controls trace decay and credit distribution

---

## Eligibility Trace Types

### 1. Accumulating Trace

The standard eligibility trace mechanism used in most applications.

**Update Rule:**

$$
z_t = \gamma \lambda z_{t-1} + \nabla \hat{Q}(S_t, A_t, w)
$$

For tile coding with binary features:

$$
z_t[i] = \gamma \lambda z_{t-1}[i] + 1 \quad \text{(for active tiles)}
$$

**Properties:**
- Accumulates credit for repeatedly visited state-action pairs
- Trace grows with repeated visits
- Standard choice for most problems

---

### 2. Dutch Trace

Used in True Online TD(λ) for better convergence properties.

**Update Rule:**

$$
z_t = \gamma \lambda z_{t-1} + \left(1 - \alpha \gamma \lambda z_{t-1}^T x_t\right) x_t
$$

For tile coding:

$$
\text{coefficient} = 1 - \alpha \gamma \lambda \sum_{i \in \text{active}} z_{t-1}[i]
$$

$$
z_t[i] = \gamma \lambda z_{t-1}[i] + \text{coefficient} \quad \text{(for active tiles)}
$$

**Properties:**
- Compensates for feature overlap in tile coding
- Prevents over-counting of overlapping features
- Improves convergence stability
- Approximates true online learning

---

### 3. Replacing Trace

Alternative to accumulating traces that prevents unbounded growth.

**Update Rule:**

$$
z_t[i] = \begin{cases}
1 & \text{if feature } i \text{ is active} \\
\gamma \lambda z_{t-1}[i] & \text{if feature } i \text{ is inactive}
\end{cases}
$$

**Properties:**
- Replaces rather than accumulates trace values
- Prevents trace from growing unboundedly
- Trace value capped at 1 for active features
- Useful when revisiting states frequently

---

### 4. Replacing Trace with Clearing

Extends replacing trace with action-specific clearing for control problems.

**Update Rule:**
Same as replacing trace, plus clearing step:

$$
z_t[\text{tiles of non-selected actions}] = 0
$$

**Properties:**
- Maintains traces only for selected action
- Clears traces for all other actions
- Prevents interference between different actions
- Essential for action-value methods
- Ensures credit assignment to correct action

**Rationale:**
In tile coding for Q(s,a), different actions use different tiles. Clearing prevents credit from bleeding between actions.

---

## Function Approximation: Tile Coding

**Configuration:**
- 8 tilings for continuous state representation
- Hash table (IHT) with 2048 maximum indices
- Position and velocity normalized to tile coordinates
- Action encoded as integer feature for action-specific tiles

**Advantages:**
- Efficient representation of continuous states
- Generalization through overlapping tiles
- Fast computation for linear function approximation
- Natural integration with eligibility traces

---

## Key Observations

**Eligibility Traces and Credit Assignment:**
- Traces enable efficient credit assignment over long action sequences
- Essential for Mountain Car where many steps precede reward
- λ controls how far back credit is assigned

**Trace Type Comparison:**
- **Accumulating**: Standard choice, good general performance
- **Dutch**: Better convergence, especially with tile coding overlap
- **Replacing**: Prevents unbounded growth, useful for revisited states
- **Replacing with Clearing**: Best for control, prevents action interference

**λ Parameter Trade-off:**
- λ = 0: Pure one-step SARSA (slow learning in Mountain Car)
- λ = 1: Monte Carlo-like (high variance, slow convergence)
- Intermediate λ (e.g., 0.9): Best balance for Mountain Car
- Optimal λ depends on problem structure

**Tile Coding Interaction:**
- Dutch traces compensate for feature overlap
- Accumulating traces may over-count overlapping features
- Feature representation affects trace effectiveness

**Learning Performance:**
- Episode length decreases as policy improves
- Convergence speed varies by trace type
- Value function gradually approximates cost-to-go

---

## Analysis Capabilities

The implementation supports comprehensive analysis:

- **Trace comparison**: Learning curves across four trace types
- **λ parameter study**: Performance for different λ values
- **Convergence analysis**: Episodes and steps to reach goal consistently
- **Episode length tracking**: Learning progress visualization
- **Value function plots**: Cost-to-go surface across state space
- **Trace evolution**: How traces change during episodes


