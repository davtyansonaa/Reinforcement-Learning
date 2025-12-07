# Random Walk with Eligibility Traces

This project implements three core eligibility-trace methods from **Chapter 12 — Eligibility Traces** in *Reinforcement Learning: An Introduction* by *Richard S. Sutton & Andrew G. Barto*.

It demonstrates how **eligibility traces** bridge Monte Carlo and TD methods, unifying n-step bootstrapping approaches through the λ-parameter.

The project includes:

- **Random Walk environment** (5-state episodic task)
- **Off-line λ-return** (forward-view definition)
- **TD(λ)** (backward-view with accumulating traces)
- **True Online TD(λ)** (exact forward-backward equivalence)
- Reproduction of **Figures 12.3, 12.6, and 12.8**

---

## Key Features

- Implements the **standard 5-state Random Walk** from Sutton & Barto
- Shows **off-line λ-return** as the forward-view definition
- Demonstrates **TD(λ)** backward-view equivalence
- Implements **True Online TD(λ)** for exact online learning
- Reproduces qualitative behavior from **Chapter 12**

---

## Environment: Random Walk

The environment is a simple episodic task designed to illustrate eligibility traces:

**States:**
- Non-terminal states: A, B, C, D, E (represented as 1-5)
- Terminal states: Left (0), Right (6)

**Transitions:**
- From each non-terminal state, agent moves left or right with equal probability
- Episodes start in the center state (C)
- Episodes terminate upon reaching either terminal state

**Rewards:**
- +1 on transition to right terminal
- 0 otherwise

**True Value Function:**
Forms a straight line: V(A) = 1/6, V(B) = 2/6, V(C) = 3/6, V(D) = 4/6, V(E) = 5/6

**Function Approximation:**
Linear function approximation with state-based features.

---

## Algorithms

### Off-line λ-Return — Forward View

The λ-return provides the forward-view definition of eligibility traces:

$$
G^\lambda_t = (1-\lambda)\sum_{n=1}^{T-t-1} \lambda^{n-1} G_t^{(n)} + \lambda^{T-t-1} G_t
$$

where the n-step return is:

$$
G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})
$$

**Update rule (after episode completion):**

$$
w \leftarrow w + \alpha \sum_t \bigl( G_t^{(\lambda)} - V(S_t) \bigr) x(S_t)
$$

**Characteristics:**
- Waits until episode ends for full trajectory information
- Single batch update per episode
- Defines the target that TD(λ) approximates
- λ=0 reduces to TD(0), λ=1 reduces to Monte Carlo

---

### TD(λ) — Backward View

TD(λ) uses eligibility traces to approximate the forward-view λ-return in an online fashion:

**Accumulating eligibility trace:**

$$
e_t = \gamma \lambda e_{t-1} + x(S_t)
$$

**TD error:**

$$
\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
$$

**Update rule (at each time step):**

$$
w_{t+1} = w_t + \alpha \delta_t e_t
$$

**Characteristics:**
- Updates at every time step (online)
- Traces accumulate credit for recent states
- Forward-backward equivalence holds in expectation for complete episodes
- More efficient than off-line λ-return

**Forward-Backward Equivalence:**
For episodic tasks with Monte Carlo initialization, TD(λ) produces the same total update as off-line λ-return, as demonstrated in **Figure 12.6**.

---

### True Online TD(λ) — Exact Equivalence

True Online TD(λ) corrects TD(λ) to achieve exact forward-backward equivalence at every time step, not just in expectation:

**Dutch trace:**

$$
e_t = \gamma \lambda e_{t-1} + (1 - \alpha \gamma \lambda e_{t-1}^\top x(S_t)) x(S_t)
$$

**Update rule:**

$$
w_{t+1} = w_t + \alpha \delta_t e_t + \alpha (w_t^\top x(S_t) - w_{t-1}^\top x(S_t))(e_t - x(S_t))
$$

where the TD error includes a correction term:

$$
\delta_t = R_{t+1} + \gamma w_t^\top x(S_{t+1}) - w_t^\top x(S_t)
$$

**Characteristics:**
- Matches forward view exactly at each step
- Dutch trace prevents over-counting of features
- More stable and accurate than standard TD(λ)
- Minimal computational overhead over TD(λ)

---

## Key Observations

**Role of λ:**
- λ = 0: Pure TD(0) (one-step bootstrapping)
- λ = 1: Pure Monte Carlo (no bootstrapping)
- 0 < λ < 1: Blend of TD and MC (controls bias-variance tradeoff)

**Eligibility Traces:**
- Provide credit assignment to recently visited states
- Unify n-step methods through continuous λ parameter
- Enable efficient online learning with Monte Carlo-like returns

**Method Comparison:**
- **Off-line λ-return**: Defines ideal target, requires complete episodes
- **TD(λ)**: Efficient online approximation, equivalent in expectation
- **True Online TD(λ)**: Exact equivalence, best for online learning

**Experimental Results:**
- All methods converge to true value function
- True Online TD(λ) shows smoother, more accurate convergence
- Higher λ values reduce bias but increase variance
- Optimal λ depends on problem characteristics

---

## Results

The implementation reproduces three key figures from Chapter 12:

| Figure | Description |
|--------|-------------|
| **12.3** | Off-line λ-return value estimates for different λ values |
| **12.6** | Forward-backward equivalence demonstration for TD(λ) |
| **12.8** | True Online TD(λ) performance comparison |

Each generated plot matches the qualitative behavior shown in the corresponding book figure, demonstrating correct implementation of the algorithms.

---

## Conclusion

This project reproduces the core eligibility-trace algorithms from **Chapter 12** of Sutton & Barto using the Random Walk environment.

It demonstrates how **eligibility traces** provide a unified framework for temporal difference learning, bridging the gap between one-step TD methods and Monte Carlo approaches through the λ parameter. The progression from **off-line λ-return** to **TD(λ)** to **True Online TD(λ)** shows the evolution toward more efficient and accurate online learning methods.

Eligibility traces remain fundamental to modern reinforcement learning, enabling efficient credit assignment and flexible control over the bias-variance tradeoff in value estimation.