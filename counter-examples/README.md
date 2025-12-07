# Baird's Counterexample

This project implements **Baird's Counterexample** from **Chapter 11 — Off-policy Methods with Function Approximation** in *Reinforcement Learning: An Introduction* by *Richard S. Sutton & Andrew G. Barto*.

It demonstrates how **off-policy TD(0)** diverges when combined with **linear function approximation** and **bootstrapping**, even in a simple 7-state MDP.

The project includes:

- **Baird's 7-state star MDP**
- **Semi-gradient off-policy TD(0)** (divergence demonstration)
- **Semi-gradient Dynamic Programming** (divergence despite being model-based)
- **Gradient-TD (TDC / GTD(0))** (convergent)
- **Expected TDC** (convergent)
- **Emphatic TD(0)** (convergent)
- **RMSVE** and **RMSPBE** evaluation metrics

---

## Key Features

- Implements the **canonical 7-state counterexample** revealing the deadly triad
- Shows divergence of **semi-gradient off-policy TD(0)** and **semi-gradient DP**
- Implements stable alternatives: **TDC**, **Expected TDC**, and **Emphatic TD**
- Computes **RMSVE** and **RMSPBE** for convergence analysis
- Reproduces qualitative behavior from **Chapter 11**

---

## Baird's 7-State Counterexample

This minimal MDP reveals a fundamental instability when function approximation, bootstrapping, and off-policy learning combine.

**States:**
- **Upper states (0-5):** Feature *i* = 2, last feature = 1
- **Lower state (6):** Second-to-last feature = 1, last feature = 2

**Actions:**
- **Dashed:** Uniformly transitions to states 0-5
- **Solid:** Transitions to state 6

**Policies:**
- **Target policy:** Always chooses solid action (deterministic)
- **Behavior policy:** Chooses solid with probability 1/7, dashed with probability 6/7

**Rewards:** Always 0

**True value function:** Identically 0 for all states

This configuration creates the **deadly triad** that causes semi-gradient TD methods to **diverge**.

---

## Error Metrics

### RMSVE — Root Mean Square Value Error

Measures prediction accuracy relative to true values:

$$
\text{RMSVE} = \sqrt{\sum_s \mu(s) (V̂(s, w) - V_\pi(s))^2}
$$

### RMSPBE — Root Mean Square Projected Bellman Error

Measures distance to TD fixed point (minimized by TDC):

$$
\text{RMSPBE} = \sqrt{\sum_s \mu(s) [\Pi(\text{Bellman error}(s))]^2}
$$

---

## Algorithms

### Semi-gradient Off-Policy TD(0) — Divergent

Implements the standard off-policy TD update with importance sampling:

$$
w \leftarrow w + \alpha \rho \delta x(s)
$$

where $\rho = \pi(a|s) / b(a|s)$ is the importance sampling ratio.

**Result:** Weights grow unboundedly (reaching 200+ when true value is 0).

---

### Semi-gradient Dynamic Programming — Divergent

Performs expected updates over all states using the full model:

- Eliminates sampling variance
- Still exhibits divergence due to semi-gradient nature
- Confirms semi-gradient as the root problem

---

### Gradient-TD (TDC / GTD(0)) — Convergent

Uses dual weight vectors to follow the true gradient:

- **Primary weights** $w$: parameterize value function
- **Secondary weights** $v$: estimate gradient correction

TDC minimizes the **Mean Square Projected Bellman Error** and **converges** to the TD fixed point.

---

### Expected TDC — Convergent

Computes expected updates using full model knowledge:

- Deterministic updates at each sweep
- Eliminates sampling noise
- Smooth, fast convergence

---

### Emphatic TD(0) — Convergent

Uses emphasis weighting to ensure convergence:

- Maintains emphasis trace $M_t$
- Reweights updates by accumulated importance ratios
- Converges through state distribution rebalancing

---

## Key Observations

**The Deadly Triad:**
1. **Function Approximation** — Linear features with limited capacity
2. **Bootstrapping** — Using estimates instead of Monte Carlo returns
3. **Off-Policy Learning** — Data from behavior policy, evaluating target policy

Any two factors are safe; all three together cause divergence in semi-gradient methods.

**Experimental Results:**
- Semi-gradient TD(0) and DP both **diverge** with unbounded weight growth
- TDC, Expected TDC, and Emphatic TD all **converge** with bounded weights
- TDC achieves near-zero RMSPBE at convergence
- Emphatic TD achieves near-zero RMSVE at convergence

---

## Conclusion

This project reproduces **Baird's Counterexample**, demonstrating the fundamental instability of **semi-gradient off-policy TD** under the deadly triad.

It provides working implementations of stable alternatives (**TDC**, **Expected TDC**, **Emphatic TD**) that achieve provable convergence through gradient correction or emphasis reweighting, matching the theoretical results from **Chapter 11** of Sutton & Barto.