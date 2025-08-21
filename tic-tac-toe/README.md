# Tic-Tac-Toe Reinforcement Learning

A complete implementation of Tic-Tac-Toe using reinforcement learning with temporal difference (TD) learning. The project features AI agents that learn optimal strategies through self-play and can compete against human players.

## Overview

This project implements a reinforcement learning approach to master Tic-Tac-Toe using value function approximation and temporal difference learning. The AI agents learn through extensive self-play, gradually improving their strategies until they achieve near-optimal performance.

## Features

- **Reinforcement Learning Agent**: Uses temporal difference learning with epsilon-greedy exploration
- **Complete State Space**: Pre-computes all possible game states for optimal learning
- **Self-Play Training**: Agents improve through playing against each other
- **Human vs AI**: Interactive gameplay against trained agents
- **Policy Persistence**: Save and load trained agent policies
- **Performance Analytics**: Track win rates and learning progress

## Architecture

### Core Components

**State (`state.py`)**
- Represents game board configurations as 3x3 numpy arrays
- Implements win/tie detection logic
- Generates unique hash values for state identification
- Provides state transition functionality

**Player (`player.py`)**
- `RLPlayer`: Reinforcement learning agent with configurable parameters
- `HumanPlayer`: Interactive human player interface
- Implements epsilon-greedy action selection
- Handles policy learning and persistence

**Judge (`judge.py`)**
- Game orchestrator managing player interactions
- Handles turn alternation and game flow
- Determines winners and manages game state

## Algorithm Details

### Temporal Difference Learning

The RL agent uses TD(0) learning with the update rule:

```
V(St) = V(St) + α[V(St+1) - V(St)]
```

Where:
- `V(St)` is the value estimate of state St
- `α` is the learning rate (step_size)
- The update only occurs for greedy actions

### Exploration Strategy

The agent employs epsilon-greedy exploration:
- With probability `ε`: choose random valid action (exploration)
- With probability `1-ε`: choose action leading to highest value state (exploitation)

### State Representation

- Empty cells: `0`
- First player (X): `1` 
- Second player (O): `-1`

Board positions correspond to keyboard layout:
```
| q | w | e |
| a | s | d |
| z | x | c |
```

## Installation

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd tic-tac-toe-rl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Agents

Train two RL agents through self-play:

```python
from tic_tac_toe import train

# Train for 100,000 epochs
train(epochs=100000, print_every_n=5000)
```

Training parameters:
- `epochs`: Number of training games
- `print_every_n`: Frequency of progress updates

### Agent Competition

Evaluate trained agents against each other:

```python
from tic_tac_toe import compete

# Run 1000 competition games
compete(turns=1000)
```

### Human vs AI

Play against a trained agent:

```python
from tic_tac_toe import play

# Start interactive gameplay
play()
```

During gameplay, enter keyboard keys corresponding to board positions (q, w, e, a, s, d, z, x, c).

### Complete Training Pipeline

Run the full training and evaluation pipeline:

```bash
python tic_tac_toe.py
```

This will:
1. Train agents for 100,000 epochs
2. Run 1,000 competition games
3. Start interactive human vs AI mode

## Configuration

### RL Agent Parameters

- `step_size` (α): Learning rate (default: 0.1)
- `epsilon` (ε): Exploration probability (default: 0.01 for training, 0.0 for competition)

### Training Parameters

- Training uses `epsilon=0.01` for exploration
- Competition uses `epsilon=0.0` for pure exploitation
- Default training duration: 100,000 epochs

## Performance

With proper training, the RL agents achieve:
- Near-optimal play in most positions
- Consistent tie results when both players play optimally
- Ability to exploit human errors effectively

The agents learn that Tic-Tac-Toe is a solved game where perfect play from both sides always results in a tie.

## File Structure

```
.
├── requirements.txt      # Project dependencies
├── __init__.py          # Package initialization
├── state.py             # Game state representation and logic
├── player.py            # Player implementations (RL and Human)
├── judge.py             # Game orchestration and management
├── tic_tac_toe.py       # Main training and gameplay interface
└── README.md            # Project documentation
```

## Policy Files

Trained policies are automatically saved as:
- `policy_first.bin`: First player (X) policy
- `policy_second.bin`: Second player (O) policy

These files contain the learned state value estimations and are automatically loaded during competition and human gameplay.

