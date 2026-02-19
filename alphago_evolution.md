# The Evolution of AlphaGo: From Human Games to Efficient Planning

---

## 1. AlphaGo (Original)

AlphaGo (DeepMind, 2016) uses a combination of deep learning and reinforcement learning.

### Phase 1: Supervised Learning from Human Games
- Trained on ~30 million positions from expert human games (KGS Go Server)
- Two neural networks are trained:
  - **Policy network** — predicts the move a human would play given a board position
  - **Value network** — estimates the probability of winning from a given position
- This gives AlphaGo a strong "prior" — it starts thinking like a strong human player

### Phase 2: Reinforcement Learning (Self-Play)
- The policy network plays games against earlier versions of itself
- Wins/losses propagate back as reward signals (no intermediate rewards — just win or lose)
- The network gradually improves by discovering moves that lead to more wins
- The value network is then trained on the outcomes of these self-play games

### Phase 3: Monte Carlo Tree Search (MCTS) at Inference
- At play time, AlphaGo doesn't just pick the policy network's top move
- It runs MCTS: simulates thousands of future game continuations
- The **policy network** guides which branches to explore (pruning the search space)
- The **value network** evaluates positions without playing to the end
- Results are aggregated to choose the best move

---

## 2. AlphaGo Zero (2017)

### Key Differences from the Original

| Aspect | AlphaGo (Original) | AlphaGo Zero |
|---|---|---|
| Human game data | Yes (~30M positions) | None |
| Network architecture | Separate policy + value nets | Single combined network |
| Input features | Hand-crafted (liberties, captures, etc.) | Raw board state only (stones) |
| Rollouts in MCTS | Yes (fast rollout policy) | No — value network replaces rollouts |
| Training start | Supervised learning first | Reinforcement learning from scratch |
| Peak Elo | ~3500 | ~5185 |

### Key Innovations

**No Human Knowledge** — starts with only the rules of Go and random play. Eliminates human bias and bottlenecks.

**Single Unified Network** — one network with a shared residual convolutional body and two output heads (policy + value). More efficient and internally consistent.

**No Hand-Crafted Features** — only raw stone positions (black/white/empty) for the last 8 board states. The network learns what features matter.

**No Monte Carlo Rollouts** — the value head directly estimates position strength without playing games to completion. Faster and more accurate.

**ResNet Architecture** — deeper network with skip connections trains more stably.

### The Core Learning Loop

```
Current network → MCTS → Better moves → Self-play games → Training signal → Improved network → repeat
```

MCTS acts as a "policy improvement operator" — it produces moves better than the raw network, and training the network to match those moves lifts its baseline each iteration.

### The Philosophical Shift

> Original: "Start with human knowledge, then improve beyond it."
> Zero: "Given only the rules, discover everything from first principles."

Zero rediscovered known Go theory (joseki, fuseki) and then went beyond it, finding patterns human players had never considered. Human data can be a ceiling, not just a floor.

---

## 3. AlphaZero (2017): Generalizing Beyond Go

### What Changed

AlphaGo Zero was still Go-specific. AlphaZero took the same core algorithm and made it **game-agnostic** — a single system that could master any perfect-information two-player zero-sum game, given only the rules.

Tested on three games:
- **Go** — previously conquered
- **Chess** — dominated by Stockfish for years
- **Shogi** (Japanese chess) — dominated by Elmo

Achieved superhuman performance in all three, training from scratch in hours to days.

### How Generalization Was Achieved

**No Game-Specific Logic** — the only inputs needed are:
- A function to generate legal moves from any position
- A function to detect terminal states (win/loss/draw)
- The board representation

**Reused Architecture** — the same residual network design applies across games; only input/output dimensions change.

**Universal Engine** — the MCTS + self-play loop is identical across games regardless of their structure.

### Why This Was Surprising — Chess Specifically

Chess engines like Stockfish were the product of decades of hand-tuned evaluation functions, optimized alpha-beta search, and extensive opening databases. AlphaZero trained for **9 hours** with no chess knowledge and beat Stockfish convincingly — with notably more positional and sacrificial play.

### Limitations

| Limitation | Detail |
|---|---|
| Perfect information only | Cannot handle hidden state (poker, StarCraft) |
| Two-player zero-sum only | No multi-agent or cooperative settings |
| Finite, discrete action spaces | Not directly applicable to continuous control |
| Self-play requires a clear win condition | Doesn't extend to open-ended goals |
| Computationally expensive | Requires massive hardware for training |

### The Meta-Claim

AlphaZero demonstrated that a single, simple learning algorithm — self-play + MCTS + deep networks — is sufficient to reach or exceed the pinnacle of human achievement in a class of games, without any domain knowledge beyond the rules.

---

## 4. MuZero (2019): Learning Without the Rules

### The Core Innovation

AlphaZero still required the game rules to be provided. MuZero removes this requirement entirely — it learns to **plan without being given a model of the environment**.

> AlphaZero: "Give me the rules and I'll master the game."
> MuZero: "Don't tell me the rules. I'll figure out what I need to know."

### The Three Learned Functions

MuZero replaces the known simulator with three neural networks:

| Network | Input | Output | Purpose |
|---|---|---|---|
| **Representation** `h` | Raw observation (pixels, board) | Hidden state `s` | Encode the environment into a latent space |
| **Dynamics** `g` | Hidden state + action | Next hidden state + reward | Simulate "what happens if I do this" |
| **Prediction** `f` | Hidden state | Policy + value | Evaluate positions and suggest moves |

MCTS operates entirely inside this learned latent space — it never touches the real environment during search.

### How Planning Works

```
Real observation → h → hidden state s₀
                              │
              ┌───────────────┼──────────────────┐
              │ MCTS unrolls in latent space      │
              │                                   │
         s₀ ──g(a₁)──► s₁ ──g(a₂)──► s₂ ...    │
              │         f(s₁)          f(s₂)      │
              │       (policy+value)  (policy+val) │
              └───────────────────────────────────┘
                              │
                        Best action selected
```

### What This Unlocks

**Works on Atari (Visual Input, No Known Rules)** — tested on 57 Atari games with raw pixel input, no access to the game engine. Matched or exceeded previous state-of-the-art on nearly all.

**Task-Relevant, Not Reality-Faithful Model** — the latent state doesn't have to represent the true game state accurately. It only needs to be useful for predicting rewards and values. More flexible and often more efficient.

### Handling Partial Observability

MuZero's handling of partial observability is real but limited — it is not a formal POMDP solver.

**Frame Stacking** — the representation network receives a stack of recent observations (e.g., last 8 frames for Atari), allowing inference of velocity, direction, and trajectory.

**Hidden State Accumulates Context** — the hidden state `s` is a dense learned vector that can encode arbitrary context from observation history, including inferred latent structure never directly observed.

**What MuZero Does NOT Do:**

| Formal POMDP Concept | MuZero's Approach |
|---|---|
| Explicit belief state | No — single deterministic hidden vector |
| Bayesian update of uncertainty | No |
| Active information gathering | No |
| Stochastic hidden state handling | Poorly |

### Comparison Table

| Aspect | AlphaGo Zero | AlphaZero | MuZero |
|---|---|---|---|
| Human data | No | No | No |
| Rules provided | Yes (Go only) | Yes (any game) | No |
| Environment simulator | External | External | Learned |
| Visual/pixel input | No | No | Yes |
| Games mastered | Go | Go, Chess, Shogi | Go, Chess, Shogi, Atari (57) |

---

## 5. Stochastic MuZero (2021): Planning Over Uncertain Futures

### The Problem It Solves

Standard MuZero uses a deterministic dynamics model:

```
s_{t+1} = g(s_t, a_t)
```

Given the same hidden state and action, it always predicts the same next state. This fails when the environment has genuine randomness — card draws, dice rolls, stochastic opponent behavior, or any hidden variable influencing outcomes.

### The Core Addition: A Stochastic Latent Variable

The dynamics model becomes:

```
s_{t+1} = g(s_t, a_t, z_t)
```

Where `z_t` is a **discrete latent variable** representing the "random outcome" of the transition — the part the agent cannot control or predict with certainty.

Two networks handle `z`:

| Network | Role | Used When |
|---|---|---|
| **Encoder** `e(z \| o_{t+1}, s_t, a_t)` | Maps actual next observation to a discrete code | During **training** only |
| **Prior** `p(z \| s_t, a_t)` | Predicts distribution over z without seeing the future | During **planning** |

### Training vs. Planning

**During Training** — the encoder has access to the real next observation. It compresses what actually happened into a discrete code `z_t`, fed into the dynamics model for accurate transitions. The prior is simultaneously trained to predict `z_t` without seeing `o_{t+1}`:

```
Real trajectory:  o_t → a_t → o_{t+1}
                              ↓
                    encoder → z_t
                              ↓
        g(s_t, a_t, z_t) → s_{t+1}
```

A KL divergence loss keeps prior and encoder aligned (structurally similar to a VAE):

```
L_KL = KL( encoder(z | o_{t+1}, s_t, a_t) || prior(z | s_t, a_t) )
```

**During Planning** — the encoder is unavailable. The prior samples possible outcomes:

```
p(z | s_t, a_t) → sample z₁, z₂, z₃, ...
                         ↓
        g(s_t, a_t, zᵢ) → multiple possible next states
```

### The MCTS Tree Structure Changes

Standard MuZero has only decision nodes. Stochastic MuZero adds **chance nodes**, producing an expectimax-style tree:

```
Decision node (agent acts)
    │
    ├── action a₁
    │       │
    │   Chance node (environment responds)
    │       ├── z₁ (prob p₁) → s_{t+1}^1 → ...
    │       ├── z₂ (prob p₂) → s_{t+1}^2 → ...
    │       └── z₃ (prob p₃) → s_{t+1}^3 → ...
    │
    └── action a₂
            │
        Chance node
            ├── z₁ → ...
            └── z₂ → ...
```

Values at decision nodes are computed as expectations over chance node outcomes, allowing the agent to reason about **distributions of futures**.

### Why Discrete Latents

- **Tractable enumeration** — MCTS can explicitly branch over a finite set of outcomes
- **Stable training** — discrete variables with straight-through estimators or Gumbel-softmax are well-understood
- **Compression** — forces identification of distinct outcome categories rather than encoding noise

### What Stochastic MuZero Can Handle

| Scenario | Standard MuZero | Stochastic MuZero |
|---|---|---|
| Deterministic games (Chess, Go) | Yes | Yes (z collapses to a point) |
| Visual noise / frame variation | Poorly | Better |
| Random events (card draws, dice) | No | Yes |
| Stochastic opponent policies | No | Partially |
| Deep hidden state (poker) | No | Still limited |

### Remaining Limitations

- If outcomes are truly unstructured random, the prior may not be much better than uniform and the tree branches explosively
- Still no explicit belief state — no distribution over possible world states, only over next transitions
- For long-range hidden state (e.g., remaining cards in a deck), single-step stochastic variables are insufficient

---

## 6. MuZero Reanalyse: Improving Training Efficiency

### The Problem It Solves

Standard MuZero training has a fundamental staleness problem.

The training pipeline works like this:

```
Actor (self-play with MCTS) → Replay Buffer → Learner (network updates)
```

Trajectories sit in the replay buffer and are sampled repeatedly for training. But the **MCTS policy and value targets stored in the buffer were computed by an older version of the network**. As the network improves, those old targets become increasingly inaccurate guides — yet they're still being used for training.

This is the off-policy target problem: the data is from the right environment, but the labels are stale.

### The Core Idea

Reanalyse decouples two things that standard training conflates:

| Thing | Standard MuZero | Reanalyse |
|---|---|---|
| **Trajectory** (observations + actions) | Generated fresh by self-play | Reused from replay buffer |
| **MCTS targets** (policy + value) | Stored at collection time | **Recomputed with current network** |

Instead of using stored targets, Reanalyse takes an old trajectory and **reruns MCTS on it using the latest network parameters**, producing fresh targets. The network is trained on old trajectories with new labels.

### How It Works Step by Step

```
1. Sample an old trajectory from replay buffer:
        (o₀, a₀, r₀), (o₁, a₁, r₁), ..., (oₙ, aₙ, rₙ)

2. For each position oₜ in the trajectory:
        hθ(oₜ) → sₜ           (encode with current network)
        MCTS(sₜ, current θ) → fresh πₜ, vₜ

3. Train current network on:
        - Old rewards rₜ      (ground truth, unchanged)
        - Fresh policy πₜ     (from current MCTS)
        - Fresh value vₜ      (from current MCTS)
```

Step 2 requires **no environment interaction** — it runs entirely in the learned latent space. This makes reanalysis cheap compared to generating new self-play games.

### The Training Architecture

```
┌─────────────────┐   trajectories    ┌────────────────┐
│   Actor         │ ────────────────► │  Replay Buffer │
│  (self-play)    │                   │                │
└─────────────────┘                   └───────┬────────┘
                                              │ sample old trajectories
┌─────────────────┐   fresh targets   ┌───────▼────────┐
│   Reanalyser    │ ────────────────► │    Learner     │
│ (MCTS in latent)│                   │ (network update)│
└─────────────────┘                   └────────────────┘
```

The ratio of reanalysed to fresh data is a hyperparameter. A high reanalyse ratio — most training from reanalysed data — significantly improves sample efficiency.

### Why This Improves Efficiency

**Fresh Targets Without Fresh Trajectories** — generating a new self-play game requires many environment steps and wall-clock time proportional to game length. Reanalysing an existing trajectory requires only MCTS in latent space — no environment interaction needed.

**Effectively On-Policy Labels from Off-Policy Data** — trajectory data remains off-policy (collected under old parameters), but targets are always current. This reduces the bias from stale value estimates, a major source of training instability.

**Higher Data Utilisation** — old trajectories that would otherwise contribute diminishing returns can be refreshed and made useful again. The replay buffer becomes a persistent, reusable asset rather than a decaying one.

### Measured Impact

- Achieves comparable performance with **~10× fewer environment interactions** on some Atari games
- Particularly impactful in **data-limited regimes** — when generating unlimited self-play data isn't feasible
- Improves training stability by reducing target variance
- Makes MuZero practical for domains where environment interaction is expensive (robotics, slow simulators)

**EfficientZero (2021)** built directly on Reanalyse, adding self-supervised consistency losses and data augmentation on top — achieving human-level Atari performance with only **2 hours of game experience**, roughly 500× more sample efficient than the original MuZero.

### The Conceptual Point

Reanalyse separates two costs that are usually bundled together:

> **Exploration cost** — interacting with the environment to collect trajectories
> **Labelling cost** — computing accurate targets for those trajectories

Standard RL pays both costs together on every sample. Reanalyse pays the exploration cost once and the labelling cost repeatedly — refreshing labels as the network improves, squeezing more value from each trajectory the environment ever generated.

---

## The Full Conceptual Arc

```
AlphaGo           — human knowledge + rules + search
AlphaGo Zero      — rules + search, no human knowledge
AlphaZero         — rules + search, generalized across games
MuZero            — search only, rules learned from experience
Stochastic MuZero — learned rules, stochastic transitions, distributional planning
MuZero Reanalyse  — learned rules, stochastic-capable, sample-efficient training
```

Each step removes another human-provided assumption or computational bottleneck. The progression is a systematic answer to: *how much can an agent learn about the world purely from experience, and how efficiently can it do so?*
