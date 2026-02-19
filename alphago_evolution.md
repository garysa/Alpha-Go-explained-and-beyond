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

---

## Appendix: A Brief History of DeepMind

### Founding (2010)

DeepMind was founded in London in **November 2010** by three co-founders:

- **Demis Hassabis** — neuroscientist, chess prodigy, and game developer; provided the scientific vision
- **Shane Legg** — machine learning researcher focused on general intelligence and AI safety
- **Mustafa Suleyman** — entrepreneur responsible for operations and applied AI (later left to co-found Inflection AI)

The stated mission from the outset was to "solve intelligence, and then use that to solve everything else" — an unusually explicit commitment to artificial general intelligence as an end goal, not just a research direction.

### Early Work and Google Acquisition (2013–2014)

DeepMind's early research focused on reinforcement learning applied to classic Atari games. In **2013** they published results showing a single neural network learning to play multiple Atari titles from raw pixels at or above human level — a striking demonstration that general-purpose learning algorithms could master diverse tasks without task-specific engineering.

In **January 2014**, Google acquired DeepMind for a reported **£400–500 million** (~$650M), making it one of the largest AI acquisitions at the time. DeepMind retained operational independence and remained headquartered in London.

### Deep Q-Networks and Nature Paper (2015)

The **Deep Q-Network (DQN)** paper, published in *Nature* in February 2015, formalised the Atari results and introduced the techniques (experience replay, target networks) that made deep reinforcement learning stable. It became one of the most cited papers in the field and established DeepMind as a leading research lab alongside OpenAI and academic groups.

### AlphaGo and the Go Milestone (2016)

In **March 2016**, AlphaGo defeated world champion Lee Sedol 4–1 in a live match watched by an estimated 200 million people. Go had been considered AI-complete — too complex for computers to master for decades. The match is widely regarded as a turning point in public perception of AI capability.

### AlphaGo Zero and AlphaZero (2017)

**AlphaGo Zero** (October 2017) demonstrated that the system could surpass human play with no human data whatsoever — only the rules of Go and self-play. It defeated the original AlphaGo 100–0.

**AlphaZero** (December 2017) generalised this to Chess and Shogi, mastering both from scratch in hours and defeating the best domain-specific engines. Chess in particular shocked the community — decades of hand-crafted engine development undone in a single training run.

### AlphaFold and Protein Folding (2018–2021)

DeepMind entered structural biology with **AlphaFold**, which won the CASP13 protein structure prediction competition in **2018**.

**AlphaFold2** won CASP14 in **2020** by a historic margin — achieving accuracy comparable to experimental techniques for most protein targets, effectively solving a 50-year grand challenge in biology.

In **2021**, DeepMind released a publicly accessible database of predicted structures for virtually the entire human proteome and later expanded to over **200 million proteins** across nearly all known organisms — one of the most significant scientific contributions of the decade.

### MuZero and the Planning Era (2019–2021)

**MuZero** (2019) extended AlphaZero's approach to environments without known rules, learning its own world model. **Stochastic MuZero** and **MuZero Reanalyse** (both 2021) added stochastic environment handling and dramatically improved sample efficiency — making the approach practical beyond game-playing into slower, more expensive simulators.

### Expanding Scope: Gato, Science, and Robotics (2022–2023)

**Gato** (May 2022) was a single transformer model trained across 604 tasks — text, images, games, robot control — demonstrating that a single architecture and training regime could handle radically diverse domains.

**Chinchilla** (2022) established new scaling laws for large language models, showing that most models of the era were undertrained relative to their compute budget. This reshaped how the entire field approached LLM training.

**RoboCat** (2023) showed a self-improving robotic agent that could adapt to new robot arms and new tasks from a small number of demonstrations.

### Merger with Google Brain → Google DeepMind (2023)

In **April 2023**, Google merged DeepMind with **Google Brain** — its other premier AI research division — to form **Google DeepMind**. Demis Hassabis became CEO of the combined entity. The merger consolidated Google's AI research under a single organisation, positioning it to compete more directly with OpenAI and Anthropic.

In **June 2023**, **AlphaDev** discovered a new sorting algorithm — embedded in the output of a reinforcement learning agent — that is **70% faster** for short sequences. It was incorporated directly into the LLVM compiler, making it one of the first times AI-discovered algorithms entered production computing infrastructure.

### Nobel Prize and Scientific AI (2024)

In **October 2024**, **Demis Hassabis** and **John Jumper** were awarded the **Nobel Prize in Chemistry** for AlphaFold2's contribution to protein structure prediction — the first Nobel Prize awarded to work primarily driven by AI methods. David Baker shared the prize for separate work on computational protein design.

**SIMA** (2024) introduced a general AI agent capable of following natural language instructions across diverse 3D environments and video games, without game-specific training — a step toward instruction-following agents that generalise across embodied tasks.

### Gemini, Robotics, and Weather (2025)

**Gemini 2.5** (March 2025) marked a major capability leap in Google DeepMind's large language model family, with strong performance across reasoning, coding, and multimodal tasks.

**Gemini Robotics** (March 2025) brought the Gemini model family into physical embodiment, enabling robots to interact with the real world through language-conditioned manipulation.

In **July 2025**, an advanced version of Gemini with extended thinking achieved **gold-medal standard at the International Mathematical Olympiad** — matching the best human competitors on problems requiring creative, multi-step mathematical reasoning.

**Weather Lab** (mid-2025) applied stochastic neural networks trained on 45 years of global weather data to probabilistic forecasting, outperforming traditional physics-based models during the 2025 Atlantic hurricane season.

**AlphaEvolve** (May 2025) used large language models to design and evolve algorithms autonomously — discovering improvements to sorting, matrix multiplication, and other fundamental computational problems.

---

### DeepMind's Arc in One View

| Period | Focus | Landmark |
|---|---|---|
| 2010–2013 | Founding, early RL | Atari games from pixels |
| 2014–2015 | Google acquisition, DQN | Nature DQN paper |
| 2016–2017 | Board games | AlphaGo beats Lee Sedol; AlphaZero |
| 2018–2021 | Biology, planning | AlphaFold2; MuZero |
| 2022–2023 | Generalist models, science | Gato; Chinchilla; Google Brain merger |
| 2024 | Recognition, embodiment | Nobel Prize; SIMA |
| 2025 | LLMs, robotics, science tools | Gemini 2.5; IMO gold; AlphaEvolve |

DeepMind's trajectory is the story of a lab that began with a single conviction — that intelligence is a general phenomenon that can be studied and engineered — and pursued it from Atari to the Nobel Prize, from Go boards to protein databases, from learned world models to physical robots.

---

## Appendix: How a Neural Network Works

Every system described in this document — policy networks, value networks, dynamics models, representation functions — is built from neural networks. Understanding what a neural network actually is clarifies why these systems can learn at all.

### The Basic Idea

A neural network is a mathematical function that maps inputs to outputs. What makes it powerful is that it is **parameterised** — it has millions of adjustable numbers (called weights) — and **learnable** — those weights can be tuned by exposure to data until the function produces the right outputs for given inputs.

The name comes from a loose analogy to biological neurons, though modern neural networks are better understood as highly flexible function approximators than as brain simulations.

### Neurons and Layers

The building block is the **neuron** (or node). A single neuron does three things:

```
1. Receives several numerical inputs:  x₁, x₂, x₃, ...
2. Computes a weighted sum:            z = w₁x₁ + w₂x₂ + w₃x₃ + ... + b
3. Applies a nonlinear function:       output = f(z)
```

The weights `w` and bias `b` are the learnable parameters. The nonlinear function `f` is called an **activation function** — common choices are ReLU (`max(0, z)`) or tanh. Without nonlinearity, stacking many neurons would still only produce a linear function, severely limiting what could be learned.

Neurons are arranged in **layers**:

```
Input layer → Hidden layer(s) → Output layer

[x₁]          [h₁]               [y₁]
[x₂]    →     [h₂]      →        [y₂]
[x₃]          [h₃]
              [h₄]
```

Each neuron in a layer receives inputs from every neuron in the previous layer (in a fully connected network). The hidden layers transform the input into increasingly abstract representations. The output layer produces the final answer — a probability distribution over moves, a value estimate, a predicted reward, etc.

### Forward Pass

Running an input through the network — computing the output — is called a **forward pass**. It is just arithmetic: multiply, add, apply activation function, repeat for each layer. This is fast and can be parallelised efficiently on GPUs.

### Learning: Backpropagation and Gradient Descent

The network learns by adjusting its weights to reduce error on training data. This requires two steps:

**1. Measure the error**

A **loss function** compares the network's output to the correct answer and returns a single number representing how wrong the prediction was. For example:
- Cross-entropy loss for predicted probability distributions (policy head)
- Mean squared error for predicted values (value head)

**2. Compute gradients and update weights**

**Backpropagation** uses the chain rule of calculus to compute, for every weight in the network, how much that weight contributed to the error. This produces a **gradient** — a direction in weight-space that increases the error.

**Gradient descent** then moves the weights in the *opposite* direction — slightly reducing the error:

```
w ← w − α · ∂Loss/∂w
```

Where `α` (the learning rate) controls the step size. Repeat over many batches of training data and the weights converge toward values that produce accurate outputs.

In practice, **stochastic gradient descent (SGD)** or its variants (Adam, RMSProp) update weights using small random batches rather than the full dataset, which is faster and often generalises better.

### Deep Networks and Depth

A **deep** neural network simply has many hidden layers — typically dozens to hundreds in modern systems. Depth allows the network to learn **hierarchical representations**:

- Early layers detect low-level patterns (edges, local stone configurations)
- Middle layers detect higher-level structures (groups, threats, spatial relationships)
- Later layers detect abstract strategic concepts (influence, territory, initiative)

This hierarchy emerges from training — it is not programmed in. The network discovers that representing the world in layers of increasing abstraction is useful for making accurate predictions.

### Convolutional Neural Networks (CNNs)

For spatial inputs like a Go board or Atari screen, fully connected layers are inefficient — every pixel connected to every neuron ignores spatial structure. **Convolutional layers** instead apply a small learned filter across the entire input:

```
Filter (3×3):        Applied across board:
┌───┬───┬───┐        detects the same pattern
│ w │ w │ w │   →   wherever it appears
│ w │ w │ w │        (translation equivariance)
│ w │ w │ w │
└───┴───┴───┘
```

This drastically reduces parameters and exploits the fact that the same pattern (a capturing sequence, a ladder) is meaningful wherever it appears on the board. AlphaGo, AlphaZero, and MuZero all use convolutional layers as their primary feature extractors.

### Residual Networks (ResNets)

Very deep networks suffer from the **vanishing gradient problem** — gradients shrink as they are propagated back through many layers, making early layers learn very slowly.

**Residual connections** (skip connections) provide a shortcut:

```
Input x
  │
  ├──────────────────────┐
  │                      │
  ▼                      │
[Layer 1]                │  (identity shortcut)
[Layer 2]                │
  │                      │
  ▼                      │
  + ◄────────────────────┘
  │
Output: F(x) + x
```

The network learns the *residual* `F(x)` — what to add to the input — rather than a full transformation from scratch. Gradients flow directly through the shortcut, enabling training of networks hundreds of layers deep. AlphaGo Zero and all subsequent DeepMind systems use ResNets.

### How This Connects to AlphaGo

Every component described in this document is a neural network doing one of these jobs:

| Component | Network Type | Task |
|---|---|---|
| Policy network | CNN / ResNet | Input: board state → Output: probability over moves |
| Value network | CNN / ResNet | Input: board state → Output: single win probability |
| Representation function `h` | CNN / ResNet | Input: raw observations → Output: latent hidden state |
| Dynamics function `g` | ResNet | Input: hidden state + action → Output: next state + reward |
| Prediction function `f` | MLP head | Input: hidden state → Output: policy + value |
| Encoder / Prior (Stochastic MuZero) | ResNet | Input: observations → Output: latent variable distribution |

The sophistication of AlphaGo Zero, MuZero, and their successors lies not in exotic network architectures but in **what the networks are trained to predict**, **how training data is generated** (self-play, reanalysis), and **how planning (MCTS) interacts with learned functions** to produce behaviour far better than the raw network alone.
