# AlphaGo Explained and Beyond

A structured walkthrough of DeepMind's AlphaGo family of systems — from the original AlphaGo to MuZero Reanalyse — explaining how each generation learns, what problem it solves, and how it differs from its predecessor.

## Contents

The main document [`alphago_evolution.md`](alphago_evolution.md) covers six systems in sequence:

| System | Year | Key Contribution |
|---|---|---|
| **AlphaGo** | 2016 | First superhuman Go AI; combines supervised learning from human games with MCTS and reinforcement learning |
| **AlphaGo Zero** | 2017 | Removes all human data; learns Go entirely from self-play using a single unified network |
| **AlphaZero** | 2017 | Generalises the Zero approach to any perfect-information game (Go, Chess, Shogi) without game-specific knowledge |
| **MuZero** | 2019 | Removes the need for known rules; learns its own model of the environment and plans inside a latent space |
| **Stochastic MuZero** | 2021 | Extends MuZero to stochastic environments by introducing a learned discrete latent variable for uncertain transitions |
| **MuZero Reanalyse** | 2021 | Dramatically improves sample efficiency by relabelling old replay buffer trajectories with fresh MCTS targets |

## Themes

- How self-play and MCTS combine as a general learning engine
- The progressive removal of human-provided assumptions (data, rules, simulators)
- How planning under uncertainty evolves from deterministic to stochastic models
- The separation of exploration cost from labelling cost to improve data efficiency

## Audience

Written for readers with a general interest in AI and machine learning. Mathematical concepts are introduced where necessary but the focus is on intuition, architecture, and the conceptual progression between systems.
