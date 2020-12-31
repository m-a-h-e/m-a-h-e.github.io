---
tags: python,numpy,neural-network,reinforcement learning,deep Q learning,DQN,DDQN,Dueling
mathjax: true
---
# Deep Q Network (DQN) Architectures

## Basic Q Network
  - state to vector of Q values related to all actions, or
  - state and action to scalar Q(s, a) value

## Basic DQN Algorithm

1. Execute a policy step and store the experienced data-point $$[s, a, r, s']$$ into an experience buffer.
2. Sample a batch of unconnected experience data-points from the experience buffer and use it to optimize (train) the policy network.

The fact, that - to optimize the policy - the experienced data is not used directly, but through sampling of unconnected data batches from an experience buffer, stabilizes the policy network behavior and the policy learning process.

## Double DQN Architecture

[Deep Reinforcement Learning with Double Q-learning (Google DeepMind)](https://arxiv.org/pdf/1509.06461.pdf)

- double (target and behavior networks)
  - prevents "tail chasing" / oscillating policy
  - stabilizes learning process

To stabilize the training process, two independent networks are used for policy execution and policy optimization.
The algorithm - once in a while - switches the function of the networks, or copies the optimized policy parameters into the policy execution network, to keep both of them up-to-date.

## Dueling Network Architecture

[Dueling Network Architectures for Deep Reinforcement Learning (Google DeepMind)](http://proceedings.mlr.press/v48/wangf16.pdf)

- dueling (value advantage decomposition)
  - helps the network to independently learn state value and action advantages which is often more simple
  - makes independent generalization of state value and action advantages possible
  - uses aggregation module to combine state value and action advantages to Q values

