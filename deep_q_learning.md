---
tags: python,numpy,neural-network,reinforcement learning,deep Q learning,DQN,DDQN,Dueling
mathjax: true
---
# Deep Q Network (DQN) Learning

- basic Q network
  - state to vector of Q values related to all actions, or
  - state and action to scalar Q(s, a) value

## Double DQN Algorithm

- double (target and behavior networks)
  - prevents "tail chasing" / oscillating policy
  - stabilizes learning process

## Dueling Network Architecture

- dueling (value advantage decomposition)
  - helps the network to independently learn state value and action advatages, which is often more simple
  - makes independent generalization of state value and action advantages possible
  - uses aggregation module to combine state value and action advantages to Q values

