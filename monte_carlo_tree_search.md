---
tags: python,numpy,neural-network,reinforcement learning
mathjax: true
---
# Monte Carlo Tree Search Algorithms

- an algorithm based on node selection, node expansion, Monte Carlo rollouts and reward backpropagation
- table based algorithm that limits its usability to more or less small problems with limited and discrete state space and moderate branching factor

## The Alpha Zero Algorithm
- uses generalizing value and policy neural networks to target fully observable deterministic problems with very large state space and large branching factor
  - a value network predicts the winner of the game
  - a policy network generates an action probability vector
  - value and policy function may be implemented using a combined (shared weights) neural network with two heads generating the scalar state value and the action probability vector
  - both value and policy function in conjunction represent a model that can be used for planning
- the algorithm incorporates a Monte Carlo Tree Search like element to generate simulated experiance, used to optimize the value and policy network
- self-play against randomly chosen previous versions of itself, is the one and only mechanism used to improve the playing strength of the algorithm

