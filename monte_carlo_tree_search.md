---
tags: python,numpy,neural-network,reinforcement learning
mathjax: true
---
# Monte Carlo Tree Search Algorithms

- Monte Carlo Tree Search
  - an algorithm based on node selection, node expansion, Monte Carlo rollouts and reward backpropagation
  - table based algorithm, which limits its usability to more or less small problems with limited and discrete state space and moderate branching factor
- The Alpha Zero Algorithm
  - uses generalizing value and policy neural networks to target fully observable deterministic problems with very large state space and large branching factor
    - the value network predicts the winner of the game
    - the policy network generates an action probability vector
    - both networks in conjunction create a model, which can be used for planning
  - the algorithm incorporates a Monte Carlo Tree Search like element to generate simulated experiance, used to optimize the value and policy networks
  - self-play against randomly chosen previous versions of itself, is the one and only mechanism used to improve the playing strength of the algorithm

