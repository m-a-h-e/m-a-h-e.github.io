---
tags: python,numpy,neural-network,reinforcement learning
mathjax: true
---
# Monte Carlo Methods (MC)

- samples (state, action, reward) data from the environment following a given (usually epsilon-greedy or random) policy
- averages total future reward for each visited state over a number of complete episode trajectories
  - the averaging may be done in an incremental manner to limit the amount of memory needed by the algorithm
- the Monte Carlo algorithm requires the episodes to terminate to get valid future reward values without bootstrapping
- Exploring Starts chosses a random trajectory start state to allow all states to appear in trajectories
- MDP graphs that include nodes from which a terminal state cannot be reached, cannot be handled by this algorithm which has the requirement that trajectories have to terminate

