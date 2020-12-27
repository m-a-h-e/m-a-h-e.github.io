---
tags: python,numpy,neural-network,reinforcement learning
mathjax: true
---
# An Introduction to Reinforcement Learning

Reinforcement Learning is a learning method that optimizes the execution of actions in an environment based on its observable state information to maximize the reward returned from the environment.
Usually, this task has to be accomplished - tabula rasa - without prior knowledge about the environment.

{:.caption .img}
![reinforcement learning base diagram](/assets/images/reinforcement_learning_base_diagram.png)
Reinforcement Learning Base Diagram

Actions are executed by a reinforcement learning agent exploiting the already collected experience but also exploring the environment searching for possibilities to maximize the collected reward.

The optimal balance between exploitation and exploration is a basic problem in reinforcement learning.
That means to leverage valuable knowledge and in parallel allow to discover unknown territory that may lead to improved behavior in the sense of larger reward to be collected.

>[Reinforcement Learning - The Book (MIT Press, Cambridge)](http://incompleteideas.net/book/the-book.html)

>[An Introduction to Deep Reinforcement Learning](https://arxiv.org/pdf/1811.12560.pdf)

## The Markov Decision Process (MDP)

>[Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process)

### The Markov property

- all information about the environment is available through its observable state
  - position, velocity, acceleration, etc.
- there is no hidden influence from the past to the future behavior of the environment, meaning the environment does not have a memory state that is excluded from its observable state
- the future behavior of the environment only depends on its current state and the actions executed in it
- a process with these properties is called a Markov Decision Process

## Reinforcement Learning Algorithms

- Monte Carlo RL Methods (MC)

- Temporal Difference Learning (TD)
  - The Bellman Equation
  - The Q Learning Algorithm
  - The SARSA Algorithm
  - The Expected SARSA Algorithm
  - The Dyna-Q Algorithm

- Deep Q Network (DQN) Architectures

- Policy Gradient Methods

- Monte Carlo Tree Search Algorithms
  - The Alpha Zero Algorithm

- Cross Entropy RL Methods
  - learns policy by filtering out low-reward episodes trajectory data and favor high-reward episodes trajectory data

{:.caption .img}
[![Reinforcement Learning, MIT 6.S191](https://img.youtube.com/vi/nZfaHIxDD5w/0.jpg)](https://www.youtube.com/watch?v=nZfaHIxDD5w)
[Alexander Amini](https://www.mit.edu/~amini/) - Reinforcement Learning, MIT 6.S191 (2020)

{:.caption .img}
[![The Basics of Reinforcement Learning, McGill University](https://img.youtube.com/vi/313kbpBq8Sg/0.jpg)](https://www.youtube.com/watch?v=313kbpBq8Sg)
[Joelle Pineau](https://www.cs.mcgill.ca/~jpineau/) - The Basics of Reinforcement Learning, McGill University

{:.caption .img}
[![Deep Reinforcement Learning, Stanford University](https://img.youtube.com/vi/lvoHnicueoE/0.jpg)](https://www.youtube.com/watch?v=lvoHnicueoE)
[Serena Yeung](https://ai.stanford.edu/~syyeung/) - Deep Reinforcement Learning, Stanford University

{:.caption .img}
[![TensorFlow and Deep Reinforcement Learning](https://img.youtube.com/vi/t1A3NTttvBA/0.jpg)](https://www.youtube.com/watch?v=t1A3NTttvBA)
Martin Görner - TensorFlow and Deep Reinforcement Learning

{:.caption .img}
[![Reinforcement Learning Course - DeepMind & UCL](https://img.youtube.com/vi/ISk80iLhdfU/0.jpg)](https://www.youtube.com/watch?v=ISk80iLhdfU&list=PLqYmG7hTraZBKeNJ-JE_eyJHZ7XgBoAyb)
Reinforcement Learning Course - DeepMind & UCL

## Glossary

- environment
  - deterministic state change
  - stochastic state change
  - deterministic reward
  - stochastic reward
  - environment dynamics
  - branching factor
    - number of possible actions in a given state

- agent
  - state
    - observation of the environment, sometimes also called node e.g. in MCTS
  - reward
    - a reward or penalty signal returned by the environment associated with the state and action chosen by the agent
    - reward is usually a sparse signal and may be delayed relative to the triggering agent action(s)
  - action
    - an action executed in the environment resulting in a state change, sometimes in combination with a reward signal
  - model-free algorithms
  - model-based algorithms
  - experience replay buffer
    - buffer of experiance data used for learning based on batch sampling
    - increases sample efficiency by using buffer entries more than once during neural network training
    - using sampled batches of training data from the replay buffer results in more stable training than directly using potentially correlated sequential observations
    - prioritized replay for faster learning
  
- state value function V(s)
  - state-action value function Q(s, a)
  - advantage function A(s, a)

- policy
  - maps a state to an action or a vector of action probabilities: $$\pi(s) = a$$
  - deterministic policy
  - stochastic policy
  - greedy policy
    - a deterministic exploitation policy without any portion of random exploration
  - epsilon-greedy policy
    - a mostly deterministic exploitation policy with a usually small portion of random exploration
  - on-policy
    - if the data used for learning is generated following the current policy
  - off-policy
    - if the data used for learning is generated by a policy that differs from the current behavior policy
      - if the data is generated using a slightly different policy, e.g. Q learning uses a greedy policy in the next state to generate learning data while following an epsilon-greedy behavior policy
      - if the data is put into an experiance buffer that is then used by the learning process for sampling random batches
      - if two networks are used to disentangle behavior and target policy
  - policy evaluation
    - update of state values or state-action values given a fixed policy, usually until all values converge and do not change anymore
  - policy improvement
    - policy update based on state values or state-action values
  - (general) policy iteration
    - process of interaction between policy evaluation and policy improvement
  - optimistic initial values
    - enforces environment exploration until all values have converged to their smaller optimal values keeping only the policy based portion of exploration
    - in contrast pessimistic initial values suppress exploration and the only source of exploration may be the random action selection based on the policy
  - how to learn a policy
    - learn policy directly
    - learn state action values and infer policy
    - learn environment model and infer policy by planning

- objective function
  - a reward function that can be optimized to maximize the expected discounted future reward
  - discount factor
    - determines the effect of future reward values depending on their distance from a state
    - a parameter usually chosen between 0.95 and 1.00
    - the closer to 1, the higher the impact of distant reward values

- model
  - a representation of the environment that enables planning

## Miscellaneous Topics

- [A Beginner's Guide to Deep Reinforcement Learning (A.I. Wiki)](https://pathmind.com/wiki/deep-reinforcement-learning)
- actor critic
- async advantage actor critic (A3C)
- Learning values across many orders of magnitude "adaptive normalization"
- A Distributional Perspective on Reinforcement Learning
- Recurrent DQN


