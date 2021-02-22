---
tags: python,numpy,neural-network,reinforcement learning
mathjax: true
---
# The Bellman Equation

[The Bellman Equation](https://en.wikipedia.org/wiki/Bellman_equation) defines the value $$V^\pi(s)$$ of a given state as the sum of
- the immediate reward $$r(s, a)$$ received when executing action $$a$$ in state $$s$$ following policy $$\pi$$
- the value $$V^\pi(s')$$ of state $$s'$$ reached by executing the policy step

This gives the following bootstrap state value evaluation rule:

$$V^\pi(s) = r(s, a^\pi) + V^\pi(s')$$

for deterministic environments.

For stochastic environments, in which the same action $$a$$ executed in the same state $$s$$ may result in different next states $$s'$$, the transition probabilities $$p$$ have to be taken into account.

The next state value $$V^\pi(s')$$ becomes the sum over all next states, that can be reached from state $$s$$ executing policy action $$a^\pi$$, weighted with the related state transition probabilities $$p$$:

$$V^\pi(s) = r(s, a^\pi) + \sum_{s'} p(s'|s, a^\pi) V^\pi(s')$$

Because the value of each state depends on the value of the related next state, it becomes the sum of future rewards collected while executing a given policy.

To prevent infinite state values, the sum of future rewards often gets discounted by a factor $$\gamma$$ smaller than 1:

$$V^\pi(s) = r(s, a^\pi, s') + \gamma \sum_{s'} p(s'|s, a^\pi) V^\pi(s')$$

The Q state-action value (state-action pair quality) related to state $$s$$ and action $$a$$ is defined as:

$$Q(s, a^\pi) = r(s, a^\pi, s') + \gamma \sum_{s'} p(s'|s, a^\pi) \max_{a'} Q(s', a')$$

which can be transformed into a $$Q$$ value 
[Temporal Difference (TD) Learning](https://en.wikipedia.org/wiki/Temporal_difference_learning)
rule with:

$$G^\pi = r(s, a^\pi, s') + \gamma \sum_{s'} p(s'|s, a^\pi) \max_{a'} Q(s', a')$$

and the learning rate $$\alpha$$, we get the following $$Q$$ value update rule:

$$Q_{t+1}(s, a^\pi) = (1 - \alpha) Q_{t}(s, a^\pi) + \alpha G^\pi_{t}$$

>The state transition probabilities are a property of the environment and must not be explicitly implemented by a TD learning algorithm. The transition probabilities are implicitly modelled (emerge) by doing a lot of averaging TD steps with the same state action pairs and probably different next states.

## TD Learning Algorithms

The following temporal difference learning algorithms are based on the Bellman Equation:

- [Q-Learning Algorithm](q_learning_algorithm.md)
- [SARSA Algorithm](sarsa_algorithm.md)
- [Expected SARSA Algorithm](expected_sarsa_algorithm.md)
- [Dyna-Q Algorithm](dyna_q_algorithm.md)

>Please see the [Gridworld Examples](gridworld_examples.md) for details.
