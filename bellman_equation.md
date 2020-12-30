---
tags: python,numpy,neural-network,reinforcement learning
mathjax: true
---
# The Bellman Equation

[The Bellman Equation](https://en.wikipedia.org/wiki/Bellman_equation) defines the value $$V^\pi(s)$$ of a given state as the sum of
- the immediate reward $$r(s, \pi(s))$$ executing one policy $$\pi$$ step from state $$s$$
- and the value $$V^\pi(s')$$ of the state $$s'$$ reached by executing the policy step

This results in the following bootstrapping state value evaluation rule

$$V^\pi(s) = r(s, \pi(s)) + V^\pi(s')$$

with the optimal policy definition

$$\pi(s) = arg \max_{a} Q(s, a) = a^\pi$$

we get

$$V^\pi(s) = r(s, a^\pi) + V^\pi(s')$$

for deterministic environments.

For stochastic environments, in which the same action executed in the same state may result in different next states, the transition probabilities have to be taken into account.
The next state value $$V^\pi(s)$$ becomes the sum over all next states, that can be reached from state $$s$$ executing action $$a$$, weighted by the related transition probabilities

$$V^\pi(s) = r(s, a^\pi) + \sum_{s'} p(s'|s, a^\pi) V^\pi(s')$$

Because the next state value also depends on the value of its next state, the state values become the sum of all future rewards collected while executing the given policy.
To prevent infinit state values in non-terminated environments, the future reward gets discounted by a factor $$\gamma$$ smaller than 1.0

$$V^\pi(s) = r(s, a^\pi, s') + \gamma \sum_{s'} p(s'|s, a^\pi) V^\pi(s')$$

The Q state-action value related to state $$s$$ and action $$a$$ is defined as

$$Q^\pi(s, a) = r(s, a^\pi, s') + \gamma \sum_{s'} p(s'|s, a^\pi) \max_{a'} Q^\pi(s', a')$$

which can be transformed into a $$Q$$ value 
[Temporal Difference (TD) Learning](https://en.wikipedia.org/wiki/Temporal_difference_learning)
rule with

$$G^\pi = r(s, a^\pi, s') + \gamma \sum_{s'} p(s'|s, a^\pi) \max_{a'} Q^\pi(s', a')$$

and learning rate $$\alpha$$

$$Q^\pi_{t+1}(s, a) = (1 - \alpha) Q^\pi_{t}(s, a) + \alpha G^\pi_{t}$$

>The state transistion probability values are a property of the environment and must not be explicitly implemented by a TD learning algorithm. The transition probabilities are implicitly modelled (emerge) by doing a lot of averaging TD steps with the same state action pairs and probably different next states.

