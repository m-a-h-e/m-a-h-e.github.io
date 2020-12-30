---
tags: python,numpy,neural-network,reinforcement learning
mathjax: true
---
# The SARSA Algorithm

- model free algorithm
- similar to Q learning algorithm, but samples reward based on policy and adds policy related Q value of new state
- SARSA algorithms are called on-policy, because the experiance used for learning is aquired following the current policy

## SARSA Example Implementation

Please see my [Svelte TD Learning Repository](https://github.com/maideas/svelte-td-learning) for the complete code and the interactive [Gridworld Examples](gridworld_examples.md) for more information.

```javascript
const SarsaQTableUpdate = (state, a, r, stateNext, aNext) => {
  let g;
  if (mazeComp.isTerminal(stateNext)) {
    g = r;
  } else {
    g = r + gamma * mazeComp.getQValue(stateNext, aNext);
  }
  let q = (1.0 - alpha) * mazeComp.getQValue(state, a) + alpha * g;
  mazeComp.setQValue(state, a, q);
};

const runSarsaEpisodeStep = (state, a) => {
  let stateNext;
  let aNext, r;
  if (mazeComp.isTerminal(state)) {
    runEpisode();  // run next episode (calls runSarsaEpisode)
  } else {
    stepTimer = setTimeout(() => {
      [stateNext, r] = mazeComp.step(state, a);
      aNext = mazeComp.getEpsilonGreedyAction(stateNext, epsilon);
      SarsaQTableUpdate(state, a, r, stateNext, aNext);
      state = [...stateNext];
      a = Number(aNext);
      runSarsaEpisodeStep(state, a);
    }, 0);
  }
};

const runSarsaEpisode = () => {
  let state = mazeComp.getRandomStartState();
  let a = mazeComp.getEpsilonGreedyAction(state, epsilon);
  runSarsaEpisodeStep(state, a);
};
```

