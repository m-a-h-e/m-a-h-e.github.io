---
tags: python,numpy,neural-network,reinforcement learning
mathjax: true
---
# The Q Learning Algorithm

- the Bellman Equation can be adapted to work as a policy optimization rule
- model free algorithm
- samples reward based on policy and adds maximum Q value of new state that may not be equal to the Q value related to the action chosen following the current behavior policy
- because of this discrepancy, Q learning is called an off-policy algorithm

## Q Learning Example Implementation

Please see my [Svelte TD Learning Repository](https://github.com/maideas/svelte-td-learning) for the complete code and the interactive [Gridworld Examples](gridworld_examples.md) for more information.

```javascript
const QLearningQTableUpdate = (state, a, r, stateNext) => {
  let g;
  if (mazeComp.isTerminal(stateNext)) {
    g = r;
  } else {
    g = r + gamma * mazeComp.getMaxQValue(stateNext);
  }
  let q = (1.0 - alpha) * mazeComp.getQValue(state, a) + alpha * g;
  mazeComp.setQValue(state, a, q);
};

const runQLearningEpisodeStep = (state) => {
  let stateNext;
  let a, r;
  if (mazeComp.isTerminal(state)) {
    runEpisode();  // run next episode (calls runQLearningEpisode)
  } else {
    stepTimer = setTimeout(() => {
      a = mazeComp.getEpsilonGreedyAction(state, epsilon);
      [stateNext, r] = mazeComp.step(state, a);
      QLearningQTableUpdate(state, a, r, stateNext);
      state = [...stateNext];
      runQLearningEpisodeStep(state);
    }, 0);
  }
};

const runQLearningEpisode = () => {
  let state = mazeComp.getRandomStartState();
  runQLearningEpisodeStep(state);
};
```

[Berkeley CS 294-112: Deep Reinforcement Learning (Advanced Q Learning)](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_7_advanced_q_learning.pdf)

