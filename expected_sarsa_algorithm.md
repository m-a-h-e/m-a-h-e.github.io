---
tags: python,numpy,neural-network,reinforcement learning
mathjax: true
---
# The Expected SARSA Algorithm

- model free algorithm
- same as SARSA algorithm, but in addition, takes action sampling probabilities into account

## Expected SARSA Example Implementation

Please see my [Svelte TD Learning Repository](https://github.com/maideas/svelte-td-learning) for the complete code and the interactive [Gridworld Examples](gridworld_examples.md) for more information.

```javascript
const ExpectedSarsaQTableUpdate = (state, a, r, stateNext, aNext) => {
  let g;
  if (mazeComp.isTerminal(stateNext)) {
    g = r;
  } else {
    let vNextExpected = 0.0;
    let prob;
    // aNext is not used to calculate the expected next state value

    // each action has a base probability to be selected of epsilon divided
    // by the number of actions (random action selection case of e-greedy)
    prob = epsilon / numA;
    for (let a = 0; a < numA; a++) {
      vNextExpected += prob * mazeComp.getQValue(stateNext, a);
    }

    // the maximum Q action has a probability of (1 - epsilon)
    // to be selected (greedy action selection case of e-greedy)
    prob = 1.0 - epsilon;
    let aNextGreedy = mazeComp.getPolicy(stateNext);
    vNextExpected += prob * mazeComp.getQValue(stateNext, aNextGreedy);

    // G with respect to expected value V of next state
    g = r + gamma * vNextExpected;
  }
  let q = (1.0 - alpha) * mazeComp.getQValue(state, a) + alpha * g;
  mazeComp.setQValue(state, a, q);
};

const runExpectedSarsaEpisodeStep = (state, a) => {
  let stateNext;
  let aNext, r;

  if (mazeComp.isTerminal(state)) {
    runEpisode();  // run next episode (calls runExpectedSarsaEpisode)
  } else {
    stepTimer = setTimeout(() => {
      [stateNext, r] = mazeComp.step(state, a);
      aNext = mazeComp.getEpsilonGreedyAction(stateNext, epsilon);
      ExpectedSarsaQTableUpdate(state, a, r, stateNext, aNext);
      state = [...stateNext];
      a = Number(aNext);
      runExpectedSarsaEpisodeStep(state, a);
    }, 0);
  }
};

const runExpectedSarsaEpisode = () => {
  let state = mazeComp.getRandomStartState();
  let a = mazeComp.getEpsilonGreedyAction(state, epsilon);
  runExpectedSarsaEpisodeStep(state, a);
};
```

