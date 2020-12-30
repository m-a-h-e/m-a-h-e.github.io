---
tags: python,numpy,neural-network,reinforcement learning
mathjax: true
style:
  - src: https://maideas.github.io/svelte-td-learning/build/bundle.css
script:
  - src: https://cdn.plot.ly/plotly-latest.min.js
  - src: https://maideas.github.io/svelte-td-learning/build/bundle.js
    flags: defer
---
# Temporal Difference (TD) Learning

[Temporal Difference Learning](https://en.wikipedia.org/wiki/Temporal_difference_learning)

- algorithm for state value or state-action value update after each temporal step (temporal difference)

## The Bellman Equation

[The Bellman Equation](https://en.wikipedia.org/wiki/Bellman_equation) defines the value $$V(s)$$ of a given state as the sum of the immediate reward $$r(s, \pi(s))$$ executing a given policy $$\pi$$ and the value $$V(s')$$ of the state reached by executing the policy.

This gives a bootstrapping state value evaluation rule

$$V^\pi(s) = r(s, \pi(s)) + V^\pi(s')$$

with the policy $$\pi(s) = a$$

$$V(s) = r(s, a) + V(s')$$

for deterministic environments.

For stochastic environments, in which the same action executed in the same state may result in different next states, the transition probabilities have to be taken into account.
The next state value $$V(s)$$ becomes the sum over all next states, that can be reached from state $$s$$ executing action $$a$$, weighted by the related transition probabilities

$$V(s) = r(s, a) + \sum_{s'} p(s'|s, a) V(s')$$

Because the next state value also depends on the value of its next state, the state values become the sum of all future rewards collected while executing the given policy.
To prevent infinit state values in non-terminated environments, the future reward gets discounted by a factor $$\gamma$$ smaller than 1.0

$$V(s) = r(s, a, s') + \gamma \sum_{s'} p(s'|s, a) V(s')$$

The optimal Q state-action value related to state $$s$$ and action $$a$$ is defined as

$$Q^*(s, a) = r(s, a, s') + \gamma \sum_{s'} p(s'|s, a) \max_{a'} Q^*(s', a')$$


## The Q Learning Algorithm

- the Bellman Equation can be adapted to work as a policy optimization rule
- model free algorithm
- samples reward based on policy and adds maximum Q value of new state that may not be equal to the Q value related to the action chosen following the current behavior policy
- because of this discrepancy, Q learning is called an off-policy algorithm

### Q Learning Example Implementation

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

## The SARSA Algorithm

- model free algorithm
- similar to Q learning algorithm, but samples reward based on policy and adds policy related Q value of new state
- SARSA algorithms are called on-policy, because the experiance used for learning is aquired following the current policy

### SARSA Example Implementation

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

## The Expected SARSA Algorithm

- model free algorithm
- same as SARSA algorithm, but in addition, takes action sampling probabilities into account

### Expected SARSA Example Implementation

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

## The Dyna-Q Algorithm

- model based temporal difference algorithm which uses real and simulated experiance
  - the model implements a (state, action) to (new state, reward) mapping
  - real experiance is sampled from the environment
  - simulated experiance is generated using already collected real experiance
  - both real and simulated experiance are used to optimize the value function and policy of the agent

### Dyna-Q Example Implementation

```javascript
let envModel =
  Array.from({ length: numX }, () =>
    Array.from({ length: numY }, () =>
      Array.from({ length: numA }, () => null
    )
  )
);
let seenStateActions = [];

const DynaQModelUpdate = (state, a, r, stateNext) => {
  let x = state[0];
  let y = state[1];
  let seen = false;
  for (let n = 0; n < seenStateActions.length; n++) {
    if (
      seenStateActions[n][0] == x &&
      seenStateActions[n][1] == y &&
      seenStateActions[n][2] == a
    ) {
      seen = true;
      break;
    }
  }
  if (!seen) {
    seenStateActions.push([state, a]);
  }
  envModel[x][y][a] = [stateNext, r];
};

const DynaQGetModelStateAction = () => {
  let i = mazeComp.getRandomInt(seenStateActions.length);
  return seenStateActions[i];
};

const runDynaQEpisodeStep = (state) => {
  let stateNext;
  let a, r;
  if (mazeComp.isTerminal(state)) {
    runEpisode();  // run next episode (calls runDynaQEpisode)
  } else {
    stepTimer = setTimeout(() => {
      a = mazeComp.getEpsilonGreedyAction(state, epsilon);
      [stateNext, r] = mazeComp.step(state, a);
      QLearningQTableUpdate(state, a, r, stateNext);
      DynaQModelUpdate(state, a, r, stateNext);
      state = [...stateNext];

      // planning steps (model based Q table update steps)
      for (let n = 0; n < planningSteps; n++) {
        let mState;
        let ma, mr;
        let mStateNext;
        [mState, ma] = DynaQGetModelStateAction();
        [mStateNext, mr] = envModel[mState[0]][mState[1]][ma];
        QLearningQTableUpdate(mState, ma, mr, mStateNext);
      }
      runDynaQEpisodeStep(state);
    }, 0);
  }
};

const runDynaQEpisode = () => {
  let state = mazeComp.getRandomStartState();
  runDynaQEpisodeStep(state);
};
```

## Additional Information

[Google DeepMind AI Safety Gridworlds Paper](https://arxiv.org/pdf/1711.09883.pdf)

## Maze Example Implementation

<div id="maze-shell-1"></div>
<div id="maze-shell-2"></div>
<div id="maze-shell-3"></div>
<div id="maze-shell-4"></div>

