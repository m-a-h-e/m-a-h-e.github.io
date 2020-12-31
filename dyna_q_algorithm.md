---
tags: python,numpy,neural-network,reinforcement learning
mathjax: true
---
# The Dyna-Q Algorithm

- model based temporal difference algorithm which uses real and simulated (planned) experience
  - the model implements a (state, action) to (new state, reward) mapping
  - real experience is sampled from the environment
  - simulated experience is generated using already collected real experience
  - both real and simulated experience are used to optimize the value function and policy of the agent

## Dyna-Q Example Implementation

Please see my [Svelte TD Learning Repository](https://github.com/maideas/svelte-td-learning) for the complete code and the interactive [Gridworld Examples](gridworld_examples.md) for more information.

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

