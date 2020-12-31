---
tags: python,numpy,neural-network,reinforcement learning
mathjax: true
---
# Monte Carlo (MC) RL Methods

- samples (state, action, reward) data from the environment following a given (usually epsilon-greedy or random) policy
- averages total future reward for each visited state over a number of complete episode trajectories
  - the averaging may be done in an incremental manner to limit the amount of memory needed by the algorithm
- the Monte Carlo algorithm requires the episodes to terminate to get valid future reward values without bootstrapping
- Exploring Starts choses a random trajectory start state to allow all states to appear in trajectories
- MDP graphs that include nodes from which a terminal state cannot be reached, cannot be handled by this algorithm which has the requirement that trajectories have to terminate

## Monte Carlo Example Implementation

Please see my [Svelte TD Learning Repository](https://github.com/maideas/svelte-td-learning) for the complete code - even though the Monte Carlo algorithm is not a TD algorithm - and the interactive [Gridworld Examples](gridworld_examples.md) for more information.

```javascript
let trajectory;

const MonteCarloQTableUpdate = () => {
  for (let y = 0; y < numY; y++) {
    for (let x = 0; x < numX; x++) {
      for (let a = 0; a < numA; a++) {
        let take = false;
        let g = 0.0;
        let gammaProduct = 1.0;

        for (let n = 0; n < trajectory.length; n++) {
          if (
            x == trajectory[n][0][0] &&
            y == trajectory[n][0][1] &&
            a == trajectory[n][1]
          ) {
            take = true;
          }
          if (take) {
            g += gammaProduct * trajectory[n][2];
            gammaProduct *= gamma;
          }
        }

        if (take) {
          // incremental average Q value update
          let q = (1.0 - alpha) * mazeComp.getQValue([x, y], a) + alpha * g;
          mazeComp.setQValue([x, y], a, q);
        }
      }
    }
  }
};

const runMonteCarloEpisodeStep = (state) => {
  let stateNext;
  let a, r;

  if (mazeComp.isTerminal(state)) {
    MonteCarloQTableUpdate();
    runEpisode();  // run next episode (calls runMonteCarloEpisode)
  } else {
    stepTimer = setTimeout(() => {
      a = mazeComp.getEpsilonGreedyAction(state, epsilon);
      [stateNext, r] = mazeComp.step(state, a);
      trajectory.push([state, a, r]);
      state = [...stateNext];
      runMonteCarloEpisodeStep(state);
    }, 0);
  }
};

const runMonteCarloEpisode = () => {
  trajectory = [];
  let state = mazeComp.getRandomStartState();
  runMonteCarloEpisodeStep(state);
};
```
