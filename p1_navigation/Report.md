# Summary
This report is based on a DQN that makes use of [Experience Replay](https://arxiv.org/abs/1712.01275) to the [Unity's](https://github.com/Unity-Technologies/ml-agents) Banana navigation task. 
The project made use of a state representation; summarized by a 36D Banana environment vector. Pixel level information was not used.
The final algorithm uses a DQN network with experience replay - similar to that described in the seminal paper [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602v1.pdf).

# Learning Algorithm
The main scripts employed are:
* ```dqn_agent.py```
* ```model.py```
* ```Navigation_Pixels.ipynb```


## 1. Experience Replay
Each trajectory (s_t, a_t, r_t, s_t+1) gotten from interaction with the agents environment is stored in a replay buffer. 
These transitions are then sampled from randomly during gradient updates of the DQN agent. 

This has a couple of advantages:
*   It Allows for smoothing the training distributions over past experiences, 
    by allowing each step to be sampled from; multiple times
    during weight updates.
*   Sampling the environment sequentially in time, results in strongly correlated samples caused by 
    time based environmental interactions. Randomly sampling these trajectories
    helps break this correlations and helps find more robust learning features.
*   Helps the agent get out poor action based feedback loops caused by falling into local minima. 
    This is done by getting an average over previous states and smoothing out learning and oscillations.

## 2. Epsilon Greedy
The DQN agent described above is Greedy in the limit with Infinite exploration (GLIE), as the epsilon greedy approach is used alongside annealing. Initially by annealinb epsilon from 1.0, at a rate of 0.995 after each episode.

# Rewards Result
The agent achieves a score of roughly 13.0 after 500 episodes as evidence by the graph below:
![plot of rewards](./results.png')
# Ideas for Future Work
The number of episodes required to learn the task, can most likely be improved by further augmenting the 
DQN agent with some of the following architectural hacks.
- [Prioritized Experienced Replay](https://arxiv.org/abs/1511.05952)
- [Dueling Network Architecture](https://arxiv.org/pdf/1511.06581.pdf)

# Trained model
The neural weightof the trained agent can be found in the link below
[Trained model (DDQN)](./checkpoint.pth)
