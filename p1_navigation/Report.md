# Summary
This report is based on a DQN that makes use of Experience replay to the Banana navigation task. 
The project made use of a 37D state representation summarized by the Banana environment. Pixel level information was not used.
The final algorithm uses a DQN network with experience replay - similar to that described in the seminal paper [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602v1.pdf).

# Learning Algorithm
The main scripts employed are:
* ```dqn_agent.py```:

    Contains the Experience Replay buffer, and Deep Q network logic. This has the hyper-parameters:
    1. BUFFER_SIZE = 1e5,
    2. BATCH_SIZE 64,
    3. GAMMA = 0.99,
    4. TAU=1e-3,
    5. LR=5e-4,
    6. UPDATE_EVERY=4

* ```model.py```:

    1. Contains a simple 3 layer MLP using RELU activations.

    2. The MLPs input layer has an input states size of 37 
    
    3. and its output layer is comprised of size 4 given the action state space.

* ```Navigation_Pixels.ipynb```

    Stitches together all the moving parts and initiates the Agents training.


## 1. Experience Replay
Each trajectory (s_t, a_t, r_t, s_t+1) gotten from interaction with the agent's environment is stored in a replay buffer. 
These transitions are then sampled from randomly during gradient updates of the DQN agent. 

This has a couple of advantages:
*   It Allows for smoothing the training distributions over past experiences, 
    by allowing each step to be sampled from multiple times
    during weight updates.
*   Sampling the environment results in strongly correlated samples caused by 
    time based environmental interactions. Randomly sampling these trajectories
    helps break this correlations.
*   Helps the agent get out poor action based feedback loops caused by falling into local minima. 
    This is done by getting an average over previous states and smoothing out learning and oscillations.

## 2. Epsilon Greedy
The DQN agent described above is Greedy in the limit with Infinite exploration (GLIE), as the epsilon greedy approach is used by annealing the epsilon function initially from 1.0, at a rate of 0.995 after each episode.

# Rewards Result
The agent achieves a score of roughly 13.0 after 500 episodes as evidence by the graph below:
![Reward Plots](results.png)
# Ideas for Future Work
The number of episodes required to learn the task, can most likely be improved by further augmenting the 
DQN agent with some of the following architectural hacks.
- [Prioritized Experienced Replay](https://arxiv.org/abs/1511.05952):
This has the added advantage of allowing the model to prioritize certain experience more often - rather than randomly sampling experiences from out replay buffer. This could lead to faster more robust convergence.
- [Dueling Network Architecture](https://arxiv.org/pdf/1511.06581.pdf)
In order for the model to learn how best to evaluate a state in the Q-table, it must be able to estimate two primitives. Namely: given a state, the advantage of taking an action given that state, and the state value. This is represented by the equation:
```Q(s, a) = A(a, s)+V(s)``` This allows the model to decouple these two state aspects by redefining the training objective as a multi-class problem. I.e. we can learn what states are valuable, and what actions are valuable given such states.

# Trained model
The neural weights of the trained agent can be found in the link below
[Trained model (DQN)](./checkpoint.pth)
