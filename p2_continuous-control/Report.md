# Reacher Report
This report is based on a DDPG algorithm for a continuous control task - Reacher task; whereby the goal is to teach a robot arm(with 2 d.o.f.) how to follow a randomly moving target. The final algorithm is based on the paper: [Continuous control with Deep RL](https://arxiv.org/pdf/1509.02971.pdf). The implemented algorithm is heavily based of the exercises within Udacity's Continuous Control section.

## Learning Algorithm particulars
The main scripts employed are:
* ```ddpg_agent.py```:

    Contains the Experience Replay buffer, and Deep Q network logic. This has the hyper-parameters:
    1. BUFFER_SIZE = 1e5,
    2. BATCH_SIZE 128,
    3. GAMMA = 0.99,
    4. TAU=1e-3,
    5. LR_ACTOR = 2e-4,
    6. LR_CRITIC = 2e-4
    7. WEIGHT_DECAY = 0

* ```model.py```:

    1. Contains 2 similar small 3 layer MLPs using RELU activations (one for the Actor, the other the Critic).

    2. The last and second last layers are passed through a batch normalization.
    
    3. The first two layers make use of RELU activation functions.
    
    4. The last layers of these MLPs make use of TANH activation functions.

* ```Continuous_Control.ipynb```

    Stitches together all the moving parts and initiates the Agents training.

### Unique model architecture details
The DDPG Actor-critic model is comprised of 4 neural networks, these are:
1. The actor
2. The critic
3. The actor target network
4. The critic target network
5. A replay buffer
6. Soft weight updates
7. Ornstein Uhlenbeck noise

### How the model learns
1. The actor and critic network, alongside their targets networks are randomly initialized.
2. During an episode at a time step, the actor network is given the current state and returns a value that is added to the Ornstein noise. This is then stored in the Replay Buffer in the form of a tuple (State, Action, Reward, NextState). 
4. The critic network is evaluated at the new state with an action given by the actor network evaluated at the new state. 
5. Should the replay buffer be 'full' - a weight update through gradient ascent is instatiated. Both target networks are updated, first the critic then the actor. This takes form of the mean squared error between the expected Q value and the actual Q value during a transition. This guides the actor network toward a better policy.
6. After we have updated both our target networks we use soft updates to update our main actor critic networks.

# Rewards Result
The agent achieves a score of roughly 30 after roughly 700 episodes as evidenced by the graph below:
![Reward Plots](./score_graphs.jpg)

# Ideas for Future Work
Continuous state space problems have been solved by other creative problems such as PPO, A3C, or D4PG.
Each of these have the added advantage of parallelizing training across multiple agents, thereby speeding up
the accumulation of experience with which to train the actor-critic methods.

# Trained model
The neural weights of the trained actor-critic agent can be found in the links below
Trained model weights (DDPG) - [actor](./checkpoint_actor.pth), [critic](./checkpoint_critic.pth)
