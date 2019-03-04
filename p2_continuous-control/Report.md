# Reacher Report
This report is based on a DDPG algorithm for a continuous control task, whereby the goal is to teach a robot arm(with 2 d.o.f.) how to 
follow a randomly moving target. The final algorithm is based on the paper: [Continuous control with Deep RL](https://arxiv.org/pdf/1509.02971.pdf). The final algorithm that resulted in the following training graph was heavily based of the exercises within Udacity's Continuous Control section.

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

# Unique model architecture details
The DDPG Actor-critic model has the following unique adventageous designs

DDPG is an approach for reinforcement learning environments where an agent uses a policy as a function that has a probability of taking an action given a state. For following through with this action, the agent receives a reward. A policy is said to be good if it results in a large reward over an episode of the environment. The optimal policy is the policy that maximizes the the reward obtained while following the policy. This environment being continuous would make using a Q-Learning Network approach fail where instead you are trying to figure out what is the best specific action. Instead we use two neural networks to approximate two values. The first network is called the Actor which is used to approximate the optimal policy where as the second network is the Critic which tries to estimate the reward from following that approximately optimal policy. This becomes a “try the policy” then “evaluate the policy” then “improve the policy” loop where the actor tries the policy and the critic evaluates the policy. The improving step comes from the actor and critic network updating through their loss functions.

This is the general structure of what the method uses but there are some additional moving parts behind the scenes. This implementation uses Ornstein Uhlenbeck noise, a replay buffer, target networks, and soft updating.

When first starting the training process, the actor and critic network are randomly initialized. During an episode at a time step, the actor network is given the current state and returns a value that is added to the Ornstein noise. This action is taken giving a new state as well as a reward for taking the previous action. This is then stored in the Replay Buffer in the form of a tuple (State, Action, Reward, NextState). Once the replay buffer has enough transitions, a random sample of them is taken and is used to help the critic network. The critic network is evaluated at the new state with an action given by the actor network evaluated at the new state. This value can be thought of as an approximation of the next reward from taking the next state, or expected Q value. Then both networks are updated, first the critic then the actor. This takes form of the mean squared error between the expected Q value and the actual Q value during a transition. The actor uses gradient ascent to update the actor network towards a better policy. Another more subtle issue that is run into is that if we update our networks every chance we get, it has the potential to become very unstable. To mitigate this problem, we will actually use two neural networks for both the actor and critic, although in the end we only care about one. The networks are our main actor network, main critic network, our target actor network and our target critic network. The target networks are held static for a fixed number of training steps while training the main networks as the goal of what the main network should be working towards. When the target networks are updated, they are pushed towards the main networks but do not get replaced completely by the main networks. This is called soft updating and makes things run much more smoothly.

# Rewards Result
The agent achieves a score of roughly 30 after roguhly 700 episodes as evidence by the graph below:
![Reward Plots](./score_graphs.jpg)

# Ideas for Future Work
Continuous state space problems have been solved by other creative problems such as PPO, A3C, or D4PG.
Each of these have the added advantage of parallelizing training across multiple agents, thereby speeding up
the accumulation of experience with which to train the actor-critic methods.

# Trained model
The neural weights of the trained actor-critic agent can be found in the links below
Trained model weights (DDPG) - [actor](./checkpoint_actor.pth), [critic](./checkpoint_critic.pth)
