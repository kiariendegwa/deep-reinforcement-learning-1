# Multi-agent Tennis using Multi- Agent Deep Deterministic Policy Gradients (MADDPG).

### Goal
Given 2 agents interacting throught the game of Tennis, the goal of both agents is to maintain a rally for as long as possible.
This setting results in complex system dynamics that need to take into account the opposing/collaborating agents evolving policy.
For each successful serve each player gets a reward of either +/-0.01. The environment is solved once a score of 0.5 is achieved across 100 consecutive episodes.

### The Tennis Environment
* The environment is comprised of 8 state variables - describing position and velocity of the ball.
* The action space is comprised of 2 actions - movement to/fro from the net and up/down.

## Moving parts of the MADDPG algorithm

### Refresher on Actor-critic methods

Policy-based methods - estimate optimal policy directly from environmental states using gradient ascent. 
This makes for an agent, that is high bias and a low variance.
 
On the other hand value-based methods - estimate optimal policy from Q-table of state-action pairs using gradient descent. 
This makes for high variance and low bias.

An actor-critic method, takes the best of both worlds by combining the strengths of both models.
Thereby resulting in a system with low bias and low variance.

### MADDPG Overview

This is a general form and subsequently an extension of the [DDPG](http://proceedings.mlr.press/v32/silver14.pdf).
Each agent in the system is comprised of an actor-critic architecture, accompanied by a local and target network for gradient stabilization.

However; unlike prior implementations of of multi-agent DDPG - whereby the agents are entirely independent with 
separate critic networks; this algorithm has a single critic network, shared across all agents within the system. 
This has the added advantage of simply-fying execution by allowing for centralized training and decentralized action. Consequently each agents policy continues to be applied based on its own state observation.

### Ornstein-Uhlenbeck noise
Each action carried out by the exploratory phase of the *actor* is subject to noise that is time correlated - 
allowing the agent to explore nearby states for longer durations.

### Experience replay
The agents share a single memory buffer. This is used to update their joint critic model. Thus allowing the agents to share a single store of memory although competing against each other - policy wise. Each corresponding buffer entry (s, a, r, s') is completed decorrelated by 
randomly sampling when training the critic. This allows for gradient stabilization; as it stops catastrophic forgetting from taking place as a stack based algorithm would be destabilized by time correlated sequential tuples.

#### Neural-net architecture of the agents
All agents - 2 agents (2 actor networks, 1 critic) and their accompanying local and target networks). Share a single neural architecture comprised of:
* 2 fully connected NNs with ReLu activation
* Actor LR: 2e-4, Critic LR: 2e-4
* Adam Optimizer
* Tau: 10^-3
* Batch size: 128
* Replay buffer: 1e5
* Weight decay: 0 
* Gamma: 0.99

## Final trained algorithm curves and weights
The algorithm converges after ~2600 episodes as shown in the graphs below:
![Optimization graphs](./score_graphs.jpg) 

The resultant agents model weights can be found here:

* [actor_agent 0](checkpoint_actor_agent_0.pth)
* [critic agent 0](checkpoint_critic_agent_0.pth)
* [actor agent 1](checkpoint_actor_agent_1.pth)
* [critic agent 1](checkpoint_critic_agent_1.pth)

## Future improvements:
A number of different tweaks to the algorithm could be applied to improve sample efficiency. Below is a list of the most 
immediate fixes I would attend to:

* Prioritized experience replay: 
  
  This could improve convergence speed.

* Further Hyper-parameter tuning: 
  
  There are a lot of hyper-parameters in this model. Maybe using some meta-learning approach could be used to solve this.
