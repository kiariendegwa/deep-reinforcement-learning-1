# Multi-agent Tennis using Multi- Agent Deep Deterministic Policy Gradients (MADDPG).

## Goal
Given 2 agents interacting throught the game of Tennis, the goal of both agents is to maintain a rally for as long as possible.

## Challenge
This setting results in complex system dynamics that need to take into account the opposing/collaborating agents evolving policy.
For each succesful serve each player gets a reward of either +/-0.01. 
The environment is solved once a score of 0.5 is achieved across 100 consecutive episodes.

## The Tennis Environment
The environment is comprised of 8 state variables - describing position and velocity of the ball.
The action space is comprised of 2 actions - movement to/fro from the net and up/down.

## Moving parts of the MADDPG algorithm

## Overview
This is a general form and thereby extension of the [DDPG](http://proceedings.mlr.press/v32/silver14.pdf).
Each agent in the system is comprised of an actor-critic architecture, accompanied by a local and target network for gradient stabilization.

However, unlike prior implementations of of multi-agent DDPG - whereby the agents are entirely independant with seperate critic networks.; this alogrithm has a single critic network, shared across all agents within the system. This has the added advantage of simplyfying execution by allowing for centralized training and decentralized action. Consequently each agents policy is applied to its own state observation.

## Ornstein-Uhlenbeck noise
Each action carried out by the exploratory phase of the *actor* is subject to noise that is time correlated - allowing the agent to explore nearby states for longer durations.

## Experience replay
The agents share a single buffer. This is used to update their joint critic model. Thus allowing the agents to share a single store of memory although competing against each other - policy wise. Each corresponding buffer entry (s, a, r, s') is completed decorrelated by 
randomly sampling when training the critic. This allows for gradient stabilization; as it stops catastrophic forgetting from taking place as a stack based algorithm would be destabilized by time correlated sequential tuples.

### Neural net architecture of the agents

### Caveats:
* Actor and critic models use separate neural networks for both agents. Rather than as described in 
[MADDPG](https://arxiv.org/pdf/1706.02275.pdf). This design decision was made to make the project simpler and more intuitive.

## Future improvements:
