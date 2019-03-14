# Multi-agent Tennis using Multi- Agent Deep Deterministic Policy Gradients (MADDPG).

## Goal
Given 2 agents interacting throught the game of Tennis, the goal of both agents is to maintain a rally for as long as possible.

## Challenges
This setting results in complex system dynamics that need to take into account the opposing/collaborating agents evolving policy.
For each succesful serve each player gets a reward of either +/-0.1. 

## The Tennis Environmnet
The environment is comprised of 8 state variables - describing position and velocity of the ball.
The action space is comprised of 2 actions - movement to/fro from the net and up/down.

## Overview of the MADDPG algorithm
Each agent is comprised of an actor-critic architecture. 
Whereby the actor is comprised of function parameterized by &thetasym;

### Neural net architecture of the agents

### Caveats:
Actor and critic models use separate neural networks for both agents. Rather than as described in 
[MADDPG](https://arxiv.org/pdf/1706.02275.pdf). This design decision was made to make the project simpler and more intuitive.

## Future improvements:
