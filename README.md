# Experimenting the impact of stochasticity and policy choice on TD(0) Agents behavior

**Please see Experiments_Report.pdf where we explain our work**  
---

* Authors: Lahrach Salma, Neveu Pierre
* Course: Reinforcement Learning
* University: PSL - Paris Dauphine

## Introduction

The objective of this project is to implement Time Difference (TD) learning
agents in environments of different complexities. We took advantage of this project
to learn how to design a code architecture that ensures flexibility in the context of im
plementing reinforcement learning agents. Therefore, we aimed for the most modular
implementation possible and it allowed us for dynamic integration of new agents, poli-
cies, environments, and experimental setups.  

Initially, we focused on Q-learning and SARSA for Mountain Car. An interpretation
of these algorithms would be that Q-Learning, as an off-policy one, is not realistic / too
optimistic because we cannot ensure that he learns what would happen in reality.
However, we quickly observed that in Mountain Car, their value updates are very sim-
ilar. This can be attributed to the use of an ε-greedy policy in such environment as
they require a small ε, which results in similar action selection and update across both
algorithms (if ε = 0.01, SARSA will select the argmax action 99% of the time and both al-
gorithms will have a very similar behavior). Moreover, Mountain Car is a deterministic
environment. All of this preliminary observations have led us to ask ourselves:
1. What is the impact of the stochasticity of the environment on the behavior of SARSA
and Q-Learning ?
2. How does the policy used influence their behavior ?  

The set of possible experiments that could answer these questions is very large so we will
focus on comparing ε-greedy policy and softmax policy. As Mountain Car doesn’t
allow much the randomness to be varied, we will use the Cliff Walking environment and
modify its implementation to vary the slip factor. We optimized our agents on both
environments under both policies.  

We will first present our results of their behavior in the complex but mostly deterministic
environment of Mountain Car.  
We will then try to identify, with the Cliff Walking environment, to what extent policy
used the stochasticity of the environment influence them.