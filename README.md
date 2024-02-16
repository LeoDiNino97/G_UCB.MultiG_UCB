# G_UCB.MultiG_UCB

This repository contains the reproduction of the results of the following papers: 
+ Multi-armed Bandit Learning on a Graph, Tianpeng Zhangy, Kasper Johansson, Na Li, 2023
+ Cooperative Multi-Agent Graph Bandits: UCB Algorithm and Regret Analysis, Phevos Paschalidis, Runyu Zhang, Na Li, 2024

The G-UCB algorithm is a new approach on classic Hoeffding-UCB methods for multi-armed bandits injecting dynamic constraints on the decision process. 
Specifically, if we embed both the state space and the action space on the nodes set of a graph, we can model a policy $pi(s_t) \in \mathcal{N}_{s_t}$: in other words the available actions given a certain state are the ones encoded in neighborhood of the current state. 
