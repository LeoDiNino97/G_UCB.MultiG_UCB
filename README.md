# G_UCB.MultiG_UCB
![image](https://github.com/LeoDiNino97/G_UCB.MultiG_UCB/assets/93933755/297206be-3573-4d3b-bc0e-0be9331e6212)

This repository contains the reproduction of the results of the following papers: 
+ Multi-armed Bandit Learning on a Graph, Tianpeng Zhangy, Kasper Johansson, Na Li, 2023
+ Cooperative Multi-Agent Graph Bandits: UCB Algorithm and Regret Analysis, Phevos Paschalidis, Runyu Zhang, Na Li, 2024

The G-UCB algorithm is a new approach on classic Hoeffding-UCB methods for multi-armed bandits injecting dynamic constraints on the decision process modeled through a graph: the neighborhood of a given state represents the available actions. 

The procedure in the single agent scenario has strong convergence properties: same for the multi-agents generalization, even if the adjoint programs are not proven to be optimal.
The repository contains the following: 
  + ```core.py``` contains the classes defining the stochastic reward graph environment and 5 agents learning on it:
      + ```G-UCB``` is the procedure of interes
      + ```Local-UCB``` is a standard upper-confidence-bound adapted on a graph environment;
      + ```Local-TS``` is a standard Thompson sampling for bayesian bandits adapted on a graph environment;
      + ```G-QL``` is a graph Q-learning procedure;
      + ```G-QLucb``` is a graph Q-learning procedure with a bonus term derived from an upper-confidence bound estimate.
  + ```mag.py``` contains the classes defining the multi-agents generalization;
  + ```topologies.py``` contains 6 different topologies the 5 agents where tested on and the definition of a syntethic robotic environment;
  + ```plots.ipynb``` contains some visualization for the results;
  + ```report.pdf``` contains some more details on the procedures. 

## Bibliography 

[1] Multi-armed Bandit Learning on a Graph, Tianpeng Zhangy, Kasper Johansson, Na Li, 2023

[2] Cooperative Multi-Agent Graph Bandits: UCB Algorithm and Regret Analysis, Phevos Paschalidis, Runyu Zhang, Na Li, 2024

[3] Near-optimal regret bounds for reinforcement learning, T. Jaksch, R. Ortner, and P. Auer, Journal of Machine Learning Research, vol. 11, no. 51, pp. 1563–1600, 2010

[4] Is q-learning provably efficient?, C. Jin, Z. Allen-Zhu, S. Bubeck, and M. I. Jordan, in Advances in Neural Information Processing, vol. 31. Curran Associates, Inc., 2018

