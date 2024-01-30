import numpy as np 
import networkx as nx
from topologies import * 

import gurobipy as gp
from gurobipy import GRB

'''
The environment for the implementation is a syntehtic robotic environment for a drone distributing internet connection. 
The parameters are initialized according to the paper's synthetic simulations.
'''

class Multi_GUCB:
    def __init__(self, K=10, N=5, max_iter = 1.5e5):
        # Initialize the clock and the hard upper bound on the number of iterations
        self.t = 1
        self.max_iter = int(max_iter)

        # Initialize the number of arms and a list of arms
        self.K = K
        self.nodes = np.array(list(range(K)))

        # Define the players
        self.N = []
        for i in range(N):
            self.N.append("P"+str(i))
        self.N = np.array(self.N)

        '''
        # Grid topology
        _, edges = grid(self.K) 
        self.E = edges
        '''
        
        # Define the topology 
        self.E = [
            (0,1),
            (0,2),
            (0,3),
            (1,3),
            (1,4),
            (2,3),
            (2,5),
            (2,6),
            (3,5),
            (3,4),
            (4,5),
            (4,8),
            (5,6),
            (5,7),
            (5,8),
            (6,7),
            (7,8),
            (7,9)
            ]
        
        for i in range(self.nodes.shape[0]):
            self.E.append((i,i))

        '''
        # Erdos-Ranyi random graph
        self.A = np.zeros((self.K,self.K))
        for i in range(self.K):
            for j in range(i, self.K):    
                edge = np.random.binomial(1,0.05,1)
                if edge == 1:
                    self.A[i,j] = edge
                    self.A[j,i] = edge
                    self.E.append((i,j))
        '''
        
        #_________Graph representation___________
        self.A = np.zeros((self.K,self.K))                                                                  # Adjacency matrix
        for i in range(self.K):
            for j in range(self.K):
                if (self.nodes[i],self.nodes[j]) in self.E:
                    self.A[i,j] = 1
                    self.A[j,i] = 1

        self.D = np.diag(np.sum(self.A, axis=1))                                                            # Degree matrix
        self.L = self.D - self.A      
        '''
        # Enforcing connectivity
        stop = False
        while not stop:
            self.D = np.diag(np.sum(self.A, axis=1))                                                            # Degree matrix
            self.L = self.D - self.A                                                                   # Laplacian matrix
            eigs = np.linalg.eigvals(self.L)
            connected = (np.sum(np.isclose(eigs, 0.0, atol=1e-10)) == 1).all()

            if not connected:
                i,j = np.random.choice(range(0,self.K),2,replace=False)
                self.A[i,j] = 1
                self.A[j,i] = 1
            else:
                stop = True
        '''
        self.adjList = {u:[v for v in self.nodes if (u,v) in self.E or (v,u) in self.E] for u in self.nodes}        # Adjacency lists
        #________________________________________


        self.meanRewards = {node:np.random.uniform(0.25,0.75) for node in self.nodes}
        self.mean_rewards = np.array(list(self.meanRewards.values()))

        # The initial position of each agent is a random node on the graph
        self.tracker = {agent:np.random.choice(self.nodes) for agent in self.N}
        self.dualTracker = {node:[agent for agent in self.N if self.tracker[agent] == node] for node in self.nodes}

        # Agent register
        self.samples_register = {node:[self.get_reward(node) for _ in self.N] for node in self.nodes}

        # Agent counter 
        self.C = np.array([len(self.dualTracker[node]) for node in self.nodes])
                
        # Samples counter 
        self.samples_counter = np.ones(self.nodes.shape[0])

        # Regret tracker
        self.regret = np.zeros(self.max_iter)


    def get_reward(self,node):
        return np.random.normal(self.meanRewards[node],np.sqrt(0.06))
    
    def set_tracker(self, agent, node):
        '''
        This method updates the tracking attributes according to a certain action for a certain agent
        '''
        prev = self.tracker[agent]
        self.tracker[agent] = node
        self.dualTracker[prev].remove(agent)
        self.dualTracker[node].append(agent)
        self.C = np.array([len(self.dualTracker[node]) for node in self.nodes])

    def set_multiset(self, C):
        '''
        This method outputs a multiset of arms according to a certain vector of agents allocation
        '''
        # Arm multiset
        S = []
        for node in self.nodes:
            for _ in range(C[node]):
                S += [node]

        return S
    
    def confidence_radius(self):
        return np.sqrt(2*np.log(self.t)/self.samples_counter)
    
    def get_UCB(self):
        U = np.zeros(self.nodes.shape[0])
        b = self.confidence_radius()
        for i in range(self.nodes.shape[0]):
            U[i] = np.mean(np.array(self.samples_register[i])) + b[i]
        return U
    
    def cost_setter(self):
        W = {edge:0 for edge in self.E}
        U = self.get_UCB()
        for edge in self.E:
            W[edge] = np.max(U - U[edge[1]])
        return W

    def weight_function(self, x, k):
        '''
        This method defines the weighted sum of the rewards as a non-decreasing concave function with f(0) = 0, f(1) = 1
        '''
        return np.log(x*(2+k)/20 + 1)/np.log((2+k)/20 + 1)
    
    def set_policies(self):
        '''
        We need for each agent the minimum path arborescence rooted in its origin node
        '''
        sp_trees = {agent:[] for agent in self.N}
        
        for agent in self.N:
            for destination in [node for node in self.nodes]:
                tot_cost, policy, _ = self.dijkstra(self.tracker[agent], destination)
                d = {}
                d['Origin'] = self.tracker[agent]
                d['Destination'] = destination
                d['Cost'] = tot_cost
                d['Policy'] = policy

                sp_trees[agent].append(d)
        
        return sp_trees
    
    def allocation_optimizer(self):
        '''
        The optimization problem to efficiently dispatch the agents over the graph has been implemented in Gurobi through a piecewise linear approximation 
        resulting in a mixed integer linear program
        '''
        # Initialize a Gurobi environment
        opt = gp.Model("allocation") 

        # Get the UCB as a surrogate of the true mean
        U = self.get_UCB()

        # By construction these are the bounds over the decision variables
        lob = 0
        upb = len(self.N)

        variable_names = ['c'+str(k) for k in range(self.K)]

        # A map for the variables and their name
        variables = {
            name_: opt.addVar(
                vtype = GRB.INTEGER, 
                name = name_, 
                lb=lob,
                ub=upb
            ) for name_ in variable_names
            }

        # Piecewise linear approximation
        for k in range(self.K):

            npts = 1000
            pts = []
            f_lines = []

            for i in range(npts):
                p = lob + (upb - lob) * i / (npts - 1)
                pts.append(p)

                # By default Gurobi is a minimizer, so we append the negated values to maximize the objective function
                f_lines.append(-self.weight_function(p,k)*U[k])

            opt.setPWLObj(variables['c'+str(k)], pts, f_lines)

        # The only constrain is that the sum of the vector C must be equal to the number of agents
        opt.addConstr(gp.quicksum(variables[name] for name in variable_names) == len(self.N), 
                      name = 'consistency')
        
        opt.setParam('OutputFlag', 0)

        opt.optimize()
        
        # This populates a vector with the optimal solution 
        C_star = []
        for _, var in variables.items():
            C_star.append(int(var.x))

        return np.array(C_star)


    def adjoint_program(self, Ce):
        '''
        This method realize the shortest path + minimum weighted matching over the induced complete bipartite graph
        '''
        # We need a complete bipartite graph between the set of agents and the multiset of arms accounting for their multiplicity
        # Reimplement the logic for multiset of nodes in the complete bipartite graph

        S = self.set_multiset(Ce)

        # This dictionary handles a mapping between a set of unique identifiers for the nodes in the new complete bipartite graph and the nodes in the multiset
        Se = {i:node for i, node in enumerate(S)}

        # Building the complete bipartite graph
        GB = nx.Graph()
        GB.add_nodes_from(self.N, bipartite=0)
        GB.add_nodes_from(list(Se.keys()), bipartite=1)
        
        # Retrieve the policies for the agents and each node in the optimal allocation
        sp_trees = self.set_policies()
        edges_weights = []
        for agent in self.N:
            for k in Se.keys():
                edges_weights.append((agent,
                                      k,
                                      sp_trees[agent][Se[k]]['Cost']))
        
        GB.add_weighted_edges_from(edges_weights)
        min_W_matching =  nx.min_weight_matching(GB, weight='weight')
        policies = {agent:None for agent in self.N}

        for tup in min_W_matching:
            agent = tup[0] if isinstance(tup[0],str) else tup[1]
            assigned = Se[tup[1]] if isinstance(tup[0],str) else Se[tup[0]]
            ps = sp_trees[agent]

            d = list(filter(lambda x: x['Destination'] == assigned, ps))
            policies[agent] = d[0]

        return policies


    def get_neighboors(self,s):
        return self.adjList[s]
    

    def dijkstra(self, src, dst):
        '''
        A shortest path Dijkstra method
        '''
        distance = np.full(self.K, np.inf)
        distance[src] = 0

        pred = {node:None for node in self.nodes}

        Q = {node:np.inf for node in self.nodes}
        Q[src] = 0
        
        W = self.cost_setter()
        while len(Q) != 0:
            curr = min(Q, key=Q.get)
            for neighbour in self.get_neighboors(curr):
                temp = distance[curr] + W[(curr,neighbour)] if (curr,neighbour) in self.E else distance[curr] + W[(neighbour,curr)]
                if temp < distance[neighbour]:
                    distance[neighbour] = temp
                    pred[neighbour] = curr
                    Q[neighbour] = temp
            del(Q[curr])
            
        path = []
        tot_cost = distance[dst]

        pointer = dst
        while pointer != src:
            path.insert(0,(pred[pointer],pointer))
            pointer = pred[pointer]
        return tot_cost, path, pred   
    
    def main_program(self):
        '''
        The main method for the class handling the learning phase
        '''
        Ce = self.allocation_optimizer()

        idxs = np.where(Ce > 0)[0]
        k_min = idxs[np.argmin(Ce[idxs])]
        n_min = int(self.samples_counter[k_min])
        policies = self.adjoint_program(Ce)
        
        # Main episode - Each agent traverse to the assigned node
        ep = max(list(map(lambda x: len(x[1]['Policy']), policies.items())))

        for j in range(ep):
            Rs = np.zeros(self.K)
            for i in range(self.N.shape[0]):
                agent = self.N[i]
                pi = policies[agent]['Policy']
                if len(pi) == 0:
                    x_kt = self.get_reward(self.tracker[agent])
                    if Rs[self.tracker[agent]] == 0:
                        self.samples_register[self.tracker[agent]].append(x_kt)  
                    Rs[self.tracker[agent]] = self.meanRewards[self.tracker[agent]] if Rs[self.tracker[agent]] == 0 else Rs[self.tracker[agent]]
                else: 
                    self.set_tracker(agent, pi[0][1])
                    policies[agent]['Policy'].pop(0)
                    x_kt = self.get_reward(self.tracker[agent])
                    if Rs[self.tracker[agent]] == 0:
                        self.samples_register[self.tracker[agent]].append(x_kt)  
                    Rs[self.tracker[agent]] = self.meanRewards[self.tracker[agent]] if Rs[self.tracker[agent]] == 0 else Rs[self.tracker[agent]]
                    self.samples_register[self.tracker[agent]].append(x_kt)
            
            R = self.weight_function(self.C, self.nodes)*Rs
            self.regret[self.t] = self.regret[self.t-1] + np.sum(self.weight_function(Ce, self.nodes)*self.mean_rewards - R)

            self.samples_counter += np.ones(self.nodes.shape[0])*(self.C > 0)
            self.t += 1
            if self.t == self.max_iter:
                break 
        
        # Doubling-termination of the episode - Sample till doubling the number of samples on the less sampled arm
        for i in range(n_min):
            Rs = np.zeros(self.K)
            for j in range(self.N.shape[0]):
                agent = self.N[j]
                x_kt = self.get_reward(self.tracker[agent])
                if Rs[self.tracker[agent]] == 0:
                        self.samples_register[self.tracker[agent]].append(x_kt)  
                Rs[self.tracker[agent]] = self.meanRewards[self.tracker[agent]] if Rs[self.tracker[agent]] == 0 else Rs[self.tracker[agent]]
                self.samples_register[self.tracker[agent]].append(x_kt)
                self.samples_counter[self.tracker[agent]] += 1


            R = self.weight_function(self.C, self.nodes)*Rs
            self.regret[self.t] = self.regret[self.t-1] + np.sum(self.weight_function(Ce, self.nodes)*self.mean_rewards - R)

            self.samples_counter += np.ones(self.nodes.shape[0])*(self.C > 0)
            self.t += 1
            if self.t == self.max_iter:
                break 

    def learning(self):
        while self.t < self.max_iter:
            self.main_program()


    






