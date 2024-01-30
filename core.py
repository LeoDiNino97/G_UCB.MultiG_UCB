import numpy as np 

'''
The environment for the implementation is a random reward graph implemented using only numpy built-ins
'''
class RR_Graph: 
    def __init__(self, N, E, fc = None):
        # List of nodes
        self.N = N
        self.n = len(self.N)

        # List of edges
        self.E = E

        # Initial node
        self.s0 = self.N[0]

        # Representations of the graph
        self.A = np.zeros((self.n,self.n))                                                                  # Adjacency matrix
        for i in range(self.n):
            for j in range(self.n):
                if (self.N[i],self.N[j]) in self.E:
                    self.A[i,j] = 1
                    self.A[j,i] = 1

        self.D = np.diag(np.sum(self.A, axis=1))                                                            # Degree matrix

        self.L = self.D - self.A                                                                            # Laplacian matrix
        
        self.adjList = {u:[v for v in self.N if (u,v) in self.E or (v,u) in self.E] for u in self.N}        # Adjacency lists

        # Reward distribution over a node 
        if fc:
            self.meanRewards = {node:np.random.uniform(0.5,1.5) for node in self.N}
        else:
            self.meanRewards = {node:np.random.uniform(0.5,9.5) for node in self.N}

        self.mean = np.array(list(self.meanRewards.values()))
        self.optimum = np.max(self.mean)
        self.optimum_node = np.argmax(self.mean)
        
        # Empirical attributes are useful for further implementations
        self.empiricalMeans = np.array(list(self.meanRewards.values()))
        self.mu_Star = np.max(self.empiricalMeans)
        self.s_Star = np.argmax(self.empiricalMeans)

        #self.rewardDistros = {node:stats.truncnorm(loc = self.meanRewards[node], scale = 1, a = - self.meanRewards[node], b = 1 - self.meanRewards[node]) for node in self.N}
        
        # Cost vector
        self.C = np.zeros(self.n)
        for i in range(self.n):
            self.C[i] = self.mu_Star - self.empiricalMeans[i]

        # Offline Policy 
        _, self.pi, self.tree = self.dijkstra(self.s0, self.s_Star)

    def get_reward(self, s): 
        return np.random.uniform(self.meanRewards[s]-0.5, self.meanRewards[s]+0.5)
    
    def get_neighboors(self,s):
        return self.adjList[s]
    
    def start_setter(self, node):
        self.s0 = node

    def cost_setter(self):
        self.mu_Star = np.max(self.empiricalMeans)
        self.s_Star = np.argmax(self.empiricalMeans)
        for i in range(self.n):
            self.C[i] = self.mu_Star - self.empiricalMeans[i]
    
    def policy_setter(self):
        _, self.pi, _ = self.dijkstra(self.s0, self.s_Star)

    def dijkstra(self, src, dst):
        '''
        A Dijkstra method for shortest path 
        '''
        distance = np.full(self.n, np.inf)
        distance[src] = 0

        pred = {node:None for node in self.N}

        Q = {node:np.inf for node in self.N}
        Q[src] = 0
        
        while len(Q) != 0:
            curr = min(Q, key=Q.get)
            for neighbour in self.get_neighboors(curr):
                temp = distance[curr] + self.C[neighbour]
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
    
'''
The following classes implement the agents whose performance in terms of regret were compared
'''

class G_UCB:
    def __init__(self, nodes, edges, fc, max_iter=2e4):
        self.graph = RR_Graph(nodes, edges, fc)
        
        self.curr = self.graph.N[0]
        self.t = 1
        
        # Initialization of the UCB vector through one random sampling from each node
        self.U = self.initialization()
        self.curr = self.graph.N[0]

        self.graph_setter()

        self.samples = {node:[self.U[node]] for node in self.graph.N}

        self.max_iter = int(max_iter)
        self.regret = np.zeros(self.max_iter)
        self.rewards = np.zeros(self.max_iter)
        self.visited = np.zeros(self.max_iter)

    def initialization(self):
        '''
        The initialization of the environment is a full graph visit 
        '''
        U = np.zeros(self.graph.n)
        visited = np.zeros(self.graph.n)

        U[self.curr] = self.graph.get_reward(self.curr)
        visited[self.curr] = 1

        while np.any(1-visited):
            self.curr = np.random.choice(self.graph.get_neighboors(self.curr))
            if visited[self.curr] == 0:
                U[self.curr] = self.graph.get_reward(self.curr)
                visited[self.curr] = 1    
        return U
                
    def graph_setter(self):
        '''
        This methods leverages the upper confidence bound as a surrogate for the true mean 
        '''
        for node in self.graph.N:
            self.graph.empiricalMeans[node] = self.U[node]
        self.graph.cost_setter()

    def UCB_eval(self):
        for node in self.graph.N:
            self.U[node] = np.mean(np.array(self.samples[node])) + np.sqrt(2*np.log(self.t)/len(self.samples[node]))
        
    def train(self):
        self.graph.start_setter(self.curr)
        self.UCB_eval()
        self.graph.s0 = self.curr
        self.graph_setter()
        self.graph.policy_setter()
        policy = self.graph.pi.copy()

        # Main episode
        while self.U[self.curr] < np.max(self.U):
            self.curr = policy[0][1]
            r = self.graph.get_reward(self.curr)
            self.samples[self.curr].append(r)
            self.rewards[self.t] = r
            self.visited[self.t] = self.graph.mean[self.curr]
            self.regretter()
            self.t += 1
            if self.t == self.max_iter:
                break 
            policy.pop(0)
        
        # Doubling termination
        N = len(self.samples[self.curr])
        while len(self.samples[self.curr]) < 2*N and self.t < self.max_iter:
            r = self.graph.get_reward(self.curr)
            self.samples[self.curr].append(r)
            self.rewards[self.t] = r
            self.visited[self.t] = self.graph.mean[self.curr]
            self.regretter()
            self.t += 1
            if self.t == self.max_iter:
                break 
        
    def training(self):
        while self.t < self.max_iter:
            self.train()

    def regretter(self):
        self.regret[self.t] = self.regret[self.t-1] + self.graph.optimum - self.visited[self.t]



class G_QL:
    '''
    This is a class to implement Q-learning on a graph
    '''
    def __init__(self, nodes, edges, fc, alpha = 0.5, gamma = 0.9, epsilon = 0.9, max_iter=2e4):
        self.graph = RR_Graph(nodes, edges, fc)

        # Initialization of the Q-table masked with the adjacency matrix
        self.Q = np.zeros((self.graph.n,self.graph.n))
        for u in self.graph.N:
            for v in self.graph.N:
                if (u,v) not in self.graph.E and (v,u) not in self.graph.E and u != v:
                    self.Q[u,v] = -np.inf
                    self.Q[u,v] = -np.inf
        
        self.curr = self.graph.N[0]
        self.t = 1
        self.n = np.zeros(self.graph.n)

        self.max_iter = int(max_iter)
        self.regret = np.zeros(self.max_iter)
        self.rewards = np.zeros(self.max_iter)
        self.visited = np.zeros(self.max_iter)

        self.alpha = alpha
        self.gamma = gamma
          

    def QL(self):
        # Explore VS Exploit
        epsilon = 1.5*(3*self.graph.n + 1) / (3*self.graph.n + self.t)      
        p = np.random.uniform(low=0, high=1)
        if p >= epsilon: 
            # Be greedy in the policy
            self.curr = np.argmax(self.Q[self.curr])
        else:
            # Explore!
            self.curr = np.random.choice(self.graph.get_neighboors(self.curr))
        
        # Q-learning: observe reward and update the table 
        
        r = self.graph.get_reward(self.curr)
        for node in self.graph.get_neighboors(self.curr):

            q_t = self.Q[node, self.curr]
            assert(np.isfinite(q_t))
            self.Q[node, self.curr] = (1-self.alpha) * q_t + self.alpha * (r + self.gamma * np.max(self.Q[self.curr,:]))
    
        self.visited[self.t] = self.graph.mean[self.curr]
        self.regretter()
        self.t += 1

    def training(self):
        while self.t < self.max_iter:
            self.QL()

    def regretter(self):
        self.regret[self.t] = self.regret[self.t-1] + self.graph.optimum - self.visited[self.t]



class QL_UCB:
    '''
    This method implements a UCB infused Q-learning procedure on a graph
    '''
    def __init__(self, nodes, edges, fc, gamma = 0.9, c=1, max_iter=2e4):
        self.graph = RR_Graph(nodes, edges, fc)

        # Initialization of the Q-table masked with the adjacency matrix
        self.Q = np.zeros((self.graph.n,self.graph.n))
        for u in self.graph.N:
            for v in self.graph.N:
                if (u,v) not in self.graph.E and (v,u) not in self.graph.E and u != v:
                    self.Q[u,v] = -np.inf
                    self.Q[u,v] = -np.inf
        
        self.curr = self.graph.N[0]
        self.t = 1

        self.max_iter = int(max_iter)
        self.regret = np.zeros(self.max_iter)
        self.rewards = np.zeros(self.max_iter)
        self.visited = np.zeros(self.max_iter)
        self.edges_visits = {edge:0 for edge in self.graph.E}

        self.c = c
        self.gamma = gamma
        self.H = 1/self.gamma

    def QL(self):
        # Explore VS Exploit
        epsilon = 1.5*(3*self.graph.n + 1) / (3*self.graph.n + self.t)      
        p = np.random.uniform(low=0, high=1)

        pred = self.curr
        if p >= epsilon: 
            # Be greedy in the policy
            self.curr = np.argmax(self.Q[self.curr])
        else:
            # Explore!
            self.curr = np.random.choice(self.graph.get_neighboors(self.curr))

        try:
            self.edges_visits[(pred,self.curr)] += 1
        except KeyError:
            self.edges_visits[(self.curr,pred)] += 1

        # Q-learning: observe reward and update the table 
         
        # Confidence parameter
        p = 1/np.sqrt(self.t) 
        
        epsilon = 0.7*(3*self.graph.n + 1) / (6*len(self.graph.E) + self.t)
        r = self.graph.get_reward(self.curr)

        for node in self.graph.get_neighboors(self.curr):
            
            q_t = self.Q[node, self.curr]
            assert(np.isfinite(q_t))
            edge = (self.curr,node) if (self.curr,node) in self.graph.E else (node,self.curr)
            t = np.max([1,self.edges_visits[edge]])

            # Confidence bonus
            b = self.c * np.sqrt(self.H ** 3 * np.log(self.graph.n * 2 *len(self.graph.E) * t/p) / t)                        
            alpha = (self.H+1)/(self.H+t)

            self.Q[node, self.curr]  = (1-alpha) * q_t + alpha * (r + self.gamma * np.max(self.Q[self.curr,:]) + b)

        self.visited[self.t] = self.graph.mean[self.curr]
        self.regretter()
        self.t += 1

    def training(self):
        while self.t < self.max_iter:
            self.QL()

    def regretter(self):
        self.regret[self.t] = self.regret[self.t-1] + self.graph.optimum - self.visited[self.t]



class LOCAL_UCB:
    '''
    Standard Hoeffding-UCB for MABs
    '''
    def __init__(self, nodes, edges, fc, max_iter=2e4):
        self.graph = RR_Graph(nodes, edges, fc)

        # Initialization of the UCB vector through one random sampling from each node
        self.U = np.array([self.graph.get_reward(node) for node in self.graph.N])
        
        self.curr = self.graph.N[0]
        self.t = 1
        self.n = np.zeros(self.graph.n)

        self.samples = {node:[self.U[node]] for node in self.graph.N}

        self.max_iter = int(max_iter)
        self.regret = np.zeros(self.max_iter)
        self.rewards = np.zeros(self.max_iter)
        self.visited = np.zeros(self.max_iter)

    def UCB_eval(self, nodes):
        for node in nodes:
            self.U[node] = np.mean(np.array(self.samples[node])) + np.sqrt(2*np.log(self.t)/len(self.samples[node]))

    def step(self):
        neighs = self.graph.get_neighboors(self.curr)
        self.UCB_eval(neighs)
        self.curr = neighs[np.argmax(self.U[neighs])]
        r = self.graph.get_reward(self.curr)
        self.samples[self.curr].append(r)
        self.rewards[self.t] = r
        self.visited[self.t] = self.graph.mean[self.curr]
        self.regretter()
        self.t += 1
        
    def training(self):
        while self.t < self.max_iter:
            self.step()

    def regretter(self):
        self.regret[self.t] = self.regret[self.t-1] + self.graph.optimum - self.visited[self.t]



class LOCAL_TS:
    '''
    Standard Thompson Sampling procedure for MABs
    '''
    def __init__(self, nodes, edges, fc, mu0 = 0, var0 = 1, var = 1, max_iter=2e4):
        self.graph = RR_Graph(nodes, edges, fc)

        self.var0 = var0
        self.mu0 = mu0
        self.var = var
        
        self.curr = self.graph.N[0]
        self.t = 1

        self.samples = {node:[self.graph.get_reward(node)] for node in self.graph.N}

        self.max_iter = int(max_iter)
        self.regret = np.zeros(self.max_iter)
        self.rewards = np.zeros(self.max_iter)
        self.visited = np.zeros(self.max_iter)

    def get_parameters(self):
        # Get posterior parameters 
        sums = np.array([sum(self.samples[node]) for node in self.graph.N])
        ns = np.array([len(self.samples[node]) for node in self.graph.N])

        var1 = 1/(self.var0 + ns/self.var)
        mu1 = var1*(self.mu0/self.var0 + sums/self.var)

        return mu1, var1

    def step(self):
        neighs = self.graph.get_neighboors(self.curr)
        mu1, var1 = self.get_parameters()

        # Posterior sampling
        ts_sample = np.random.normal(mu1,np.sqrt(var1))[neighs]
        self.curr = neighs[np.argmax(ts_sample)]

        # Reward observation
        r = self.graph.get_reward(self.curr)
        self.samples[self.curr].append(r)
        self.rewards[self.t] = r
        self.visited[self.t] = self.graph.mean[self.curr]
        self.regretter()
        self.t += 1
        
    def training(self):
        while self.t < self.max_iter:
            self.step()

    def regretter(self):
        self.regret[self.t] = self.regret[self.t-1] + self.graph.optimum - self.visited[self.t]

'''
class UCRL_2:
    def __init__(self, nodes, edges, fc, delta = 0.01, epsilon = 0.001, max_iter = 20000):
        self.graph = RR_Graph(nodes, edges, fc)

        # Initialization of the UCB vector through one random sampling from each node
        self.U = np.array([self.graph.get_reward(node) for node in self.graph.N])

        self.graph_setter()
        
        self.S = self.graph.n
        self.A = 2*len(self.graph.E)

        self.curr = self.graph.N[0]
        self.t = 1
        self.n = np.zeros(self.graph.n)

        self.sums = np.zeros(self.graph.n)
        self.num_samples = np.ones(self.graph.n)

        self.epsilon = epsilon
        self.delta = delta
        self.max_iter = max_iter
        
        self.regret = np.zeros(self.max_iter)
        self.rewards = np.zeros(self.max_iter)
        self.visited = np.zeros(self.max_iter)

    def graph_setter(self):
        for node in self.graph.N:
            self.graph.empiricalMeans[node] = self.U[node]
        self.graph.cost_setter()

    def UCB_eval(self):

        ucb = self.sums/self.num_samples + np.sqrt(7*np.log(2*self.S*self.A*self.t/self.delta)/(2*self.num_samples))
        return ucb

    def EVI_planning(self, means, epsilon):
        u = np.zeros(self.graph.n)
        policy = {}

        iter_count = 0
        max_iter = min(1e2, 1 / epsilon)

        while iter_count <= max_iter:
            iter_count += 1
            u_old = u.copy()

            for node in self.graph.N:
                best_nb_u = np.max(u_old[self.graph.get_neighboors(node)])
                policy[node] = self.graph.get_neighboors(node)[np.argmax(u_old[self.graph.get_neighboors(node)])]
                u[node] = means[node] + best_nb_u

            if np.max(u - u_old) - np.min(u - u_old) < self.epsilon:
                break

        return policy, u
    
    def train(self):
        
        # Compute the UCB defined in Jacksh(2008).
        ucb = self.UCB_eval()

        # Compute the policy.
        self.t += 1
        epsilon = 1/np.sqrt(self.t)  # The stopping condition of value iteration as specified in Jacksch(2008).
        policy, _ = self.EVI_planning(ucb, epsilon=epsilon)
        print(policy)

        # Keep executing the policy until the doubling terminal condition specified in Jacksh(2008) is met.
        prev_visits = np.array([len(self.samples[node]) for node in self.graph.N])
        visits_this_episode = np.zeros(self.graph.n)

        while visits_this_episode[self.curr] < np.max(np.array([1, prev_visits[self.curr]])):

            self.curr = policy[self.curr]
            visits_this_episode[self.curr] += 1

            r = self.graph.get_reward(self.curr)
            self.sums[self.curr] += r
            self.num_samples[self.curr] += 1

            self.visited[self.t] = self.graph.mean[self.curr]
            self.regretter()
            self.t += 1
                 
    def training(self):
        while self.t < self.max_iter:
            print(self.t)
            self.train()

    def regretter(self):
        self.regret[self.t] = self.regret[self.t-1] + self.graph.optimum - self.visited[self.t]
'''