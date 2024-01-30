import numpy as np 

'''
This file contains functions defining different topologies of 100 nodes
'''

def grid(n_nodes = 100, self_loop = False):
    # Define the dimensions of the grid
    rows = int(np.sqrt(n_nodes))
    cols = int(np.sqrt(n_nodes))

    # Create a grid of nodes
    nodes_grid = np.arange(rows * cols).reshape((rows, cols))

    # Initialize an empty adjacency matrix
    edges_grid = []


    # Connect horizontal edges
    for i in range(rows):
        for j in range(cols - 1):
            edges_grid.append((nodes_grid[i, j], nodes_grid[i, j + 1]))

    # Connect vertical edges
    for i in range(rows - 1):
        for j in range(cols):
            edges_grid.append((nodes_grid[i, j], nodes_grid[i + 1, j]))

    nodes_grid = [i for i in range(n_nodes)]

    if self_loop:

        for i in nodes_grid:
            edges_grid.append((i,i))

    return nodes_grid, edges_grid

def fully_connected(n_nodes = 100):
    nodes_fc = [i for i in range(n_nodes)]
    edges_fc = []

    for u in range(len(nodes_fc)):
        for w in range(u, len(nodes_fc)):
            edges_fc.append((u,w))

    return nodes_fc, edges_fc

def line(n_nodes = 100):
    nodes_ln = [i for i in range(n_nodes)]
    edges_ln = [(n_nodes-1,n_nodes-1)]

    for u in range(len(nodes_ln)-1):
        edges_ln.append((nodes_ln[u],nodes_ln[u]))
        edges_ln.append((nodes_ln[u], nodes_ln[u+1]))

    return nodes_ln, edges_ln

def circle(n_nodes = 100):

    nodes_crcl = [i for i in range(n_nodes)]
    edges_crcl = [(n_nodes-1,n_nodes-1),(0,n_nodes-1)]

    for u in range(0,len(nodes_crcl)-1):
        edges_crcl.append((nodes_crcl[u], nodes_crcl[u]))
        edges_crcl.append((nodes_crcl[u], nodes_crcl[u+1]))

    return nodes_crcl, edges_crcl

def star(n_nodes = 101, layers = 5):
    nodes_star = [i for i in range(n_nodes)]

    k = int(n_nodes/layers)
    edges_star = [(0,i) for i in range(k+1)]

    for i in range(1,n_nodes):
        edges_star.append((i,i))
        if i + k < n_nodes:
            edges_star.append((i,i+k))

    return nodes_star, edges_star

def tree(n_nodes = 100):
    nodes_tree = [i for i in range(n_nodes)]
    edges_tree = []

    for id in range(n_nodes):
        edges_tree.append((id,id))
        if id * 2 + 1 <= n_nodes:

            edges_tree.append((id,id*2))
            edges_tree.append((id,id*2+1))
    
    return nodes_tree, edges_tree

def robo_env():
    nodes_rob = [0,1,2,3,4,5,6,7,8,9]
    edges_rob = [(0,1),(0,2),(0,3),(1,3),(1,4),(2,3),(2,5),(2,6),(3,5),(3,4),(4,5),(4,8),(5,6),(5,7),(5,8),(6,7),(7,8),(7,9)]

    for i in nodes_rob:
        edges_rob.append((i,i))

    return nodes_rob, edges_rob

