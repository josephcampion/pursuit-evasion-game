#!/usr/bin/env python
# coding: utf-8

"""
Implementation of RRT* algorithm using edges and nodes in 2D space.

Date created: 7/11/21
Creator: Joseph Campion
"""

import numpy as np
import random
from matplotlib import pyplot as plt 

class Node:
    def __init__(self, position=np.array([0.0,0.0])):
        self._index = -1
        self._position = position
        self._path_cost = float("inf")
        self._parent_index = -1
        self._child_indices = []

    def get_index(self):
        return self._index

    def set_index(self, index):
        self._index = index

    def get_parent_index(self):
        return self._parent_index

    def set_parent_index(self, parent_index):
        self._parent_index = parent_index

    def get_position(self):
        return self._position

    def get_path_cost(self):
        return self._path_cost

    def set_path_cost(self, cost):
        self._path_cost = cost

    def get_children_indices(self):
        return self._child_indices

    def __repr__(self):
        return "(P={p}) V{i} at ({x},{y}) w/C={c}".format(p=self._parent_index, i=self._index, 
        x=self._position[0], y=self._position[1], c=self._path_cost)    

# TODO: add local steering to edge cost calculation

def calc_dist(x1, x2):
    return np.linalg.norm(x1 - x2)

class Edge:
    def __init__(self, parent_index, child_index, index=-1):
        self._index = index
        self._parent_index = parent_index
        self._child_index = child_index
        self._weight = float("inf")

    def get_index(self):
        return self._index

    def set_index(self, index):
        self._index = index

    def get_weight(self):
        return self._weight

    def set_weight(self, _weight):
        self._weight = weight

    def get_parent_node_index(self):
        return self._parent_node.get_index()

    def set_parent_node_index(self, parent_index):
        self._parent_index = parent_index
    
    def get_child_node_index(self):
        return self._child_node.get_index()    

    def set_child_node_index(self, child_index):
        self._child_index = child_index

    def __repr__(self):
        return "E{i} between V{p} and V{c} w={w}".format(i=self._index, p=self._parent_node.get_index(),
        c=self._child_node.get_index(), w=self._weight)


# TODO: should the nodes and edges with a graph be a dictionary? would that be faster 

class Graph:
    def __init__(self):
        self._current_node_index = 0
        self._current_edge_index = 0
        self._nodes = {}
        self._edges = {}

    def get_nodes(self):
        return self._nodes

    def get_edges(self):
        return self._edges

    def add_node(self, node):
        if isinstance(node, Node):
            # TODO: add node to graph with index and increment count
            node.set_index(self._current_node_index)
            self._nodes[self._current_node_index] = node
            self._current_node_index = self._current_node_index + 1
        else:
            print("Argument must be of type Node.")

    def add_edge(self, edge):
        if isinstance(edge, Edge):
            # TODO:
            edge.set_index(self._current_edge_index)

        else:
            print("Argument must be of type Edge.")
        


    def plot_graph(self):
        for node in self._nodes:
            X = node.get_position()
            plt.plot(X[0], X[1], marker='o', color='r')
        for edge in self._edges:
            edge_start_index = edge.get_parent_node_index()

            # x_start = 
        

################### TESTING #########################################

X0 = np.array([0.0, 0.0])
print(X0)
origin = Node(X0)
origin.set_index(0)
origin.set_path_cost(42.0)
print(origin)

G = Graph()

# [x_min, x_max, y_min, y_max]
grid_limits = np.array([-10.0, 10.0, -10.0, 10.0])

for i in range(10):
    x = (grid_limits[1] - grid_limits[0])*random.random() + grid_limits[0]
    y = (grid_limits[3] - grid_limits[2])*random.random() + grid_limits[2]
    node = Node(np.array([x, y]))
    G.add_node(node)

for i in range(25):
    i1 = random.randint(0,9)
    i2 = random.randint(0,9)
    edge = Edge(i1, i2)

    # print(ri)


N = G.get_nodes()
for i in range(len(N)):
    print(N[i])

