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

    def set_weight(self, weight):
        self._weight = weight

    def get_parent_node_index(self):
        return self._parent_index

    def set_parent_node_index(self, parent_index):
        self._parent_index = parent_index
    
    def get_child_node_index(self):
        return self._child_index    

    def set_child_node_index(self, child_index):
        self._child_index = child_index

    def __repr__(self):
        return "E{i} between V{p} and V{c} w={w}".format(i=self._index, p=self._parent_index, c=self._child_index, w=self._weight)


def calc_distance(x1, x2):
    return np.linalg.norm(x1 - x2)


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
            X1 = self._nodes[edge.get_parent_node_index()].get_position()
            X2 = self._nodes[edge.get_child_node_index()].get_position()
            distance = calc_distance(X1, X2)
            # print("Distance between ", X1, " and ", X2, ": ", distance)
            edge.set_weight(distance)
            self._edges[self._current_edge_index] = edge
            self._current_edge_index = self._current_edge_index + 1
        else:
            print("Argument must be of type Edge.")

    def plot_graph(self):
        for node in self._nodes.values():
            # print(node)
            X = node.get_position()
            plt.plot(X[0], X[1], marker='o', color='r')
        for edge in self._edges.values():
            # print(edge)
            start_node = self._nodes[edge.get_parent_node_index()]
            X_start = start_node.get_position()
            # print("x: ", X_start)
            end_node = self._nodes[edge.get_child_node_index()]
            X_end = end_node.get_position()
            plt.plot([X_start[0], X_end[0]], [X_start[1], X_end[1]], color='b', linewidth=1.0)
        plt.grid()
        plt.show()


class RRTStarGraph(Graph):

    def nearest_neighbor(self, v):
        nearest = self._nodes[0]
        print(nearest)
        min_distance = calc_distance(v, nearest.get_position())
        print("Initial distance: ", min_distance)
        for node in self._nodes.values():
            distance = calc_distance(v, node.get_position())
            print(distance)
            if distance < min_distance:
                min_distance = distance
                nearest = node
        print("Final distance: ", min_distance)
        return nearest




################### TESTING #########################################

X0 = np.array([0.0, 0.0])
print(X0)
origin = Node(X0)
origin.set_index(0)
origin.set_path_cost(42.0)
print(origin)

# G = Graph()
G = RRTStarGraph()

# [x_min, x_max, y_min, y_max]
grid_limits = np.array([-10.0, 10.0, -10.0, 10.0])

n_nodes = 50
n_edges = 100

for i in range(n_nodes):
    x = (grid_limits[1] - grid_limits[0])*random.random() + grid_limits[0]
    y = (grid_limits[3] - grid_limits[2])*random.random() + grid_limits[2]
    node = Node(np.array([x, y]))
    print(node)
    G.add_node(node)

for i in range(n_edges):
    i1 = random.randint(0,n_nodes-1)
    i2 = random.randint(0,n_nodes-1)
    edge = Edge(i1, i2)
    G.add_edge(edge)


# N = G.get_nodes()
# for i in range(len(N)):
#     print(N[i])

# E = G.get_edges()
# for i in range(len(E)):
#     print(E[i])

G.plot_graph()

v = np.array([0.0, 0.0])

nearest = G.nearest_neighbor(v)

print("Closest to the origin: ")
print(nearest)
plt.plot(nearest.get_position(), color='g', markersize=14)
plt.show()
