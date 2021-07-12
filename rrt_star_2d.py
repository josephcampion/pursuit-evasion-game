#!/usr/bin/env python
# coding: utf-8

"""
Implementation of RRT* algorithm using edges and nodes in 2D space.

Date created: 7/11/21
Creator: Joseph Campion
"""

import numpy as np 
from matplotlib import pyplot as plt 

class Node:

    def __init__(self, index, position=np.array([0.0,0.0]), parent_index=-1, path_cost=float("inf")):
        self._index = index
        self._position = position
        self._path_cost = path_cost
        self._parent_index = parent_index
        self._child_indices = []

    def get_index(self):
        return self._index

    def get_parent_index(self):
        return self._parent_index

    def get_position(self):
        return self._position

    def set_path_cost(self, cost):
        self.path_cost = cost

    def __repr__(self):
        if self._parent_index == -1:
            return "(Base) V{i} at ({x},{y}) w/C={c}".format(i=self._index, x=self._position[0], y=self._position[1], c=self._path_cost)
        else:
            return "(P={p}) V{i} at ({x},{y}) w/C={c}".format(p=self._parent_index,
            i=self._index, x=self._position[0], y=self._position[1], c=self._path_cost)    

# TODO: add local steering to edge cost calculation

def calc_dist(x1, x2):
    return np.linalg.norm(x1 - x2)

class Edge:

    def __init__(self, parent_node, child_node, index):
        self._index = index
        self._parent_node = parent_node
        self._child_node = child_node
        self._weight = calc_dist(self._parent_node.get_position(), self._child_node.get_position())

    def get_index(self):
        return self._index

    def get_weight(self):
        return self._weight

    def get_parent_node_index(self):
        return self._parent_node.get_index()
    
    def get_child_node_index(self):
        return self._child_node.get_index()    

    def __repr__(self):
        return "E{i} between V{p} and V{c} w={w}".format(i=self._index, p=self._parent_node.get_index(),
        c=self._child_node.get_index(), w=self._weight)


################### TESTING #########################################

ni = 0

n0 = Node(0)
ni = ni + 1


pos1 = np.array([3.0,-4.0])
n1 = Node(ni, pos1, n0.get_index())
ni = ni + 1

pos2 = np.array([8.0,-16.0])
n2 = Node(ni, pos2, n1.get_index())
ni = ni + 1

pos3 = np.array([-8.0,6.0])
n3 = Node(ni, pos3, n0.get_index())
ni = ni + 1


print(n0)
print(n1)
print(n2)
print(n3)
print()

e0 = Edge(n0, n1, 0)
e1 = Edge(n1, n2, 1)
e2 = Edge(n0, n3, 2)

print(e0)
print(e1)
print(e2)

