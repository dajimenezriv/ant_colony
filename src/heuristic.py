import numpy as np
from itertools import groupby


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


class Heuristic:
    def __init__(self, n_nodes, n):
        self.n_nodes = n_nodes
        # remove the starter node
        self.n = n[:-1, :-1]
        self.nodes = np.array((range(self.n_nodes)))
        self.edges_used = [[] for _ in range(len(self.nodes))]
        self.weight = 0

        self.kruskal()

    def root(self, x):
        if x == self.nodes[x]: return x
        else: return self.root(self.nodes[x])

    def kruskal(self):
        while not all_equal(self.nodes):
            # edge with lower weight
            pos = np.argmin(self.n)
            orig = pos // self.n_nodes
            dest = pos % self.n_nodes
            weight = self.n[orig, dest]
            self.n[orig, dest] = np.inf

            # get their roots
            orig_ = self.root(orig)
            dest_ = self.root(dest)
            # it means that they are not connected yet
            if orig_ != dest_:
                self.nodes[self.nodes == orig_] = dest_
                self.weight += weight

                if orig < dest: edge = (orig, dest, weight)
                else: edge = (dest, orig, weight)
                self.edges_used[orig].append(edge)
                self.edges_used[dest].append(edge)
