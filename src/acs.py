from custom_parser import parse_file
from heuristic import Heuristic

import time
import numpy as np
from copy import deepcopy


def remove_leafs(n_nodes, edges_used, weight, t, a):
        degrees = np.array([0] * n_nodes)
        for i in list(range(n_nodes)):
            degrees[i] = len(edges_used[i])

        check = True
        while check:
            check = False
            leafs = list(np.where(degrees == 1)[0])
            for leaf in leafs:
                if leaf not in t:
                    check = True
                    edge = edges_used[leaf][0]
                    edges_used[leaf] = []
                    orig = edge[0]
                    dest = edge[1]
                    w = edge[2]
                    weight -= w * a
                    degrees[orig] -= 1
                    degrees[dest] -= 1
                    if edge[1] == leaf:
                        edges_used[edge[0]].remove(edge)
                    else:
                        edges_used[edge[1]].remove(edge)

        new_edges = []
        for e in edges_used:
            new_edges += e

        edges_used = list(set([i for i in new_edges]))

        return edges_used, weight


class AntColonySystem:
    """
    MAX-MIN AS
    Edges have always the lower value as origin
    """

    def __init__(self, input_file, n_ants, alpha, beta, p, iter, print_time=False):
        self.input_file = input_file
        self.n_nodes, self.a1, self.a2, self.y, self.original_n, self.t1, self.t2 = parse_file(self.input_file)
        self.alpha = alpha
        self.beta = beta
        self.p = p # pheromone evaporation rate
        self.iter = iter
        self.print_time = print_time

        # heuristic
        heuristic = Heuristic(self.n_nodes, deepcopy(self.original_n))
        self.best_weight = heuristic.weight * (self.a1 + self.a2)
        self.best_edges1, self.best_weight = remove_leafs(self.n_nodes, deepcopy(heuristic.edges_used), self.best_weight, self.t1, self.a1)
        self.best_edges2, self.best_weight = remove_leafs(self.n_nodes, deepcopy(heuristic.edges_used), self.best_weight, self.t2, self.a2)
        shared_edges = list(set(self.best_edges1) & set(self.best_edges2))
        self.best_weight += self.y * sum([edge[2] for edge in shared_edges])

        print(f'# Best weight so far: {self.best_weight}')
        
        self.start = len(self.original_n) - 1
        self.n_ants = n_ants
        self.ants = []
        self.ants1 = []
        self.ants2 = []
        self.ants1_finished = 0
        self.ants2_finished = 0
        self.n = np.power(1 / self.original_n, self.beta) # local information
        
        # half ants for each terminal
        for i in range(self.n_ants):
            if i < self.n_ants // 2:
                ant = Ant('t1', self.t1, self.a1, self.original_n, self.n, self.start, self.alpha)
                self.ants.append(ant)
                self.ants1.append(ant)
            else:
                ant = Ant('t2', self.t2, self.a2, self.original_n, self.n, self.start, self.alpha)
                self.ants.append(ant)
                self.ants2.append(ant)

        # pheromones update with heuristic
        best_edges = list(set(self.best_edges1 + self.best_edges2))
        self.max_tau = 1/(1 - self.p) * 1/len(best_edges)
        self.min_tau = (self.max_tau * (1 - self.p ** (1/self.n_nodes))) / ((self.n_nodes/2 - 1) * self.p ** (1/self.n_nodes))
        self.tau = np.full(self.original_n.shape, self.max_tau)
        self.pheromones_update(best_edges)

    def execute(self):
        start1 = time.time()
        for _ in range(self.iter):
            start2 = time.time()
            
            # we want to stop moving when half 1/4 of each terminal finish
            moves = 0
            while self.ants1_finished < self.n_ants // 8 or self.ants2_finished < self.n_ants // 8:
                moves += 1
                self.move()

            if self.print_time:
                end = time.time()
                print(f'Whole move method: {(end - start2)}')
                print(f'Mean move method: {(end - start2) / moves}s')

            start3 = time.time()
            self.end_iteration()
            if self.print_time:
                end = time.time()
                print(f'End iteration: {end - start3}s')
                print(f'One full iteration: {end - start2}s\n')   

        if self.print_time:
            print(f'All {self.iter} iterations: {time.time() - start1}s')

    def move(self):
        for i, ant in enumerate(self.ants):
            ant.move()
            if ant.finished:
                self.ants.pop(i)
                if ant.type == 't1':
                    self.ants1_finished += 1
                else:
                    self.ants2_finished += 1

    def end_iteration(self):
        self.ants = self.ants1 + self.ants2
        self.ants1_finished = 0
        self.ants2_finished = 0

        for ant in self.ants:
            if ant.finished:
                ant.edges_used, ant.weight = remove_leafs(self.n_nodes + 1, ant.edges_used, ant.weight, ant.t, ant.a)

        best_edges = []
        best_edges1 = []
        best_edges2 = []
        best_weight = np.inf
        for ant1 in self.ants1:
            if ant1.finished:
                for ant2 in self.ants2:
                    if ant2.finished:
                        shared_edges = list(set(ant1.edges_used) & set(ant2.edges_used))
                        weight = sum([edge[2] for edge in shared_edges])
                        weight = ant1.weight + ant2.weight + self.y * weight
                        if weight < best_weight:
                            best_edges1 = ant1.edges_used
                            best_edges2 = ant2.edges_used
                            best_edges = list(set(ant1.edges_used + ant2.edges_used))
                            best_weight = weight

        if best_weight < self.best_weight:
            self.best_weight = best_weight
            self.best_edges1 = best_edges1
            self.best_edges2 = best_edges2
            print(f'# Best weight so far: {self.best_weight}')

            # update max and min pheromone values
            self.max_tau = 1/(1 - self.p) * 1/len(best_edges)
            self.min_tau = (self.max_tau * (1 - self.p ** (1/self.n_nodes))) / ((self.n_nodes/2 - 1) * self.p ** (1/self.n_nodes))

        self.pheromones_update(best_edges)

    def pheromones_update(self, best_edges):
        self.tau = self.p * self.tau
        increment = 1/len(best_edges)
        for e in best_edges:
            orig = e[0]
            dest = e[1]
            self.tau[orig, dest] += increment
            self.tau[dest, orig] += increment

        self.tau[self.tau < self.min_tau] = self.min_tau
        self.tau[self.tau > self.max_tau] = self.max_tau

        for ant in self.ants:
            ant.restart(self.tau)

    def output(self, output_file):
        with open(self.input_file, 'r') as f:
            first_line = f.readlines()[0].replace('\n', '')

        with open(f'{output_file.split(".")[0]}-{self.best_weight}.txt', 'w') as f:
            f.write(f'{first_line}\n')
            for e in self.best_edges1:
                f.write(f'S1 {e[0]} {e[1]}\n')
            for e in self.best_edges2:
                f.write(f'S2 {e[0]} {e[1]}\n')

class Ant:
    def __init__(self, type, t, a, original_n, n, start, alpha):
        self.type = type
        self.t = t
        self.a = a
        self.original_n = original_n
        self.n = n
        self.start = start
        self.alpha = alpha
        self.nodes = list(range(start + 1))

    def restart(self, tau):
        self.tau = np.power(tau, self.alpha)
        self.weight = 0
        self.orig = self.start
        self.visited = [self.start]
        self.t_visited = []
        self.edges_used = [[] for _ in range(len(self.nodes))]
        self.finished = False

    def move(self):
        if not self.finished:
            res = self.tau[self.orig] * self.n[self.orig]
            res /= np.sum(res)
            dest = np.random.choice(self.nodes, p = res)

            if dest not in self.visited:
                self.visited.append(dest)
                weight = self.original_n[self.orig, dest]
                if self.orig < dest:
                    edge = (self.orig, dest, weight)
                else:
                    edge = (dest, self.orig, weight)
                self.edges_used[self.orig].append(edge)
                self.edges_used[dest].append(edge)
                self.weight += weight * self.a

                if dest in self.t and dest not in self.t_visited:
                    self.t_visited.append(dest)

            self.orig = dest
            self.finished = len(self.t) == len(self.t_visited)
