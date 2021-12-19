import numpy as np


def parse_file(file):
    """
    The edges are stored in a matrix of n_nodes x n_nodes called n
    We add a row and a column for the starter node (-1) which helps us to start in any node
    """
    
    n_nodes = None
    a1 = None
    a2 = None
    y = None
    n = None
    t1 = []
    t2 = []

    with open(file, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '').split(' ')

            if line[0] == 'S':
                n_nodes = int(line[1])
                n = np.full((n_nodes + 1, n_nodes + 1), np.inf)
            elif line[0] == 'F':
                a1 = float(line[1])
                a2 = float(line[2])
                y = float(line[3])
            elif line[0] == 'E':
                orig = int(line[1])
                dest = int(line[2])
                weight = float(line[3])
                n[orig, dest] = weight
                n[dest, orig] = weight
            elif line[0] == 'T1':
                t1.append(int(line[1]))
            elif line[0] == 'T2':
                t2.append(int(line[1]))

    # we add edges from a new node to all nodes so all ants can start anywhere (unidirectional edges)
    for i in range(n_nodes):
        n[n_nodes, i] = 1

    return n_nodes, a1, a2, y, n, t1, t2
    