import itertools
import random

import numpy as np
from numpy.random.mtrand import rand, randn


def generate_global_variable_consensus_ilp(params):
    nb_agents = params['nb_agents']
    nb_vars = params['nb_vars']
    seed = params['seed']

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    cs = []
    As = []
    bs = []
    c_cent = np.zeros((nb_vars * nb_agents, 1))
    A_cent = np.zeros((nb_agents, nb_vars * nb_agents))
    b_cent = np.ones((nb_agents, 1))

    for i in range(nb_agents):
        c = rand(nb_vars, 1) + 0.5  # np.random.randint(1, 10000, size=(nb_vars, 1)) # + rand(nb_vars, 1) * 1e-2
        A = np.ones((1, nb_vars))
        b = np.ones((1, 1))
        cs.append(c)
        As.append(A)
        bs.append(b)
        A_cent[i, range(i * nb_vars, (i + 1) * nb_vars)] = A
        c_cent[range(i * nb_vars, (i + 1) * nb_vars), :] = c

    for j in range(nb_vars):
        for (i, k) in itertools.combinations(range(nb_agents), r=2):
            if i != k:
                line = np.zeros((1, nb_vars * nb_agents))
                line[0, i * nb_vars + j] = 1
                line[0, k * nb_vars + j] = -1
                A_cent = np.concatenate([A_cent, line])
                b_cent = np.concatenate([b_cent, np.zeros((1, 1))])
    return cs, As, bs, c_cent, A_cent, b_cent


def generate_general_form_consensus_ilp(params):
    nb_agents = params['nb_agents']
    nb_vars_per_agent = params['nb_vars_per_agent']
    nb_shared_vars = params['nb_shared_vars']
    seed = params['seed']

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    cs = []
    As = []
    bs = []
    c_cent = np.zeros((nb_vars_per_agent * nb_agents, 1))
    A_cent = np.zeros((nb_agents, nb_vars_per_agent * nb_agents))
    b_cent = np.ones((nb_agents, 1))

    mapping = {i: [] for i in range(0, nb_shared_vars)}

    for i in range(nb_agents):
        c = rand(nb_vars_per_agent, 1) + 0.5  # np.random.randint(1, 10000, size=(nb_vars, 1)) # + rand(nb_vars, 1) * 1e-2
        A = np.ones((1, nb_vars_per_agent))
        b = np.ones((1, 1))
        cs.append(c)
        As.append(A)
        bs.append(b)
        A_cent[i, range(i * nb_vars_per_agent, (i + 1) * nb_vars_per_agent)] = A
        c_cent[range(i * nb_vars_per_agent, (i + 1) * nb_vars_per_agent), :] = c
        indexes = list(range(nb_vars_per_agent))
        random.shuffle(indexes)
        for v, j in zip(range(nb_shared_vars), indexes[0:nb_shared_vars]):
            mapping[v].append((i, j))

    for g in mapping:
        for (xi, xj) in itertools.combinations(mapping[g], r=2):
            if xi != xj:
                line = np.zeros((1, nb_vars_per_agent * nb_agents))
                line[0, xi[0] * nb_vars_per_agent + xi[1]] = 1
                line[0, xj[0] * nb_vars_per_agent + xj[1]] = -1
                A_cent = np.concatenate([A_cent, line])
                b_cent = np.concatenate([b_cent, np.zeros((1, 1))])
    return cs, As, bs, c_cent, A_cent, b_cent, mapping


def generate_global_variable_consensus_lp(params):
    nb_agents = params['nb_agents']
    nb_vars = params['nb_vars']
    nb_constraints_per_agent = params['nb_constraints_per_agent']
    seed = params['seed']

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    cs = []
    As = []
    bs = []
    c_cent = np.zeros((nb_vars * nb_agents, 1))
    A_cent = np.zeros((nb_agents * nb_constraints_per_agent, nb_vars * nb_agents))
    b_cent = np.ones((nb_agents * nb_constraints_per_agent, 1))
    n = nb_vars
    m = nb_constraints_per_agent
    for i in range(nb_agents):
        c = rand(n, 1) + 0.5
        x0 = abs(randn(n, 1))
        A = abs(randn(m, n))
        b = A @ x0
        cs.append(c)
        As.append(A)
        bs.append(b)
        A_cent[i * nb_constraints_per_agent:(i + 1) * nb_constraints_per_agent, i * nb_vars:(i + 1) * nb_vars] = A
        c_cent[i * nb_vars:(i + 1) * nb_vars, :] = c
        b_cent[i * nb_vars:(i + 1) * nb_vars, :] = b

    # consensus constraints
    for j in range(nb_vars):
        for (i, k) in itertools.combinations(range(nb_agents), r=2):
            if i != k:
                line = np.zeros((1, nb_vars * nb_agents))
                line[0, i * nb_vars + j] = 1
                line[0, k * nb_vars + j] = -1
                A_cent = np.concatenate([A_cent, line])
                b_cent = np.concatenate([b_cent, np.zeros((1, 1))])
    return cs, As, bs, c_cent, A_cent, b_cent
