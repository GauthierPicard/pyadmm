import logging

import numpy as np
from cvxpy import Parameter, sum_squares, Variable, Problem, Minimize, Constant

from util import constants
from util.constants import MAX_ITER, ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE, ALPHA


class ADMMProximalLPSolver:

    def __init__(self, name, c, A, b, dims):
        logging.info(name + " started")
        logging.info(name + "'s dims = " + str(dims))
        self.name = name
        _, n = np.shape(A)
        self.n = n
        self.dims = dims
        self.xbar = Parameter(n, value=np.zeros(n))
        self.xbar_old = Parameter(n, value=np.zeros(n))
        self.u = Parameter(n, value=np.zeros(n))
        self.x = Variable(n)
        self.c = c
        self.A = A
        self.b = b
        self.f = self.c.T @ self.x
        self.rho = constants.RHO
        self.f += (self.rho / 2) * sum_squares(self.x[self.dims] - self.xbar[self.dims] + self.u[self.dims])
        logging.info(name + "'s f = " + str(self.f))
        self.prox = Problem(Minimize(self.f), [self.A @ self.x == self.b, self.x >= 0])
        self.history = {'objval': [], 'r_norm': [], 'eps_pri': [], 's_norm': [], 'eps_dual': [], 'iter': []}
        self.has_converged = False
        self.k = 0


    def solve(self):
        logging.info(self.name + " is solving proximal...")
        # if not self.has_converged:
        self.f = self.c.T @ self.x
        self.rho = constants.RHO
        self.f += (self.rho / 2) * sum_squares(self.x[self.dims] - self.xbar[self.dims] + self.u[self.dims])
        self.prox = Problem(Minimize(self.f), [self.A @ self.x == self.b, self.x >= 0])
        self.prox.solve()
        logging.info(self.name + ' sends ' + str(self.x.value) + ' to master')
        logging.info(self.name + "'s x[dims] = " + str(self.x[self.dims].value))
        return self.x.value, self.has_converged, self.prox.value, self.history

    def update(self, x):
        if not self.has_converged:
            self.xbar_old.value = self.xbar.value
            self.xbar.value = x
            self.x.value = constants.ALPHA * self.x.value + (1 - constants.ALPHA) * self.xbar_old.value
            self.xbar.value = np.maximum(np.zeros(self.n), self.x.value + self.u.value)
            self.u.value += self.x.value - self.xbar.value

        r_norm = np.linalg.norm(self.xbar[self.dims].value - self.x[self.dims].value)
        s_norm = np.linalg.norm(-self.rho * (self.xbar[self.dims].value - self.xbar_old[self.dims].value))
        eps_pri = np.sqrt(len(self.dims)) * ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * max(
            np.linalg.norm(self.x[self.dims].value),
            np.linalg.norm(-self.xbar[self.dims].value))
        eps_dual = np.sqrt(len(self.dims)) * ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * np.linalg.norm(
            self.rho * self.u[self.dims].value)

        if r_norm > constants.MU * eps_pri:
            self.rho = self.rho * constants.TAU_INCR
            self.u.value = self.u.value / constants.TAU_INCR
        if constants.MU * r_norm < eps_pri:
            self.rho = self.rho / constants.TAU_DECR
            self.u.value = self.u.value * constants.TAU_DECR

        self.history['iter'].append(self.k)
        self.history['objval'].append(self.prox.value)
        self.history['r_norm'].append(r_norm)
        self.history['s_norm'].append(s_norm)
        self.history['eps_pri'].append(eps_pri)
        self.history['eps_dual'].append(eps_dual)

        if r_norm < eps_pri and s_norm < eps_dual and not self.has_converged:
            logging.info(self.name + ': Local convergence at step ' + str(self.k))
            self.has_converged = True

        self.k += 1


def solve_consensus_lp_with_admm(cs, As, bs, max_iter=MAX_ITER,):
    n = len(cs)
    proxs = []
    xs = [None] * n
    hcs = [None] * n
    vs = [0] * n
    hs = [None] * n
    for i in range(n):
        proxs.append(ADMMProximalLPSolver("solver_" + str(i), cs[i], As[i], bs[i], [1]))

    # ADMM loop.
    for k in range(max_iter):
        nb_converged = 0
        for i in range(n):
            xs[i], hcs[i], vs[i], hi = proxs[i].solve()
            hs[i] = hi
            logging.info('Master receives ' + str(xs[i].tolist()) + ' from ' + proxs[i].name)
            nb_converged += (1 if hcs[i] else 0)

        xbar = sum(xi for xi in xs) / n

        # Scatter xbar
        for prox in proxs:
            prox.update(xbar)

        if nb_converged == n:
            logging.info("Global convergence at k=" + str(k))
            break

    solution = {}
    i = 0
    for x in list(xbar):
        solution['x_' + str(i)] = x
    return {'solution': solution, 'history': hs, 'array': np.vstack(np.array(list(xbar))),
            'extended_array': np.array(list(np.vstack(np.array(list(xbar)))) * len(cs))}


def inverse(mapping):
    m = {}
    for k, v in mapping.items():
        for (i, j) in v:
            if not i in m.keys():
                m[i] = []
            m[i].append((k, j))
    return m


def solve_general_form_consensus_lp_with_admm(cs, As, bs, max_iter=MAX_ITER, mapping={}):
    n = len(cs)
    proxs = []
    xs = [None] * n
    hcs = [None] * n
    vs = [None] * n
    hs = [None] * n
    gnippam = inverse(mapping)
    for i in range(n):
        proxs.append(ADMMProximalLPSolver("solver_" + str(i), cs[i], As[i], bs[i], dims=[dim[1] for dim in gnippam[i]]))

    var_names = ['x_' + str(i) + '_' + str(j) for i in range(n) for j in range(len(cs[i]))]

    # ADMM loop.
    for k in range(max_iter):
        nb_converged = 0
        for i in range(n):
            xs[i], hcs[i], vs[i], hi = proxs[i].solve()
            hs[i] = hi
            logging.info('Master receives ' + str(xs[i].tolist()) + ' from ' + proxs[i].name)
            nb_converged += (1 if hcs[i] else 0)

        # aggregate xi's
        xbar = np.zeros(len(mapping.keys()))
        for g in mapping:
            logging.info('Master aggregates results received from solvers...')
            xbar[g] = sum(xs[i][j] for (i, j) in mapping[g]) / len(mapping[g])
            logging.info('xbar[' + str(g) + '] = ' + str(xbar))
        # scatter xbar
        for prox, i in zip(proxs, range(0, n)):
            # build xbar_i
            logging.info('Master extracts components to send to solver ' + str(i) + "...")
            for (l, j) in gnippam[i]:
                xs[i][j] = xbar[l]
            logging.info('x_' + str(i) + ' = ' + str(xs[i]))
            prox.update(xs[i])

        if nb_converged == n:
            logging.info("Global convergence at k=" + str(k))
            break

    x = np.zeros(n*len(cs[0]))
    index = 0
    for i in range(n):
        for j in range(len(list(xs[i]))):
            x[index] = xs[i][j]
            index += 1
    solution = {}
    i = 0
    for xi in list(x):
        solution[i] = xi
        i += 1
    return {'solution': solution, 'history': hs, 'array': np.vstack(np.array(list(x))),
            'value': sum(vs),
            'var_names': var_names,
            'rho': max([prox.rho for prox in proxs])}
