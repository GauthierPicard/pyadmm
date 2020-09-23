#!/usr/bin/env python
# Standard form LP with random data
# From https://stanford.edu/~boyd/papers/admm/
import time

import numpy as np
from numpy.random import rand, randn, seed
import matplotlib.pyplot as plt

QUIET = False
MAX_ITER = 1000
ABSTOL = 1e-4
RELTOL = 1e-2


def linear_program(c, A, b, rho, alpha):
    t_start = time.time()

    m, n = np.shape(A)

    z = np.zeros((n, 1))
    u = np.zeros((n, 1))

    if not QUIET:
        print('%3s\t' % 'iter', '%10s\t' % 'r norm', '%10s\t' % 'eps pri', '%10s\t' % 's norm', '%10s\t' % 'eps dual',
              '%10s\n' % 'objective')

    history = {'objval': [], 'r_norm': [], 'eps_pri': [], 's_norm': [], 'eps_dual': [], 'iter': []}

    for k in range(0, MAX_ITER):
        # x - update
        x = np.linalg.solve(
            np.concatenate((np.block([rho * np.eye(n), A.transpose()]), np.block([A, np.zeros((m, m))]))),
            np.concatenate((rho * (z - u) - c, b)))[0:n]
        # z - update with relaxation
        zold = z
        x_hat = alpha * x + (1 - alpha) * zold
        z = np.maximum(np.zeros((n, 1)), x_hat + u)
        u = u + (x_hat - z)
        # diagnostics, reporting, termination checks
        history['iter'].append(k)
        history['objval'].append(objective(c, x)[0][0])
        history['r_norm'].append(np.linalg.norm(x - z))
        history['s_norm'].append(np.linalg.norm(-rho * (z - zold)))
        history['eps_pri'].append(np.sqrt(n) * ABSTOL + RELTOL * max(np.linalg.norm(x), np.linalg.norm(-z)))
        history['eps_dual'].append(np.sqrt(n) * ABSTOL + RELTOL * np.linalg.norm(rho * u))
        if not QUIET:
            print('%3d\t' % k, '%10.4f\t' % history['r_norm'][k], '%10.4f\t' % history['eps_pri'][k],
                  '%10.4f\t' % history['s_norm'][k],
                  '%10.4f\t' % history['eps_pri'][k], '%10.4f' % history['objval'][k])

        if (history['r_norm'][k] < history['eps_pri'][k]) and (history['s_norm'][k] < history['eps_dual'][k]):
            print("Convergence")
            break

    if not QUIET:
        t_end = time.time()
        print(str(t_end - t_start) + "sec Elapsed")
    return [z, history]


def objective(c, x):
    return c.conj().transpose() @ x


seed(0)

n = 500
m = 400

c = rand(n, 1) + 0.5
x0 = abs(randn(n, 1))
A = abs(randn(m, n))
b = A @ x0

[x, history] = linear_program(c, A, b, 1.0, 1.0)

plt.plot(history['iter'], history['objval'])
plt.ylabel('obj')
plt.xlabel('iter')
plt.show()

plt.plot(history['iter'], history['r_norm'])
plt.plot(history['iter'], history['eps_pri'], 'k--')
plt.ylabel('||r||_2')
plt.xlabel('iter')
plt.yscale('log')
plt.show()

plt.plot(history['iter'], history['s_norm'])
plt.plot(history['iter'], history['eps_dual'], 'k--')
plt.ylabel('||s||_2')
plt.xlabel('iter')
plt.yscale('log')
plt.show()