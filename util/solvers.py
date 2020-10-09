import logging

from pulp import LpProblem, LpMinimize, LpVariable, LpInteger, lpSum, LpStatus
from datetime import datetime
import numpy as np

from model.solver import Solver
from util.constants import ZEROS


class PuLPSolver(Solver):

    def solve(self):
        return solve_lp_with_pulp(self.problem.c, self.problem.A, self.problem.b, self.problem.var_names, self.problem.cat, self.problem.objective, export=False)


def solve_lp_with_pulp(c, A, b, var_names=[], cat=LpInteger, objective=LpMinimize, export=False):
    prob = LpProblem("AllocationProblem", objective)
    # create variables
    xs = []
    if not var_names:
        var_names = ["x_" + str(i).zfill(ZEROS) for i in range(len(c))]
    for i, name in zip(range(len(c)), var_names):
        xs.append(LpVariable(name, 0, 1, cat))
    # create objective
    prob += lpSum([c.tolist()[i][0] * xs[i] for i in range(len(c))]), "TotalCost"
    # create constraints
    for i in range(len(b)):
        prob += lpSum([A[i][j] * xs[j] for j in range(len(c))]) == b[i], "c_" + str(i)
    if export:
        prob.writeLP('lp_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ".lp")
    prob.solve()

    logging.info("Status:" + str(LpStatus[prob.status]))

    solution = {}
    for v in prob.variables():
        # print(v.name, "=", v.varValue)
        solution[v.name] = v.varValue

    return {'solution': solution, 'history': None, 'array': np.vstack(np.array(list(solution.values()))),
            'extended_array': np.vstack(np.array(list(solution.values()))), 'value': prob.objective.value(),
            'var_names': var_names}
