import pytest
import logging
import sys

import numpy as np

from admm.consensus.admm_consensus_lp import solve_general_form_consensus_lp_with_admm
from generators.problem_generators import generate_general_form_consensus_ilp
from util import constants
from util.solvers import solve_lp_with_pulp
from util.constants import QUIET
from util.util import display_history, decode

logging.basicConfig(level=logging.DEBUG,  # filename='test_admm_general_form_consensus.log',
                    stream=sys.stdout,
                    format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s",
                    filemode='w')


@pytest.mark.parametrize('nb_agents, nb_vars_per_agent, nb_shared_vars, seed',
                         [[a, v, s, seed] for a in range(2, 11) for s in range(1, 5) for v in range(s, 5) for seed in
                          range(100)])
def test_general_form_consensus_ilp(nb_agents, nb_vars_per_agent, nb_shared_vars, seed):
    cs, As, bs, c, A, b, mapping = generate_general_form_consensus_ilp(
        {'nb_agents': nb_agents, 'nb_vars_per_agent': nb_vars_per_agent, 'nb_shared_vars': nb_shared_vars,
         'seed': seed})
    # print(cs, As, bs, c, A, b, mapping)
    constants.RHO = nb_vars_per_agent * nb_agents * 2
    constants.THRESHOLD = 1 / (nb_vars_per_agent * nb_agents)

    var_names = ['x_' + str(i) + '_' + str(j) for i in range(len(cs)) for j in range(len(cs[i]))]

    opt_sol = solve_lp_with_pulp(c, A, b, var_names=var_names)
    admm_sol = solve_general_form_consensus_lp_with_admm(cs, As, bs, mapping=mapping)

    if not QUIET:
        for i in range(len(admm_sol['history'])):
            display_history(admm_sol['history'][i], "ADMM Proximal " + str(i))

    admm_sol_dict = decode(admm_sol['array'].transpose()[0], admm_sol['var_names'], True,
                           threshold=1 / (nb_vars_per_agent * nb_agents))
    opt_sol_dict = decode(opt_sol['array'].transpose()[0], opt_sol['var_names'], True)
    print(constants.RHO, constants.THRESHOLD)
    print("opt sol   = " + str(opt_sol_dict))
    print("opt array = " + str(opt_sol['array']))
    print("opt cost  = " + str(opt_sol['value']))
    print("admm sol  = " + str(admm_sol_dict))
    print("admm cost = " + str(admm_sol['value']))
    print("admm array= " + str(admm_sol['array']))
    print(mapping)
    # print('==================>', seed)
    index = 0
    same_admm = True

    for k in admm_sol_dict:
        same_admm = same_admm and (
                    admm_sol_dict[k] == opt_sol_dict[k])  # opt_sol_dict[list(opt_sol_dict)[mapping[index][0][1]]])
        index += 1

    assert same_admm


def test_small_general_form_consensus_ilp():
    cs = [np.array([[1, 2, 0.5]]).transpose(), np.array([[0.1, 1, 0.5]]).transpose()]
    As = [np.array([[1, 1, 1]]), np.array([[1, 1, 1]])]
    bs = [np.array([[1]]).transpose(), np.array([[1]]).transpose()]
    c = np.array([[1, 2, 0.5, 0.1, 1, 0.5]]).transpose()
    A = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1], [0, 0, 1, -1, 0, 0]])
    b = np.array([[1, 1, 0]]).transpose()
    mapping = {0: [(0, 2), (1, 0)]}

    var_names = ['x_' + str(i) + '_' + str(j) for i in range(len(cs)) for j in range(len(cs[i]))]

    opt_sol = solve_lp_with_pulp(c, A, b, var_names=var_names)
    admm_sol = solve_general_form_consensus_lp_with_admm(cs, As, bs, mapping=mapping)

    print(opt_sol['extended_array'])
    print(admm_sol['extended_array'])

    if not QUIET:
        for i in range(len(admm_sol['history'])):
            display_history(admm_sol['history'][i], "ADMM Proximal " + str(i))

    admm_sol_dict = decode(admm_sol['array'].transpose()[0], admm_sol['var_names'], True)
    opt_sol_dict = decode(opt_sol['array'].transpose()[0], opt_sol['var_names'], True)
    index = 0
    same_admm = True

    for k in admm_sol_dict:
        same_admm = same_admm and (admm_sol_dict[k] == opt_sol_dict[list(opt_sol_dict)[mapping[index][0][1]]])
        # break
        index += 1

    assert same_admm
