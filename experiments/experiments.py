import argparse
import logging
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import distance
import tqdm
from mpl_toolkits.mplot3d import Axes3D

from admm.admm_lp import lp_objective
from admm.consensus.admm_consensus_lp import solve_general_form_consensus_lp_with_admm
from admm.consensus.admm_consensus_lp_multiprocessing import \
    solve_consensus_lp_with_distributed_admm_cvxpy_multiprocessing
from generators.problem_generators import generate_general_form_consensus_ilp
from util import pbar, constants
from util.solvers import solve_lp_with_pulp
from util.util import decode
from matplotlib import cm


def main():
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-f', '--file', required=False,
                        help='file name for yaml file with parameters (default is experiments.yaml)',
                        default='experiments.yaml')
    parser.add_argument('-g', '--generate', help='generate instances', default=False, action='store_true')
    parser.add_argument('-s', '--solve',
                        help='solve instances using built-in solvers',
                        default=False, action='store_true')
    parser.add_argument('-p', '--plot', help='generate figures from pre-generated results', default=False,
                        action='store_true')
    parser.add_argument('-v', '--verbose', help='display logs to console', default=False, action='store_true')
    parser.add_argument('--overwrite',
                        help='overwrite already existing solution files, otherwise existing results are kept without running solvers',
                        default=False, action='store_true')
    args = parser.parse_args()

    if not (args.generate or args.solve or args.plot):
        parser.print_help()
        parser.error("One of -g, -s or -p must be given")

    with open(args.file, 'r') as ymlfile:
        exp_cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    secured_path = os.path.realpath(exp_cfg['path']) + '/' if os.path.realpath(exp_cfg['path'])[-1] != '/' else ''
    exp_cfg['path'] = secured_path
    if exp_cfg['dir'][-1] != '/':
        exp_cfg['dir'] = exp_cfg['dir'] + '/'

    if not os.path.exists(exp_cfg['path'] + exp_cfg['dir']):
        os.makedirs(exp_cfg['path'] + exp_cfg['dir'])

    log_handlers = [logging.FileHandler(exp_cfg['path'] + exp_cfg['dir'] + exp_cfg['log'], mode='w')]
    if args.verbose:
        log_handlers.append(logging.StreamHandler())
    logging.basicConfig(handlers=log_handlers, level=logging.DEBUG, format='%(levelname)s [%(asctime)s]: %(message)s')
    logging.info('Started')

    if args.generate:
        # generate instances and save them
        logging.info("Generating instances...")
        pass

    if args.solve:
        run_experiments(file=exp_cfg['path'] + exp_cfg['dir'] + exp_cfg['results'],
                        nb_agents_range=eval(exp_cfg['parameters']['nb_agents_range']),
                        nb_vars_range=eval(exp_cfg['parameters']['nb_vars_range']),
                        nb_shared_vars_range=eval(exp_cfg['parameters']['nb_shared_vars_range']),
                        seed_range=eval(exp_cfg['parameters']['seed_range']),
                        solvers=[eval(solver) for solver in exp_cfg['solvers']],
                        generator=eval(exp_cfg['generators'][0]))

    if args.plot:
        plot_experiments(file=exp_cfg['path'] + exp_cfg['dir'] + exp_cfg['results'],
                         nb_agents_range=eval(exp_cfg['parameters']['nb_agents_range']),
                         nb_vars_range=eval(exp_cfg['parameters']['nb_vars_range']),
                         nb_shared_vars_range=eval(exp_cfg['parameters']['nb_shared_vars_range']),
                         solvers=[eval(solver) for solver in exp_cfg['solvers']],
                         directory=exp_cfg['path'] + exp_cfg['dir'] + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '/')


def admm(c, A, b, mapping):
    return solve_general_form_consensus_lp_with_admm(c, A, b, mapping=mapping)


# TODO : add mapping
def admm_mp(c, A, b, mapping):
    return solve_consensus_lp_with_distributed_admm_cvxpy_multiprocessing(c, A, b)


def opt(c, A, b, var_names):
    return solve_lp_with_pulp(c, A, b, var_names)


def run_general_form_consensus_ilp(nb_agents, nb_vars_par_agent, nb_shared_vars=1, solvers=(opt, admm), seed=None,
                                   generator=generate_general_form_consensus_ilp, relaxation=True):
    cs, As, bs, c, A, b, mapping = generator(
        {'nb_agents': nb_agents, 'nb_vars_per_agent': nb_vars_par_agent, 'nb_shared_vars': nb_shared_vars,
         'seed': seed})

    results = {'nb_agents': nb_agents, 'nb_vars': nb_vars_par_agent, 'nb_shared_vars': nb_shared_vars}
    # constants.RHO = nb_vars_par_agent * nb_agents * 2
    # constants.THRESHOLD = 1 / (nb_vars_par_agent * nb_agents)
    for solver in solvers:
        st = time.time_ns() // 1000000
        opt_var_names = ['x_' + str(i) + '_' + str(j) for i in range(nb_agents) for j in
                         range(nb_vars_par_agent)]
        cur_sol = solver(c if solver is opt else cs, A if solver is opt else As, b if solver is opt else bs,
                         opt_var_names if solver is opt else mapping)
        et = time.time_ns() // 1000000

        same = True

        if solver is not opt:
            opt_sol = results[opt.__name__]
            cur_sol_dict = decode(cur_sol['array'].transpose()[0], cur_sol['var_names'], relaxation,
                                  threshold=constants.THRESHOLD)
            if relaxation:
                cur_sol['array'] = np.vstack(np.array([cur_sol_dict[v] for v in cur_sol_dict]))
            cur_sol['value'] = lp_objective(c, cur_sol['array'])[0][0]
            opt_sol_dict = decode(opt_sol['array'].transpose()[0], opt_sol['var_names'], relaxation)
            index = 0
            for k in cur_sol_dict:
                same = same and (cur_sol_dict[k] == opt_sol_dict[k])
                index += 1

        results[solver.__name__] = {'array': cur_sol['array'], 'time': et - st, 'value': cur_sol['value'],
                                    'convergence': same,
                                    'assignment': list(cur_sol['array'][:, 0].transpose()),
                                    'steps': 0 if solver is opt else max([len(i['iter']) for i in cur_sol['history']]),
                                    'var_names': cur_sol['var_names']}
        pbar.pbar.update(1)
    return results


def load_csv(file):
    return pd.read_csv(file)


def save_csv(df, file):
    df.to_csv(file)


def plot_2d(df, x, ys, title, x_label, y_label, labels, filename=None, show=False):
    data = df
    data = data.filter(items=[x] + ys).groupby(
        [x])  # .agg([('_mean', 'mean'), ('_min', 'min'), ('_max', 'max'), ('_std', 'std')]).reset_index()
    _mean = data.mean().reset_index()
    _min = data.min().reset_index()
    _max = data.max().reset_index()
    _std = data.std().reset_index()
    fig = plt.figure(figsize=(12, 8))
    for y, label in zip(ys, labels):
        plt.errorbar(_mean[x], _mean[y], yerr=[_min[y], _max[y]], label=label, capsize=5)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
    plt.title(title)
    plt.grid(linestyle=':')
    plt.legend()
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    if show:
        fig.show()
    plt.close()


def plot_3d(df, x, y, z, title, x_label, y_label, z_label, filename=None, show=False, scatter=False):
    data = df
    data = data.filter(items=[x, y, z]).groupby([x, y])
    mean = data.mean().reset_index()
    fig = plt.figure(figsize=(12, 8))
    ax = Axes3D(fig, azim=-115, elev=15)

    if scatter:
        ax.scatter(df[x], df[y], df[z], cmap=cm.coolwarm)
    else:
        try:
            ax.plot_trisurf(mean[x], mean[y], mean[z], antialiased=False, cmap=cm.coolwarm)
        except:
            ax.scatter(mean[x], mean[y], mean[z], antialiased=False, cmap=cm.coolwarm)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    plt.title(title)
    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()
    plt.close()


def run_experiments(file, nb_agents_range=range(5, 11, 1), nb_vars_range=range(10, 21, 2),
                    nb_shared_vars_range=range(4, 5),
                    seed_range=range(10), solvers=(opt, admm),
                    generator=generate_general_form_consensus_ilp):
    results = []

    params = [[nb_agents, nb_vars, nb_shared_vars, seed] for nb_agents in nb_agents_range for nb_shared_vars in
              nb_shared_vars_range for nb_vars in
              range(max(nb_shared_vars, nb_vars_range[0]), nb_vars_range[-1] + 1) for seed in
              seed_range]

    with tqdm.tqdm(
            total=len(params) * len(solvers),
            desc="Progress") as bar:
        pbar.pbar = bar
        for [nb_agents, nb_vars, nb_shared_vars, seed] in params:
            # constants.RHO = nb_agents
            # constants.THRESHOLD = 1 / nb_agents
            results.append(
                run_general_form_consensus_ilp(nb_agents, nb_vars, nb_shared_vars, solvers, seed,
                                               generator, True))
    df = pd.json_normalize(results, sep='_')
    # file = 'exp_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.csv'
    save_csv(df, file)


def plot_experiments(file, nb_agents_range, nb_vars_range, nb_shared_vars_range, solvers, directory='images/'):
    df = load_csv(file)
    file_name = os.path.splitext(os.path.basename(file))[0]
    # directory += datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    def f(x, y):
        if type(x) == str:
            x = eval(x)
        if type(y) == str:
            y = eval(y)
            return distance.euclidean(x, y)

    for solver in solvers:
        df[solver.__name__ + '_optimality'] = np.vectorize(lambda x, y: abs(y - x) * 100 / x if x > 1e-9 else 0)(
            df['opt_value'],
            df[solver.__name__ + '_value'])
        df[solver.__name__ + '_diff'] = abs(df['opt_value'] - df[solver.__name__ + '_value'])
        df[solver.__name__ + '_distance'] = np.vectorize(f)(
            df['opt_assignment'], df[solver.__name__ + '_assignment'])
        for nb_shared_vars in nb_shared_vars_range:
            current_dir = directory + str(nb_shared_vars) + '_shared_vars/'
            if not os.path.exists(current_dir):
                os.makedirs(current_dir)
            data = df.query('nb_shared_vars == ' + str(nb_shared_vars)).reset_index()
            plot_3d(data, 'nb_agents', 'nb_vars',
                    solver.__name__ + '_time', 'Solving time for ' + solver.__name__,
                    'agents',
                    'variables', 'time (ms)',
                    current_dir + file_name + '_' + solver.__name__ + '_time.png')
            if solver is not opt:
                plot_3d(data, 'nb_agents', 'nb_vars', solver.__name__ + '_steps',
                        'Steps to converge for ' + solver.__name__,
                        'agents',
                        'variables', 'steps',
                        current_dir + file_name + '_' + solver.__name__ + '_steps.png')
            plot_3d(data, 'nb_agents', 'nb_vars', solver.__name__ + '_optimality', solver.__name__ + ' optimality',
                    'agents',
                    'variables', 'optimality',
                    current_dir + file_name + '_' + solver.__name__ + '_optimality.png',
                    scatter=True)
            plot_3d(data, 'nb_agents', 'nb_vars', solver.__name__ + '_distance', solver.__name__ + ' distance',
                    'agents',
                    'variables', 'euclidian distance to opt',
                    current_dir + file_name + '_' + solver.__name__ + '_distance.png',
                    scatter=True)
            plot_3d(data, 'nb_agents', 'nb_vars', solver.__name__ + '_diff',
                    solver.__name__ + ' solution difference to optimal', 'agents', 'variables',
                    'cost difference to opt',
                    current_dir + file_name + '_' + solver.__name__ + '_diff.png',
                    scatter=True)

    for nb_shared_vars in nb_shared_vars_range:
        for nb_vars in nb_vars_range:
            if nb_vars < nb_shared_vars:
                break
            current_dir = directory + str(nb_shared_vars) + '_shared_vars/' + str(
                nb_vars) + '_vars/'
            if not os.path.exists(current_dir):
                os.makedirs(current_dir)
            data = df.query('nb_shared_vars == ' + str(nb_shared_vars)).reset_index().query(
                'nb_vars == ' + str(nb_vars)).reset_index()
            plot_2d(data, 'nb_agents',
                    [solver.__name__ + '_time' for solver in solvers],
                    'Solving time with ' + str(nb_vars) + ' variables', 'agents', 'time (ms)',
                    [solver.__name__ for solver in solvers],
                    current_dir + file_name + '_time.png')
            plot_2d(data, 'nb_agents',
                    [solver.__name__ + '_optimality' for solver in solvers],
                    'Optimality with ' + str(nb_vars) + ' variables', 'agents', 'Optimality',
                    [solver.__name__ for solver in solvers],
                    current_dir + file_name + '_optimality.png')
            plot_2d(data, 'nb_agents',
                    [solver.__name__ + '_distance' for solver in solvers],
                    'Euclidian distance to opt with ' + str(nb_vars) + ' variables', 'agents', 'Distance',
                    [solver.__name__ for solver in solvers],
                    current_dir + file_name + '_distance.png')

        for nb_agents in nb_agents_range:
            current_dir = directory + str(nb_shared_vars) + '_shared_vars/' + str(
                nb_agents) + '_agents/'
            if not os.path.exists(current_dir):
                os.makedirs(current_dir)
            data = df.query('nb_shared_vars == ' + str(nb_shared_vars)).reset_index().query(
                'nb_agents == ' + str(nb_agents)).reset_index()
            plot_2d(data, 'nb_vars',
                    [solver.__name__ + '_time' for solver in solvers],
                    'Solving time with ' + str(nb_agents) + ' agents', 'variables', 'time (ms)',
                    [solver.__name__ for solver in solvers],
                    current_dir + file_name + '_time.png')
            plot_2d(data, 'nb_vars',
                    [solver.__name__ + '_optimality' for solver in solvers],
                    'Optimality with ' + str(nb_agents) + ' agents', 'variables', 'optimality',
                    [solver.__name__ for solver in solvers],
                    current_dir + file_name + '_optimality.png')
            plot_2d(data, 'nb_vars',
                    [solver.__name__ + '_distance' for solver in solvers],
                    'Euclidian distance to opt with ' + str(nb_agents) + ' agents', 'inner variables', 'distance',
                    [solver.__name__ for solver in solvers],
                    current_dir + file_name + '_distance.png')
            plot_2d(data, 'nb_vars',
                    [solver.__name__ + '_steps' for solver in solvers],
                    'Steps with ' + str(nb_agents) + ' agents', 'variables', 'steps',
                    [solver.__name__ for solver in solvers],
                    current_dir + file_name + '_steps.png')


if __name__ == "__main__":
    main()  # "exp_2020-10-08-14-52-26.csv")
