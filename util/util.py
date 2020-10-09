from matplotlib import pyplot as plt

from util import constants
from util.constants import ZEROS


def decode(x, var_names=[], relax=True, threshold=constants.THRESHOLD):
    solution = {}
    i = 0
    if not var_names:
        var_names = ['x_' + str(i).zfill(ZEROS) for i in range(len(x))]
    for x_i, name in zip(x, var_names):
        solution[name] = round(x_i) if relax else x_i #1 if x_i > threshold else 0 if relax else x_i
        i += 1
    return solution


def display_history(h, title):
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(h['iter'], h['objval'])
    axs[0].set_ylabel('obj')
    axs[0].set_xlabel('iter')
    axs[1].plot(h['iter'], h['r_norm'])
    axs[1].plot(h['iter'], h['eps_pri'], 'k--')
    axs[1].set_ylabel('||r||_2')
    axs[1].set_xlabel('iter')
    axs[1].set_yscale('log')
    axs[2].plot(h['iter'], h['s_norm'])
    axs[2].plot(h['iter'], h['eps_dual'], 'k--')
    axs[2].set_ylabel('||s||_2')
    axs[2].set_xlabel('iter')
    axs[2].set_yscale('log')
    plt.suptitle(title)
    plt.show()