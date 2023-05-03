from utils import *
from doorkey_class import *
# from example import example_use_of_gym_env

import glob

def doorkey_problem(env):
    """
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env

        doorkey-6x6-direct.env
        doorkey-8x8-direct.env

        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env

    Feel Free to modify this fuction
    """

    dk = doorkey(env)
    seq = dk.__all__()

    return seq


def partA():
    known_envs = glob.glob("./envs/known_envs/*.env")
    known_envs.sort()

    for env_path in known_envs:
        seq = doorkey_problem(env_path)  # find the optimal action sequence

        name = env_path.split(".")[1].split('/')[-1]
        draw_gif_from_seq(seq, load_env(env_path)[0], path="./gif/{}.gif".format(name))  # draw a GIF & save


def partB():
    env_folder = "./envs/random_envs"
    env, info, env_path = load_random_env(env_folder)


if __name__ == "__main__":
    # example_use_of_gym_env()
    partA()
    # partB()
