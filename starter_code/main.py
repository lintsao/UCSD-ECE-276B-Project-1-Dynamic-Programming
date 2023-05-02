from utils import *
from example import example_use_of_gym_env
from doorkey import *
import glob

def main():
    known_envs = glob.glob("./envs/known_envs/*.env")
    known_envs.sort()
    # env_path = "./envs/known_envs/doorkey-6x6-normal.env"
    # env_path = "./envs/random_envs/DoorKey-8x8-16.env"

    for env_path in known_envs:
        dk = doorkey(env_path)
        dk.__all__()

if __name__ == '__main__':
    main()
