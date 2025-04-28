import numpy as np
import hydra
import ipdb
from omegaconf import DictConfig
from vid2skill.common.helper_functions.drake_helper_functions import make_env


@hydra.main(
    version_base=None,
    config_name="franka_allegro_drake_env",
    config_path="../configs/envs",
)
def main(cfg: DictConfig):
    env = make_env(cfg)
    env.reset()
    env.render()

    input("Press Enter to start the simulation...")
    for _ in range(1000):
        env.step(np.zeros(env.action_space.shape))
        env.render()
    ipdb.set_trace()


if __name__ == "__main__":
    main()
