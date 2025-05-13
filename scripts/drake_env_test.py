import hydra
from omegaconf import DictConfig

from vid2skill.common.helper_functions.drake_helper_functions import setup_environment


@hydra.main(
    version_base=None,
    config_name="franka_allegro_drake_env",
    config_path="../configs/envs",
)
def main(cfg: DictConfig):
    env = setup_environment(cfg)
    env.reset()

    for _ in range(1000):
        env.step(env.action_space.sample())
        env.render()


if __name__ == "__main__":
    main()
