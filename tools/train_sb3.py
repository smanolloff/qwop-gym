from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import safe_mean
from gym.wrappers import TimeLimit
import os
import time
import math
import tools.common as common


class LogCallback(BaseCallback):
    """Logs user-defined `info` values into tensorboard"""

    def _on_step(self) -> bool:
        for k in common.INFO_KEYS:
            v = safe_mean([ep_info[k] for ep_info in self.model.ep_info_buffer])
            self.model.logger.record(f"user/{k}", v)


def init_model(
    venv, seed, learner_cls, learner_kwargs, learning_rate, log_tensorboard, out_dir
):
    alg = None

    match learner_cls:
        case "PPO":
            from stable_baselines3 import PPO

            alg = PPO
        case "DQN":
            from stable_baselines3 import DQN

            alg = DQN
        case "QRDQN":
            from sb3_contrib import QRDQN

            alg = QRDQN
        case _:
            raise Exception("Unexpected learner_cls: %s" % learner_cls)

    model = alg(
        **dict(learner_kwargs, env=venv, learning_rate=learning_rate, seed=seed)
    )

    if log_tensorboard:
        os.makedirs(out_dir, exist_ok=True)
        log = logger.configure(folder=out_dir, format_strings=["tensorboard"])
        model.set_logger(log)

    return model


#
# A note about tensorboard logging of user-defined values in `info`:
#
# On each step, if env is done, Monitor wrapper will read `info_keywords`
# from `info` and copy them into `info["episode"]`:
# https://github.com/DLR-RM/stable-baselines3/blob/v1.8.0/stable_baselines3/common/monitor.py#L103
#
# Then, on each step, SB3 algos (PPO/DQN/...) put all `info["episode"]`
# dicts from the vec_env's step into `ep_info_buffer`:
# https://github.com/DLR-RM/stable-baselines3/blob/v1.8.0/stable_baselines3/common/base_class.py#L441
#
# This buffer can then be accessed in SB3 callbacks, which also have
# access to the SB3 log - and that's how user-defined values in `info`
# (set by QwopEnv) can be logged into tensorboard.
#
def create_vec_env(seed, max_episode_steps):
    venv = make_vec_env(
        "QwopEnv-v1",
        env_kwargs={"seed": seed},
        monitor_kwargs={"info_keywords": common.INFO_KEYS},
        wrapper_class=TimeLimit,
        wrapper_kwargs={"max_episode_steps": max_episode_steps},
    )

    return venv


def train_sb3(
    learner_cls,
    seed,
    run_id,
    learner_kwargs,
    learner_lr_schedule,
    total_timesteps,
    max_episode_steps,
    n_checkpoints,
    out_dir_template,
    log_tensorboard,
):
    venv = create_vec_env(seed, max_episode_steps)

    try:
        out_dir = out_dir_template.format(seed=seed, run_id=run_id)
        learning_rate = common.lr_from_schedule(learner_lr_schedule)

        model = init_model(
            venv=venv,
            seed=seed,
            learner_cls=learner_cls,
            learner_kwargs=learner_kwargs,
            learning_rate=learning_rate,
            log_tensorboard=log_tensorboard,
            out_dir=out_dir,
        )

        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,
            callback=[
                LogCallback(),
                CheckpointCallback(
                    save_freq=math.ceil(total_timesteps / n_checkpoints),
                    save_path=out_dir,
                    name_prefix="model",
                ),
            ],
        )

        # The CheckpointCallback kinda makes this redundant...
        common.save_model(out_dir, model)

        return {"out_dir": out_dir}
    finally:
        venv.close()
