# =============================================================================
# Copyright 2023 Simeon Manolov <s.manolloff@gmail.com>.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import sys
import os
import importlib
import yaml
import argparse
from copy import deepcopy

import tools.common as common


def main(action, cfg, tag=None):
    env_wrappers = cfg.pop("env_wrappers", {})
    env_kwargs = cfg.pop("env_kwargs", {})
    expanded_env_kwargs = common.expand_env_kwargs(env_kwargs)
    common.register_env(expanded_env_kwargs, env_wrappers)

    match action:
        case "play":
            from tools.play import play

            play(
                seed=cfg.get("seed", None) or common.gen_seed(),
                run_id=cfg.get("run_id", None) or common.gen_id(),
                fps=cfg.get("fps", 30),
            )
        case "replay":
            from tools.replay import replay

            replay(
                fps=cfg.get("fps", 30),
                reset_delay=cfg.get("reset_delay", 1),
                recordings=cfg.get("recordings", "data/recordings/*.rec"),
                steps_per_step=cfg.get("steps_per_step", 1),
            )
        case "spectate":
            from tools.spectate import spectate

            spectate(
                fps=cfg.get("fps", 30),
                reset_delay=cfg.get("reset_delay", 1),
                steps_per_step=cfg.get("steps_per_step", 1),
                model_file=cfg["model_file"],
                model_mod=cfg.get("model_mod", "stable_baselines3"),
                model_cls=cfg.get("model_cls", "PPO"),
            )
        case "train_bc":
            print_imitation_error_and_exit()
            from tools.train_bc import train_bc

            run_config = deepcopy(
                {
                    "seed": cfg.get("seed", None) or common.gen_seed(),
                    "run_id": cfg.get("run_id", None) or common.gen_id(),
                    "learner_kwargs": cfg.get("learner_kwargs", {}),
                    "n_epochs": cfg.get("n_epochs", 100),
                    "recordings": cfg.get("recordings", ["data/recordings/*.rec"]),
                    "out_dir_template": cfg.get("out_dir_template", "data/BC-{run_id}"),
                    "log_tensorboard": cfg.get("log_tensorboard", False),
                }
            )

            run_duration, run_values = common.measure(train_bc, run_config)

            print("saving run metadata...")
            common.save_run_metadata(
                action=action,
                cfg=dict(run_config, env_kwargs=env_kwargs),
                duration=run_duration,
                values=dict(run_values, env=expanded_env_kwargs),
            )
            print("saved run metadata.")
        case "train_gail" | "train_airl":
            print_imitation_error_and_exit()
            from tools.train_adversarial import train_adversarial

            trainer_cls = action.split("_")[-1].upper()
            learner_cls = cfg.get("learner_cls", "PPO")
            default_template = "data/%s_%s-{run_id}" % (
                trainer_cls,
                learner_cls,
            )

            # trainer_cls is not part of the config
            run_config = deepcopy(
                {
                    "seed": cfg.get("seed", None) or common.gen_seed(),
                    "run_id": cfg.get("run_id", None) or common.gen_id(),
                    "recordings": cfg.get("recordings", "data/recordings/*.rec"),
                    "out_dir_template": cfg.get("out_dir_template", default_template),
                    "log_tensorboard": cfg.get("log_tensorboard", False),
                    "learner_module": cfg.get("learner_module", "stable_baselines3"),
                    "learner_cls": learner_cls,
                    "learner_kwargs": cfg.get("learner_kwargs", {}),
                    "trainer_kwargs": cfg.get("trainer_kwargs", {}),
                    "total_timesteps": cfg.get("total_timesteps", 1000000),
                    "episode_len": cfg.get("episode_len", 500),
                    "learner_lr_schedule": cfg.get(
                        "learner_lr_schedule", "const_0.001"
                    ),
                }
            )

            run_duration, run_values = common.measure(
                train_adversarial, dict(run_config, trainer_cls=trainer_cls)
            )

            print("saving run metadata...")
            common.save_run_metadata(
                action=action,
                cfg=dict(run_config, env_kwargs=env_kwargs),
                duration=run_duration,
                values=dict(run_values, env=expanded_env_kwargs),
            )
            print("saved run metadata.")
        case "train_ppo" | "train_dqn" | "train_qrdqn":
            from tools.train_sb3 import train_sb3

            learner_cls = action.split("_")[-1].upper()
            default_template = "data/%s-{run_id}" % learner_cls

            # learner_cls is not part of the config
            run_config = deepcopy(
                {
                    "seed": cfg.get("seed", None) or common.gen_seed(),
                    "run_id": cfg.get("run_id", None) or common.gen_id(),
                    "model_load_file": cfg.get("model_load_file", None),
                    "out_dir_template": cfg.get("out_dir_template", default_template),
                    "log_tensorboard": cfg.get("log_tensorboard", False),
                    "learner_kwargs": cfg.get("learner_kwargs", {}),
                    "total_timesteps": cfg.get("total_timesteps", 1000000),
                    "max_episode_steps": cfg.get("max_episode_steps", 5000),
                    "n_checkpoints": cfg.get("n_checkpoints", 5),
                    "learner_lr_schedule": cfg.get(
                        "learner_lr_schedule", "const_0.003"
                    ),
                }
            )

            run_duration, run_values = common.measure(
                train_sb3, dict(run_config, learner_cls=learner_cls)
            )

            common.save_run_metadata(
                action=action,
                cfg=dict(run_config, env_kwargs=env_kwargs),
                duration=run_duration,
                values=dict(run_values, env=expanded_env_kwargs),
            )
        case "benchmark":
            from tools.benchmark import benchmark

            benchmark(steps=cfg.get("steps", 10000))
        case _:
            print("Unknown action: %s" % action)


# XXX: temporary until `imitation` release a gymnasium-compatible version
def print_imitation_error_and_exit():
    print(
        "If you want to do imitation learning, please use the `gym-compat` git branch"
    )
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", help=argparse.SUPPRESS)
    parser.add_argument(
        "-c",
        metavar="FILE",
        type=argparse.FileType("r"),
        help="config file, defaults to config/<action>.yml",
    )

    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.usage = "%(prog)s [options] <action>"
    parser.epilog = """
action:
  play              play QWOP, optionally recording actions
  replay            replay recorded game actions
  train_bc          train using Behavioral Cloning (BC)
  train_gail        train using Generative Adversarial Imitation Learning (GAIL)
  train_airl        train using Adversarial Inverse Reinforcement Learning (AIRL)
  train_ppo         train using Proximal Policy Optimization (PPO)
  train_dqn         train using Deep Q Network (DQN)
  train_qrdqn       train using Quantile Regression DQN (QRDQN)
  spectate          watch a trained model play QWOP, optionally recording actions
  benchmark         evaluate the actions/s achievable with this env
  help              print this help message

examples:
  qwop-gym.py play
  qwop-gym.py -c config/play.yml play
"""

    args = parser.parse_args()

    if args.c is None:
        args.c = open(os.path.join("config", f"{args.action}.yml"), "r")

    print("Loading configuration from %s" % args.c.name)
    cfg = yaml.safe_load(args.c)
    args.c.close()

    main(args.action, cfg)
