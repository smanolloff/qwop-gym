import stable_baselines3 as sb3
import sb3_contrib
import json5 as json

from wandb.integration.sb3 import WandbCallback

import env_registrar

import numpy as np
import gym
import wandb
import importlib
import os
import sys


def step_decay_fn(initial_value, frac, decays=10, min_value=0):
    milestones = [step / (decays + 1) for step in range(1, decays + 1)]
    # [0.9, 0.8, ... 0.1] for 9 decays
    milestones.reverse()

    def func(progress_remaining: float) -> float:
        value = initial_value
        for m in milestones:
            if progress_remaining < m:
                value *= frac
                if value < min_value:
                    break
            else:
                break
        return value

    return func


def init_vec_env():
    version = wandb.config["common"]["env"]["version"]
    env_id = "%s-%s" % (wandb.config["common"]["env"]["id"], version)

    try:
        env_kwargs = wandb.config["env_kwargs"]
    except KeyError:
        env_kwargs = {}

    vec_env_kwargs = wandb.config["common"]["env"]["vec_env_kwargs"].copy()
    vec_env_kwargs["vec_env_cls"] = getattr(
        sb3.common.vec_env, vec_env_kwargs["vec_env_cls"]
    )
    return sb3.common.env_util.make_vec_env(
        env_id, env_kwargs=env_kwargs, **vec_env_kwargs
    )


def init_model(run, load_path=None):
    mod_name = wandb.config["common"].get("mod", "sb3")
    mod = globals()[mod_name]
    alg_class = getattr(mod, wandb.config["common"]["alg"])

    kwargs = wandb.config["model"].copy()
    kwargs["tensorboard_log"] = wandb.config["common"]["tensorboard_log"].format(
        run=run
    )
    kwargs["device"] = wandb.config["common"]["device"]

    lr_config = kwargs.pop("learning_rate_config")

    match lr_config["schedule"]:
        case "const":
            kwargs["learning_rate"] = lr_config["initial_value"]
        case "lin_decay":
            kwargs["learning_rate"] = step_decay_fn(
                lr_config["initial_value"],
                lr_config["step"],
                lr_config["decays"],
                lr_config["min_value"],
            )

    kwargs["env"] = init_vec_env()

    load_path = wandb.config["common"]["model_load_file"]

    if load_path:
        print("Load model: %s" % load_path)
        return alg_class.load(load_path, **kwargs, reset_num_timesteps=False)
    else:
        return alg_class(**kwargs)


def train_model(run, model):
    total_timesteps = wandb.config["common"]["total_timesteps"]

    version = wandb.config["common"]["env"]["version"]
    n_envs = wandb.config["common"]["env"]["vec_env_kwargs"]["n_envs"]
    cb_mod = importlib.import_module(f"src.{version}.callback")

    cb = cb_mod.CryptoCallback(
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        desired_logs=wandb.config["common"]["desired_logs"],
        model_save_path=wandb.config["common"]["model_save_path"].format(run=run),
    )

    learn_kwargs = {
        "callback": cb,
        "total_timesteps": total_timesteps,
        "progress_bar": wandb.config["common"]["progress_bar"],
    }

    try:
        model.learn(**learn_kwargs)
    finally:
        model.env.close()


def train(run):
    # print(json.dumps(wandb.config.as_dict(), indent=4))
    print("!11111!!!!!!!!!!!!!!!!!")
    exit(1)
    model = init_model(run)
    train_model(run, model)


if __name__ == "__main__":
    print("!!!222222!!!!!!!!!!!!!!!")
    exit(1)
    env_registrar.register()

    if len(sys.argv) > 1 and sys.argv[1] == "offline":
        with open("config/ppo-manual.json", "r") as f:
            cfg = json.loads(f.read())

        run = wandb.init(
            sync_tensorboard=True, project="qwop", mode="offline", config=cfg
        )
    else:
        run = wandb.init(sync_tensorboard=True, project="qwop")

    fname = "%s/config.json" % wandb.config["common"]["model_save_path"].format(run=run)
    print(f"Create {fname}")

    d = os.path.dirname(fname)
    if d and not os.path.isdir(d):
        os.makedirs(d)

    with open(fname, "w", encoding="utf-8") as f:
        json.dump(wandb.config.as_dict(), f, indent=4)
    train(run)
