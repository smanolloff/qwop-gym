import importlib
import time
import datetime
import os
import re
import glob
import gym
import yaml
import random
import string
import numpy as np

from src.env.v1.env import Env

# Keys of user-defined metrics in the `info` dict
INFO_KEYS = ("time", "distance", "avgspeed", "is_success")


class Clock:
    """A better alternative to pygame.Clock for our use-case"""

    def __init__(self, fps):
        self.fps = fps
        self.min_interval = 1 / fps
        self.last_tick_at = time.time()

    def tick(self):
        tick_at = time.time()
        interval = tick_at - self.last_tick_at
        sleep_for = self.min_interval - interval

        if sleep_for > 0:
            time.sleep(sleep_for)

        self.last_tick_at = tick_at + sleep_for


def expand_env_kwargs(env_kwargs):
    env_include_cfg = env_kwargs.pop("__include__", None)

    if env_include_cfg:
        with open(env_include_cfg, "r") as f:
            env_kwargs = dict(yaml.safe_load(f), **env_kwargs)

    return env_kwargs


def register_env(env_kwargs={}, env_wrappers=[]):
    def wrapped_env_creator(**kwargs):
        env = Env(**kwargs)

        for wrapper in env_wrappers:
            wrapper_mod = importlib.import_module(wrapper["module"])
            wrapper_cls = getattr(wrapper_mod, wrapper["cls"])
            env = wrapper_cls(env, **wrapper.get("kwargs", {}))

        return env

    gym.envs.register(
        id="QwopEnv-v1", entry_point=wrapped_env_creator, kwargs=env_kwargs
    )


def gen_seed():
    return int(np.random.default_rng().integers(2**31))


def gen_id():
    population = string.ascii_lowercase + string.digits
    return str.join("", random.choices(population, k=8))


def load_recordings(rec_file_patterns):
    recs = []

    for rfp in rec_file_patterns:
        for rec_file in sorted(glob.glob(rfp)):
            rec = load_recording(rec_file)

            if len(rec["episodes"]) > 0:
                recs.append(rec)

    if len(recs) == 0:
        print("No recordings found")
        exit(1)

    return recs


def load_recording(recfile):
    # print("Loading recording: %s" % recfile)

    episodes = []

    with open(recfile) as f:
        rechead = f.readline()

        # The seed of the recording is required in order to
        # ensure the actions are replayed deterministically
        m = re.match("^seed=(\\d+)$", rechead)
        assert m, "Failed to parse header for recording: %s" % recfile
        seed = int(m.group(1))

        # episode is a list of actions: [1, 1, 0, 4, 0, 2, ...]
        episode = []

        for line in f:
            if line.rstrip() == "X":
                episodes.append(episode)
                episode = []
            else:
                episode.append(int(line))

    if len(episodes) == 0:
        print("Empty recording %s" % recfile)
    else:
        print(
            "Loaded %d episodes with seed=%d from %s " % (len(episodes), seed, recfile)
        )

    return {"file": recfile, "seed": seed, "episodes": episodes}


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


def lr_from_schedule(schedule):
    match schedule["fn"]:
        case "const":
            return schedule["initial_value"]
        case "step_decay":
            return step_decay_fn(
                schedule["initial_value"],
                schedule["step"],
                schedule["decays"],
                schedule["min_value"],
            )
        case _:
            print(
                "Invalid config value for learner_lr_schedule.fn: %s" % schedule["fn"]
            )
            exit(1)


def play_model(env, fps, model):
    done = False
    normfps = 30  # ~ game fps at "normal" speed
    obs = env.reset()
    t1 = time.time()
    clock = Clock(fps)
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        clock.tick()


def save_model(out_dir, model):
    os.makedirs(out_dir, exist_ok=True)
    model_file = os.path.join(out_dir, "model.zip")
    model.save(model_file)


def save_config(out_dir, config):
    os.makedirs(out_dir, exist_ok=True)
    config_file = os.path.join(out_dir, "config.yml")

    with open(config_file, "w") as f:
        f.write(yaml.safe_dump(config))


def measure(func, kwargs):
    t1 = time.time()
    retval = func(**kwargs)
    t2 = time.time()

    return t2 - t1, retval


def save_run_metadata(action, cfg, duration, values):
    out_dir = values["out_dir"]
    metadata = dict(values, action=action, config=cfg, duration=duration)

    print("Output directory: %s" % out_dir)
    os.makedirs(out_dir, exist_ok=True)
    md_file = os.path.join(out_dir, "metadata.yml")

    with open(md_file, "w") as f:
        f.write(yaml.safe_dump(metadata))
