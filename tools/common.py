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

from src.env.v1.qwop_env import QwopEnv

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


class RecordWrapper(gym.Wrapper):
    def __init__(self, env, rec_file, overwrite, max_time, min_distance, complete_only):
        super().__init__(env)

        if os.path.exists(rec_file) and not overwrite:
            raise Exception("rec_file already exists: %s" % rec_file)

        print("Recording to %s" % rec_file)
        self.handle = open(rec_file, "w")
        self.handle.write("seed=%d\n" % env.seedval)
        self.overwrite = overwrite
        self.max_time = max_time or 999
        self.min_distance = min_distance or 0
        self.complete_only = complete_only
        self.actions = []
        self.discarded_episodes = []

    def step(self, action):
        obs, reward, terminated, info = self.env.step(action)
        self.actions.append(str(action))

        if terminated:
            ep_info = "episode with time=%.2f and distance=%.2f" % (
                info["time"],
                info["distance"],
            )
            incomplete = self.env.r_for_terminate and action == 15

            # only write actions to file if this was NOT a manual reset
            if self.complete_only and incomplete:
                print("Discarded %s (incomplete)" % ep_info)
                self.discarded_episodes.append(self.actions)
            elif info["time"] > self.max_time:
                print("Discarded %s (max_time exceeded)" % ep_info)
                self.discarded_episodes.append(self.actions)
            elif info["distance"] < self.min_distance:
                print("Discarded %s (min_distance not reached)" % ep_info)
                self.discarded_episodes.append(self.actions)
            else:
                # Dump discarded episodes actions only when there is
                # a regular episode after them
                if len(self.discarded_episodes) > 0:
                    print("Dump %d discarded episodes" % len(self.discarded_episodes))
                    episodes = [ep + ["X"] for ep in self.discarded_episodes]
                    actions = [a for ep in episodes for a in ep]
                    self.handle.write("\n".join(actions) + "\n")
                    self.discarded_episodes = []

                self.handle.write("\n".join(self.actions) + "\n*\n")

                if incomplete:
                    print("Recorded incomplete %s" % ep_info)
                else:
                    print("Recorded %s" % ep_info)

            self.actions = []
        elif info.get("manual_restart"):
            self.actions = []

        return obs, reward, terminated, info


def expand_env_kwargs(env_kwargs):
    env_include_cfg = env_kwargs.pop("__include__", None)

    if env_include_cfg:
        with open(env_include_cfg, "r") as f:
            env_kwargs = yaml.safe_load(f) | env_kwargs

    return env_kwargs


def register_env(env_kwargs={}, env_wrappers=[]):
    def wrapped_env_creator(**kwargs):
        env = QwopEnv(**kwargs)

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


def out_dir_from_template(tmpl, seed, run_id):
    out_dir = tmpl.format(seed=seed, run_id=run_id)

    if os.path.exists(out_dir):
        raise Exception("Output directory already exists: %s" % out_dir)

    return out_dir


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

        # episode is dict of {"skip": [bool], "actions": [...]}
        actions = []
        n_episodes = 0

        for line in f:
            line = line.rstrip()

            if line == "*":
                episodes.append({"skip": False, "actions": actions})
                n_episodes += 1
                actions = []
            elif line == "X":
                episodes.append({"skip": True, "actions": actions})
                n_episodes += 1
                actions = []
            else:
                actions.append(int(line))

    if n_episodes == 0:
        print("Empty recording %s" % recfile)
    else:
        print("Loaded %d episodes with seed=%d from %s " % (n_episodes, seed, recfile))

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


def play_model(env, fps, steps_per_step, model, obs):
    done = False
    normfps = 30  # ~ game fps at "normal" speed
    t1 = time.time()
    clock = Clock(fps)
    done = False

    print("play start")
    while not done:
        action, _states = model.predict(obs)
        for _ in range(steps_per_step):
            obs, reward, done, info = env.step(action)
            clock.tick()
            if done:
                break
    print("play end")


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
