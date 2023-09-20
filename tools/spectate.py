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

import gym
import time
import importlib
import tools.common as common


def load_model(mod_name, cls_name, file):
    print("Loading %s model from %s" % (cls_name, file))
    mod = importlib.import_module(mod_name)

    if cls_name == "BC":
        return mod.BC.reconstruct_policy(file)

    return getattr(mod, cls_name).load(file)


def spectate(
    fps,
    reset_delay,
    steps_per_step,
    model_file,
    model_mod,
    model_cls,
):
    model = load_model(model_mod, model_cls, model_file)
    env = gym.make("QwopEnv-v1")

    try:
        while True:
            obs = env.reset()
            common.play_model(env, fps, steps_per_step, model, obs)
            time.sleep(reset_delay)
    finally:
        env.close()
