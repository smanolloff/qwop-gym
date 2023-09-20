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
import tools.common as common


class Replayer:
    def __init__(self, actions):
        self.actions = actions
        self.iterator = iter(actions)
        self.i = 0

    def predict(self, _obs):
        self.i += 1
        return (next(self.iterator), None)


def bulk_skip(env, steps_per_step, model):
    done = False

    # Disable auto-draw
    old_auto_draw = env.unwrapped.auto_draw
    env.unwrapped.auto_draw = False

    if hasattr(env, "disable_verbose_wrapper"):
        env.disable_verbose_wrapper()

    print("skip start")
    while not done:
        action, _ = model.predict(None)
        for _ in range(steps_per_step):
            _, _, done, _ = env.step(action)
            if done:
                break
    print("skip env")

    if hasattr(env, "enable_verbose_wrapper"):
        env.enable_verbose_wrapper()

    env.unwrapped.auto_draw = old_auto_draw


def replay(fps, recordings, reset_delay, steps_per_step):
    env = None
    episode_ended_at = None

    try:
        for rec in common.load_recordings(recordings):
            print("Replaying episodes from %s" % rec["file"])

            if env:
                obs = env.reload(rec["seed"])
            else:
                env = gym.make("QwopEnv-v1", seed=rec["seed"])
                obs = env.reset()

            for i, episode in enumerate(rec["episodes"], 1):
                model = Replayer(episode["actions"])

                if episode["skip"]:
                    print("Skipping episode %d" % i)
                    bulk_skip(env, steps_per_step, model)
                    obs = env.reset()
                else:
                    if episode_ended_at:
                        sleep_for = reset_delay - (time.time() - episode_ended_at)
                        if sleep_for > 0:
                            time.sleep(sleep_for)
                    print("Playing episode %d" % i)

                    common.play_model(env, fps, steps_per_step, model, obs)
                    episode_ended_at = time.time()
                    obs = env.reset()

                # Recorded episodes should termiate at exactly the last action
                assert next(model.iterator, None) is None, f"Trailing actions"

        time.sleep(reset_delay)
    finally:
        if env:
            env.close()
