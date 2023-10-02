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

import gymnasium as gym
import time
import tools.common as common


def replay(fps, recordings, reset_delay, steps_per_step):
    env = None
    episode_ended_at = None

    try:
        for rec in common.load_recordings(recordings):
            print("Replaying episodes from %s" % rec["file"])

            if env:
                obs, _ = env.reset(seed=rec["seed"])
            else:
                env = gym.make("QwopEnv-v1", seed=rec["seed"])
                # 2 resets are needed as gymnasium.utils.play also
                # calls it twice after init
                env.reset()
                obs, _ = env.reset()

            for i, episode in enumerate(rec["episodes"], 1):
                model = common.Replayer(episode["actions"])

                if episode["skip"]:
                    print("Skipping episode %d" % i)
                    common.skip_episode(env, steps_per_step, model)
                    obs, _ = env.reset()
                else:
                    if episode_ended_at:
                        sleep_for = reset_delay - (time.time() - episode_ended_at)
                        if sleep_for > 0:
                            time.sleep(sleep_for)
                    print("Playing episode %d" % i)

                    common.play_model(env, fps, steps_per_step, model, obs)
                    episode_ended_at = time.time()
                    obs, _ = env.reset()

                # Recorded episodes should termiate at exactly the last action
                assert next(model.iterator, None) is None, f"Trailing actions"

        time.sleep(reset_delay)
    finally:
        if env:
            env.close()
