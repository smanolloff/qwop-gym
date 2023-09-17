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

    def predict(self, _obs):
        return (next(self.iterator), None)


def replay(fps, recordings, reset_delay):
    env = None

    try:
        for rec in common.load_recordings(recordings):
            if env:
                env.reload(rec["seed"])
            else:
                env = gym.make("QwopEnv-v1", seed=rec["seed"])
                env.reset()

            for i, episode in enumerate(rec["episodes"], start=1):
                print("Replaying episode %d from %s" % (i, rec["file"]))
                model = Replayer(episode)
                common.play_model(env, fps, 1, model)
                time.sleep(reset_delay)

                # Recorded episodes should termiate at exactly the last action
                assert next(model.iterator, None) is None, f"Trailing actions"
    finally:
        if env:
            env.close()
