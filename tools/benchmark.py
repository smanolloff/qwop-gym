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


def benchmark(steps):
    env = gym.make("QwopEnv-v1")

    try:
        env.reset()
        time_start = time.time()

        for i in range(steps):
            _obs, _rew, term, _info = env.step(0)

            if term:
                env.reset()

            if i % 1000 == 0:
                percentage = (i / steps) * 100
                print("\r%d%%..." % percentage, end="", flush=True)

        seconds = time.time() - time_start
        sps = steps / seconds
        print("\n\n%.2f steps/s (%s steps in %.2f seconds)" % (sps, steps, seconds))
    finally:
        env.close()
