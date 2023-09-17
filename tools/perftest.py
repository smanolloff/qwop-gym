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

import random
import time
import gym


def perftest():
    try:
        n_steps = 100000
        env = gym.make("QwopEnv-v2")
        env.reset()
        t1 = time.time()

        for i in range(n_steps):
            (obs, rew, term, inf) = env.step(random.randint(0, 3))
            if term:
                env.reset()

        t2 = time.time()
        print("Done %d in %.2fs" % (n_steps, t2 - t1))

    finally:
        if locals().get("env"):
            env.close()
