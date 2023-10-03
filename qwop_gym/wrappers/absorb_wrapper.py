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


class AbsorbWrapper(gym.Wrapper):
    """Return terminated=False and the last non-terminal observation forever"""

    def __init__(self, env):
        super().__init__(env)
        self.terminal_return = None

    def reset(self, *args, **kwargs):
        self.terminal_return = None
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        if self.terminal_return is not None:
            return self.terminal_return

        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated:
            self.terminal_return = (obs, reward, False, truncated, info)

        return obs, reward, False, truncated, info
