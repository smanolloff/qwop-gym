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

import os
import gymnasium as gym


class RecordWrapper(gym.Wrapper):
    def __init__(self, env, rec_file, overwrite, max_time, min_distance):
        super().__init__(env)

        if os.path.exists(rec_file) and not overwrite:
            raise Exception("rec_file already exists: %s" % rec_file)

        os.makedirs(os.path.dirname(rec_file), exist_ok=True)

        print("Recording to %s" % rec_file)
        self.handle = open(rec_file, "w")
        self.handle.write("seed=%d\n" % env.unwrapped.seedval)
        self.overwrite = overwrite
        self.max_time = max_time or 999
        self.min_distance = min_distance or 0
        self.actions = []
        self.discarded_episodes = []

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.actions.append(str(action))

        if terminated:
            ep_info = "episode with time=%.2f" % info["time"]
            ep_info += " and distance=%.2f" % info["distance"]
            incomplete = action == self.env.unwrapped.action_t

            if incomplete:
                print("Discarded %s (incomplete)" % ep_info)
                self.discarded_episodes.append(self.actions)
            elif info["time"] > self.max_time:
                print("Discarded %s (max_time exceeded)" % ep_info)
                self.discarded_episodes.append(self.actions)
            elif info["distance"] < self.min_distance:
                print("Discarded %s (min_distance not reached)" % ep_info)
                self.discarded_episodes.append(self.actions)
            else:
                # Dump discarded episodes only when there is
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

        return obs, reward, terminated, truncated, info
