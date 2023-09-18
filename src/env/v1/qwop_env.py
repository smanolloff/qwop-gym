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

import socket
import numpy as np
from multiprocessing import Process
import gym
import itertools
import functools
import PIL
import PIL.Image
import io
import struct

from .util.wsproto import WSProto, to_bytes
from .util.server import WSServer
from .util.client import WSClient, WSClientMock
from .util.log import Log

BYTES_RESET = to_bytes(WSProto.H_CMD) + to_bytes(WSProto.CMD_RST)
BYTES_DRAW = to_bytes(WSProto.H_CMD) + to_bytes(WSProto.CMD_DRW)
BYTES_RENDER = to_bytes(WSProto.H_CMD) + to_bytes(WSProto.CMD_DRW | WSProto.CMD_IMG)
BYTES_RELOAD = to_bytes(WSProto.H_RLD)
INT_OBS = int(WSProto.H_OBS)
INT_IMG = int(WSProto.H_IMG)
INT_JPG = int(WSProto.IMG_JPG)

# the numpy data type
# it seems pytorch is optimized for float32
DTYPE = np.float32


class Reaction:
    def __init__(self, flags, time, distance, data, ndata):
        self.data = data
        self.ndata = ndata
        self.time = time
        self.distance = distance
        self.game_over = bool(flags & WSProto.OBS_END)

        # NOTE: Normally, `OBS_SUC` and `OBS_END` are both set whenever the
        #       athlete steps beyond the finish line (100 meters) or if the
        #       athlete reaches 105 meters without stepping at all.
        self.is_success = bool(flags & WSProto.OBS_SUC)


class Normalizable:
    def __init__(self, name, limit_min, limit_max):
        self.name = name
        self.limit_min = limit_min
        self.limit_max = limit_max
        self.center = DTYPE((limit_min + limit_max) / 2)
        self.maxdev = DTYPE(limit_max - self.center)
        self.min = DTYPE(0)
        self.max = DTYPE(0)

    def normalize(self, value):
        norm = (value - self.center) / self.maxdev

        if value > self.max:
            # assert value < self.limit_max, f"{self.name} max overflow: {value}"
            self.max = value
        elif value < self.min:
            # assert value > self.limit_min, f"{self.name} min overflow: {value}"
            self.min = value

        return norm

    def denormalize(self, norm):
        return norm * self.maxdev + self.center


class QwopEnv(gym.Env):
    metadata = {"render.modes": ["browser", "rgb_array"]}

    def __init__(
        self,
        render_mode="browser",
        driver=None,
        browser=None,
        failure_cost=10,
        success_reward=50,
        time_cost_mult=10,
        frames_per_step=1,
        stat_in_browser=False,  # statistics in the browser
        text_in_browser=None,
        game_in_browser=True,  # show the game itself in the browser
        reload_on_reset=False,  # reload web page on reset calls
        auto_draw=False,  # browser draw on each step
        r_for_terminate=False,
        seed=None,
        browser_mock=False,
        noop=None,
    ):
        seedval = seed or np.random.default_rng().integers(2**31)
        assert seedval >= 0 and seedval <= np.iinfo(np.int32).max
        self.seedval = int(seedval)

        if browser_mock:
            self.client = WSClientMock()
        else:
            sock = socket.socket()
            sock.bind(("localhost", 0))
            server = WSServer(
                sock,
                seed=self.seedval,
                stepsize=frames_per_step,
                stat_in_browser=stat_in_browser,
                game_in_browser=game_in_browser,
                text_in_browser=text_in_browser,
                driver=driver,
                browser=browser,
            )
            self.proc = Process(target=server.start)
            self.proc.start()
            self.client = WSClient(sock.getsockname()[1])

        self.screen = None
        self.auto_draw = auto_draw
        self.r_for_terminate = r_for_terminate
        self.reload_on_reset = reload_on_reset

        self._set_keycodes()

        self.render_mode = render_mode
        self.action_space = gym.spaces.Discrete(len(self.action_bytes))
        self.observation_space = gym.spaces.Box(
            shape=(60,), low=-1, high=1, dtype=DTYPE
        )

        self.speed_rew_mult = DTYPE(0.01)
        self.time_cost_mult = DTYPE(time_cost_mult)
        self.failure_cost = DTYPE(failure_cost)
        self.success_reward = DTYPE(success_reward)

        self.pos_x = Normalizable("pos_x", DTYPE(-10), DTYPE(1050))
        self.pos_y = Normalizable("pos_y", DTYPE(-10), DTYPE(10))
        self.angle = Normalizable("angle", DTYPE(-6), DTYPE(6))
        self.vel_x = Normalizable("vel_x", DTYPE(-20), DTYPE(60))
        self.vel_y = Normalizable("vel_y", DTYPE(-25), DTYPE(60))

        zeros = np.zeros(self.observation_space.shape, dtype=DTYPE)
        self.noop_reaction = Reaction(0, 0, 0, zeros, zeros)
        self.action_r = self.action_space.n - 1

        self.steps = 0
        self.last_reaction = self.noop_reaction
        self.last_reward = DTYPE(0)
        self.total_reward = DTYPE(0)

    def _set_keycodes(self):
        self.keycodes = [ord(x) for x in ["q", "w", "o", "p"]]
        self.keyflags = [
            WSProto.CMD_K_Q,
            WSProto.CMD_K_W,
            WSProto.CMD_K_O,
            WSProto.CMD_K_P,
        ]

        # Tuples of 0, 1, 2, 3 and 4 pressed keys
        keycodes_c0 = list(itertools.combinations(self.keycodes, 0))  # 0
        keycodes_c1 = list(itertools.combinations(self.keycodes, 1))  # 4: Q, W, O, P
        keycodes_c2 = list(itertools.combinations(self.keycodes, 2))  # 6: QW, QO, ...
        keycodes_c3 = list(itertools.combinations(self.keycodes, 3))  # 4: QWO, QWP, ...
        keycodes_c4 = list(itertools.combinations(self.keycodes, 4))  # 1: QWOP

        keyflags_c0 = list(itertools.combinations(self.keyflags, 0))
        keyflags_c1 = list(itertools.combinations(self.keyflags, 1))
        keyflags_c2 = list(itertools.combinations(self.keyflags, 2))
        keyflags_c3 = list(itertools.combinations(self.keyflags, 3))
        keyflags_c4 = list(itertools.combinations(self.keyflags, 4))

        keycodes_c = keycodes_c0 + keycodes_c1 + keycodes_c2 + keycodes_c3 + keycodes_c4
        keyflags_c = keyflags_c0 + keyflags_c1 + keyflags_c2 + keyflags_c3 + keyflags_c4

        self.keycodes_c = keycodes_c
        self.keyflags_c = keyflags_c

        # convert each tuple an integer (key flags) with an STP flag set
        self.action_flags = [
            functools.reduce(lambda a, e: a | e, t, WSProto.CMD_STP) for t in keyflags_c
        ]

        if self.auto_draw:
            # add draw flag to each command
            self.action_flags = [x | WSProto.CMD_DRW for x in self.action_flags]

        # convert each integer to a 2-bytestring: <CMD header> + <CMD flags>
        self.action_bytes = [
            to_bytes(WSProto.H_CMD) + to_bytes(k) for k in self.action_flags
        ]

        if self.r_for_terminate:
            # used in manual play:
            # replace the Q+W+O+P key-combination with the "R" key
            # and make it terminate the env instead (to be restarted)
            keycodes_c4 = [(ord("r"),)]
            keycodes_c[-1] = (ord("r"),)

    def seed(self, seed=None):
        # Can't re-seed -- the game has already been initialized
        # (forcing browser reload could fix that though)
        print(
            "WARNING: seed=%s ignored in seed() calls, call reload() instead" % seed
        )
        super()

    def reset(self):
        self._reset_env()
        reaction = self._restart_game(reload_page=self.reload_on_reset)
        return reaction.ndata

    # A custom type of "reset" that changes QWOP's seed
    # (it can be changed ONLY if reloading the page)
    def reload(self, seed):
        assert seed >= 0 and seed <= np.iinfo(np.int32).max
        Log.log("Re-loading env with new seed: %d" % seed)
        self.seedval = seed
        self._restart_game(reload_page=True)
        self.reset()

    def _reset_env(self):
        self.steps = 0
        self.last_reaction = self.noop_reaction
        self.last_reward = DTYPE(0)
        self.total_reward = DTYPE(0)

    def _restart_game(self, reload_page=False):
        if reload_page:
            data = self.client.send(BYTES_RELOAD + to_bytes(self.seedval, 4))
            assert data[0] == WSProto.H_ACK, f"expected an ACK header, got: {data[0]}"

        return self._build_reaction(self.client.send(BYTES_RESET))

    def step(self, action):
        self.steps += 1

        reaction = self._perform_action(action)

        reward = self._calc_reward(reaction, self.last_reaction)
        terminated = reaction.game_over or (
            self.r_for_terminate and action == self.action_r
        )

        info = {
            "time": reaction.time,
            "distance": reaction.distance,
            "avgspeed": reaction.distance / reaction.time,
            "is_success": reaction.is_success,
        }

        self.last_reward = reward  # QWOP stats
        self.total_reward += reward  # QWOP stats
        self.last_reaction = reaction  # needed for reward calc

        return reaction.ndata, reward, terminated, info

    def _perform_action(self, action):
        data = (
            self.action_bytes[action]
            + to_bytes(self.steps, 2)
            + struct.pack("=f", self.last_reward)
            + struct.pack("=f", self.total_reward)
        )

        resp = self.client.send(data)
        return self._build_reaction(resp)

    def _build_reaction(self, data):
        assert data[0] == INT_OBS, f"expected an OBS header, got: {data[0]}"

        flags = data[1]
        floats = np.frombuffer(data[2:], dtype=DTYPE)
        time = floats[0]
        distance = floats[1]
        obsdata = floats[2:]  # 60 floats (12 bodyparts, 5 floats per part)
        nobsdata = self._normalize(obsdata)
        return Reaction(flags, time, distance, obsdata, nobsdata)

    def _normalize(self, obs):
        length = len(obs)
        nobs = np.zeros(length, dtype=DTYPE)

        for i in range(0, len(obs), 5):
            nobs[i] = self.pos_x.normalize(obs[i])
            nobs[i + 1] = self.pos_y.normalize(obs[i + 1])
            nobs[i + 2] = self.angle.normalize(obs[i + 2])
            nobs[i + 3] = self.vel_x.normalize(obs[i + 3])
            nobs[i + 4] = self.vel_y.normalize(obs[i + 4])

        # Clamp values to -1..1
        np.clip(nobs, -1, 1, out=nobs)
        return nobs

    # r = reaction, lr = last_reaction
    def _calc_reward(self, reaction, last_reaction):
        ds = reaction.distance - last_reaction.distance
        dt = reaction.time - last_reaction.time
        v = ds / dt
        rew = v * self.speed_rew_mult - dt * self.time_cost_mult

        if reaction.game_over:
            if reaction.is_success:
                rew += self.success_reward
            else:
                rew -= self.failure_cost

        return rew

    def render(self, render_mode="browser"):
        match self.render_mode:
            case "rgb_array":
                return self.render_rgb()
            case "browser":
                return self.render_browser()
            case None:
                gym.logger.warn("No render mode set")
            case _:
                gym.logger.warn("Render mode not implemented: %s" % self.render_mode)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

        self.client.close()

        if self.proc and self.proc.is_alive():
            self.proc.terminate()

    def render_browser(self):
        self.client.send(BYTES_DRAW)

    def render_bytes(self):
        data = self.client.send(BYTES_RENDER)
        assert data[0] == INT_IMG, f"expected an IMG header, got: {data[0]}"
        assert data[1] == INT_JPG, f"expected JPEG format, got: {data[1]}"
        return data[2:]

    def render_img(self):
        PIL.Image.open(io.BytesIO(self.render_bytes())).show()

    def render_rgb(self):
        return np.array(PIL.Image.open(io.BytesIO(self.render_bytes())))

    def get_keys_to_action(self):
        return dict(
            [(tuple(sorted(keys)), i) for (i, keys) in enumerate(self.keycodes_c)]
        )
