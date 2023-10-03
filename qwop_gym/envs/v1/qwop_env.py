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
import multiprocessing
import gymnasium as gym
import itertools
import functools
import PIL
import PIL.Image
import io
import struct
import logging

from .util.wsproto import WSProto, to_bytes
from .util.wsserver import WSServer
from .util.wsclient import WSClient, WSClientMock
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
    """
    A Gym environment for Bennet Foddy's game called _QWOP_.

    browser: Path to the web browser executable.
    driver: Path to the chromedriver executable.
    render_mode: Either "browser" and "rgb_array":
        With "browser", a call to `.render()` renders the current frame
        in the browser and `None` is returned. With "rgb_array", the
        same happens, but the frame itself is returned as RGB image data.
    failure_cost: Subtracted from the reward at the end of unsuccessful
        episodes.
    success_reward: Added to the reward at the end of successful episodes.
    time_cost_mult: Multiplier for the amount subtracted from the reward
        at each step.
    frames_per_step: Number of frames to advance per call to `.step`
        (aka. frameskip).
    stat_in_browser: Display a table with various game stats in the browser
        next to the game area.
    game_in_browser: Display the game area itself in the browser.
    text_in_browser: Display a static text next to the game area in the
        browser.
    reload_on_reset: Perform a page reload on each call to `.reset`
        (aka. "hard reset").
    auto_draw: Automatically draw the current frame on each call to `.step`.
    reduced_action_set: Reduce possible actions from 16 to just 9:
        Genuine set: (none),Q,W,O,P,QW,QO,QP,WO,WP,OP,QWO,QWP,QOP,WOP,QWOP
        Reduced set: (none),Q,W,O,P,QW,QP,WO,OP.
    t_for_terminate: Map an additional action to the T key for terminating
        the env (useful when a human is playing)
    loglevel: Logger level (DEBUG|INFO|WARN|ERROR).
    seed: Initial seed for QWOP.min.js's RNG.
    """

    metadata = {"render_modes": ["browser", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        browser=None,
        driver=None,
        render_mode="browser",
        failure_cost=10,
        success_reward=50,
        time_cost_mult=10,
        frames_per_step=1,
        stat_in_browser=False,
        game_in_browser=True,
        text_in_browser=None,
        reload_on_reset=False,
        auto_draw=False,
        reduced_action_set=False,
        t_for_terminate=False,
        loglevel="WARN",
        seed=None,
        browser_mock=False,
    ):
        seedval = seed or np.random.default_rng().integers(2**31)
        assert seedval >= 0 and seedval <= np.iinfo(np.int32).max
        self.seedval = int(seedval)

        self.frames_per_step = frames_per_step

        if browser_mock:
            self.client = WSClientMock()
        else:
            if browser is None:
                raise ValueError(
                    "please specify a valid path to a chrome-based browser executable"
                    + " via the `browser` constructor argument"
                )
            if driver is None:
                raise ValueError(
                    "please specify a valid path to a chromedriver executable via"
                    + " the `driver` constructor argument"
                )

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
                loglevel=loglevel,
            )
            self.shutdown = multiprocessing.Event()
            self.proc = multiprocessing.Process(
                target=server.start, kwargs={"shutdown": self.shutdown}
            )
            self.proc.start()
            self.client = WSClient(sock.getsockname()[1], loglevel, self.shutdown)

        self.auto_draw = auto_draw
        self.t_for_terminate = t_for_terminate
        self.reload_on_reset = reload_on_reset
        self.logger = Log.get_logger(__name__, loglevel)

        self.reduced_action_set = reduced_action_set
        self._set_keycodes()

        self.render_mode = render_mode
        self.action_space = gym.spaces.Discrete(len(self.action_cmdflags))
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

        if self.t_for_terminate:
            # "T" is always the last action
            self.action_t = self.action_space.n - 1
        else:
            self.action_t = None

        self.steps = 0
        self.last_reaction = self.noop_reaction
        self.last_reward = DTYPE(0)
        self.total_reward = DTYPE(0)

        self.logger.info("Initialized with seed: %d" % self.seedval)

    def _set_keycodes(self):
        keymap = {
            "q": WSProto.CMD_K_Q,
            "w": WSProto.CMD_K_W,
            "o": WSProto.CMD_K_O,
            "p": WSProto.CMD_K_P,
        }

        keycodes = [ord(x) for x in keymap.keys()]
        keyflags = keymap.values()

        # All possible key combinations represented as lists of tuples
        self.keycodes_c = (
            list(itertools.combinations(keycodes, 0))  # 0
            + list(itertools.combinations(keycodes, 1))  # 4: Q, W, O, P
            + list(itertools.combinations(keycodes, 2))  # 6: QW, QO, ...
            + list(itertools.combinations(keycodes, 3))  # 4: QWO, QWP, ...
            + list(itertools.combinations(keycodes, 4))  # 1: QWOP
        )

        self.keyflags_c = (
            list(itertools.combinations(keyflags, 0))
            + list(itertools.combinations(keyflags, 1))
            + list(itertools.combinations(keyflags, 2))
            + list(itertools.combinations(keyflags, 3))
            + list(itertools.combinations(keyflags, 4))
        )

        if self.reduced_action_set:
            # Remove useless key combinations instead of
            # letting agents work it out on their own
            redundant_combinations = [
                ("q", "o"),
                ("w", "p"),
                ("q", "w", "o"),
                ("q", "w", "p"),
                ("q", "o", "p"),
                ("w", "o", "p"),
                ("q", "w", "o", "p"),
            ]

            for rc in redundant_combinations:
                self.keycodes_c.remove(tuple(ord(x) for x in rc))
                self.keyflags_c.remove(tuple(keymap.get(x) for x in rc))

        # Key combinations represented as WSProto cmdflags
        self.action_cmdflags = [
            functools.reduce(lambda a, e: a | e, t, 0) for t in self.keyflags_c
        ]

        if self.t_for_terminate:
            # add an extra "T" key which terminates env immediately
            self.keycodes_c.append((ord("t"),))
            self.keyflags_c.append(0)
            self.action_cmdflags.append(0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._reset_env()
        needs_reload = self.reload_on_reset

        if seed is not None:
            # QWOP's seed can be changed ONLY if reloading the page
            assert seed >= 0 and seed <= np.iinfo(np.int32).max
            self.seedval = seed
            needs_reload = True

        reaction = self._restart_game(reload_page=needs_reload)
        return reaction.ndata, self._build_info(reaction)

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
        terminated = reaction.game_over or action == self.action_t
        info = self._build_info(reaction)

        self.last_reward = reward  # QWOP stats
        self.total_reward += reward  # QWOP stats
        self.last_reaction = reaction  # needed for reward calc

        return reaction.ndata, reward, terminated, False, info

    def _perform_action(self, action):
        cmdflags = WSProto.CMD_STP | self.action_cmdflags[action]

        if self.auto_draw:
            cmdflags |= WSProto.CMD_DRW

        data = (
            to_bytes(WSProto.H_CMD)
            + to_bytes(cmdflags)
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
        rew = v * self.speed_rew_mult - dt * self.time_cost_mult / self.frames_per_step

        if reaction.game_over:
            if reaction.is_success:
                rew += self.success_reward
            else:
                rew -= self.failure_cost

        return rew

    def _build_info(self, reaction):
        return {
            "time": reaction.time,
            "distance": reaction.distance,
            "avgspeed": reaction.distance / reaction.time,
            "is_success": reaction.is_success,
        }

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
        self.client.close()

        if self.proc and self.proc.is_alive():
            self.shutdown.set()
            self.proc.join(timeout=2)
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
