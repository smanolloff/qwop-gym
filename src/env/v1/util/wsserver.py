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

import signal
import uuid
import socket
import asyncio
import pathlib
import websockets
import sys
import os
import urllib.parse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService

from .wsproto import WSProto, to_bytes
from .log import Log


class Peer:
    def __init__(self, name):
        self.ws = None
        self.other = None
        self.name = name
        self.ua = None


class WSServer:
    def __init__(
        self,
        sock,
        seed,
        driver,
        browser,
        stepsize,
        stat_in_browser,
        text_in_browser,
        game_in_browser,
        loglevel,
        manual_client=False,
    ):
        seedmin = -9007199254740991  # js Number.MIN_SAFE_INTEGER
        seedmax = 9007199254740991  # js Number.MAX_SAFE_INTEGER
        assert (
            seed >= seedmin and seed <= seedmax
        ), f"seed must be between {seedmin} and {seedmax}"

        self.sock = sock
        self.seed = seed
        self.stat_in_browser = stat_in_browser
        self.game_in_browser = game_in_browser
        self.text_in_browser = text_in_browser
        self.driver = driver
        self.browser = browser
        self.stepsize = stepsize
        self.loglevel = loglevel

        self._steps = 0
        self._event = asyncio.Event()

        self._jspeer = Peer("js")
        self._pypeer = Peer("py")
        self._jspeer.other = self._pypeer
        self._pypeer.other = self._jspeer
        self._peers = {}
        self._manual_client = manual_client
        self._window = None
        self._driver = None
        self._initialized = False

    def cleanup_and_exit(self):
        if not self._shutdown.is_set():
            self.logger.info("Shutting down")

        self._shutdown.set()
        self._event.set()
        if not self._future.done():
            self._future.set_result(0)

    def start(self, shutdown):
        # must set logger here, as .start() is called in another process
        self.logger = Log.get_logger(__name__, self.loglevel)
        self._shutdown = shutdown
        self._future = asyncio.Future()

        loop = asyncio.get_event_loop()
        loop.create_task(self.check_shutdown())

        if os.name == "posix":
            loop.add_signal_handler(signal.SIGINT, self.cleanup_and_exit)
            loop.add_signal_handler(signal.SIGTERM, self.cleanup_and_exit)

        with self.sock:
            while not shutdown.is_set():
                loop.run_until_complete(self._start())

            if self._driver:
                self._driver.quit()

    async def check_shutdown(self):
        while not self._shutdown.is_set():
            await asyncio.sleep(0.1)
        self.logger.info("Shutting down")
        self.cleanup_and_exit()

    async def _start(self):
        async with websockets.serve(self.handler, sock=self.sock) as server:
            self.port = server.sockets[0].getsockname()[1]

            self.logger.info("Listening on port %d" % self.port)

            task = None

            # wait until driver connects
            if self.driver and self.browser:
                task = asyncio.create_task(self._launch_browser())
                await asyncio.wait_for(self._event.wait(), timeout=30)

            if task and task.exception():
                self.logger.error(task.exception())
                self.cleanup_and_exit()

            await self._future
            server.close()

    def build_url(self):
        file = pathlib.Path(__file__).parents[3].joinpath("game", "QWOP.html")
        url = "file://%s" % file
        url += "?port=%d" % self.port
        url += "&seed=%d" % self.seed
        url += "&game=%d" % self.game_in_browser
        url += "&stat=%d" % self.stat_in_browser
        url += "&text=%s" % urllib.parse.quote_plus(self.text_in_browser or "")
        url += "&intro=0"
        url += "&stepsize=%d" % self.stepsize

        return url

    async def _launch_browser(self):
        self.logger.info("Launching web browser...")

        options = webdriver.ChromeOptions()
        options.add_argument("allow-file-access-from-files")
        options.add_argument("allow-cross-origin-auth-prompt")
        options.add_argument("user-agent=Chrome-%s" % uuid.uuid4())
        options.add_argument("disable-infobars")
        options.add_argument("disable-extensions")
        options.add_argument("disable-popup-blocking")
        options.add_argument("disable-notifications")

        if self.game_in_browser and self.stat_in_browser:
            options.add_argument("window-size=1160,585")
            options.add_argument("window-position=650,130")
        else:
            options.add_argument("window-size=660,585")
            options.add_argument("window-position=650,130")

        # keeping a ref to the driver keeps it running
        # also, it allows to be explicitly closed on exit

        # https://www.selenium.dev/documentation/webdriver/getting_started/upgrade_to_selenium_4/#python-1
        service = ChromeService(executable_path=self.driver)

        # https://community.brave.com/t/is-there-a-selenium-driver-for-brave-browser/49696/4
        options.binary_location = self.browser
        options.add_argument("--incognito")

        self._driver = webdriver.Chrome(service=service, options=options)
        self._window = self._driver.window_handles[0]
        self._driver.get(self.build_url())
        self._initialized = True

    def _maybe_relaunch_browser(self):
        if not self._initialized:
            self.cleanup_and_exit()
            return

        try:
            if self._driver and self._window in self._driver.window_handles:
                # window is alive
                return
            else:
                self.logger.error("Browser window not found")
        except Exception as e:
            self.logger.error(str(e))
            pass

        # window is dead
        if self._event.is_set():
            self._event.clear()
            asyncio.create_task(self._launch_browser())

    async def _register_peer(self, ws, peer_id):
        ua = ws.request_headers.get("user-agent")

        match peer_id:
            case WSProto.REG_JS:
                if self._jspeer.ws:
                    await self._jspeer.ws.close()

                self._jspeer.ws = ws
                self._jspeer.ua = ua
                self._peers[ws] = self._jspeer

                await self.send(self._jspeer, to_bytes(WSProto.H_ACK))

                self.logger.info("Browser (js client) registration ACK")
                self._event.set()  # Unblock self._start()

            case WSProto.REG_PY:
                if self._pypeer.ws:
                    await self._pypeer.ws.close()

                self._pypeer.ws = ws
                self._pypeer.ua = ua
                self._peers[ws] = self._pypeer

                # If py client reconnects, maybe the browser is dead
                self._maybe_relaunch_browser()

                if not self._manual_client:
                    self.logger.info("Waiting for browser...")
                    await asyncio.wait_for(self._event.wait(), timeout=60)

                await self.send(self._pypeer, to_bytes(WSProto.H_ACK))
                self.logger.info("QwopEnv (py client) registration ACK")

    async def _reload(self, src_peer, seed):
        assert src_peer == self._pypeer, "Received RELOAD from a non-py peer"
        self.logger.info("Reloading browser page with new seed: %s" % seed)

        self.seed = seed

        self.logger.info("Closing JS connection")

        ws = self._jspeer.ws
        self._jspeer.ws = None
        self._event.clear()

        await ws.close()
        self._driver.get(self.build_url())
        self.logger.info("Waiting for browser ready...")
        await asyncio.wait_for(self._event.wait(), timeout=5)
        await self.send(self._pypeer, to_bytes(WSProto.H_ACK))

    async def handler(self, ws):
        try:
            await self._handler(ws)
        except Exception as e:
            self.logger.error(str(e))

    async def _handler(self, ws):
        ua = ws.request_headers.get("user-agent")
        self.logger.info("Connection from: %s" % ua)

        async for data in ws:
            src = self._peers.get(ws)
            self.logger.debug(Log.format_inbound(data, src))

            header = data[0]
            payload = data[1:]

            match header:
                # put most common match cases on top
                case WSProto.H_OBS | WSProto.H_CMD:
                    await self.send(src.other, data)
                case WSProto.H_REG:
                    await self._register_peer(ws, payload[0])
                case WSProto.H_RLD:
                    await self._reload(src, int.from_bytes(payload[0:4], sys.byteorder))
                case WSProto.H_LOG:
                    longfmt = self.logger.level == logging.DEBUG
                    self.logger.info(Log.format_remote(payload, src, longfmt))
                case WSProto.H_ERR:
                    raise Exception("JS error: %s" % data[1:].decode())
                case _:
                    await self.send(src.other, data)

    async def send(self, peer, data):
        self.logger.debug(Log.format_outbound(data, peer))
        await peer.ws.send(data)


if __name__ == "__main__":
    ws = WSServer(
        sock=socket.socket(),
        manual_client=True,
        seed=0,
        browser="/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        driver="/Users/simo/Projects/qwop/vendor/chromedriver",
    )

    with ws.sock:
        ws.start()
