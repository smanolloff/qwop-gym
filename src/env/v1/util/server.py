import signal
import uuid
import socket
import asyncio
import pathlib
import websockets
import sys
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
        stepsize=1,
        stat_in_browser=False,
        text_in_browser=None,
        game_in_browser=True,
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

        self._steps = 0
        self._event = asyncio.Event()

        self._jspeer = Peer("js")
        self._pypeer = Peer("py")
        self._jspeer.other = self._pypeer
        self._pypeer.other = self._jspeer
        self._peers = {}
        self._manual_client = manual_client

    def cleanup(self, *args):
        self.driver.quit()
        if not self._future.done():
            self._future.set_result(0)

    def start(self):
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)

        with self.sock:
            asyncio.run(self._start())

    async def _start(self):
        self._future = asyncio.Future()

        async with websockets.serve(self.handler, sock=self.sock) as server:
            self.port = server.sockets[0].getsockname()[1]

            Log.log("Listening on port %d" % self.port)

            #
            # XXX: running ws.py in a fullscreen terminal means that
            #      the browser will NOT render anything if terminal is focused
            #       => screenshots will be a black box
            #       => better to run terminal in windowed mode
            #

            # wait until driver connects
            if self.driver and self.browser:
                asyncio.create_task(self._start_game())

                # XXX: I can't figure out why the code hangs forever
                #      when this task raises an exception :(
                await asyncio.wait_for(self._event.wait(), timeout=30)

            # run forever
            await self._future

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

    async def _start_game(self):
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

        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.get(self.build_url())
        Log.log("Browser started...")

    async def _register_peer(self, ws, peer_id):
        ua = ws.request_headers.get("user-agent")

        match peer_id:
            case WSProto.REG_JS:
                if self._jspeer.ws:
                    Log.log("error: already registered: JS peer %s" % ua)
                    await self.send(self._jspeer, to_bytes(WSProto.H_REJ))
                    await ws.close()
                else:
                    self._jspeer.ws = ws
                    self._jspeer.ua = ua
                    self._peers[ws] = self._jspeer

                    await self.send(self._jspeer, to_bytes(WSProto.H_ACK))

                    Log.log("Browser ready")
                    self._event.set()  # Unblock self._start()

            case WSProto.REG_PY:
                if self._pypeer.ws:
                    Log.log("error: already registered: PY peer %s" % ua)
                    await self.send(self._pypeer, to_bytes(WSProto.H_REJ))
                    await ws.close()
                else:
                    self._pypeer.ws = ws
                    self._pypeer.ua = ua
                    self._peers[ws] = self._pypeer

                    if not self._manual_client:
                        Log.log("Waiting for browser ready...")
                        await asyncio.wait_for(self._event.wait(), timeout=60)

                    await self.send(self._pypeer, to_bytes(WSProto.H_ACK))

    async def _reload(self, src_peer, seed):
        assert src_peer == self._pypeer, "Received RELOAD from a non-py peer"
        self.seed = seed

        Log.log("Closing JS connection")

        ws = self._jspeer.ws
        self._jspeer.ws = None
        self._event.clear()

        await ws.close()
        self.driver.get(self.build_url())
        Log.log("Waiting for browser ready...")
        await asyncio.wait_for(self._event.wait(), timeout=5)
        await self.send(self._pypeer, to_bytes(WSProto.H_ACK))

    async def handler(self, ws):
        try:
            await self._handler(ws)
        except Exception as e:
            self._future.set_exception(e)
            # pass

    async def _handler(self, ws):
        ua = ws.request_headers.get("user-agent")
        Log.log("Connection from: %s" % ua)

        async for data in ws:
            src = self._peers.get(ws)
            Log.log_inbound(data, src)

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
                    Log.log_remote(payload, src)
                case WSProto.H_ERR:
                    raise Exception("JS error: %s" % data[1:].decode())
                case _:
                    await self.send(src.other, data)

    async def send(self, peer, data):
        Log.log_outbound(data, peer)
        await peer.ws.send(data)


if __name__ == "__main__":
    ws = WSServer(socket.socket(), manual_client=True)
    gamefile = "/Users/simo/Projects/qwop/game/QWOP.html"

    with ws.sock:
        ws.start(gamefile, True)
