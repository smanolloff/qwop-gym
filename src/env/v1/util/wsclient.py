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

import numpy as np
import time
from websockets.sync import client
from .wsproto import WSProto, to_bytes
from .log import Log
import sys


class WSClient:
    def __init__(self, port, loglevel):
        self.port = port
        self.logger = Log.get_logger(__name__, loglevel)
        self.connect()

    def connect(self):
        while True:
            try:
                self._connect_attempt()
                self.logger.debug("Connected")
                break
            except Exception as e:
                self.logger.warn("Failed to connect: %s" % str(e))
                time.sleep(5)
                pass

    def _connect_attempt(self):
        self.ws = client.connect(f"ws://localhost:{self.port}", open_timeout=10)
        out = to_bytes(WSProto.H_REG) + to_bytes(WSProto.REG_PY)

        self.ws.send(out)
        data = self.ws.recv()

        if data[0] != WSProto.H_ACK:
            exp = np.binary_repr(WSProto.H_ACK)
            got = np.binary_repr(data[0])
            raise Exception("Header error: expected %s, got: %s" % (exp, got))

    def send(self, data):
        while True:
            try:
                self.ws.send(data)
                return self.ws.recv(timeout=5)
            except Exception as e:
                self.logger.error("Failed to send/receive: %s" % str(e))
                try:
                    self.close()
                except Exception as e1:
                    self.logger.error("Failed to close connection: %s" % str(e1))

                self.logger.info("Reconnecting in 5s...")
                time.sleep(5)
                self.connect()

    def close(self):
        self.ws.close()
        self.ws.recv_events_thread.join()


class WSClientMock:
    def send(self, _data):
        return self.recv()

    def recv(self):
        return (
            to_bytes(WSProto.H_OBS)
            + to_bytes(0)
            + np.zeros(65, dtype=np.float32).tobytes()
        )

    def close(self):
        pass
