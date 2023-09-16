import numpy as np
import time
from websockets.sync import client
from .wsproto import WSProto, to_bytes
from .log import Log
import sys


class WSClient:
    def __init__(self, port):
        self.port = port
        max_retries = 5
        sleep = 5  # seconds
        timeout = 10  # seconds
        retries = 0

        while True:
            try:
                self.ws = client.connect(f"ws://localhost:{port}", open_timeout=timeout)
                out = to_bytes(WSProto.H_REG) + to_bytes(WSProto.REG_PY)
                retries = 0

                while True:
                    if retries >= max_retries:
                        raise Exception(
                            "WS registration failed after %s attempts" % retries
                        )

                    self.ws.send(out)
                    data = self.ws.recv()

                    if data[0] == WSProto.H_ACK:
                        Log.log("Registration successful")
                        return

                    Log.log(
                        "(%d) Failed to register: %s"
                        % (retries, np.binary_repr(data[0]))
                    )
                    retries += 1

            except Exception as e:
                if retries >= max_retries:
                    raise

                retries += 1
                Log.log("(%d) Failed to connect: %s" % (retries, e))
                time.sleep(sleep)
                pass

    def send(self, data):
        self.ws.send(data)
        return self.ws.recv()

    def recv(self):
        return self.ws.recv()

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
