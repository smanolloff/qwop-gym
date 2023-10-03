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

import logging
import hashlib
from datetime import datetime
from rich.logging import RichHandler

from .wsproto import WSProto


class RelativeTimeFormatter(logging.Formatter):
    def format(self, record):
        reltime = record.relativeCreated
        record.reltime = "%.2f" % (record.relativeCreated / 1000)
        return super().format(record)


class Log:
    # 0 = print ALL messages
    # 1 = print a (short) version of ALL messages
    # 2 = filter out in/outbound messages, print the rest
    # 3 = print only the explicit LOG calls
    # 4 = silence

    LEVEL = 4
    DELIM_OUT = ">" * 79
    DELIM_IN = "<" * 79
    DELIM_REMOTE = ":" * 79

    # for logging purposes
    HMAP = {
        WSProto.H_REG: "H_REG",
        WSProto.H_ACK: "H_ACK",
        WSProto.H_REJ: "H_REJ",
        WSProto.H_CMD: "H_CMD",
        WSProto.H_OBS: "H_OBS",
        WSProto.H_IMG: "H_IMG",
        WSProto.H_LOG: "H_LOG",
        WSProto.H_ERR: "H_ERR",
        WSProto.H_RLD: "H_RLD",
    }

    REGMAP = {
        WSProto.REG_JS: "REG_JS",
        WSProto.REG_PY: "REG_PY",
    }

    CMDMAP = {
        WSProto.CMD_STP: "STP",
        WSProto.CMD_K_Q: "K_Q",
        WSProto.CMD_K_W: "K_W",
        WSProto.CMD_K_O: "K_O",
        WSProto.CMD_K_P: "K_P",
        WSProto.CMD_RST: "RST",
        WSProto.CMD_IMG: "IMG",
        WSProto.CMD_DRW: "DRW",
    }

    OBSMAP = {
        WSProto.OBS_PAS: "PAS",
        WSProto.OBS_END: "END",
    }

    IMGMAP = {
        WSProto.IMG_JPG: "JPG",
        WSProto.IMG_PNG: "PNG",
    }

    def get_logger(name, level):
        logger = logging.getLogger(name.split(".")[-1])
        logger.setLevel(getattr(logging, level))

        fmt = "-- %(reltime)ss [%(name)s] %(levelname)s: %(message)s"
        formatter = RelativeTimeFormatter(fmt)

        loghandler = logging.StreamHandler()
        loghandler.setLevel(logging.DEBUG)
        loghandler.setFormatter(formatter)
        logger.addHandler(loghandler)

        return logger

    def format_remote(msg, client, longfmt):
        client_name = client.name if client else "??"

        if longfmt:
            return "%s\n[ %s ] %s" % (
                Log.DELIM_REMOTE,
                client_name,
                msg.decode("utf-8"),
            )
        else:
            return "[ %s ] %s" % (client_name, msg.decode("utf-8"))

    def format_outbound(data, client):
        return Log.format_allbound(data, client, "out")

    def format_inbound(data, client):
        return Log.format_allbound(data, client, "in")

    def format_allbound(data, client, direction):
        client_name = client.name if client else "??"

        if direction == "in":
            delim = Log.DELIM_IN
            arrow = "<--"
        else:
            delim = Log.DELIM_OUT
            arrow = "-->"

        header_uint = data[0]
        header_name = Log.HMAP.get(header_uint)
        header_repr = f"{header_uint:08b}"

        if header_uint == WSProto.H_REG:
            header2_uint = data[1]
            header2_name = Log.REGMAP.get(header2_uint)
            header2_repr = f"{header2_uint:08b}"

            line2 = "%-16s | %-16s |" % (header_name, header2_name)
            line3 = "%-16s | %-16s | %s" % (
                header_repr,
                header2_repr,
                Log.data_repr(data[2:]),
            )
        elif header_uint == WSProto.H_CMD:
            header2_uint = data[1]
            names = []
            for k, v in Log.CMDMAP.items():
                if header2_uint & k:
                    names.append(v)

            header2_name = "+".join(names)
            header2_repr = f"{header2_uint:08b}"

            line2 = "%-16s | %-16s | (data)" % (header_name, header2_name)
            line3 = "%-16s | %-16s | %s" % (
                header_repr,
                header2_repr,
                Log.data_repr(data[2:]),
            )
        elif header_uint == WSProto.H_OBS:
            header2_uint = data[1]
            names = []
            for k, v in Log.OBSMAP.items():
                if header2_uint & k:
                    names.append(v)

            header2_name = "+".join(names)
            header2_repr = f"{header2_uint:08b}"

            # line2 = "%-16s | %-16s | %-16s | md5" % (header_name, header2_name, "(data)")
            # line3 = "%-16s | %-16s | %-16s | %s" % (header_repr, header2_repr, Log.data_repr(data[2:]), Log.md5(data[2:]))
            line2 = "%-16s | %-16s | %-16s" % (header_name, header2_name, "(data)")
            line3 = "%-16s | %-16s | %-16s" % (
                header_repr,
                header2_repr,
                Log.data_repr(data[2:]),
            )
        elif header_uint == WSProto.H_IMG:
            header2_uint = data[1]
            header2_name = Log.IMGMAP.get(header2_uint)
            header2_repr = f"{header2_uint:08b}"

            # line2 = "%-16s | %-16s | %-16s | md5" % (header_name, header2_name, "(data)")
            # line3 = "%-16s | %-16s | %-16s | %s" % (header_repr, header2_repr, Log.data_repr(data[2:]), Log.md5(data[2:]))
            line2 = "%-16s | %-16s | %-16s" % (header_name, header2_name, "(data)")
            line3 = "%-16s | %-16s | %-16s" % (
                header_repr,
                header2_repr,
                Log.data_repr(data[2:]),
            )
        else:
            # line2 = "%-16s | %-16s | md5" % (header_name, "(data)")
            # line3 = "%-16s | %-16s | %s" % (header_repr, Log.data_repr(data[1:]), Log.md5(data[2:]))
            line2 = "%-16s | %-16s" % (header_name, "(data)")
            line3 = "%-16s | %-16s" % (header_repr, Log.data_repr(data[1:]))

        delim = "-" * 79

        if Log.LEVEL == 0:
            return "%s\n[%s%s] %s\n[%s%s] %s" % (
                delim,
                arrow,
                client_name,
                line2,
                arrow,
                client_name,
                line3,
            )
        else:
            return "[%s%s] %s" % (arrow, client_name, line2)

    def data_repr(data):
        x = data.hex()
        return x if len(x) <= 16 else (x[:13] + "...")

    def md5(data):
        md5 = hashlib.md5()
        md5.update(data)
        return md5.hexdigest()
