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

import pathlib
import os


class Writer:
    def __init__(self, srcdir, dstdir, overwrite=None, browser_path=None, driver_path=None):
        os.makedirs(dstdir, exist_ok=True)

        self.srcdir = srcdir
        self.dstdir = dstdir
        self.overwrite = overwrite
        self.browser_path = browser_path
        self.driver_path = driver_path

    def maybe_write(self, fname, callback=None):
        srcfile = self.srcdir / fname
        dstfile = self.dstdir / fname

        with open(srcfile, "r") as f:
            content = f.read()

        if os.path.exists(dstfile) and not self.can_overwrite(dstfile):
            return

        content = callback(content) if callback else content

        print("Create %s" % dstfile)
        with open(dstfile, "w") as f:
            f.write(content)

    def can_overwrite(self, dstfile):
        if self.overwrite is not None:
            return self.overwrite

        while True:
            print(
                "\nFile %s already exists.\nOverwrite? [y/n/all/none] " % dstfile,
                end="",
            )

            match input():
                case "y" | "Y":
                    return True
                case "n" | "N":
                    return False
                case "all":
                    self.overwrite = True
                    return True
                case "none":
                    self.overwrite = False
                    return False
                case _:
                    print('Please answer with "y", "n", "all" or "none".')

    def replace_paths(self, content):
        if self.browser_path is None:
            print("Path to chrome-based browser: ", end="")
            self.browser_path = input()

        content = content.replace("/path/to/browser", self.browser_path)

        if self.driver_path is None:
            print("Path to chromedriver: ", end="")
            self.driver_path = input()

        content = content.replace("/path/to/chromedriver", self.driver_path)

        return content


def bootstrap():
    print("Performing initial setup...")

    w1 = Writer(
        pathlib.Path(__file__).parent / "templates",
        pathlib.Path("config"),
    )

    w1.maybe_write("env.yml", w1.replace_paths)
    w1.maybe_write("benchmark.yml")
    w1.maybe_write("play.yml")
    w1.maybe_write("record.yml")
    w1.maybe_write("replay.yml")
    w1.maybe_write("spectate.yml")
    w1.maybe_write("train_a2c.yml")
    w1.maybe_write("train_airl.yml")
    w1.maybe_write("train_bc.yml")
    w1.maybe_write("train_dqn.yml")
    w1.maybe_write("train_gail.yml")
    w1.maybe_write("train_ppo.yml")
    w1.maybe_write("train_qrdqn.yml")
    w1.maybe_write("train_rppo.yml")

    w2 = Writer(
        pathlib.Path(__file__).parent / "templates" / "wandb",
        pathlib.Path("config") / "wandb",
        w1.overwrite,
        browser_path=w1.browser_path,
        driver_path=w1.driver_path
    )

    w2.maybe_write("a2c.yml", w2.replace_paths)
    w2.maybe_write("dqn.yml", w2.replace_paths)
    w2.maybe_write("gail.yml", w2.replace_paths)
    w2.maybe_write("ppo.yml", w2.replace_paths)
    w2.maybe_write("qrdqn.yml", w2.replace_paths)
    w2.maybe_write("rppo.yml", w2.replace_paths)

    gamefile = (
        pathlib.Path(__file__).parents[1] / "envs" / "v1" / "game" / "QWOP.min.js"
    )

    if not os.path.exists(gamefile):
        print(
            """

To finish setup, run the following command:

    curl -sL https://www.foddy.net/QWOP.min.js | qwop-gym patch

or you can save QWOP.min.js locally and run `qwop-gym patch QWOP.min.js`.
"""
        )
