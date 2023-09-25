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

import sys
import os

if __name__ == "__main__":
    replacements = [
        (
            "!function()",
            "function QWOP()",
        ),
        (
            "this.doneLoading=!0",
            'this.doneLoading=!0,this.app.app.window.handle.dispatchEvent(new Event("doneLoading"))',
        ),
        (
            "&&m.shutdown()",
            "",
        ),
        (
            "t.preprocess_sound_meta(t.pack.sounds),",
            "",
        ),
        (
            "t.load(t.pack.sounds,s(t,t.create_sound)),",
            "",
        ),
        (
            "o.Core.__super__=C.AppFixedTimestep",
            "o.Core.__super__=C.App",
        ),
        (
            "o.Core.prototype=t(C.AppFixedTimestep.prototype",
            "o.Core.prototype=t(C.App.prototype",
        ),
        (
            "this.request_update(),!0",
            "this.__manual_mode||this.request_update(),!0",
        ),
        (
            "u.main()",
            "u.main(),QWOP.__i=i",
        ),
        (
            "}();",
            "};",
        ),
    ]

    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            source = f.read()
    else:
        source = sys.stdin.read()

    for old, new in replacements:
        if old not in source:
            raise Exception("Could not find substring '%s'" % old)
        source = source.replace(old, new)

    outfile = os.path.join(os.path.dirname(__file__), "QWOP.min.js")
    print("Wrote %s" % outfile)
    with open(outfile, "w") as f:
        f.write(source)
