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

import importlib
import string
import random
import sys
import os
import numpy as np
import ast
import time
import wandb


if __name__ == "__main__":
    seed = int(np.random.default_rng().integers(2**31))
    run_id = os.environ["WANDB_RUN_ID"]

    # Fetch and expand `out_dir_template`
    cfgstr = next(v[4:] for v in sys.argv if v.startswith("--c="))
    cfg = ast.literal_eval(cfgstr)
    out_dir = cfg["out_dir_template"].format(run_id=run_id, seed=seed)
    out_dir = os.path.join(os.path.dirname(__file__), out_dir)
    print("out dir: %s" % out_dir)

    # 4: Patch tensorboard before wandb.init
    # (to suppresses wandb warnings from adversarial training)
    # DISABLED: it makes lognames from pure PPO training too long
    # wandb.tensorboard.patch(root_logdir=out_dir)
    wandb.init(sync_tensorboard=True)

    action = wandb.config["action"]
    config = dict(
        wandb.config["c"],
        run_id=run_id,
        seed=seed,
        out_dir=out_dir,
    )

    main = importlib.import_module("qwop-gym").main
    main(action, config)

    wandb.save(f"{out_dir}/model.zip", base_path="/")
    wandb.save(f"{out_dir}/metadata.yml", base_path="/")
