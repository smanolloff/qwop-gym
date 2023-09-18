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

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.algorithms import bc
from imitation.util.util import make_vec_env
from imitation.util import logger
import numpy as np
import os
import time
import tools.common as common


def train_model(venv, seed, learner_kwargs, transitions, n_epochs, out_dir, log_tensorboard):
    venv.env_method("reload", seed)
    rng = np.random.default_rng(seed)
    log = None

    if log_tensorboard:
        os.makedirs(out_dir, exist_ok=True)
        log = logger.configure(folder=out_dir, format_strs=["tensorboard"])

    kwargs = dict(
        learner_kwargs,
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        demonstrations=transitions,
        rng=rng,
        custom_logger=log,
    )

    model = bc.BC(**kwargs)
    model.train(n_epochs=n_epochs, progress_bar=True)

    return model


def get_actions_and_sample_until_fns(rec):
    # rec is a {"file": rec_file, "seed": seed, "episodes": [[1, 3, 1], [...]]}

    state = {"no_more_recordings": False, "gi": 0}
    episodes_iter = iter([iter(actions) for actions in rec["episodes"]])
    episodes_completed = 0
    episodes_total = len(rec["episodes"])
    global_state = {
        "cur_episode": next(episodes_iter),
        "next_episode": next(episodes_iter, None),
        "episodes_completed": 0,
        "episodes_total": len(rec["episodes"]),
        "action_no": 0,
    }

    def get_actions(_observations, state, dones):
        assert len(dones) == 1, "vec envs with n_envs>1 are not supported"

        # print("actions taken: %d" % global_state["action_no"])

        if dones[0]:
            # each episode list contains actions exactly until episode ends
            action = next(global_state["cur_episode"], None)
            assert (
                action is None
            ), f"Expected end of episode, but have action {action} -- check seeds"

            # should never happen if `sample_until` has been returning false
            assert global_state["next_episode"] is not None, f"Expected more episodes"

            global_state.update(
                {
                    "cur_episode": global_state["next_episode"],
                    "next_episode": next(episodes_iter, None),
                    "episodes_completed": global_state["episodes_completed"] + 1,
                }
            )

        action = next(global_state["cur_episode"], None)
        global_state["action_no"] += 1

        assert action is not None, f"Unexpected end of recording -- check seeds"

        return [action], state

    def sample_until(_trajectories):
        return global_state["next_episode"] is None

    return get_actions, sample_until


def create_vec_env(seed):
    vec_env_kwargs = {
        "env_name": "QwopEnv-v1",
        # needed for computing rollouts later
        "post_wrappers": [lambda env, _: RolloutInfoWrapper(env)],
        # script will not work with n_envs>1
        "n_envs": 1,
        "rng": np.random.default_rng(),
    }

    venv = make_vec_env(env_make_kwargs={"seed": seed}, **vec_env_kwargs)

    return venv


def collect_transitions(venv, recs):
    rollouts = []
    for rec in recs:
        print("Collecting transitions from %s" % rec["file"])
        rng = np.random.default_rng(rec["seed"])
        venv.env_method("reload", rec["seed"])

        get_actions_fn, sample_until_fn = get_actions_and_sample_until_fns(rec)
        env_rollouts = rollout.rollout(get_actions_fn, venv, sample_until_fn, rng=rng)
        rollouts.extend(env_rollouts)

    transitions = rollout.flatten_trajectories(rollouts)
    print(f"Collected a total of {len(transitions)} transitions")

    return transitions


def save_model(out_dir, model):
    os.makedirs(out_dir, exist_ok=True)
    policy_file = os.path.join(out_dir, "model.zip")
    model.save_policy(policy_file)


def train_bc(seed, run_id, n_epochs, recordings, out_dir_template, learner_kwargs, log_tensorboard):
    recs = common.load_recordings(recordings)
    venv = create_vec_env(seed)

    try:
        out_dir = common.out_dir_from_template(out_dir_template, seed, run_id)
        transitions = collect_transitions(venv, recs)
        model = train_model(
            venv=venv,
            seed=seed,
            learner_kwargs=learner_kwargs,
            transitions=transitions,
            n_epochs=n_epochs,
            out_dir=out_dir,
            log_tensorboard=log_tensorboard,
        )

        save_model(out_dir, model)

        return {
            "recordings": list(map(lambda rec: rec["file"], recs)),
            "out_dir": out_dir,
        }
    finally:
        venv.close()
