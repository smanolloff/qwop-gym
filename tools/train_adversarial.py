from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.adversarial.gail import GAIL
from imitation.util.util import make_vec_env
from imitation.util import logger
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import safe_mean
import numpy as np
import importlib
import os
import time
import gym
import tools.common as common


class AbsorbWrapper(gym.Wrapper):
    """Return done=False and the last non-terminal observation forever"""

    def __init__(self, env):
        super().__init__(env)
        self.terminal_return = None

    def reset(self, *args, **kwargs):
        self.terminal_return = None
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        if self.terminal_return is not None:
            return self.terminal_return

        obs, reward, terminated, info = self.env.step(action)

        if terminated:
            self.terminal_return = (obs, reward, False, info)

        return obs, reward, False, info


def get_actions_and_sample_until_fns(rec, episode_len):
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
        "steps": 0,
        "steps_this_episode": 0,
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
                    "steps_this_episode": 0,
                }
            )

        action = next(global_state["cur_episode"], None)
        global_state["steps"] += 1
        global_state["steps_this_episode"] += 1

        if global_state["steps_this_episode"] == episode_len:
            assert (
                action is None
            ), f"Unexpected end of recording at step {global_state['steps_this_episode']}"

        # need to return a valid action
        return [action or 0], state

    def sample_until(_trajectories):
        return global_state["next_episode"] is None

    return get_actions, sample_until


def create_inf_length_env(**kwargs):
    """Infinite-length variant of QwopEnv-v1.

    In the event of early episode completion (i.e., the athlete falls),
    we enter an absorbing state that repeats the final observation and reward.
    """
    env = gym.make("QwopEnv-v1", **kwargs)
    env = AbsorbWrapper(env)
    return env


#
# See note in train_sb3.create_vec_env()
#
# Unfortunately, `imitation`'s `make_vec_env` does accept `monitor_kwargs`
# and it always wraps the env within a default Monitor wrapper.
# As a workaround, we wrap the env in a second Monitor, then remove
# the first one.
#
# Unfortunately #2, `imitation` hardcodes its own learner callback
# meaning we can't provide an SB3 callback which logs metrics.
# Addressed in https://github.com/HumanCompatibleAI/imitation/pull/786
# Until then, we use `imitation`'s simple callback (a callable),
# which gets called much less often, but could serve as a workaround.
#
def create_fixed_length_vec_env(max_episode_steps, seed):
    gym.envs.register(id="QwopInfEnv-v1", entry_point=create_inf_length_env)

    vec_env_kwargs = {
        "env_name": "QwopInfEnv-v1",
        "post_wrappers": [
            lambda env, _: gym.wrappers.TimeLimit(env, max_episode_steps),
            lambda env, _: Monitor(env, info_keywords=common.INFO_KEYS),
            # needed for computing rollouts later
            lambda env, _: RolloutInfoWrapper(env),
        ],
        # script will not work with n_envs>1
        "n_envs": 1,
        "rng": np.random.default_rng(),
    }

    venv = make_vec_env(env_make_kwargs={"seed": seed}, **vec_env_kwargs)

    # Find the Monitor with info_keywords == ()
    # then remove it from the chain
    prev_env = venv.envs[0]
    env = venv.envs[0].env
    while env.__class__ != Monitor or env.info_keywords != ():
        prev_env = env
        env = env.env

    prev_env.env = env.env

    return venv


def collect_rollouts(venv, episode_len, recs):
    rollouts = []

    for rec in recs:
        print("Collecting rollouts from %s" % rec["file"])
        rng = np.random.default_rng(rec["seed"])
        venv.env_method("reload", rec["seed"])

        get_actions_fn, sample_until_fn = get_actions_and_sample_until_fns(
            rec, episode_len
        )
        env_rollouts = rollout.rollout(get_actions_fn, venv, sample_until_fn, rng=rng)
        rollouts.extend(env_rollouts)

    print(f"Collected a total of {len(rollouts)} rollouts")
    return rollouts


def init_trainer(trainer_cls, kwargs):
    match trainer_cls:
        case "GAIL":
            return GAIL(**kwargs)
        case "AIRL":
            return AIRL(**kwargs)
        case _:
            raise Exception("Unknown trainer class: %s" % trainer_cls)


def train_learner(
    venv,
    seed,
    rollouts,
    learner,
    trainer_cls,
    trainer_kwargs,
    total_timesteps,
    out_dir,
    log_tensorboard,
):
    venv.env_method("reload", seed)
    rng = np.random.default_rng(seed)
    log = None

    if log_tensorboard:
        os.makedirs(out_dir, exist_ok=True)
        log = logger.configure(folder=out_dir, format_strs=["tensorboard"])

    reward_net = BasicShapedRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )

    trainer = init_trainer(
        trainer_cls,
        dict(
            trainer_kwargs,
            venv=venv,
            demonstrations=rollouts,
            gen_algo=learner,
            reward_net=reward_net,
            custom_logger=log,
        ),
    )

    def cb(_i):
        for k in common.INFO_KEYS:
            v = safe_mean([ep_info[k] for ep_info in learner.ep_info_buffer])
            learner.logger.record(f"user/{k}", v)

    trainer.train(total_timesteps=total_timesteps, callback=cb)


def init_learner(venv, seed, module_name, cls_name, lr_schedule, kwargs):
    lr = common.lr_from_schedule(lr_schedule)
    mod = importlib.import_module(module_name)
    klass = getattr(mod, cls_name)

    return klass(**dict(kwargs, env=venv, learning_rate=lr, seed=seed))


def train_adversarial(
    trainer_cls,
    seed,
    run_id,
    recordings,
    out_dir_template,
    log_tensorboard,
    learner_module,
    learner_cls,
    learner_lr_schedule,
    learner_kwargs,
    trainer_kwargs,
    episode_len,
    total_timesteps,
):
    recs = common.load_recordings(recordings)

    # Adversarial training works with fixed-length envs
    venv = create_fixed_length_vec_env(episode_len, seed)

    try:
        out_dir = out_dir_template.format(seed=seed, run_id=run_id)
        rollouts = collect_rollouts(venv, episode_len, recs)

        learner = init_learner(
            venv=venv,
            seed=seed,
            module_name=learner_module,
            cls_name=learner_cls,
            lr_schedule=learner_lr_schedule,
            kwargs=learner_kwargs,
        )

        train_learner(
            venv=venv,
            seed=seed,
            rollouts=rollouts,
            learner=learner,
            trainer_cls=trainer_cls,
            trainer_kwargs=trainer_kwargs,
            total_timesteps=total_timesteps,
            out_dir=out_dir,
            log_tensorboard=log_tensorboard,
        )

        common.save_model(out_dir, learner)

        return {
            "recordings": list(map(lambda rec: rec["file"], recs)),
            "out_dir": out_dir,
        }
    finally:
        venv.close()
