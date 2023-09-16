import os
import time
import gym
import gym.utils.play
import tools.common as common


class RecordWrapper(gym.Wrapper):
    def __init__(self, env, rec_file):
        super().__init__(env)

        print("Recording to %s" % rec_file)
        self.handle = open(rec_file, "w")
        self.handle.write("seed=%d\n" % env.seedval)
        self.actions = []

    def step(self, action):
        obs, reward, terminated, info = self.env.step(action)
        self.actions.append(str(action))

        if terminated:
            # only write actions to file if this was NOT a manual reset
            if not (self.env.r_for_terminate and action == 15):
                # not a manual reset => write actions
                self.handle.write("\n".join(self.actions))
                self.handle.write("\nX\n")

            self.actions = []
        elif info.get("manual_restart"):
            self.actions = []

        return obs, reward, terminated, info


def record(seed, run_id, fps, out_file_template):
    env = gym.make("QwopEnv-v1", seed=seed)

    os.makedirs(os.path.dirname(out_file_template), exist_ok=True)
    rec_file = out_file_template.format(seed=seed, run_id=run_id)

    try:
        env = RecordWrapper(env, rec_file)

        # Unfortunately, this will immediately reset on termination
        # (gym.utils.play() does not allow control over this)
        gym.utils.play.play(env, fps=fps)
    finally:
        print("Recording saved as %s" % rec_file)
        env.close()
