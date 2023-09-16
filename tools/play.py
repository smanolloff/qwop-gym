import gym
import gym.utils.play


def play(fps):
    env = gym.make("QwopEnv-v1")

    try:
        # Unfortunately, this will immediately reset on termination
        # (gym.utils.play() does not allow control over this)
        gym.utils.play.play(env, fps=fps)
    finally:
        env.close()
