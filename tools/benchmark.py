import gym
import time


def benchmark(steps):
    env = gym.make("QwopEnv-v1")

    try:
        env.reset()
        time_start = time.time()

        for i in range(steps):
            _obs, _rew, term, _info = env.step(0)

            if term:
                env.reset()

            if i % 1000 == 0:
                print(".", end="", flush=True)

        seconds = time.time() - time_start
        sps = steps / seconds
        print("\n%.2f steps/s (%s steps in %.2f seconds)" % (sps, steps, seconds))
    finally:
        env.close()
