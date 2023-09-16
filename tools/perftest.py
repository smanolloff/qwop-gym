import random
import time
import gym


def perftest():
    try:
        n_steps = 100000
        env = gym.make("QwopEnv-v2")
        env.reset()
        t1 = time.time()

        for i in range(n_steps):
            (obs, rew, term, inf) = env.step(random.randint(0, 3))
            if term:
                env.reset()

        t2 = time.time()
        print("Done %d in %.2fs" % (n_steps, t2 - t1))

    finally:
        if locals().get("env"):
            env.close()
