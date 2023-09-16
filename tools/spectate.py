import gym
import time
import importlib
import tools.common as common


def load_model(mod_name, cls_name, file):
    print("Loading %s model from %s" % (cls_name, file))
    mod = importlib.import_module(mod_name)

    if cls_name == "BC":
        return mod.BC.reconstruct_policy(file)

    return getattr(mod, cls_name).load(file)


def spectate(fps, reset_delay, model_file, model_mod, model_cls):
    model = load_model(model_mod, model_cls, model_file)
    env = gym.make("QwopEnv-v1")

    try:
        while True:
            common.play_model(env, fps, model)
            time.sleep(reset_delay)
    finally:
        env.close()
