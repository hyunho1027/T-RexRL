import traceback
import tensorflow as tf
from dqn import DQN
from env import Env
import keyboard

if __name__=="__main__":
    agent = DQN()
    agent.load("models/model")
    env = Env()
    step = 0
    try:
        for ep in range(1000000):
            d = False
            score = 0
            s = env.reset()
            while not d:
                if keyboard.is_pressed('q'):
                    raise Exception("Quit.")
                a = agent.get_action([s], training=False)
                ns, r, d = env.step(a)
                step += 1
                s = ns
                score += r
            
            print(f"{ep+1} Episode / Step : {step} / Score : {score:.1f} /")

    except Exception as e:
        traceback.print_exc()
    env.close()
