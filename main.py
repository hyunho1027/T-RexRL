import tensorflow as tf
import traceback
import keyboard
from dqn import DQN
from env import Env

training = False
load_model = True

if __name__=="__main__":
    agent = DQN()
    if load_model : 
        agent.load("models/model")
    env = Env()
    step = 0
    losses = []
    try:
        for ep in range(1000000):
            d = False
            score = 0
            s = env.reset()
            while not d:
                if keyboard.is_pressed('q'):
                    raise Exception("Quit.")
                a = agent.get_action([s], training)
                ns, r, d = env.step(a)
                if training:
                    agent.append_sample(s,a,r,ns,d)
                step += 1
                s = ns
                score += r
            
            if training and step > 500:
                for _ in range(8):
                    losses.append(agent.train())

                agent.epsilon_decay()
                if ep%5==0:
                    agent.update_target()
                if ep%10==0:
                    agent.save("./models/model")
                
                loss = tf.reduce_mean(losses)
                print(f"{ep+1} Episode / Step : {step} / Score : {score:.1f} / " +\
                      f"Loss : {loss:.4f} / Epsilon : {agent.e:.4f}")
                agent.write(score, loss, agent.e, ep+1)
            else:
                print(f"{ep+1} Episode / Step : {step} / Score : {score:.1f} /")

    except Exception as e:
        traceback.print_exc()
    env.close()
