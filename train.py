import tensorflow as tf
import traceback
import keyboard
from dqn import DQN
from env import Env

if __name__=="__main__":
    agent = DQN()
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
                a = agent.get_action([s])
                ns, r, d = env.step(a)
                agent.append_sample(s,a,r,ns,d)
                step += 1
                s = ns
                score += r
            
            if step > 500:
                env.alt_tab()
                for _ in range(8):
                    losses.append(agent.train())
                env.alt_tab()

            if losses:
                agent.epsilon_decay()
                if ep%5==0:
                    agent.update_target()
                if ep%10==0:
                    agent.save("./models/model")
            
            loss = tf.reduce_mean(losses)
            print(f"{ep+1} Episode / Step : {step} / Score : {score:.1f} / Loss : {loss:.4f} / Epsilon : {agent.e:.4f}")
            agent.write(score, loss, agent.e, ep+1)

    except Exception as e:
        traceback.print_exc()
    env.close()
