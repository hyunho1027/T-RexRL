import traceback
from dqn import DQN
from env import Env

if __name__=="__main__":
    agent = DQN()
    env = Env()
    loss = None
    step = 0
    try:
        for ep in range(10000):
            d = False
            score = 0
            s = env.reset()
            while not d:
                a = agent.get_action([s])
                ns, r, d = env.step(a)
                if score > 0.5:
                    agent.append_sample(s,a,r,ns,d)
                    step += 1
                s = ns
                score += r
            
                if step > 500:
                    loss = agent.train()

            if loss:
                agent.epsilon_decay()
                if ep%5==0:
                    agent.update_target()
                if ep%10==0:
                    agent.save("./models/model")

            print(f"{ep} Episode / Stpe : {step} / Score : {score:.1f} / Loss : {loss}")
    except Exception as e:
        traceback.print_exc()
    env.close()
