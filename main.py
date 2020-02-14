from env import Env
from dqn import DQN
import traceback

agent = DQN()
env = Env()

try:
    for ep in range(10000):
        d = False
        score = 0
        s = env.reset()
        while not d:
            a = agent.get_action([s])
            ns, r, d = env.step(a)
            # print(s.shape, a, ns.shape, r, d)
            agent.append_sample(s,a,r,ns,d)
            s = ns
            score += r
        
        if ep > 100:
            loss = agent.train()
            print(f"{ep} Epi / Loss : {loss} / Score : {score+10}")

            if not ep%10 and loss is not None:
                agent.update_target()
            if not ep%20 and loss is not None:
                agent.save("./models/")
        else:
            print(f"{ep} Epi / Score : {score}")

except Exception as e:
    traceback.print_exc()
    env.close()