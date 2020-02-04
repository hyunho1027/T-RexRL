from env import Env
from dqn import DQN

agent = DQN()
env = Env()

for ep in range(10000):
    d = False
    s = env.reset()
    while not d:
        a = agent.get_action([s])
        ns, r, d = env.step(a)
        #print(s.shape, a, ns.shape, r, d)
        agent.append_sample(s,a,r,ns,d)
        s = ns

    if ep > 0:#100:
        loss = agent.train()
        print(f"{ep} Epi / Loss : {loss}")

        if not ep%10:
            agent.update_target()
