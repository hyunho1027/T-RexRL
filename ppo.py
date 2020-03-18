import tensorflow as tf
import numpy as np
import traceback
import warnings
from env import Env
warnings.simplefilter('ignore')

class Net(tf.keras.Model):
    def __init__(self):
        super(Net, self).__init__()
        self.input_size = (64, 128, 2)
        self.hidden_size = 64
        self.output_size = 2

        self.conv1 = tf.keras.layers.Conv2D(filters=self.hidden_size, kernel_size=[3,3], strides=[2,2], padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=self.hidden_size, kernel_size=[3,3], strides=[2,2], padding='same', activation='relu')
        self.flat = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.fc2 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.pi = tf.keras.layers.Dense(self.output_size, activation='softmax')
        self.v = tf.keras.layers.Dense(1)

    def call(self, x):
        x = tf.convert_to_tensor(x, tf.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.pi(x), self.v(x)

class PPO:
    def __init__(self):
        self.net = Net()
        self.mem = []
        self.gamma = 0.99
        self._lambda = 0.95
        self.epoch = 3
        self.epsilon = 0.1
        self.n_step = 32
        self.lr = 3e-4
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def get_action(self, s):
        pi, v= self.net(s)
        a = np.random.choice(self.net.output_size, p=pi[0].numpy())
        return a, pi, v

    def append_sample(self,s,a,pi,r,d):
        self.mem.append((s,a,pi,r,d))

    def train(self):
        if len(self.mem) < self.n_step:
            return None
        s = [m[0] for m in self.mem]
        a = [[m[1]] for m in self.mem]
        old_pi = [m[2][0] for m in self.mem]
        r = [[m[3]] for m in self.mem]
        d = [[1] if m[4] else [0] for m in self.mem]
        old_pi,r,d = map(lambda x: tf.convert_to_tensor(x, tf.float32), [old_pi,r,d])
        one_hot_a = tf.squeeze(tf.one_hot(a, self.net.output_size, dtype=tf.float32))
        old_pi_a = tf.reduce_sum(one_hot_a * old_pi, axis=1)
        losses = []
        for _ in range(self.epoch):
            with tf.GradientTape() as tape:
                pi, v = self.net(s)
                nv = tf.concat([v[1:], [[0]]], axis=0)
                target = r + self.gamma*(1-d)*nv
                delta = target - v
                adv = delta.numpy()
                for t in reversed(range(delta.shape[0]-1)):
                    adv[t] += (1-d[t])*self.gamma*self._lambda*adv[t+1]
                pi_a = tf.reduce_sum(one_hot_a * pi, axis=1)
                ratio = tf.exp(tf.math.log(pi_a) - tf.math.log(old_pi_a))
                surr1 = ratio * adv
                surr2 = tf.clip_by_value(ratio, 1-self.epsilon, 1+self.epsilon) * adv
                pi_loss = -tf.minimum(surr1, surr2)
                v_loss = tf.square(target - v)
                loss = pi_loss + v_loss
            grads = tape.gradient(loss, self.net.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
            losses.append(loss)
        self.mem=[]
        return tf.reduce_mean(losses)
    
    def save(self, path):
        self.net.save_weights(path+"ppo")

    def load(self, path):
        self.net.load_weights(path+"ppo")

if __name__=="__main__":
    agent = PPO()
    env = Env()

    try:
        for ep in range(10000):
            d = False
            score = 0
            s = env.reset()
            while not d:
                a, pi, v = agent.get_action([s])
                ns, r, d = env.step(a)
                if score > 0.5:
                    agent.append_sample(s,a,pi,r,d)
                s = ns
                score += r
            loss = agent.train()
            if not ep%20 and loss is not None:
                agent.save("./models/")
            print(f"{ep} Epi / Score : {score} / Loss : {loss}")
    except Exception as e:
        traceback.print_exc()
    env.close()
