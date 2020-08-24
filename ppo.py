import tensorflow as tf
import numpy as np
import traceback
import keyboard
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
        self.epsilon = 0.2
        self.lr = 2e-5
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def get_action(self, s):
        pi, v= self.net(s)
        a = np.random.choice(self.net.output_size, p=pi[0].numpy())
        return a, pi, v

    def append_sample(self,s,a,pi,r,ns,d):
        self.mem.append((s,a,pi,r,ns,d))

    def train(self):
        s = tf.convert_to_tensor([m[0] for m in self.mem])
        a = tf.convert_to_tensor([[m[1]] for m in self.mem])
        old_pi = tf.convert_to_tensor([m[2][0] for m in self.mem])
        r = tf.convert_to_tensor([[m[3]] for m in self.mem])
        ns = tf.convert_to_tensor([m[4] for m in self.mem])
        d = tf.convert_to_tensor([[1] if m[5] else [0] for m in self.mem], tf.float32)

        loss = self.train_step(s,a,old_pi,r,ns,d)
        self.mem.clear()
        return loss
    
    @tf.function
    def train_step(self, s,a,old_pi,r,ns,d):
        one_hot_a = tf.squeeze(tf.one_hot(a, self.net.output_size, dtype=tf.float32))
        old_pi_a = tf.reduce_sum(one_hot_a * old_pi, axis=1)
        for _ in range(self.epoch):
            with tf.GradientTape() as tape:
                pi, v = self.net(s)
                nv = self.net(ns)[1]
                target = r + self.gamma*(1-d)*nv
                delta = target - v
                adv = tf.identity(delta)
                for t in reversed(range(delta.shape[0]-1)):
                    adv = tf.concat([adv[:t], 
                                    [adv[t]+(1-d[t])*self.gamma*self._lambda*adv[t+1]],
                                    adv[t+1:]], axis=0)
                pi_a = tf.reduce_sum(one_hot_a * pi, axis=1)
                ratio = tf.exp(tf.math.log(pi_a) - tf.math.log(old_pi_a))
                surr1 = ratio * adv
                surr2 = tf.clip_by_value(ratio, 1-self.epsilon, 1+self.epsilon) * adv
                pi_loss = -tf.minimum(surr1, surr2)
                v_loss = tf.square(target - v)
                loss = pi_loss + v_loss
            grads = tape.gradient(loss, self.net.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
        return tf.reduce_mean(loss)

    def save(self, path):
        self.net.save_weights(path)

    def load(self, path):
        self.net.load_weights(path)

if __name__=="__main__":
    agent = PPO()
    env = Env()
    step = 0
    try:
        for ep in range(10000):
            d = False
            score = 0
            s = env.reset()
            while not d:
                if keyboard.is_pressed('q'):
                    raise Exception("Quit.")
                a, pi, v = agent.get_action([s])
                ns, r, d = env.step(a)
                agent.append_sample(s,a,pi,r,ns,d)
                s = ns
                score += r
                step += 1

            if ep%5==0:
                env.alt_tap()
                loss = agent.train()
                env.alt_tap()

            if ep%20==0:
                agent.save("./models/model")

            print(f"{ep} Epi / Step : {step} / Score : {score:.1f} / Loss : {loss:.3f}")
    except Exception as e:
        traceback.print_exc()
    env.close()
