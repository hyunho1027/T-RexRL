import tensorflow as tf
from collections import deque
import random
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
        self.q = tf.keras.layers.Dense(self.output_size, activation='softmax')

    def call(self, x):
        x = tf.convert_to_tensor(x, tf.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.q(x)

class DQN:
    def __init__(self):
        self.net = Net()
        self.target_net = Net()
        self.update_target()

        self.mem = deque(maxlen=50000)
        self.batch_size = 256

        self.e_init = 0.8
        self.e_min = 0.05
        self.e_decay = 0.9995
        self.e = self.e_init

        self.gamma = 0.99

        self.lr = 3e-4
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def get_action(self, s):
        return tf.argmax(self.net(s), axis=1)[0] if random.random() > self.e else round(random.random())

    def append_sample(self, s, a, r, ns, d):
        self.mem.append((s,a,r,ns,d))

    def train(self):
        if len(self.mem) < self.batch_size:
            return None
            
        self.e = self.e *self.e_decay if self.e > self.e_min else self.e

        mini = random.sample(self.mem, self.batch_size)
        s = [m[0] for m in mini]
        a = [m[1] for m in mini]
        r = [m[2] for m in mini]
        ns = [m[3] for m in mini]
        d = [m[4] for m in mini]

        with tf.GradientTape() as tape:
            q = tf.reduce_sum(self.net(s)*tf.one_hot(a, 2), axis=1)
            q_ns = tf.reduce_max(self.target_net(ns), axis=1)
            target_q = r + (1 - tf.constant(d, tf.float32))*self.gamma*q_ns
            loss = tf.reduce_sum(tf.square(target_q - q))
            grads = tape.gradient(loss, self.net.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
        return loss
    
    def update_target(self):
        self.target_net.set_weights(self.net.get_weights())

    def save(self, path):
        self.net.save_weights(path+"dqn")

    def load(self, path):
        self.net.load_weights(path+"dqn")
        self.target_net.set_weights(self.net.get_weights())
        
if __name__=="__main__":
    agent = DQN()
    env = Env()
    loss = None
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
                if not d:
                    s = ns
                score += r
            
            if ep > 100:
                loss = agent.train()

            if not ep%10 and loss is not None:
                agent.update_target()
            if not ep%20 and loss is not None:
                agent.save("./models/")

            print(f"{ep} Epi / Score : {score} / Loss : {loss}")

    except Exception as e:
        traceback.print_exc()
    env.close()
