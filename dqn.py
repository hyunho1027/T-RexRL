import tensorflow as tf
import numpy as np
from collections import deque
import random
import warnings
warnings.simplefilter('ignore')

class Net(tf.keras.Model):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPooling2D(2)
        self.conv2 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPooling2D(2)
        self.conv3 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.maxpool3 = tf.keras.layers.MaxPooling2D(2)
        self.flat = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(2)

    def call(self, s):
        s = tf.convert_to_tensor(s, tf.float32)
        s = self.conv1(s)    # None, 64, 128, 64
        s = self.maxpool1(s) # None, 32, 64, 64
        s = self.conv2(s)    # None, 32, 64, 64
        s = self.maxpool2(s) # None, 16, 32, 64
        s = self.conv3(s)    # None, 16, 32, 64
        s = self.maxpool3(s) # None, 8, 16, 64
        s = self.flat(s)     # None, 8*16*64
        s = self.dense1(s)   # None, 64
        s = self.dense2(s)   # None, 64
        q = self.dense3(s)   # None, 64

        return q

class DQN:
    def __init__(self):
        self.net = Net()
        self.target_net = Net()
        self.update_target()

        self.mem = deque(maxlen=5000)
        self.batch_size = 64

        self.e_init = 0.8
        self.e_min = 0.1
        self.e_decay = 0.999
        self.e = self.e_init

        self.gamma = 0.99

        self.lr = 1e-3
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def get_action(self, s):
        return tf.argmax(self.net(s), axis=1)[0] \
                    if random.random() > self.e else round(random.random())
        
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
        self.net.save_weights(path+"net")

    def load(self, path):
        self.net.load_weights(path+"net")
        self.target_net.set_weights(self.net.get_weights())

if __name__=="__main__":
    net = Net()
    q = net(tf.random.normal((5,128,128,1)))
    print(q)