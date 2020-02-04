import tensorflow as tf
import numpy as np
from collections import deque
import random

tf.keras.backend.set_floatx('float64')
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
        s = (np.array(s)-128)/128    # None, 128, 128, 1
        s = self.conv1(s)    # None, 128, 128, 64
        s = self.maxpool1(s) # None, 64, 64, 64
        s = self.conv2(s)    # None, 64, 64, 64
        s = self.maxpool2(s) # None, 32, 32, 64
        s = self.conv3(s)    # None, 32, 32, 64
        s = self.maxpool3(s) # None, 16, 16, 64
        s = self.flat(s)     # None, 16*16*64
        s = self.dense1(s)   # None, 64
        s = self.dense2(s)   # None, 64
        q = self.dense3(s)   # None, 64

        return q

class DQN:
    def __init__(self):
        self.net = Net()
        self.target_net = Net()

        self.mem = deque(maxlen=5000)
        self.batch_size = 16

        self.e_init = 0.8
        self.e_min = 0.1
        self.e_decay = 0.99
        self.e = self.e_init

        self.gamma = 0.99

        self.lr = 3e-4
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def get_action(self, s):
        return tf.argmax(self.net(s), axis=1) if tf.random.uniform([1],0,1) > self.e \
                             else tf.math.round(tf.random.uniform([1],0,1))
        
    def append_sample(self, s, a, r, ns, d):
        self.mem.append((s,a,r,ns,d))

    def compute_loss(self, true, pred):
        tf.keras.losses.Huber(true, pred)
        return loss
    
    @tf.function
    def train(self):
        if len(self.mem) < self.batch_size:
            return 0
            
        self.e = self.e *self.e_decay if self.e > self.e_min else self.e

        mini = np.array(random.sample(self.mem, self.batch_size))
        s, a, r, ns, d = mini[:,0], mini[:,1], mini[:,2], mini[:,3], mini[:,4]

        with tf.GradientTape() as tape:
            q = self.net(s)
            q_ns = self.target_net(ns)

            target_q = tf.identity(q)
            for i in range(self.batch_size):
                target_q[a[i]] = r[i] + self.gamma*(1-d[i])*q_ns[i]

            loss = self.compute_loss(target_q, q)
            grads = tape.gradient(loss, self.net.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
        return loss
    
    def update_target(self):
        self.target_net.set_weights(self.net.get_weights())

    def save(self, path):
        path = path+'/' if path[-1]!='/' else path
        self.net.save_weight(path+"net")
        self.tg_net.save_weight(path+"target")

    def load(self, path):
        path = path+'/' if path[-1]!='/' else path
        self.net.load_weight(path+"net")
        self.tg_net.load_weight(path+"target")

if __name__=="__main__":
    net = Net()
    q = net(tf.random.normal((5,128,128,1)))
    print(q)