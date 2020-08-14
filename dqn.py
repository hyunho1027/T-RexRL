import tensorflow as tf
from collections import deque
import random

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
        self.mem = deque(maxlen=5000)

        self.e_init = 0.8
        self.e_min = 0.05
        self.e = self.e_init
        self.e_decay = 0.995
        self.gamma = 0.99

        self.batch_size = 16
        self.lr = 2e-5
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def get_action(self, s):
        return tf.argmax(self.net(s), axis=1)[0] if random.random() > self.e else random.randint(0,1)

    def append_sample(self, s, a, r, ns, d):
        self.mem.append((s,a,r,ns,d))

    def train(self):
        mini = random.sample(self.mem, min(len(self.mem), self.batch_size))
        s = tf.convert_to_tensor([m[0] for m in mini])
        a = tf.convert_to_tensor([m[1] for m in mini])
        r = tf.convert_to_tensor([m[2] for m in mini])
        ns = tf.convert_to_tensor([m[3] for m in mini])
        d = tf.convert_to_tensor([m[4] for m in mini])
        return self.train_step(s,a,r,ns,d)

    @tf.function
    def train_step(self, s, a, r, ns, d):
        with tf.GradientTape() as tape:
            q = tf.reduce_sum(self.net(s)*tf.one_hot(a, 2), axis=1)
            next_q = tf.reduce_max(self.target_net(ns), axis=1)
            target_q = r + (1 - tf.cast(d, tf.float32))*self.gamma*next_q
            delta = abs(target_q - q)
            huber = delta + tf.cast(delta > 1, tf.float32)*(delta**2 - delta)
            loss = tf.reduce_mean(huber)
        grads = tape.gradient(loss, self.net.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
        return loss

    def update_target(self):
        self.target_net.set_weights(self.net.get_weights())

    def epsilon_decay(self):
        self.e = max(self.e_min, self.e*self.e_decay)

    def save(self, path):
        self.net.save_weights(path)

    def load(self, path):
        self.net.load_weights(path)
        self.target_net.set_weights(self.net.get_weights())
        