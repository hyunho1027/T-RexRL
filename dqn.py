import tensorflow as tf
from collections import deque
import random
import datetime

class Net(tf.keras.Model):
    def __init__(self):
        super(Net, self).__init__()
        self.input_size = (64, 128, 2)
        self.hidden_size = 128
        self.output_size = 2

        self.conv1 = tf.keras.layers.Conv2D(filters=self.hidden_size, kernel_size=[3,3],
                                            padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))
        self.conv2 = tf.keras.layers.Conv2D(filters=self.hidden_size, kernel_size=[3,3],
                                            padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))
        self.flat = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.fc2 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.q = tf.keras.layers.Dense(self.output_size)

    def call(self, x):
        x = tf.convert_to_tensor(x, tf.float32)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
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
        self.e_min = 0.01
        self.e = self.e_init
        self.e_decay = 0.005
        self.gamma = 0.99

        self.batch_size = 64
        self.lr = 3e-4
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.writer = tf.summary.create_file_writer(f"./summaries/{now}")

    def get_action(self, s, training=True):
        return random.randint(0,1) if random.random() < self.e and  training else tf.argmax(self.net(s), axis=1)[0]

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
        self.e = max(self.e_min, self.e-self.e_decay)

    def save(self, path):
        self.net.save_weights(path)

    def load(self, path):
        self.net.load_weights(path)
        self.target_net.set_weights(self.net.get_weights())
    
    def write(self, score, loss, epsilon, episode):
        with self.writer.as_default():
            tf.summary.scalar("model/loss", loss, step=episode)
            tf.summary.scalar("run/score", score, step=episode)
            tf.summary.scalar("run/epsilon", epsilon, step=episode)