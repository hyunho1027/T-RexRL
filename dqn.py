import tensorflow as tf
from collections import deque

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

    def call(self, obs):
        s = (obs-128)/128    # None, 128, 64, 1
        s = self.conv1(s)    # None, 128, 64, 64
        s = self.maxpool1(s) # None, 64, 32, 64
        s = self.conv2(s)    # None, 64, 32, 64
        s = self.maxpool2(s) # None, 32, 16, 64
        s = self.conv3(s)    # None, 32, 16, 64
        s = self.maxpool3(s) # None, 16,  8, 64
        s = self.flat(s)     # None, 16*8*64
        s = self.dense1(s)   # None, 64
        s = self.dense2(s)   # None, 64
        q = self.dense3(s)   # None, 64

        return q

class DQN():
    def __init__(self):
        self.net = Net()
        self.target = tf.keras.models.clone_model(self.net)

        self.mem = deque(maxlen=5000)
        self.epsilon = 1
        self.lr = 3e-4
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def get_action(self, obs):
        return self.net(obs) if tf.random.uniform([1],0,1) > self.epsilon
                             else tf.math.round(tf.random.uniform([1],0,1))
        

    def append_sample(self, s, a, r, ns, d):
        self.mem.append((s,a,r,ns,d))

    def compute_loss(self, true, pred):
        tf.keras.losses.Huber(true, pred)
        return loss

    def train(self):
        # TODO: COMPUTE TARGET_VALUE
        with tf.GradientTape() as tape:
            loss = self.compute_loss(q_target, q)
            grads = tape.gradient(loss, self.net.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
    
    def update_target(self):
        self.target = tf.keras.models.clone_model(self.net)

    def save(self, path):
        path = path+'/' if path[-1]!='/' else path
        self.net.save_weight(path+"net")
        self.tg_net.save_weight(path+"target")

    def load(self, path)
        path = path+'/' if path[-1]!='/' else path
        self.net.load_weight(path+"net")
        self.tg_net.load_weight(path+"target")

if __name__=="__main__":
    net = Net()
    q = net(tf.random.normal((1,128,64,1)))
    print(q)