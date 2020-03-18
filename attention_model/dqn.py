import tensorflow as tf
from collections import deque
import random
from PIL import Image
import traceback
import warnings
from env import Env
warnings.simplefilter('ignore')


class AttentionCNN(tf.keras.Model):
    def __init__(self):
        super(AttentionCNN, self).__init__()
        self.iSize = (64, 128, 2)
        self.hSize = 64
        self.oSize = 2
        self.fSize = 8

        self.beta = tf.Variable(tf.zeros([self.fSize]), trainable=False)
        self.gamma = tf.Variable(tf.ones([self.fSize]), trainable=False)

        self.attentionCNN1 = tf.keras.layers.Conv2D(filters=self.fSize//2, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        self.attentionCNN2 = tf.keras.layers.Conv2D(filters=self.fSize, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)

        self.Q = tf.keras.layers.Dense(self.fSize, activation='relu')
        self.K = tf.keras.layers.Dense(self.fSize, activation='relu')
        self.V = tf.keras.layers.Dense(self.fSize, activation='relu')

        self.O1 = tf.keras.layers.Dense(self.hSize, activation='relu')
        self.O2 = tf.keras.layers.Dense(self.hSize, activation='relu')
        self.O3 = tf.keras.layers.Dense(self.hSize, activation='relu')
        self.O4 = tf.keras.layers.Dense(self.oSize)

    def call(self, x):
        x = tf.convert_to_tensor(x, tf.float32)
        nnk, shape = self.attentionCNN(x)
        query, key, value, E = self.queryKeyValue(nnk, shape)
        normalizedQKV = list(map(self.layerNormalization, [query, key, value]))
        A, attentionWeight, shape = self.selfAttention(normalizedQKV[0], normalizedQKV[1], normalizedQKV[2])
        residualAE = self.residual(A, E, 2)
        maxResidualAE = self.featureWiseMax(residualAE)
        out = self.outputLayer(maxResidualAE)
        return out, attentionWeight

    def attentionCNN(self, x):
        x = self.attentionCNN1(x)
        x = self.attentionCNN2(x)
        return x, x.shape
    
    def queryKeyValue(self, nnk, shape):
        flatten = tf.reshape(nnk, [-1, shape[1]*shape[2], shape[3]])
        Q = self.Q(flatten)
        K = self.K(flatten)
        V = self.V(flatten)
        return Q, K, V, flatten
    
    def selfAttention(self, query, key, value):
        keyDimSize = float(key.shape[-1])
        key = tf.transpose(key, perm=[0, 2 ,1])
        S = tf.linalg.matmul(query, key) / tf.sqrt(keyDimSize)
        attentionWeight = tf.nn.softmax(S)
        A = tf.linalg.matmul(attentionWeight, value)
        return A, attentionWeight, A.shape
    
    def outputLayer(self, x):
        x = self.O1(x)
        x = self.O2(x)
        x = self.O3(x)
        return self.O4(x)

    def layerNormalization(self, x):
        featureShape = x.shape[-1:]
        mean, variance = tf.nn.moments(x, [2], keepdims=True)
        return self.gamma * (x - mean) / tf.sqrt(variance + 1e-8) + self.beta
    
    def residual(self, x, inp, residualTime):
        for _ in range(residualTime):
            x = x + inp
            x = self.layerNormalization(x)
        return x
    
    def featureWiseMax(self, x):
        return tf.reduce_max(x, axis=2)

class DQN:
    def __init__(self):
        self.net = AttentionCNN()
        self.target_net = AttentionCNN()
        self.update_target()

        self.mem = deque(maxlen=50000)
        self.batch_size = 64

        self.e_init = 0.8
        self.e_min = 0.05
        self.e_decay = 0.9995
        self.e = self.e_init

        self.gamma = 0.99

        self.lr = 2e-5
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def get_action(self, s):
        return tf.argmax(self.net(s)[0], axis=1)[0] if random.random() > self.e else round(random.random())

    def get_attention(self, s):
        attention = self.net(s)[1]
        attention = tf.reduce_sum(attention[0], axis=0)
        attention = tf.reshape(attention, [64, 128])
        attention = tf.cast((attention * 255), tf.uint8)
        attention = Image.fromarray(attention.numpy(), 'L')
        # attention = attention.resize((128,64), resample=Image.NEAREST)
        return attention

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
            q = tf.reduce_sum(self.net(s)[0]*tf.one_hot(a, 2), axis=1)
            q_ns = tf.reduce_max(self.target_net(ns)[0], axis=1)
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
                # print(s.shape, a, ns.shape, r, d)
                if score > 1:
                    agent.append_sample(s,a,r,ns,d)
                if not d:
                    s = ns
                score += r
            
                if ep > 500:
                    loss = agent.train()

            agent.get_attention([s]).save(f"./attention/{ep}.png")
            if not ep%10 and loss is not None:
                agent.update_target()
            if not ep%20 and loss is not None:
                agent.save("./models/")

            print(f"{ep} Epi / Score : {score} / Loss : {loss}")

    except Exception as e:
        traceback.print_exc()
    env.close()
