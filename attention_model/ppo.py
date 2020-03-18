import tensorflow as tf
import numpy as np
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
        self.fSize = 24

        self.beta = tf.Variable(tf.zeros([self.fSize]), trainable=False)
        self.gamma = tf.Variable(tf.ones([self.fSize]), trainable=False)

        self.attentionCNN1 = tf.keras.layers.Conv2D(filters=self.fSize//2, kernel_size=[3,3], strides=[2,2], padding='same', activation='relu')
        self.attentionCNN2 = tf.keras.layers.Conv2D(filters=self.fSize, kernel_size=[3,3], strides=[2,2], padding='same', activation='relu')

        self.Q = tf.keras.layers.Dense(self.fSize, activation='relu')
        self.K = tf.keras.layers.Dense(self.fSize, activation='relu')
        self.V = tf.keras.layers.Dense(self.fSize, activation='relu')

        self.O1 = tf.keras.layers.Dense(self.hSize, activation='relu')
        self.O2 = tf.keras.layers.Dense(self.hSize, activation='relu')

        self.pi = tf.keras.layers.Dense(self.oSize, activation='softmax')
        self.v = tf.keras.layers.Dense(1)

    def call(self, x):
        x = tf.convert_to_tensor(x, tf.float32)
        nnk, shape = self.attentionCNN(x)
        query, key, value, E = self.queryKeyValue(nnk, shape)
        normalizedQKV = list(map(self.layerNormalization, [query, key, value]))
        A, attentionWeight, shape = self.selfAttention(normalizedQKV[0], normalizedQKV[1], normalizedQKV[2])
        residualAE = self.residual(A, E, 2)
        maxResidualAE = self.featureWiseMax(residualAE)
        pi, v = self.outputLayer(maxResidualAE)
        return pi, v, attentionWeight

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
        return self.pi(x), self.v(x)

    def layerNormalization(self, x):
        mean, variance = tf.nn.moments(x, [2], keepdims=True)
        return self.gamma * (x - mean) / tf.sqrt(variance + 1e-8) + self.beta
    
    def residual(self, x, inp, residualTime):
        for _ in range(residualTime):
            x = x + inp
            x = self.layerNormalization(x)
        return x
    
    def featureWiseMax(self, x):
        return tf.reduce_max(x, axis=2)
        
class PPO:
    def __init__(self):
        self.net = AttentionCNN()
        self.mem = []
        self.gamma = 0.99
        self._lambda = 0.95
        self.epoch = 3
        self.epsilon = 0.1
        self.n_step = 32
        self.lr = 5e-5
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def get_action(self, s):
        pi, v, _ = self.net(s)
        a = np.random.choice(self.net.oSize, p=pi[0].numpy())
        return a, pi, v

    def get_attention(self, s):
        attention = self.net(s)[2]
        attention = tf.reduce_sum(attention[0], axis=0)
        attention = tf.reshape(attention, [16, 32])
        attention = tf.cast((attention * 255), tf.uint8)
        attention = Image.fromarray(attention.numpy(), 'L')
        attention = attention.resize((128,64), resample=Image.NEAREST)
        return attention

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
        one_hot_a = tf.squeeze(tf.one_hot(a, self.net.oSize, dtype=tf.float32))
        old_pi_a = tf.reduce_sum(one_hot_a * old_pi, axis=1)
        losses = []
        for _ in range(self.epoch):
            with tf.GradientTape() as tape:
                pi, v, _ = self.net(s)
                nv = tf.concat([v[1:], [[0]]], axis=0)
                target = r + self.gamma*(1.0 - d)*nv
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
                a, pi, _ = agent.get_action([s])
                ns, r, d = env.step(a)

                if score > 0.5:
                    agent.append_sample(s,a,pi,r,d)
                
                s = ns
                score += r
            
            loss = agent.train()
            agent.get_attention([s]).save(f"./attention/ppo/{ep}.png")

            if not ep%20 and loss is not None:
                agent.save("./models/")

            print(f"{ep} Epi / Score : {score} / Loss : {loss}")

    except Exception as e:
        traceback.print_exc()
    env.close()
