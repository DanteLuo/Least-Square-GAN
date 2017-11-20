import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


class LSGAN():
    def __init__(self):
        self.z_dim = 64
        self.h1_dim_g = 128

        self.goal_dim = 784
        self.h1_dim_d = 128

        self.scaling = 1
        self.iteration = 0
        self.imageid = 0

        self.X = tf.placeholder(tf.float64, shape=[None, self.goal_dim])
        self.Z = tf.placeholder(tf.float64, shape=[None, self.z_dim])

        with tf.variable_scope('G'):
            W1 = tf.get_variable(name='W1', shape=[self.z_dim, self.h1_dim_g], dtype=tf.float64,
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable(name='b1', shape=[self.h1_dim_g], dtype=tf.float64,
                                 initializer=tf.initializers.truncated_normal())
            W2 = tf.get_variable(name='W2', shape=[self.h1_dim_g, self.goal_dim], dtype=tf.float64,
                                 initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable(name='b2', shape=[self.goal_dim], dtype=tf.float64,
                                 initializer=tf.initializers.truncated_normal())
            self.theta_G = [W1, W2, b1, b2]

        with tf.variable_scope('D'):
            W1 = tf.get_variable(name='W1', shape=[self.goal_dim, self.h1_dim_d], dtype=tf.float64,
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable(name='b1', shape=[self.h1_dim_d], dtype=tf.float64,
                                 initializer=tf.initializers.truncated_normal())
            W2 = tf.get_variable(name='W2', shape=[self.h1_dim_d, 1], dtype=tf.float64,
                                 initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable(name='b2', shape=[1], dtype=tf.float64,
                                 initializer=tf.initializers.truncated_normal())
            self.theta_D = [W1, W2, b1, b2]

    def sample_z(self,m):
        return np.random.uniform(-1., 1., size=[m,self.z_dim])

    def generator(self,z):
        with tf.variable_scope('G'):
            h1 = tf.nn.relu(tf.matmul(z,self.theta_G[0])+self.theta_G[2])
            y = tf.nn.sigmoid(tf.matmul(h1,self.theta_G[1])+self.theta_G[3])
        return y

    def discriminator(self,x):
        with tf.variable_scope('D'):
            h1 = tf.nn.relu(tf.matmul(x, self.theta_D[0]) + self.theta_D[2])
            out = tf.matmul(h1, self.theta_D[1]) + self.theta_D[3]
        return out

    def build_model(self, lr=0.001):
        self.G_sample = self.generator(self.Z)

        self.D_real = self.discriminator(self.X)
        self.D_fake = self.discriminator(self.G_sample)

        self.D_loss = 0.5 * (tf.reduce_mean((self.D_real-1)**2)+
                             tf.reduce_mean((self.D_fake)**2))
        self.G_loss = 0.5 * (tf.reduce_mean((self.D_fake-1)**2))

        self.D_solver = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.D_loss, var_list=self.theta_D)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.G_loss, var_list=self.theta_G)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, x_real, num_x_fake, num_iterations=20):
        for it in range(num_iterations):
            D_loss_curr = 0
            for itt in range(3):
                X_mb = x_real[(it*128+itt*32):((itt+1)*32+it*128),:]
                z_mb = self.sample_z(num_x_fake)

                _, D_loss_curr = self.sess.run([self.D_solver,self.D_loss],
                                                feed_dict={self.X: X_mb,
                                                           self.Z: z_mb})

            z_mb = self.sample_z(num_x_fake)
            X_mb = x_real[it*128+96:(it+1)*128,:]
            _, G_loss_curr = self.sess.run([self.G_solver,self.G_loss],
                                           feed_dict={self.X: X_mb,
                                                      self.Z: z_mb})

            if self.iteration % 1000 == 0:
                print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'
                      .format(self.iteration, D_loss_curr, G_loss_curr))

                samples = self.sess.run(self.G_sample, feed_dict={self.Z: self.sample_z(16)})
                fig = self.plot(samples)
                plt.savefig('out/{}.png'
                            .format(str(self.imageid).zfill(3)), bbox_inches='tight')
                self.imageid += 1
                plt.close(fig)

            self.iteration += 1

    def plot(self,samples):
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        return fig

    def save_model(self, path=None):
        if path == None:
            path = os.getcwd()+"/model/"

        if not os.path.exists(path):
            os.makedirs(path)

        saver = tf.train.Saver()
        save_path = saver.save(sess=self.sess,save_path=path)
        print("Model saved in file: %s" % save_path)

    def close_session(self):
        self.sess.close()


