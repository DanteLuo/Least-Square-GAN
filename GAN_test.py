from LSGAN_Mnist import LSGAN
from tensorflow.examples.tutorials.mnist import input_data
import os


if not os.path.exists('out/'):
    os.makedirs('out/')

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

lsgan = LSGAN()
lsgan.build_model()
for it in range(50000):
    X_mb, _ = mnist.train.next_batch(20*96+20*32)
    lsgan.train(x_real=X_mb,num_x_fake=32)

lsgan.save_model()
lsgan.close_session()
