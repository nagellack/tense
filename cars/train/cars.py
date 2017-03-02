import tensorflow as tf
import numpy as np
from scipy.ndimage import imread
from glob import glob
import PIL
from PIL import Image
import time
import os

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', 'input', 'Input Directory.')
flags.DEFINE_string('output_dir', 'output', 'Output Directory.')

def run_training():
    ###IMPORT IMAGES and LABELS
    imagefiles= glob(os.path.join(FLAGS.input_dir, '*.jpg'))
    print "path",os.path.join(FLAGS.input_dir, '*.jpg')
    print "path2",FLAGS.input_dir
    print "files",os.listdir(FLAGS.input_dir)
    print "nFiles: ",len(imagefiles)
    image0 = Image.open(imagefiles[0]).resize((600, 400), PIL.Image.ANTIALIAS)
    length = 2
    if length !=0:
        all_data = np.zeros((length,400,600,3))
    else:
        all_data = np.zeros((len(imagefiles),400,600,3))
    for i in range(len(all_data)):
        all_data[i,:,:,:]=Image.open(imagefiles[i]).resize((600, 400), PIL.Image.ANTIALIAS)
    labels = np.genfromtxt('input/train_perfect_preds.txt', delimiter=',')[0:len(all_data)]
    print np.shape(labels)
    y = np.zeros((len(all_data),196))
    for i in range(len(y)):
        y[i,int(labels[i])]=1
    print "nLabels ",len(labels)
    
    ###INITIALISE ANN
    #first layer

    sess = tf.InteractiveSession()
    
    n_1 = 3
    n_2 = 6
    n_o = 10
    wind = 5
    wstep=1
    x = tf.placeholder(tf.float32, [None,400,600,3])
    W_conv1 = weight_variable([wind, wind, 3, n_1])
    b_conv1 = bias_variable([n_1])
    #x_image = tf.reshape(x, [-1,28,28,3])#!!!
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1,wstep) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    print "h_conv1",np.shape(h_conv1)
    print "h_pool1",np.shape(h_pool1)

    #second layer
    W_conv2 = weight_variable([wind, wind, n_1, n_2])
    b_conv2 = bias_variable([n_2])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2,wstep) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    print "h_conv1",np.shape(h_conv2)
    print "h_pool1",np.shape(h_pool2)

    #third
    sha=np.shape(h_pool2)
    print sha
    W_fc1 = weight_variable([int(sha[1]) * int(sha[2]) * n_2, n_o])
    b_fc1 = bias_variable([n_o])
    h_pool2_flat = tf.reshape(h_pool2, [-1, int(sha[1]) * int(sha[2]) * n_2])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #readout layer
    W_fc2 = weight_variable([n_o, 196])
    b_fc2 = bias_variable([196])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y_ = tf.placeholder(tf.float32, [None, 196])

    
    ###TRAIN ANN
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())
    for i in range(5):
        #batch = mnist.train.next_batch(50)
        #if i%100 == 0:
            #train_accuracy = accuracy.eval(feed_dict={x:all_data, y_: y, keep_prob: 1.0})
            #print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: all_data, y_: y, keep_prob: 0.5})
    train_accuracy = accuracy.eval(feed_dict={x:all_data, y_: y, keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    #print("test accuracy %g"%accuracy.eval(feed_dict={
    #    x: all_data, y_: y, keep_prob: 1.0}))
    

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W,wstep):
  return tf.nn.conv2d(x, W, strides=[1, wstep,wstep, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def main(_):
    t0=time.time()
    run_training()
    print time.time()-t0

if __name__ == "__main__":
    tf.app.run()