import numpy as np
import os, sys
if os.path.exists('../../Python-Lib/'):
	sys.path.insert(1, '../../Python-Lib')
import tensorflow as tf
import math, time
from BasicModel import *
import RoadData

def getData(num):
	aa, bb = [], []
	for i in range(num):
		a, b = RoadData.GetData((256, 256), 30, False)
		aa.append(a)
		bb.append(b)
	return np.array(aa), np.array(bb)

img = tf.placeholder(tf.float32, [None, 256, 256, 3])
gt = tf.placeholder(tf.float32, [None, 32, 32, 3])

loss = Model('train', img, gt)
pred = Model('test', img)

optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4)
train = optimizer.minimize(loss)
saver = tf.train.Saver(max_to_keep = 1)
init = tf.global_variables_initializer()

# Launch graph
with tf.Session() as sess:
	# Create loggers
	train_loss = open('./LossTrain.out', 'w')
	valid_loss = open('./LossValid.out', 'w')
	if not os.path.exists('./Model/'):
		os.makedirs('./Model/')

	# Restore weights
	if len(sys.argv) == 2 and sys.argv[1] != None:
		saver.restore(sess, './Model/%s.ckpt' % sys.argv[1])
		iter_obj = range(int(sys.argv[1]) + 1, 10000)
	else:
		sess.run(init)
		iter_obj = range(10000)

	# Main loop
	for i in iter_obj:
		aa, bb = getData(8)
		feed_dict = {img: aa - 160, gt: bb}

		# Training and get result
		init_time = time.time()
		_, loss_train = sess.run([train, loss], feed_dict)
		cost_time = time.time() - init_time
		train_loss.write('Train Iter %d, %.6lf, %.3lfs\n' % (i, loss_train, cost_time))
		train_loss.flush()

		# Validation
		if i % 200 == 0:
			aa, bb = getData(8)
			feed_dict = {img: aa - 160, gt: bb}
			init_time = time.time()
			loss_valid = sess.run(loss, feed_dict)
			cost_time = time.time() - init_time
			valid_loss.write('Valid Iter %d, %.6lf, %.3lfs\n' % (i, loss_valid, cost_time))
			valid_loss.flush()
			saver.save(sess, './Model/%s.ckpt' % i)



