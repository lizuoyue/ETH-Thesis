import numpy as np
import os, sys
if os.path.exists('../../Python-Lib/'):
	sys.path.insert(1, '../../Python-Lib')
import tensorflow as tf
import math, time
from BasicModel import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import RoadData

def getData(num):
	aa, bb, cc = [], [], []
	for i in range(num):
		a, b, c = RoadData.GetData((256, 256), 30, False)
		aa.append(a)
		bb.append(b)
		cc.append(c)
	return np.array(aa), np.array(bb), np.array(cc)

img = tf.placeholder(tf.float32, [None, 256, 256, 3])
gt = tf.placeholder(tf.float32, [None, 32, 32, 3])
msk = tf.placeholder(tf.float32, [None, 32, 32])

loss = Model('train', img, gt, msk)
pred = Model('test', img)

optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4)
train = optimizer.minimize(loss[0] + loss[1])
saver = tf.train.Saver(max_to_keep = 1)
init = tf.global_variables_initializer()

# Launch graph
with tf.Session() as sess:
	# Create loggers
	train_loss = open('./LossTrain.out', 'w')
	valid_loss = open('./LossValid.out', 'w')
	if not os.path.exists('./Model/'):
		os.makedirs('./Model/')
	if not os.path.exists('./Result/'):
		os.makedirs('./Result/')

	# Restore weights
	if len(sys.argv) == 2 and sys.argv[1] != None:
		saver.restore(sess, './Model/%s.ckpt' % sys.argv[1])
		iter_obj = range(int(sys.argv[1]) + 1, 10000)
	else:
		sess.run(init)
		iter_obj = range(10000)

	# Main loop
	for i in iter_obj:
		aa, bb, cc = getData(8)
		feed_dict = {img: aa - 160, gt: bb, msk: cc}

		# Training and get result
		init_time = time.time()
		_, (loss1, loss2) = sess.run([train, loss], feed_dict)
		cost_time = time.time() - init_time
		train_loss.write('Train Iter %d, %.6lf, %.6lf, %.3lfs\n' % (i, loss1, loss2, cost_time))
		train_loss.flush()

		# Validation
		if i % 200 == 0:
			aa, bb, cc = getData(8)
			feed_dict = {img: aa - 160, gt: bb, msk: cc}
			init_time = time.time()
			loss1, loss2 = sess.run(loss, feed_dict)
			cost_time = time.time() - init_time
			valid_loss.write('Valid Iter %d, %.6lf, %.6lf, %.3lfs' % (i, loss1, loss2, cost_time))
			valid_loss.flush()
			saver.save(sess, './Model/%s.ckpt' % i)

		# Test
		if i % 200 == 0:
			aa, bb, cc = getData(8)
			feed_dict = {img: aa - 160}
			sss, lll = sess.run(pred, feed_dict)
			for j in range(8):
				plt.figure()
				plt.imshow(aa[j])
				plt.axis('equal')
				plt.savefig('./Result/%d-%d.pdf' % (i, j))
				plt.figure()
				plt.imshow(sss[j, ..., 0])
				plt.axis('equal')
				plt.savefig('./Result/%d-%d-s.pdf' % (i, j))
				plt.figure()
				Y, X = np.mgrid[0: 32, 0: 32]
				U, V = lll[j, :, :, 0], lll[j, :, :, 1]
				Q = plt.quiver(X, 32 - Y, U, -V)
				plt.axis('equal')
				plt.savefig('./Result/%d-%d-l.pdf' % (i, j))





