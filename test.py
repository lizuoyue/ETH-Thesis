import tensorflow as tf
import math

y = 1
p = 0.999999999

a = tf.constant([y],tf.float32)
b = tf.constant([p],tf.float32)
c = tf.losses.log_loss(labels = a, predictions = b)

with tf.Session() as sess:
	print(sess.run(c))

print(-y*math.log(p)-(1-y)*math.log(1-p))