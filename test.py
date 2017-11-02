import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def generatePolygon(img_size = (224, 224), padding_div = 16, polygon_type = 'tri'):
	if polygon_type == 'cir':
		padding_div /= 2
	n_row = img_size[0]
	n_col = img_size[1]
	half_x = int(n_col / 2)
	half_y = int(n_row / 2)
	pad_x = int(n_col / padding_div)
	pad_y = int(n_row / padding_div)
	color = (255, 0, 0)
	img = Image.new('RGB', img_size, color = (255, 255, 255))
	draw = ImageDraw.Draw(img)
	if polygon_type == 'tri':
		if np.random.rand() > 0.5:
			p1x = np.random.randint(pad_x, half_x - pad_x)
			p2x = np.random.randint(half_x + pad_x, n_col - pad_x)
			p3x = np.random.randint(pad_x, n_col - pad_x)
			p1y = np.random.randint(pad_y, half_y - pad_y)
			p2y = np.random.randint(pad_y, half_y - pad_y)
			p3y = np.random.randint(half_y + pad_y, n_row - pad_y)
		else:
			p1x = np.random.randint(pad_x, half_x - pad_x)
			p2x = np.random.randint(half_x + pad_x, n_col - pad_x)
			p3x = np.random.randint(pad_x, n_col - pad_x)
			p1y = np.random.randint(half_y + pad_y, n_row - pad_y)
			p2y = np.random.randint(half_y + pad_y, n_row - pad_y)
			p3y = np.random.randint(pad_y, half_y - pad_y)
		polygon = [(p1x, p1y), (p2x, p2y), (p3x, p3y)]
		draw.polygon(polygon, fill = color, outline = color)
	elif polygon_type == 'qua':
		p1x = np.random.randint(pad_x, half_x - pad_x)
		p2x = np.random.randint(half_x + pad_x, n_col - pad_x)
		p3x = np.random.randint(half_x + pad_x, n_col - pad_x)
		p4x = np.random.randint(pad_x, half_x - pad_x)
		p1y = np.random.randint(pad_y, half_y - pad_y)
		p2y = np.random.randint(pad_y, half_y - pad_y)
		p3y = np.random.randint(half_y + pad_y, n_row - pad_y)
		p4y = np.random.randint(half_y + pad_y, n_row - pad_y)
		polygon = [(p1x, p1y), (p2x, p2y), (p3x, p3y), (p4x, p4y)]
		draw.polygon(polygon, fill = color, outline = color)
	elif polygon_type == 'ell':
		p1x = np.random.randint(pad_x, half_x - pad_x)
		p2x = np.random.randint(half_x + pad_x, n_col - pad_x)
		p1y = np.random.randint(pad_y, half_y - pad_y)
		p2y = np.random.randint(half_y + pad_y, n_row - pad_y)
		bbox = [(p1x, p1y), (p2x, p2y)]
		draw.ellipse(bbox, fill = color, outline = color)
	else:
		return
	noise = np.random.normal(0, 50, (n_row, n_col, 3))
	background = np.array(img)
	new_img = background + noise
	new_img = np.array((new_img - np.amin(new_img)) / (np.amax(new_img) - np.amin(new_img)) * 255.0, dtype = np.uint8)
	new_img = Image.fromarray(new_img)
	# new_img.show()
	return (np.array(new_img) / 255.0)

def model(x, y, n_class):
	input_layer = tf.reshape(x, [-1, 224, 224, 3])
	label = tf.reshape(y, [-1])

	conv1_1 = tf.layers.conv2d(
		inputs = input_layer,
		filters = 64,
		kernel_size = (3, 3),
		padding = 'same',
		activation = tf.nn.relu
	)
	conv1_2 = tf.layers.conv2d(
		inputs = conv1_1,
		filters = 64,
		kernel_size = (3, 3),
		padding = 'same',
		activation = tf.nn.relu
	)
	pool1 = tf.layers.max_pooling2d(
		inputs = conv1_2,
		pool_size = (2, 2),
		strides = 2
	)
	conv2_1 = tf.layers.conv2d(
		inputs = pool1,
		filters = 128,
		kernel_size = (3, 3),
		padding = 'same',
		activation = tf.nn.relu
	)
	conv2_2 = tf.layers.conv2d(
		inputs = conv2_1,
		filters = 128,
		kernel_size = (3, 3),
		padding = 'same',
		activation = tf.nn.relu
	)
	pool2 = tf.layers.max_pooling2d(
		inputs = conv2_2,
		pool_size = (2, 2),
		strides = 2
	)
	conv3_1 = tf.layers.conv2d(
		inputs = pool2,
		filters = 256,
		kernel_size = (3, 3),
		padding = 'same',
		activation = tf.nn.relu
	)
	conv3_2 = tf.layers.conv2d(
		inputs = conv3_1,
		filters = 256,
		kernel_size = (3, 3),
		padding = 'same',
		activation = tf.nn.relu
	)
	conv3_3 = tf.layers.conv2d(
		inputs = conv3_2,
		filters = 256,
		kernel_size = (3, 3),
		padding = 'same',
		activation = tf.nn.relu
	)
	pool3 = tf.layers.max_pooling2d(
		inputs = conv3_3,
		pool_size = (2, 2),
		strides = 2
	)
	conv4_1 = tf.layers.conv2d(
		inputs = pool3,
		filters = 512,
		kernel_size = (3, 3),
		padding = 'same',
		activation = tf.nn.relu
	)
	conv4_2 = tf.layers.conv2d(
		inputs = conv4_1,
		filters = 512,
		kernel_size = (3, 3),
		padding = 'same',
		activation = tf.nn.relu
	)
	conv4_3 = tf.layers.conv2d(
		inputs = conv4_2,
		filters = 512,
		kernel_size = (3, 3),
		padding = 'same',
		activation = tf.nn.relu
	)
	pool4 = tf.layers.max_pooling2d(
		inputs = conv4_3,
		pool_size = (2, 2),
		strides = 2
	)
	conv5_1 = tf.layers.conv2d(
		inputs = pool4,
		filters = 512,
		kernel_size = (3, 3),
		padding = 'same',
		activation = tf.nn.relu
	)
	conv5_2 = tf.layers.conv2d(
		inputs = conv5_1,
		filters = 512,
		kernel_size = (3, 3),
		padding = 'same',
		activation = tf.nn.relu
	)
	conv5_3 = tf.layers.conv2d(
		inputs = conv5_2,
		filters = 512,
		kernel_size = (3, 3),
		padding = 'same',
		activation = tf.nn.relu
	)
	pool4 = tf.layers.max_pooling2d(
		inputs = conv4_3,
		pool_size = (2, 2),
		strides = 2
	)
	part1_pool = tf.layers.max_pooling2d(
		inputs = pool2,
		pool_size = (2, 2),
		strides = 2
	)
	part1 = tf.layers.conv2d(
		inputs = part1_pool,
		filters = 128,
		kernel_size = (3, 3),
		padding = 'same',
		activation = tf.nn.relu
	)
	part2 = tf.layers.conv2d(
		inputs = pool3,
		filters = 128,
		kernel_size = (3, 3),
		padding = 'same',
		activation = tf.nn.relu
	)
	part3 = tf.layers.conv2d(
		inputs = conv4_3,
		filters = 128,
		kernel_size = (3, 3),
		padding = 'same',
		activation = tf.nn.relu
	)
	part4_conv = tf.layers.conv2d(
		inputs = conv5_3,
		filters = 128,
		kernel_size = (3, 3),
		padding = 'same',
		activation = tf.nn.relu
	)
	part4 = tf.image.resize_images(
		images = part4_conv,
		size = [28, 28]
	)
	part = tf.concat([part1, part2, part3, part4], 3)
	part_new = tf.layers.conv2d(
		inputs = part,
		filters = 128,
		kernel_size = (3, 3),
		padding = 'same',
		activation = tf.nn.relu
	)
	# Diff
	part_flat = tf.reshape(part_new, [-1, 28 * 28 * 128])
	dense_1 = tf.layers.dense(
		inputs = part_flat,
		units = 1024,
		activation = tf.nn.relu
	)
	dense_2 = tf.layers.dense(
		inputs = dense_1,
		units = 128,
		activation = tf.nn.relu
	)
	logits = tf.layers.dense(
		inputs = dense_2,
		units = n_class
	)
	pred_class = tf.argmax(input = logits, axis = 1)
	pred_prob = tf.nn.softmax(logits)
	onehot_labels = tf.one_hot(indices = tf.cast(label, tf.int32), depth = n_class)
	loss = tf.losses.softmax_cross_entropy(onehot_labels = onehot_labels, logits = logits)
	return pred_class, pred_prob, loss

if __name__ == '__main__':
	random.seed(3142857)
	x = tf.placeholder(tf.float32)
	y = tf.placeholder(tf.int32)
	res = model(x, y, 3)
	optimizer = tf.train.AdamOptimizer(learning_rate = 0.0005)
	train = optimizer.minimize(res[2])
	init = tf.global_variables_initializer()
	batch = 30
	n_iter = 10000
	with tf.Session() as sess:
		sess.run(init)
		for i in range(n_iter):
			xy_train = []
			for j in range(int(batch / 3)):
				xy_train.append((generatePolygon(polygon_type = 'tri'), 0))
				xy_train.append((generatePolygon(polygon_type = 'qua'), 1))
				xy_train.append((generatePolygon(polygon_type = 'ell'), 2))
			random.shuffle(xy_train)
			x_train = [item[0] for item in xy_train]
			y_train = [item[1] for item in xy_train]
			# for j in range(batch):
			# 	plt.imshow(x_train[j])
			# 	plt.show()
			print(np.array(y_train))
			feed_dict = {x: x_train, y: y_train}
			sess.run(train, feed_dict)
			pred_class, pred_prob, loss = sess.run(res, feed_dict)
			print(pred_class)
			acc = sum(pred_class == y_train) / float(batch)
			print(i, loss, float('%.2lf' % acc))
			
