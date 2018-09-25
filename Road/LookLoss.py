import os, time
import matplotlib.pyplot as plt
import numpy as np

def mov_avg(li, n):
	assert(len(li) >= n)
	s = sum(li[0: n])
	res = [s / float(n)]
	for i in range(n, len(li)):
		s += (li[i] - li[i - n])
		res.append(s / float(n))
	return res

def process(filename, n = 2000):
	with open(filename) as f:
		lines = [line.strip().split(', ') for line in f.readlines()]
		loss_cnn = [float(line[1]) for line in lines]
		loss_rnn = [float(line[2]) for line in lines]
	return mov_avg(loss_cnn, n), mov_avg(loss_rnn, n)

if __name__ == '__main__':
	# os.popen('scp leonhard:~/Master-Thesis/Road/tmp/LossTrain.out ./LossTrain1.out')
	# os.popen('scp leonhard:~/Master-Thesis/Road/tmp/LossValid.out ./LossValid1.out')
	# os.popen('scp leonhard:~/Master-Thesis/Road/LossTrain.out ./LossTrainChicago.out')
	# os.popen('scp leonhard:~/Master-Thesis/Road/LossValid.out ./LossValidChicago.out')
	# time.sleep(10)
	# quit()

	n_val = 200
	loss_cnn, loss_rnn = process('LossValid.out')
	loss_cnn_val, loss_rnn_val = process('LossValid.out', n_val)
	l = len(loss_cnn)
	l_val = len(loss_cnn_val)

	# plt.plot(range(l), loss_cnn, label = 'CNN')
	# plt.plot(range(l), loss_rnn, label = 'RNN')
	plt.plot((np.array(range(l_val)) + n_val) * 100, loss_cnn_val, label = 'CNN Val')
	plt.plot((np.array(range(l_val)) + n_val) * 100, loss_rnn_val, label = 'RNN Val')

	plt.title('Training Loss')
	# plt.ylim(ymin = 0, ymax = 1.5)
	# plt.xlim(xmin = 200000)
	plt.legend(loc = 'upper right')
	plt.show()




