import os, time
import matplotlib.pyplot as plt
import numpy as np

def mov_avg(li, n = 100):
	assert(len(li) >= n)
	s = sum(li[0: n])
	res = [s / float(n)]
	for i in range(n, len(li)):
		s += (li[i] - li[i - n])
		res.append(s / float(n))
	return res

def process(filename):
	with open(filename) as f:
		lines = [line.strip().split(', ') for line in f.readlines()]
		loss_cnn = [float(line[1]) for line in lines]
		loss_rnn = [float(line[2]) for line in lines]
	return mov_avg(loss_cnn), mov_avg(loss_rnn)

if __name__ == '__main__':
	# os.popen('scp leonhard:~/Master-Thesis/Road/LossTrain.out ./LossTrainChicago.out')
	# quit()

	loss_cnn, loss_rnn = process('LossTrainChicago.out')
	l = min(len(loss_cnn), len(loss_rnn))

	plt.plot(range(l), loss_cnn, label = 'CNN')
	plt.plot(range(l), loss_rnn, label = 'RNN')

	plt.title('Training Loss')
	# plt.ylim(ymin = 0, ymax = 0.75)
	# plt.xlim(xmin = 84000)
	plt.legend(loc = 'upper right')
	plt.show()



