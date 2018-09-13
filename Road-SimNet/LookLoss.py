import os, time
import matplotlib.pyplot as plt
import numpy as np

def mov_avg(li, n = 1000):
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
		loss_b = [float(line[1]) for line in lines]
		loss_v = [float(line[2]) for line in lines]
		loss_sim = [float(line[3]) for line in lines]
		acc = [float(line[4]) for line in lines]
	return mov_avg(loss_b), mov_avg(loss_v), mov_avg(loss_sim), mov_avg(acc)

if __name__ == '__main__':
	os.popen('scp leonhard:~/Master-Thesis/Road-SimNet/LossTrain.out ./LossTrainChicago.out')
	time.sleep(10)

	loss_b, loss_v, loss_sim, acc = process('LossTrainChicago.out')
	l = len(loss_b)

	plt.plot(range(l), loss_b, label = 'Boundary')
	plt.plot(range(l), loss_v, label = 'Vertices')
	plt.plot(range(l), loss_sim, label = 'SIM')
	plt.plot(range(l), acc, label = 'Acc')
	plt.plot([0, l], [1, 1], '--')
	plt.plot([0, l], [0.8, 0.8], '--')

	plt.title('Training Loss and SIM Accuracy')
	# plt.ylim(ymin = 0, ymax = 0.75)
	# plt.xlim(xmin = 84000)
	plt.legend(loc = 'upper right')
	plt.show()



