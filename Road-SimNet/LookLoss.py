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

def process(filename, n = 1000):
	with open(filename) as f:
		lines = [line.strip().split(', ') for line in f.readlines()]
		loss_b = [float(line[1]) for line in lines]
		loss_v = [float(line[2]) for line in lines]
		loss_sim = [float(line[3]) for line in lines]
		acc = [float(line[4]) for line in lines]
	return mov_avg(loss_b, n), mov_avg(loss_v, n), mov_avg(loss_sim, n), mov_avg(acc, n)

if __name__ == '__main__':
	# os.popen('scp leonhard:~/Master-Thesis/Road-SimNet/LossTrain.out ./LossTrainChicago.out')
	# os.popen('scp leonhard:~/Master-Thesis/Road-SimNet/LossValid.out ./LossValidChicago.out')
	# time.sleep(10)

	n_val = 300
	loss_b, loss_v, loss_sim, acc = process('LossTrainChicago.out')
	loss_b_val, loss_v_val, loss_sim_val, acc_val = process('LossValidChicago.out', n_val)
	l = len(loss_b)
	l_val = len(loss_b_val)

	plt.plot(range(l), loss_b, label = 'Boundary')
	plt.plot(range(l), loss_v, label = 'Vertices')
	plt.plot(range(l), loss_sim, label = 'SIM')
	plt.plot((np.array(range(l_val)) + n_val) * 100, loss_b_val, label = 'Boundary Val')
	plt.plot((np.array(range(l_val)) + n_val) * 100, loss_v_val, label = 'Vertices Val')
	plt.plot((np.array(range(l_val)) + n_val) * 100, loss_sim_val, label = 'SIM Val')

	plt.title('Training Loss')
	# plt.ylim(ymin = 0, ymax = 0.75)
	# plt.xlim(xmin = 84000)
	plt.legend(loc = 'upper right')
	plt.show()

	plt.plot(range(l), acc, label = 'Acc')
	plt.plot((np.array(range(l_val)) + n_val) * 100, acc_val, label = 'Acc Val')
	plt.plot([0, l], [1, 1], '--')
	plt.plot([0, l], [0.8, 0.8], '--')

	plt.title('SIM Accuracy')
	# plt.ylim(ymin = 0, ymax = 0.75)
	# plt.xlim(xmin = 84000)
	plt.legend(loc = 'upper right')
	plt.show()



