import matplotlib.pyplot as plt
plt.switch_backend('agg')

def save(var1, var2, file):
	plt.gcf().clear()
	plt.plot(var1[-2000:], 'b')
	plt.plot(var2[-2000:], 'r')
	plt.savefig(file)
	return

def mean(li):
	return sum(li) / float(len(li))

if __name__ == '__main__':
	p = 300
	f = open('./PolyRNN.out', 'r')
	lines = f.readlines()
	f.close()
	cnn = []
	rnn = []
	full = []
	cnn_m = []
	rnn_m = []
	full_m = []
	for line in lines:
		a, b, c, d = line.strip().split(', ')
		cnn.append(float(b))
		rnn.append(float(c))
		full.append(float(d))
		cnn_m.append(mean(cnn[max(len(cnn) - p, 0):]))
		rnn_m.append(mean(rnn[max(len(rnn) - p, 0):]))
		full_m.append(mean(full[max(len(full) - p, 0):]))	
	save(cnn, cnn_m, './plot-cnn.png')
	save(rnn, rnn_m, './plot-rnn.png')
	save(full, full_m, './plot-full.png')
