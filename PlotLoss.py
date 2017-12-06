import matplotlib.pyplot as plt
plt.switch_backend('agg')

def save(var1, var2, file):
	plt.plot(var1)
	plt.plot(var2)
	plt.savefig(file)
	plt.gcf().clear()
	return

def mean(li):
	return sum(li) / float(len(li))

if __name__ == '__main__':
	p = 500
	f = open('./PolyRNN.out', 'r')
	lines = f.readlines()
	f.close()
	cnn = rnn = full = []
	cnn_m = rnn_m = full_m = []
	for i, line in enumerate(lines):
		a, b, c, d = line.strip().split(', ')
		cnn.append(float(b))
		rnn.append(float(c))
		full.append(float(d))
		cnn_m.append(mean(cnn[max(len(cnn) - 200, 0):]))
		rnn_m.append(mean(cnn[max(len(rnn) - 200, 0):]))
		full_m.append(mean(cnn[max(len(full) - 200, 0):]))	
	save(cnn, cnn_m, './cnn.pdf')
	save(rnn, rnn_m, './rnn.pdf')
	save(full, full_m, './full.pdf')
