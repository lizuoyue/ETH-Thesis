import matplotlib.pyplot as plt
plt.switch_backend('agg')

def save(var1, var2, file):
	plt.gcf().clear()
	plt.plot(var1[-3000:], 'b')
	plt.plot(var2[-3000:], 'r')
	plt.savefig(file)
	return

def mean(li):
	return sum(li) / float(len(li))

def plot(lines, mode):
	cnn = []
	rnn = []
	full = []
	cnn_m = []
	rnn_m = []
	full_m = []
	for line in lines:
		if line.strip() == '':
			continue
		a, b, c, d = line.strip().split(', ')
		if a.find(mode) >= 0:
			cnn.append(float(b))
			rnn.append(float(c))
			full.append(float(d))
			cnn_m.append(mean(cnn[max(len(cnn) - p, 0):]))
			rnn_m.append(mean(rnn[max(len(rnn) - p, 0):]))
			full_m.append(mean(full[max(len(full) - p, 0):]))
	save(cnn, cnn_m, './plot-%s-cnn.png' % mode)
	save(rnn, rnn_m, './plot-%s-rnn.png' % mode)
	save(full, full_m, './plot-%s-full.png' % mode)

if __name__ == '__main__':
	p = 200
	f = open('./PolygonRNN.out', 'r')
	lines = f.readlines()
	f.close()
	plot(lines, 'Train')
	plot(lines, 'Valid')
