import numpy as np
import math, sys
import matplotlib.pyplot as plt

nn = 250

TABLEAU20 = [
	(174, 199, 232), (255, 187, 120), (152, 223, 138), (255, 152, 150), (197, 176, 213),
	(196, 156, 148), (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229),
]
# blue orange green red purple
# brown pink gray yellow bg
TABLEAU20_DEEP = [
	( 31, 119, 180), (255, 127,  14), ( 44, 160,  44), (214,  39,  40), (148, 103, 189),
	(140,  86,  75), (227, 119, 194), (127, 127, 127), (188, 189,  34), ( 23, 190, 207),
]
TABLEAU20 = [(r/255.0,g/255.0,b/255.0) for r,g,b in TABLEAU20]
TABLEAU20_DEEP = [(r/255.0,g/255.0,b/255.0) for r,g,b in TABLEAU20_DEEP]

def moving_average(a, n):
	ret = np.cumsum(a)
	ret[n: ] = ret[n: ] - ret[: -n]
	return ret[n - 1: ] / n

def read(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()
	lines = [line.strip().split(',')[1: ] for line in lines if line.strip()]
	d = dict([(key, np.array([float(line[i]) for line in lines])) for i, key in enumerate(['cls', 'reg', 'cnn', 'rnn'])])
	d['rpn'] = (d['cls'] + d['reg'])/2
	d['ply'] = (d['cnn'] + d['rnn'])/2
	d['sum'] = (d['rpn'] + d['ply'])/2
	for k in d:
		d[k] = moving_average(d[k], nn)
	return d

s2l = {'cls':'Anchor Classification','reg':'Anchor Regression','cnn':'Per-pixel Mask','rnn':'Polygon Vertices','rpn':'RPN','ply':'PolygonRNN','sum':'Full Loss'}
for city in ['Zurich', 'Chicago']:
	plt.figure()
	md1 = read('../HybridModel/Loss%s1Train.out' % city)
	md2 = read('../HybridModel/Loss%s2Train.out' % city)
	for i, k in enumerate(['cls', 'reg', 'cnn', 'rnn', 'rpn', 'ply', 'sum']):
		plt.plot(np.arange(nn, 25001), md1[k], c = TABLEAU20_DEEP[i], label = s2l[k] + ' Hybrid Version')
		plt.plot(np.arange(nn, 25001), md2[k], c = TABLEAU20[i], label = s2l[k] + ' Two-step Version')
	plt.ylim(ymin = 0, ymax = 9)
	plt.xlabel('Iteration', fontname = 'Arial')
	plt.ylabel('Loss', fontname = 'Arial')
	plt.xticks(fontname = 'Arial')
	plt.yticks(fontname = 'Arial')
	plt.title('Loss Decrease for Dataset %s (#Avg. = 250)' % city, fontname = 'Arial')
	lgd = plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), prop={'size': 9, 'family':'Arial'})
	plt.savefig('%s.pdf' % city, bbox_extra_artists=(lgd,), bbox_inches='tight')


