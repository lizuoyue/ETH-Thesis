import h5py
import numpy as np

res = []
data = h5py.File('../../vgg16_weights.h5', 'r')
for i in range(0, 40):
	l = 'layer_%d' % i
	if l in data.keys():
		if list(data[l].keys()):
			w = data[l]['param_0']
			b = data[l]['param_1']
			if len(w.shape) == 4:
				w = np.transpose(np.array(w), (2, 3, 1, 0))
				b = np.array(b)
				print(w.shape, b.shape)
				res.append((w, b))
np.save('../../VGG16ConvWeights.npy', res)
