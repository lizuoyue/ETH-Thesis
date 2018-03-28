import h5py
import numpy as np

res1 = []
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
				res1.append((w, b))
np.save('../VGG16ConvWeights1.npy', res1)

res2 = []
ls = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3', '4_1', '4_2', '4_3', '5_1', '5_2', '5_3']
data = np.load('../../vgg16_weights.npz', 'r')
for i in ls:
	w = data['conv' + i + '_W']
	b = data['conv' + i + '_b']
	print(w.shape, b.shape)
	res2.append((w, b))
np.save('../VGG16ConvWeights2.npy', res2)
