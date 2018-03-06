import numpy as np
import os, sys, math
from PIL import Image

if __name__ == '__main__':
	assert(len(sys.argv) == 2)
	path = sys.argv[1]
	file_list = os.popen('ls %s' % path).read().strip().split('\n')
	assert(len(file_list) % 5 == 0)
	num_w = int(math.ceil(math.sqrt(len(file_list) / 5)))
	num_h = num_w
	patch_size = (224, 224)
	name = ['-0-img.png', '-1-bound-pred.png', '-2-vertices-pred.png', '-3-vertices-merge.png', '-4-vertices-link.png']
	img = [np.zeros((num_h * patch_size[1], num_w * patch_size[0], 3), dtype = np.uint8) for i in range(5)]
	for i in range(num_h):
		for j in range(num_w):
			idx = i * num_w + j
			if idx >= int(len(file_list) / 5):
				break
			for k in range(5):
				org = np.array(Image.open(path + '/%d' % idx + name[k]).resize(patch_size, resample = Image.BICUBIC))[..., 0: 3]
				img[k][i * patch_size[1]: (i + 1) * patch_size[1], j * patch_size[0]: (j + 1) * patch_size[0], ...] = org
	for k in range(5):
		Image.fromarray(img[k]).save(path + '/_' + name[k][1: ])
