import sys, glob, time
import numpy as np
from PIL import Image

if __name__ == '__main__':
	assert(len(sys.argv) == 2)
	city_name = sys.argv[1]
	filenames = glob.glob('../../Buildings%s/*/roadmap.png' % city_name)
	count = 0
	for filename in filenames:
		img = Image.open(filename)
		roadmap = np.array(img, dtype = np.int32)
		choose = (roadmap[..., 0] < 150) & (roadmap[..., 1] > 220) & (roadmap[..., 2] < 150)
		if choose.sum() > 0:
			count += 1
	print(count)
