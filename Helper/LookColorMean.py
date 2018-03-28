import sys, glob
import numpy as np
from PIL import Image

def func(img_type, city_name):
	files = glob.glob('../../%s%s/*/img.png' % (img_type, city_name))
	files = np.random.choice(files, size = 2000, replace = False)
	res = [[], [], []]
	for idx, file in enumerate(files):
		print(idx)
		img = np.array(Image.open(file), np.int32)
		for i in range(3):
			res[i].append(np.mean(img[..., i]))
	return [np.mean(res[i]) for i in range(3)]

city_name = sys.argv[1]
print(func('Buildings', city_name))
print(func('Areas', city_name))
