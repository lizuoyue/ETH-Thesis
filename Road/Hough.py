import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageDraw

img = Image.new('P', (256, 256), color = 255)
draw = ImageDraw.Draw(img)
draw.line([(-10, 100), (100, -10)], fill = 0, width = 20)
draw.line([(50, 50), (256, 256)], fill = 0, width = 20)
draw.line([(150, 150), (300, 0)], fill = 0, width = 20)
img.show()
img = np.array(img.resize((32, 32), resample = Image.BICUBIC))

res = np.zeros((361, 361))
theta = [i * (1.0/360) * math.pi for i in range(361)]
for i in range(img.shape[0]):
	for j in range(img.shape[1]):
		if img[i, j] < 128:
			for k, th in enumerate(theta):
				val = (j * math.cos(th) + i * math.sin(th)) * 8
				if val >= 0 and val <= 360:
					res[int(val), k] += 1
res = res / res.max()

plt.imshow(res)
plt.show()

