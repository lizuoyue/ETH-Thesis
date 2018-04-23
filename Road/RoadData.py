import math, random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
from scipy.stats import multivariate_normal

def pepper(img):
	row, col, ch = img.shape
	mean, var = 0, 1000
	gauss = np.random.normal(mean, var ** 0.5, img.shape)
	noisy = img + gauss
	noisy = (noisy - noisy.min()) / (noisy.max() - noisy.min())
	return np.array(noisy * 255, np.uint8)

def get_crossing(s1, s2):
	xa, ya = s1[0][0], s1[0][1]
	xb, yb = s1[1][0], s1[1][1]
	xc, yc = s2[0][0], s2[0][1]
	xd, yd = s2[1][0], s2[1][1]
	a = np.matrix([
		[xb - xa, -(xd - xc)],
		[yb - ya, -(yd - yc)]
	])
	delta = np.linalg.det(a)
	if np.fabs(delta) < 1e-6:
		return None
	c = np.matrix([
		[xc - xa, -(xd - xc)],
		[yc - ya, -(yd - yc)]
	])
	d = np.matrix([
		[xb - xa, xc - xa],
		[yb - ya, yc - ya]
	])
	lamb = np.linalg.det(c) / delta
	miu = np.linalg.det(d) / delta
	if lamb <= 1 and lamb >= 0 and miu >= 0 and miu <= 1:
		x = xc + miu * (xd - xc)
		y = yc + miu * (yd - yc)
		return (x, y)
	else:
		return None

def extend_seg(seg):
	return [tuple(2 * np.array(seg[0]) - np.array(seg[1])), tuple(2 * np.array(seg[1]) - np.array(seg[0]))]

def make_ellipse(p, pad = 10):
	return [(p[0] - pad, p[1] - pad), (p[0] + pad, p[1] + pad)]

def direction(seg, seg_w):
	d = np.array(seg[1]) - np.array(seg[0])
	dd = d / np.sqrt(np.dot(d, d))
	ll = np.array([dd[1], -dd[0]])
	rr = np.array([-dd[1], dd[0]])
	l = np.array(seg[0]) + seg_w / 4.0 * ll
	r = np.array(seg[0]) + seg_w / 4.0 * rr
	seg1 = (tuple(l + d), tuple(l))
	seg2 = (tuple(r), tuple(r + d))
	return (seg1, seg2)

def dir_field(seg, w, h, seg_w):
	res = np.zeros((h, w, 2), np.float32)
	img = Image.new('P', (w, h), color = 255)
	draw = ImageDraw.Draw(img)
	draw.line(extend_seg(seg), fill = 0, width = int(math.ceil(seg_w / 2)))
	d = np.array(seg[1]) - np.array(seg[0])
	dd = d / np.sqrt(np.dot(d, d))
	res[np.where(np.array(img) < 128)] = dd
	return res

def GetData(img_size, max_path_width, show = False):
	num_per_edge = random.randint(1, 2)

	w, h = img_size
	ww, hh = int(w / 8.0), int(h / 8.0)
	pad_w = math.ceil(1.0 / num_per_edge * w / 4)
	pad_h = math.ceil(1.0 / num_per_edge * h / 4)
	l = [(0, random.randint(int(1.0 / num_per_edge * h * i) + pad_h, int(1.0 / num_per_edge * h * (i + 1)) - pad_h)) for i in range(num_per_edge)]
	u = [(random.randint(int(1.0 / num_per_edge * w * i) + pad_w, int(1.0 / num_per_edge * w * (i + 1)) - pad_w), 0) for i in range(num_per_edge)]
	r = [(w, random.randint(int(1.0 / num_per_edge * h * i) + pad_h, int(1.0 / num_per_edge * h * (i + 1)) - pad_h)) for i in range(num_per_edge)]
	d = [(random.randint(int(1.0 / num_per_edge * w * i) + pad_w, int(1.0 / num_per_edge * w * (i + 1)) - pad_w), h) for i in range(num_per_edge)]
	p = l + u + r + d
	random.shuffle(p)
	img = Image.new('RGB', img_size, color = (255, 255, 255))
	road_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
	draw = ImageDraw.Draw(img)
	segs = []
	dirfld = np.zeros((h, w, 2), np.float32)
	pts = []
	for x, y in zip(p[: num_per_edge * 2], p[num_per_edge * 2: ]):
		if x[0] == 0 and y[0] == 0 or x[0] == w and y[0] == w or x[1] == 0 and y[1] == 0 or x[1] == h and y[1] == h:
			continue
		segs.append([x, y])
	for seg in segs:
		seg_w = random.randint(int(max_path_width * 0.6), max_path_width)
		seg1, seg2 = direction(seg, seg_w)
		dirfld += dir_field(seg1, w, h, seg_w)
		dirfld += dir_field(seg2, w, h, seg_w)
		draw.line(extend_seg(seg), fill = road_color, width = seg_w)
		# draw.ellipse(make_ellipse(seg[0]), fill = (128, 128, 128), outline = (0, 0, 0))
		# draw.ellipse(make_ellipse(seg[1]), fill = (128, 128, 128), outline = (0, 0, 0))
		pts.append(seg[0])
		pts.append(seg[1])
	for i in range(len(segs)):
		for j in range(i, len(segs)):
			inter = get_crossing(segs[i], segs[j])
			if inter:
				pts.append(inter)
				# draw.ellipse(make_ellipse(inter), fill = (128, 128, 128), outline = (0, 0, 0))
	img = pepper(np.array(img))

	if show:
		plt.figure()
		plt.imshow(img)
		plt.show()

	Y, X = np.mgrid[0: ww, 0: hh]
	U = cv2.resize(dirfld[..., 0], dsize = (ww, hh), interpolation = cv2.INTER_LINEAR)
	V = cv2.resize(dirfld[..., 1], dsize = (ww, hh), interpolation = cv2.INTER_LINEAR)
	N = np.sqrt(U ** 2 + V ** 2)
	N[N < 1e-6] = np.inf
	U, V = U / N, V / N

	if show:
		plt.figure()
		Q = plt.quiver(X, 32 - Y, U, -V)
		plt.axis('equal')
		plt.show()

	pos = np.empty((hh, ww, 2))
	pos[..., 1], pos[..., 0] = np.mgrid[0: ww, 0: hh]
	heatmap = np.zeros((hh, ww))
	for pt in pts:
		rv = multivariate_normal(np.array(pt) / 8 - np.array([0.5, 0.5]), [[1, 0], [0, 1]])
		heatmap = np.maximum(heatmap, rv.pdf(pos))
	heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

	if show:
		plt.figure()
		plt.imshow(heatmap)
		plt.axis('equal')
		plt.show()

	gt = np.zeros((hh, ww, 3))
	gt[..., 0], gt[..., 1], gt[..., 2] = heatmap, U, V
	return img, gt

for i in range(2000):
	img = GetData((256, 256), 30, False)




