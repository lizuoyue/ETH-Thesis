import os, time, glob
import numpy as np
import random, math
from PIL import Image, ImageDraw
from scipy import spatial

cities = [
	'boston', 'chicago', 'amsterdam', 'denver', 'kansas city',
	'la', 'montreal', 'new york', 'paris', 'pittsburgh',
	'saltlakecity', 'san diego', 'tokyo', 'toronto', 'vancouver'
]
biases = [
	[ 1, -1, 3, 1], [-1, -2, 1, 0], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1],
	[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1],
	[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1]
]
files = [
	# 'out_ours/%s_0.60.out.graph',
	# 'out_ours/%s_0.65.out.graph',
	# 'out_ours_uc/%s_0.70.out.graph',
	'out_ours_cu/%s_0.70.out.graph',
	# 'out_ours/%s_0.75.out.graph',
	# 'out_ours/%s_0.80.out.graph',
	# 'out_roadtracer/%s.0.2_0.2.newreconnect.graph',
	# 'out_roadtracer/%s.0.25_0.25.newreconnect.graph',
	'out_roadtracer/%s.0.3_0.3.newreconnect.graph',
	# 'out_roadtracer/%s.0.35_0.35.newreconnect.graph',
	# 'out_roadtracer/%s.0.4_0.4.newreconnect.graph',
	'out_deep/%s.fix.connected.graph',
	# 'segmentation/thr10/%s.pix.graph',
	# 'segmentation/thr20/%s.pix.graph',
	# 'segmentation/thr30/%s.pix.graph',
	# 'segmentation/thr40/%s.pix.graph',
]
ths = [1, 12, 16, 18, 20, 22, 24]

def read_graph_dense_v(filename, bias, width):
	with open(filename) as f:
		lines = f.readlines()
	for i, line in enumerate(lines):
		if line == '\n':
			sep = i
			break
	v = lines[:sep]
	e = lines[sep+1:]
	v = [tuple(float(item) for item in line.strip().split()) for line in v]
	e = [tuple(int(item) for item in line.strip().split()) for line in e]
	img = Image.new('P', (8192, 8192), color = 0)
	draw = ImageDraw.Draw(img)
	bx, by = bias
	for s, t in e:
		(x1, y1), (x2, y2) = v[s], v[t]
		draw.line([x1 - bx, y1 - by, x2 - bx, y2 - by], fill = 255, width = width)
	return np.array(img, np.uint8) > 128

def score(gt, dt, th):
	for t in th:
		gt_yes, gt_total = 0, 0
		dt_yes, dt_total = 0, 0
		for city in gt:
			gt_yes += np.logical_and(gt[city][1], dt[city][t]).sum()
			gt_total += gt[city][1].sum()
			dt_yes += np.logical_and(dt[city][1], gt[city][t]).sum()
			dt_total += dt[city][1].sum()
		print(t, dt_yes / dt_total, gt_yes / gt_total)
	return

if __name__ == '__main__':
	gt = {}
	for city, bias in zip(cities, biases):
		gt[city] = {}
		for th in ths:
			gt[city][th] = read_graph_dense_v('dataset/data/graphs/%s.graph' % city, (bias[0] * 4096, bias[1] * 4096), th)

	for file in files:
		dt = {}
		for city, bias in zip(cities, biases):
			dt[city] = {}
			for th in ths:
				dt[city][th] = read_graph_dense_v(file % city, (bias[0] * 4096, bias[1] * 4096), th)
		print(file)
		score(gt, dt, ths[1:])






