import numpy as np
import sys, glob, time
import math, random, cv2
from PIL import Image, ImageDraw

def readPolygon(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()
		polygon = [tuple([int(i) for i in line.strip().split()]) for line in lines]
	return polygon

def enhanceImage(img):
	lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	l, a, b = cv2.split(lab)
	clahe = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (8, 8))
	cl = clahe.apply(l)
	limg = cv2.merge((cl, a, b))
	return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def autoCanny(img, sigma = 0.33):
	v = np.median(img)
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edge = cv2.Canny(img, lower, upper)
	return edge


class PolygonShiftHelper(object):
	def __init__(self, path, alpha = 1):
		# 
		self.img = cv2.imread(path + 'img.png')
		self.polygon = readPolygon(path + 'polygon.txt')
		self.roadmap = cv2.imread(path + 'roadmap.png')
		self.polygon_i = np.array([v[1] for v in self.polygon], np.float32)
		self.polygon_j = np.array([v[0] for v in self.polygon], np.float32)
		if alpha != 1:
			c_i = (self.polygon_i.min() + self.polygon_i.max()) / 2
			c_j = (self.polygon_j.min() + self.polygon_j.max()) / 2
			self.polygon_i = self.polygon_i * alpha + c_i * (1 - alpha)
			self.polygon_j = self.polygon_j * alpha + c_j * (1 - alpha)
		self.polygon_i = np.array(self.polygon_i, np.int32)
		self.polygon_j = np.array(self.polygon_j, np.int32)
		self.polygon = [(self.polygon_j[k], self.polygon_i[k]) for k in range(len(self.polygon))]
		self.alpha = alpha

		# Face indices
		mask = Image.new('P', (self.img.shape[0], self.img.shape[1]), color = 0)
		draw = ImageDraw.Draw(mask)
		draw.polygon(self.polygon, fill = 255, outline = 255)
		self.face_idx = np.nonzero(np.array(mask))

		# Edge indices
		mask = Image.new('P', (self.img.shape[0], self.img.shape[1]), color = 0)
		draw = ImageDraw.Draw(mask)
		draw.polygon(self.polygon, fill = 0, outline = 255)
		self.edge_idx = np.nonzero(np.array(mask))

		#
		self.img_enh = enhanceImage(self.img)
		# self.img_edge = autoCanny(cv2.GaussianBlur(self.img, (5, 5), 0))
		self.img_edge = cv2.Canny(self.img, 100, 200)
		self.img_edge = cv2.dilate(self.img_edge, None)
		self.img_edge = cv2.dilate(self.img_edge, None)

		# 
		# dst = cv2.cornerHarris(self.img_edge, 2, 3, 0.04)
		# dst = cv2.cornerHarris(cv2.cvtColor(enhanceImage(self.img), cv2.COLOR_BGR2GRAY), 5, 7, 0.04)
		dst = cv2.cornerHarris(cv2.cvtColor(self.img_enh, cv2.COLOR_BGR2GRAY), 3, 5, 0.04)
		dst = cv2.dilate(dst, None)
		dst = cv2.dilate(dst, None)
		dst = cv2.dilate(dst, None)
		dst = cv2.dilate(dst, None)
		self.img_corner = dst > 0.01 * dst.max()

		# 
		# self.img_edge = cv2.dilate(self.img_edge, np.ones((1, 1), np.uint8), iterations = 1)
		self.mean = self.img[self.face_idx[0], self.face_idx[1]].mean(axis = 0)

		self.img_map_edge = (self.roadmap[..., 0] < 150) & (self.roadmap[..., 1] > 200) & (self.roadmap[..., 2] < 150)
		self.img_map_edge = np.array(self.img_map_edge, np.uint8) * 255
		self.img_map_edge = cv2.dilate(self.img_map_edge, None)
		self.img_map_edge = cv2.dilate(self.img_map_edge, None)

		# 
		return

	def var(self, shift_i, shift_j):
		return self.img_enh[self.face_idx[0] + shift_i, self.face_idx[1] + shift_j, ...].var(axis = 0).mean()

	def edge(self, shift_i, shift_j):
		return 255 - self.img_edge[self.edge_idx[0] + shift_i, self.edge_idx[1] + shift_j, ...].mean()

	def mapEdge(self, shift_i, shift_j):
		return 255 - self.img_map_edge[self.edge_idx[0] + shift_i, self.edge_idx[1] + shift_j, ...].mean()

	def corner(self, shift_i, shift_j, th = 0.9):
		num = self.img_corner[self.polygon_i + shift_i, self.polygon_j + shift_j].sum()
		return (num / len(self.polygon) >= th)

	def cornerInside(self, shift_i, shift_j):
		corner_edge = self.img_corner[self.edge_idx[0] + shift_i, self.edge_idx[1] + shift_j].sum()
		corner_all = self.img_corner[self.face_idx[0] + shift_i, self.face_idx[1] + shift_j].sum()
		return (corner_all - corner_edge) / len(self.face_idx)

	def dist(self, shift_i, shift_j):
		m = self.img[self.face_idx[0] + shift_i, self.face_idx[1] + shift_j, ...].mean(axis = 0)
		return np.sqrt(np.mean((m - self.mean) ** 2))

	def ground(self, shift_i, shift_j):
		color = self.img[self.face_idx[0] + shift_i, self.face_idx[1] + shift_j, ...] - np.array([30, 50, 30])
		return 255 - np.sqrt((color ** 2).mean())


class PolygonShiftProcessor(object):
	def __init__(self, city_name):
		self.building_list = glob.glob('../../Buildings%s/*/' % city_name)
		self.building_list.sort()
		return

	def shift(self, building_idx, alphas = [1, 1.05], show = False):
		# 
		var_d, edge_d, map_d, corner_d, dist_d, ground_d = {}, {}, {}, {}, {}, {}
		path = self.building_list[building_idx]
		helper = [PolygonShiftHelper(path, alpha = a) for a in alphas]

		# 
		keys = []
		for idx, obj in enumerate(helper):
			#
			shift_i_min = -obj.polygon_i.min()
			shift_i_max = -obj.polygon_i.max() + obj.img.shape[0]
			shift_j_min = -obj.polygon_j.min()
			shift_j_max = -obj.polygon_j.max() + obj.img.shape[1]
			diff_x = int((obj.polygon_i.max() - obj.polygon_i.min()) * 0.75)
			diff_y = int((obj.polygon_j.max() - obj.polygon_j.min()) * 0.75)

			#
			search_range_i = range(max(shift_i_min, -diff_y, -32), min(shift_i_max, diff_y, 32))
			search_range_j = range(max(shift_j_min, -diff_x, -32), min(shift_j_max, diff_x, 32))

			for i in search_range_i:
				for j in search_range_j:
					if obj.corner(i, j) or (i == 0 and j == 0):
						keys.append((idx, i, j))
						var_d[(idx, i, j)] = obj.var(i, j)
						edge_d[(idx, i, j)] = obj.edge(i, j)
						map_d[(idx, i, j)] = obj.mapEdge(i, j)
						corner_d[(idx, i, j)] = obj.cornerInside(i, j)
						dist_d[(idx, i, j)] = obj.dist(i, j)
						ground_d[(idx, i, j)] = obj.ground(i, j)

		min_v = min(   var_d.items(), key = lambda x: x[1])[1]
		max_v = max(   var_d.items(), key = lambda x: x[1])[1]
		min_e = min(  edge_d.items(), key = lambda x: x[1])[1]
		max_e = max(  edge_d.items(), key = lambda x: x[1])[1]
		min_m = min(   map_d.items(), key = lambda x: x[1])[1]
		max_m = max(   map_d.items(), key = lambda x: x[1])[1]
		min_c = min(corner_d.items(), key = lambda x: x[1])[1]
		max_c = max(corner_d.items(), key = lambda x: x[1])[1]
		min_d = min(  dist_d.items(), key = lambda x: x[1])[1]
		max_d = max(  dist_d.items(), key = lambda x: x[1])[1]
		min_g = min(ground_d.items(), key = lambda x: x[1])[1]
		max_g = max(ground_d.items(), key = lambda x: x[1])[1]
		
		li = [(
			23 * (   var_d[k] - min_v) / (max_v - min_v + 1) + \
			99 * (  edge_d[k] - min_e) / (max_e - min_e + 1) + \
			47 * (   map_d[k] - min_m) / (max_m - min_m + 1) + \
			10 * (corner_d[k] - min_c) / (max_c - min_c + 1) + \
			18 * (  dist_d[k] - min_d) / (max_d - min_d + 1) + \
			27 * (ground_d[k] - min_g) / (max_g - min_g + 1),  k
		) for k in keys]
		score, (idx, shift_i, shift_j) = min(li, key = lambda x: x[0])
		edge_score = edge_d[(idx, shift_i, shift_j)]
		# print(idx, shift_i, shift_j, score, edge_score)

		with open(self.building_list[building_idx] + 'shift.txt', 'w') as f:
			f.write('%d %d %d\n' % (idx, shift_i, shift_j))
			f.write('%lf %lf\n' % (score, edge_score))

		if show:
			img = Image.open(path + 'img.png')
			new_poly = [(a + shift_j, b + shift_i) for a, b in helper[idx].polygon]
			mask = Image.new('RGBA', img.size, color = (255, 255, 255, 0))
			draw = ImageDraw.Draw(mask)
			draw.polygon(new_poly, fill = (255, 0, 0, 128), outline = (255, 0, 0, 128))
			merge = Image.alpha_composite(img, mask)
			merge.show()

		return


if __name__ == '__main__':
	assert(len(sys.argv) == 4)
	city_name, idx_beg, idx_end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
	obj = PolygonShiftProcessor(city_name)
	for i in range(idx_beg, min(idx_end, len(obj.building_list))):
		print(i)
		obj.shift(i, show = False)

