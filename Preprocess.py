import numpy as np
import sys, glob, math, cv2, time
from PIL import Image, ImageDraw

class ImageShift(object):

	def __init__(self, img, polygon):
		# Original data
		self.img = img[..., 0: 3]
		self.polygon = polygon
		self.polygon_i = np.array([p[1] for p in polygon])
		self.polygon_j = np.array([p[0] for p in polygon])

		# Face indices
		self.img_enh = self.imageEnhance(self.img)
		mask = Image.new('P', (self.img.shape[0], self.img.shape[1]), color = 0)
		draw = ImageDraw.Draw(mask)
		draw.polygon(polygon, fill = 255, outline = 255)
		mask = np.array(mask)
		self.face_li = np.nonzero(mask)

		# Edge indices
		self.img_edge = cv2.Canny(self.img, 100, 200)
		self.img_edge = cv2.dilate(self.img_edge, None)
		self.img_edge = cv2.dilate(self.img_edge, None)
		mask = Image.new('P', (self.img.shape[0], self.img.shape[1]), color = 0)
		draw = ImageDraw.Draw(mask)
		draw.polygon(polygon, fill = 0, outline = 255)
		mask = np.array(mask)
		self.edge_li = np.nonzero(mask)
		# Image.fromarray(self.img_edge).show()

		# Corner candidates
		dst = cv2.cornerHarris(cv2.cvtColor(self.img_enh, cv2.COLOR_BGR2GRAY), 3, 5, 0.04)
		dst = cv2.dilate(dst, None)
		dst = cv2.dilate(dst, None)
		dst = cv2.dilate(dst, None)
		dst = cv2.dilate(dst, None)
		self.corner_map = dst > 0.01 * dst.max()
		self.img_with_corner = np.copy(self.img)
		self.img_with_corner[self.corner_map] = [255, 0, 0]
		# Image.fromarray(self.img_with_corner).show()

		# Mean
		self.mean = self.img[self.face_li[0], self.face_li[1]].mean(axis = 0)

		# 
		return

	def imageEnhance(self, img):
		img = np.array(img)
		lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
		l, a, b = cv2.split(lab)
		clahe = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (8, 8))
		cl = clahe.apply(l)
		limg = cv2.merge((cl, a, b))
		return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

	def var(self, shift_i, shift_j):
		return self.img_enh[self.face_li[0] + shift_i, self.face_li[1] + shift_j, ...].var(axis = 0).mean()

	def edge(self, shift_i, shift_j):
		return 255 - self.img_edge[self.edge_li[0] + shift_i, self.edge_li[1] + shift_j, ...].mean()

	def dist(self, shift_i, shift_j):
		m = self.img[self.face_li[0] + shift_i, self.face_li[1] + shift_j, ...].mean(axis = 0)
		d = np.sqrt(np.mean((m - self.mean) ** 2))
		return d

	def ground(self, shift_i, shift_j):
		color = self.img[self.face_li[0] + shift_i, self.face_li[1] + shift_j, ...] - np.array([30, 30, 30])
		return 255 - np.sqrt((color ** 2).mean())

	def corner(self, shift_i, shift_j):
		return np.sum(self.corner_map[self.face_li[0] + shift_i, self.face_li[1] + shift_j])

	def matchCorner(self, shift_i, shift_j, th = 0.75):
		num = np.sum(self.corner_map[self.polygon_i + shift_i, self.polygon_j + shift_j])
		if num / len(self.polygon) >= th:
			return True
		else:
			return False

class Preprocessor(object):

	def __init__(self, city_name):
		self.building_list = glob.glob('../' + city_name + '/*/')
		self.building_list.sort()
		return

	def shift(self, obj):
		#
		x = [item[0] for item in obj.polygon]
		y = [item[1] for item in obj.polygon]
		shift_i_min = -min(y)
		shift_i_max = obj.img.shape[0] - max(y)
		shift_j_min = -min(x)
		shift_j_max = obj.img.shape[1] - max(x)
		diff_x = int((max(x) - min(x)) * 0.75)
		diff_y = int((max(y) - min(y)) * 0.75)

		#
		search_range_i = range(max(shift_i_min, -diff_y, -32), min(shift_i_max, diff_y, 32))
		search_range_j = range(max(shift_j_min, -diff_x, -32), min(shift_j_max, diff_x, 32))

		#
		var_d = {}
		edge_d = {}
		dist_d = {}
		ground_d = {}
		corner_d = {}
		keys = []
		for i in search_range_i:
			for j in search_range_j:
				if obj.matchCorner(i, j):
					keys.append((i, j))
					var_d[(i, j)] = obj.var(i, j)
					edge_d[(i, j)] = obj.edge(i, j)
					dist_d[(i, j)] = obj.dist(i, j)
					ground_d[(i, j)] = obj.ground(i, j)
					corner_d[(i, j)] = obj.corner(i, j)
		if len(var_d) == 0:
			return 0, 0
		min_v = min(var_d.items(), key = lambda x: x[1])[1]
		max_v = max(var_d.items(), key = lambda x: x[1])[1]
		min_e = min(edge_d.items(), key = lambda x: x[1])[1]
		max_e = max(edge_d.items(), key = lambda x: x[1])[1]
		min_d = min(dist_d.items(), key = lambda x: x[1])[1]
		max_d = max(dist_d.items(), key = lambda x: x[1])[1]
		min_m = min(ground_d.items(), key = lambda x: x[1])[1]
		max_m = max(ground_d.items(), key = lambda x: x[1])[1]
		min_c = min(corner_d.items(), key = lambda x: x[1])[1]
		max_c = max(corner_d.items(), key = lambda x: x[1])[1]
		keys = [(
			3 * (   var_d[k] - min_v) / (max_v - min_v) + \
			7 * (  edge_d[k] - min_e) / (max_e - min_e) + \
			5 * (  dist_d[k] - min_d) / (max_d - min_d) + \
			9 * (ground_d[k] - min_m) / (max_m - min_m) + \
			1 * (corner_d[k] - min_c) / (max_c - min_c),
			k
		) for k in keys]
		shift_i, shift_j = min(keys, key = lambda x: x[0])[1]
		return shift_i, shift_j

	def batchShift(self, beg_idx, end_idx):
		for i, building in enumerate(self.building_list):
			if i < beg_idx or i >= end_idx:
				continue

			# Local test
			# time.sleep(1)
			# Image.open(building + '2-l-merge.png').show()

			# 
			img = Image.open(building + '0-l-img.png')
			f = open(building + '3-l-v.txt', 'r')
			polygon = []
			for line in f.readlines():
				if line.strip() != '':
					x, y = line.strip().split()
					polygon.append((int(x), int(y)))

			obj = ImageShift(np.array(img), polygon)
			shift_i, shift_j = self.shift(obj)

			f = open('./4-shift.txt', 'w')
			f.write('%d %d\n' & (shift_i, shift_j))
			f.close()
			print(building, shift_i, shift_j)

			# Local test
			# polygon = [(p[0] + shift_j, p[1] + shift_i) for p in polygon]
			# mask = Image.new('RGBA', img.size, color = (255, 255, 255, 0))
			# draw = ImageDraw.Draw(mask)
			# draw.polygon(polygon, fill = (255, 0, 0, 128), outline = (255, 0, 0, 128))
			# merge = Image.alpha_composite(img, mask)
			# merge.show()

		# 
		# boundary = Image.new('P', size, color = 0)
		# draw = ImageDraw.Draw(boundary)
		# draw.polygon(polygon, fill = 0, outline = 255)
		# draw.line(polygon + [polygon[0]], fill = 255, width = 3) # <- For thicker outline
		# for point in polygon:
			# draw.ellipse((point[0] - 1, point[1] - 1, point[0] + 1, point[1] + 1), fill = 255)
		# if save:
			# boundary.save('../%s/%d/3-b.png' % (self.city_name, building_id))
		# boundary = ut.pil2np(boundary, show)

		# 
		# vertices = Image.new('P', size, color = 0)
		# draw = ImageDraw.Draw(vertices)
		# draw.point(polygon, fill = 255)
		# for point in polygon:
			# draw.ellipse((point[0] - 1, point[1] - 1, point[0] + 1, point[1] + 1), fill = 255)
		# if save:
			# vertices.save('../%s/%d/4-v.png' % (self.city_name, building_id))
		# vertices = ut.pil2np(vertices, show)

		# 
		# vertex_list = []
		# for i in range(len(polygon_s)):
		# 	vertex = Image.new('P', size_s, color = 0)
		# 	draw = ImageDraw.Draw(vertex)
		# 	draw.point([polygon_s[i]], fill = 255)
		# 	if save:
		# 		vertex.save('../%s/%d/5-v%s.png' % (self.city_name, building_id, str(i).zfill(2)))
		# 	vertex = ut.pil2np(vertex, show)
		# 	vertex_list.append(vertex)
		# vertex_list.append(np.zeros(size_s, dtype = np.float32))
		# vertex_list = np.array(vertex_list)

if __name__ == '__main__':
	city_name = sys.argv[1]
	obj = Preprocessor(city_name)
	obj.batchShift(int(sys.argv[2]), int(sys.argv[3]))
