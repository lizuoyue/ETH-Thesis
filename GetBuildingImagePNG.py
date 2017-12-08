import numpy as np
import io, os, sys, cv2
import requests, math
from PIL import Image, ImageDraw, ImageFilter
ut = __import__('Utility')
bli = __import__('GetBuildingListOSM')

class BuildingImageDownloader(object):

	def __init__(self, google_keys_filename, city_name):
		f = open(google_keys_filename, 'r')
		self.keys = [item.strip() for item in f.readlines()]
		f.close()
		self.city_name = city_name

	def dist(self, p1, p2):
		return math.fabs(p1[0] - p2[0]) + math.fabs(p1[1] - p2[1])

	def colorDist(self, c1, c2):
		s = 0.0
		for i in range(3):
			s += (c1[i] - c2[i]) ** 2
		return math.sqrt(s)

	def centroid(self, p1, p2):
		x = (p1[0] * p1[2] + p2[0] * p2[2]) / (p1[2] + p2[2])
		y = (p1[1] * p1[2] + p2[1] * p2[2]) / (p1[2] + p2[2])
		return (math.floor(x), math.floor(y), p1[2] + p2[2])

	def norm(self, p):
		l = math.sqrt(p[0] ** 2 + p[1] ** 2)
		return (p[0] / l, p[1] / l)

	def centerRight(self, p1, p2, l):
		direction = (p1[1] - p2[1], p2[0] - p1[0])
		direction = self.norm(direction)
		x = math.floor((p1[0] + p2[0]) / 2 + l * direction[0])
		y = math.floor((p1[1] + p2[1]) / 2 + l * direction[1])
		return (x, y)

	def imageAugmentation(self, img):
		img = np.array(img)
		lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
		l, a, b = cv2.split(lab)
		clahe = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (8, 8))
		cl = clahe.apply(l)
		limg = cv2.merge((cl, a, b))
		final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
		return final

	def var(self, img, polygon):
		img = np.array(img)
		color = []
		for i, j in polygon:
			color.append(img[i, j, ...])
		mean = np.mean(np.array(color), axis = 0)
		color_dist = [self.colorDist(c, mean) for c in color]
		var = sum(color_dist) / len(color_dist)
		return var

	def edge(self, img, polygon):
		img = np.array(img)
		li = []
		for i, j in polygon:
			li.append(np.sum(img[i, j, ...]))
		return sum(li)

	def shiftStep(self, img, polygon, mode):
		if mode == 'var':
			search_range = range(-1, 2)
			min_var = sys.maxsize
			for i in search_range:
				for j in search_range:
					new_polygon = [(p[0] + i, p[1] + j) for p in polygon]
					var = self.var(img, new_polygon)
					if var < min_var:
						min_var = var
						shift_i = i
						shift_j = j
		if mode == 'edge':
			search_range = range(-2, 3)
			max_edge = 0
			for i in search_range:
				for j in search_range:
					new_polygon = [(p[0] + i, p[1] + j) for p in polygon]
					edge = self.edge(img, new_polygon)
					if edge > max_edge:
						max_edge = edge
						shift_i = i
						shift_j = j
		return shift_i, shift_j

	def shift(self, img, polygon, mode):
		shift_i = 0
		shift_j = 0
		if mode == 'edge':
			step_num = 6
			img = cv2.Canny(img[..., 0: 3], 100, 200)
			mask = Image.new('P', (img.shape[0], img.shape[1]), color = 0)
			draw = ImageDraw.Draw(mask)
			draw.polygon(polygon, fill = 0, outline = 255)
			mask = np.array(mask)
		if mode == 'var':
			step_num = 8
			img = self.imageAugmentation(img[..., 0: 3])
			mask = Image.new('P', (img.shape[0], img.shape[1]), color = 0)
			draw = ImageDraw.Draw(mask)
			draw.polygon(polygon, fill = 255, outline = 255)
			mask = np.array(mask)
		polygon = []
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				if mask[i, j] > 128:
					polygon.append((i, j))
		for i in range(step_num):
			new_polygon = [(p[0] + shift_i, p[1] + shift_j) for p in polygon]
			step_i, step_j = self.shiftStep(img, new_polygon, mode)
			if step_i == 0 and step_j == 0:
				break
			shift_i += step_i
			shift_j += step_j
		return shift_i, shift_j

	def getBuildingAerialImage(self, idx, building, scale = 2, size = (224, 224), show = False, save = True, building_id = None):
		# Decide the parameters
		pad = 100
		zoom = 20

		# Check inputs
		assert(scale == 1 or scale == 2)
		if scale == 2:
			assert(size[0] % 2 == 0)
			assert(size[1] % 2 == 0)
			size_g = (int(size[0] / 2) + pad, int(size[1] / 2) + pad)
		else:
			size_g = (size[0] + pad * 2, size[1] + pad * 2)
		if save:
			assert(building_id != None)

		# Create new folder
		if not os.path.exists('../%s/%d' % (self.city_name, building_id)):
			os.makedirs('../%s/%d' % (self.city_name, building_id))

		# Decide the tight bounding box
		min_lon, max_lon = 200.0, -200.0
		min_lat, max_lat = 100.0, -100.0
		for lon, lat in building:
			min_lon = min(lon, min_lon)
			min_lat = min(lat, min_lat)
			max_lon = max(lon, max_lon)
			max_lat = max(lat, max_lat)
		c_lon = (min_lon + max_lon) / 2.0
		c_lat = (min_lat + max_lat) / 2.0

		# Decide the zoom level
		left, up = ut.lonLatToPixel(min_lon, max_lat, zoom + scale - 1)
		right, down = ut.lonLatToPixel(max_lon, min_lat, zoom + scale - 1)
		while right - left > math.floor(size[0] * 0.9) or down - up > math.floor(size[1] * 0.9): # <- Decide the padding
			zoom -= 1
			assert(zoom > 0)
			left, up = ut.lonLatToPixel(min_lon, max_lat, zoom + scale - 1)
			right, down = ut.lonLatToPixel(max_lon, min_lat, zoom + scale - 1)
		print('%d, %d, Final zoom = %d.' % (idx, building_id, zoom))
		
		while True:
			try:
				data = requests.get(
					'https://maps.googleapis.com/maps/api/staticmap?' 			+ \
					'maptype=%s&' 			% 'satellite' 						+ \
					'center=%.7lf,%.7lf&' 	% (c_lat, c_lon) 					+ \
					'zoom=%d&' 				% zoom 								+ \
					'size=%dx%d&' 			% size_g						 	+ \
					'scale=%d&' 			% scale 							+ \
					'format=%s&' 			% 'png32' 							+ \
					'key=%s' 				% self.keys[idx % len(self.keys)] 	  \
				).content
				img = np.array(Image.open(io.BytesIO(data)))
				break
			except:
				print('Try again to get the image.')
				pass
		img = img[pad: img.shape[0] - pad, pad: img.shape[1] - pad, ...]

		# Compute polygon's vertices
		bbox = ut.BoundingBox(c_lon, c_lat, zoom, scale, size)
		polygon = []
		for lon, lat in building:
			px, py = bbox.lonLatToRelativePixel(lon, lat)
			if not polygon or self.dist(polygon[-1], (px, py, 1)) > 0:
				polygon.append((px, py, 1))
			else:
				pass
				# polygon[-1] = self.centroid(polygon[-1], (px, py, 1))
		polygon = [(item[0], item[1]) for item in polygon]
		if polygon[-1] == polygon[0]:
			polygon.pop()

		shift_i, shift_j = self.shift(img, polygon, mode = 'var')
		print(shift_i, shift_j)
		polygon = [(p[0] + shift_j, p[1] + shift_i) for p in polygon]

		img = Image.fromarray(img)
		mask = Image.new('RGBA', img.size, color = (255, 255, 255, 0))
		draw = ImageDraw.Draw(mask)
		draw.polygon(polygon, fill = (255, 0, 0, 128), outline = (255, 0, 0, 128))
		# polygon: (col, row)
		merge = Image.alpha_composite(img, mask)
		if save:
			img.save('../%s/%d/0-img.png' % (self.city_name, building_id))
			# mask.save('../%s/%d/1-mask.png' % (self.city_name, building_id))
			merge.save('../%s/%d/2-merge.png' % (self.city_name, building_id))
		img = ut.pil2np(img, show)
		mask = ut.pil2np(mask, show)
		# merge = ut.pil2np(merge, show)

		# Decide the order of vertices
		inner_count = 0
		for i in range(len(polygon)):
			x, y = self.centerRight(polygon[i - 1], polygon[i], 5)
			try:
				inner_count += (np.sum(mask[y, x, 1: 3]) > 1.0) # <- The pixel is not red
			except:
				inner_count += 1
		if inner_count / len(polygon) < 0.5:
			polygon.reverse()

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
		#
		# Just save text to save storage
		f = open('../%s/%d/5-v.txt' % (self.city_name, building_id), 'w')
		for point in polygon:
			f.write('%d %d\n' % point)
		f.close()

		# Return
		if show:
			print(img.shape)
			# print(boundary.shape)
			# print(vertices.shape)
			# print(vertex_list.shape)
		return# img, mask, merge, boundary, vertices, vertex_list

if __name__ == '__main__':
	assert(len(sys.argv) == 4)
	city_name = sys.argv[1]
	idx_beg = int(sys.argv[2])
	idx_end = int(sys.argv[3])
	objCons = bli.BuildingListConstructor(range_vertices = (4, 20), filename = './BuildingList-%s.npy' % city_name)
	objDown = BuildingImageDownloader('./GoogleMapAPIKey.txt', city_name)
	id_list = [k for k in objCons.building]
	id_list.sort()
	for i, building_id in enumerate(id_list):
		if i < idx_beg or i >= idx_end:
			continue
		objDown.getBuildingAerialImage(i, objCons.building[building_id], building_id = building_id)

