import numpy as np
import lxml.etree as ET
import io, os, sys, time
import requests, math, random
from PIL import Image, ImageDraw
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

	def getBuildingAerialImage(self, idx, building, scale = 2, size = (224, 224), show = False, save = True, building_id = None):
		# Decide the parameters
		pad = 100
		zoom = 19
		size_s = (math.floor(size[0] / 8), math.floor(size[1] / 8))

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
				print('Try again to get image.')
				time.sleep(1)
				pass
		img = img[pad: img.shape[0] - pad, pad: img.shape[1] - pad, ...]

		# Compute polygon's vertices
		bbox = ut.BoundingBox(c_lon, c_lat, zoom, scale, size)
		polygon = []
		polygon_s = []
		for lon, lat in building:
			px, py = bbox.lonLatToRelativePixel(lon, lat)
			px_s, py_s = math.floor(px / 8), math.floor(py / 8)
			if not polygon_s or self.dist(polygon_s[-1], (px_s, py_s)) > 0:
				polygon.append((px, py, 1))
				polygon_s.append((px_s, py_s, 1))
			else:
				pass
				# polygon[-1] = self.centroid(polygon[-1], (px, py, 1))
				# polygon_s[-1] = self.centroid(polygon_s[-1], (px_s, py_s, 1))
		polygon = [(item[0], item[1]) for item in polygon]
		polygon_s = [(item[0], item[1]) for item in polygon_s]
		if polygon[-1] == polygon[0] or polygon_s[-1] == polygon_s[0]:
			polygon.pop()
			polygon_s.pop()

		# 
		img = Image.fromarray(img)
		mask = Image.new('RGBA', img.size, color = (255, 255, 255, 0))
		draw = ImageDraw.Draw(mask)
		draw.polygon(polygon, fill = (255, 0, 0, 128), outline = (255, 0, 0, 128))
		merge = Image.alpha_composite(img, mask)
		if save:
			img.save('../%s/%d/0-img.png' % (self.city_name, building_id))
			mask.save('../%s/%d/1-mask.png' % (self.city_name, building_id))
			merge.save('../%s/%d/2-merge.png' % (self.city_name, building_id))
		img = ut.pil2np(img, show)
		mask = ut.pil2np(mask, show)
		merge = ut.pil2np(merge, show)

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
			polygon_s.reverse()

		# 
		boundary = Image.new('P', size_s, color = 0)
		draw = ImageDraw.Draw(boundary)
		draw.polygon(polygon_s, fill = 0, outline = 255)
		if save:
			boundary.save('../%s/%d/3-b.png' % (self.city_name, building_id))
		boundary = ut.pil2np(boundary, show)

		# 
		vertices = Image.new('P', size_s, color = 0)
		draw = ImageDraw.Draw(vertices)
		draw.point(polygon_s, fill = 255)
		if save:
			vertices.save('../%s/%d/4-v.png' % (self.city_name, building_id))
		vertices = ut.pil2np(vertices, show)

		# 
		vertex_list = []
		for i in range(len(polygon_s)):
			vertex = Image.new('P', size_s, color = 0)
			draw = ImageDraw.Draw(vertex)
			draw.point([polygon_s[i]], fill = 255)
			if save:
				vertex.save('../%s/%d/5-v%s.png' % (self.city_name, building_id, str(i).zfill(2)))
			vertex = ut.pil2np(vertex, show)
			vertex_list.append(vertex)
		vertex_list.append(np.zeros(size_s, dtype = np.float32))
		vertex_list = np.array(vertex_list)

		# Return
		if show:
			print(img.shape)
			print(boundary.shape)
			print(vertices.shape)
			print(vertex_list.shape)
		return img, mask, merge, boundary, vertices, vertex_list

if __name__ == '__main__':
	assert(len(sys.argv) == 4)
	city_name = sys.argv[1]
	idx_beg = int(sys.argv[2])
	idx_end = int(sys.argv[3])
	objCons = bli.BuildingListConstructor(range_vertices = (4, 20), filename = './BuildingList-%s.npy' % city_name)
	objDown = BuildingImageDownloader('./GoogleMapAPIKey.txt')
	id_list = [k for k in objCons.building]
	id_list.sort()
	for i, building_id in enumerate(id_list):
		if i < idx_beg or i >= idx_end:
			continue
		objDown.getBuildingAerialImage(i, objCons.building[building_id], building_id = building_id)

