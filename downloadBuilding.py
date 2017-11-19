import numpy as np
import lxml.etree as ET
import io, os, sys, requests, math, random
from PIL import Image, ImageDraw
ut = __import__('utility')

class BuildingImageDownloader(object):

	def __init__(self, google_keys_filename):
		f = open(google_keys_filename, 'r')
		self.keys = [item.strip() for item in f.readlines()]
		f.close()

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
		if not os.path.exists('../Dataset/%d' % building_id):
			os.makedirs('../Dataset/%d' % building_id)

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

		# 
		img = Image.fromarray(img)
		mask = Image.new('RGBA', img.size, color = (255, 255, 255, 0))
		draw = ImageDraw.Draw(mask)
		draw.polygon(polygon, fill = (255, 0, 0, 128), outline = (255, 0, 0, 128))
		merge = Image.alpha_composite(img, mask)
		if save:
			img.save('../Dataset/%d/0-img.png' % building_id)
			mask.save('../Dataset/%d/1-mask.png' % building_id)
			merge.save('../Dataset/%d/2-merge.png' % building_id)
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
			boundary.save('../Dataset/%d/3-b.png' % building_id)
		boundary = ut.pil2np(boundary, show)

		# 
		vertices = Image.new('P', size_s, color = 0)
		draw = ImageDraw.Draw(vertices)
		draw.point(polygon_s, fill = 255)
		if save:
			vertices.save('../Dataset/%d/4-v.png' % building_id)
		vertices = ut.pil2np(vertices, show)

		# 
		vertex_list = []
		for i in range(len(polygon_s)):
			vertex = Image.new('P', size_s, color = 0)
			draw = ImageDraw.Draw(vertex)
			draw.point([polygon_s[i]], fill = 255)
			if save:
				vertex.save('../Dataset/%d/5-v%s.png' % (building_id, str(i).zfill(2)))
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

class BuildingListConstructor(object):

	def __init__(self, range_vertices, filename = None):
		self.building = {}
		self.range_vertices = range_vertices
		assert(range_vertices[0] >= 3)
		assert(range_vertices[0] <= range_vertices[1])
		if filename:
			self.loadBuildingList(filename)
		return

	def saveBuildingList(self, filename):
		np.save(filename, self.building)
		return

	def printBuildingList(self):
		print(self.building)
		return

	def printBuildingListLen(self):
		print('Totally %d buidings.' % len(self.building))
		return

	def loadBuildingList(self, filename):
		d = np.load(filename).item()
		for item in d:
			if item in self.building:
				pass
			else:
				building = d[item]
				if len(building) >= self.range_vertices[0] and len(building) <= self.range_vertices[1]:
					self.building[item] = building
		self.printBuildingListLen()
		return

	def getBuildingList(self):
		return [self.building[item] for item in self.building]

	def resetBuildingList(self):
		self.building = {}
		return

	def addBuildingList(self, left, down, right, up):
		while True:
			try:
				osm = requests.get(
					'http://www.openstreetmap.org/api/0.6/map?bbox=' + \
					'%.7lf%%2C%.7lf%%2C%.7lf%%2C%.7lf' % \
					(left, down, right, up)
				).content
				osm = ET.fromstring(osm)
				break
			except:
				print('Try again to get .osm file.')
				pass
		node = {}
		for item in osm:
			if item.tag == 'node':
				id_str = item.attrib.get('id')
				lon = item.attrib.get('lon')
				lat = item.attrib.get('lat')
				if id_str and lon and lat:
					node[int(id_str)] = (float(lon), float(lat))
			elif item.tag == 'way':
				if item.attrib.get('visible') == 'true':
					node_list = []
					d = {}
					for sub_item in item:
						if sub_item.tag == 'nd':
							ref = sub_item.attrib.get('ref')
							if ref:
								node_list.append(node[int(ref)])
						elif sub_item.tag == 'tag':
							k = sub_item.attrib.get('k')
							v = sub_item.attrib.get('v')
							if k and v:
								d[k] = v
						else:
							pass
					if 'building' in d and d['building'] == 'yes': # <- Maybe there is other kind of building
						node_list = node_list[: -1]
						if len(node_list) >= self.range_vertices[0] and len(node_list) <= self.range_vertices[1]:
							bid = int(item.attrib.get('id'))
							if bid in self.building:
								pass
							else:
								self.building[bid] = node_list
						else:
							pass
					else:
						pass
				else:
					pass
			else:
				pass
		return

	def batchAddBuildingList(self, lon, lat, lon_step, lat_step, lon_num, lat_num):
		for i in range(lon_num):
			for j in range(lat_num):
				print('Step', i, j)
				self.addBuildingList(
					left  = lon + lon_step * i - lon_step / 2,
					down  = lat - lat_step * j - lat_step / 2,
					right = lon + lon_step * i + lon_step / 2,
					up    = lat - lat_step * j + lat_step / 2,
				)
				self.printBuildingListLen()
		return

if __name__ == '__main__':
	if False:
		objCons = BuildingListConstructor(range_vertices = (4, 20))
		objCons.batchAddBuildingList(
			lon = 8.4200,
			lat = 47.4600,
			lon_step = 0.0221822,
			lat_step = 0.0150000,
			lon_num = 10,
			lat_num = 10,
		)
		objCons.saveBuildingList('./buildingList.npy')
		objCons.printBuildingList()
	else:
		objCons = BuildingListConstructor(range_vertices = (4, 20), filename = './buildingList.npy')
		objDown = BuildingImageDownloader('./GoogleMapAPIKey.txt')
		id_list = [k for k in objCons.building]
		id_list.sort()
		for i, building_id in enumerate(id_list):
			if i < int(sys.argv[1]) or i >= int(sys.argv[2]):
				continue
			objDown.getBuildingAerialImage(i, objCons.building[building_id], building_id = building_id)

