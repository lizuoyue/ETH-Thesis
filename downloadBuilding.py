import numpy as np
import lxml.etree as ET
import io, requests, math, random
from PIL import Image, ImageDraw
ut = __import__('utility')

class BuildingImageDownloader(object):

	def __init__(self, google_keys_filename):
		self.count = 0
		f = open(google_keys_filename, 'r')
		self.keys = [item.strip() for item in f.readlines()]
		f.close()

	def getBuildingAerialImage(building, zoom = 19, scale = 2, size = (224, 224), show = False):
		z = zoom + scale - 1
		size_s = (math.floor(size[0] / 8), math.floor(size[1] / 8))
		min_lon, max_lon = 200.0, -200.0
		min_lat, max_lat = 100.0, -100.0
		for lon, lat in building:
			min_lon = min(lon, min_lon)
			min_lat = min(lat, min_lat)
			max_lon = max(lon, max_lon)
			max_lat = max(lat, max_lat)
		left, up = ut.lonLatToPixel(min_lon, max_lat, z)
		right, down = ut.lonLatToPixel(max_lon, min_lat, z)
		if right - left > math.floor(size[0] * 0.85) or down - up > math.floor(size[1] * 0.85): # <- Decide the padding
			return None
		c_lon = (min_lon + max_lon) / 2.0
		c_lat = (min_lat + max_lat) / 2.0
		while True:
			try:
				data = requests.get(
					'https://maps.googleapis.com/maps/api/staticmap?' 			+ \
					'maptype=%s&' 			% 'satellite' 						+ \
					'center=%.7lf,%.7lf&' 	% (c_lat, c_lon) 					+ \
					'zoom=%d&' 				% zoom 								+ \
					'size=%dx%d&' 			% (size[0] + 100, size[1] + 100) 	+ \
					'scale=%d&' 			% scale 							+ \
					'format=%s&' 			% 'png32' 							+ \
					'key=%s' 				% self.keys[self.count] 			  \
				).content
				img = np.array(Image.open(io.BytesIO(data)))
				break
			except:
				print('Try again to get image.')
				pass
		img = img[100: img.shape[0] - 100, 100: img.shape[1] - 100, ...]

		bbox = ut.BoundingBox(c_lon, c_lat, zoom, scale, size)
		polygon = []
		polygon_s = []
		for lon, lat in building:
			px, py = bbox.lonLatToRelativePixel(lon, lat)
			polygon.append((px, py))
			polygon_s.append((math.floor(px / 8), math.floor(py / 8)))

		# 
		img = Image.fromarray(img)
		mask = Image.new('RGBA', img.size, color = (255, 255, 255, 0))
		draw = ImageDraw.Draw(mask)
		draw.polygon(polygon, fill = (255, 0, 0, 128), outline = (255, 0, 0, 128))
		merge = Image.alpha_composite(img, mask)
		img = ut.pil2np(img, show)
		mask = ut.pil2np(mask, show)
		merge = ut.pil2np(merge, show)

		# 
		boundary = Image.new('P', size_s, color = 0)
		draw = ImageDraw.Draw(boundary)
		draw.polygon(polygon_s, fill = 0, outline = 255)
		boundary = ut.pil2np(boundary, show)

		# 
		vertices = Image.new('P', size_s, color = 0)
		draw = ImageDraw.Draw(vertices)
		draw.point(polygon_s, fill = 255)
		vertices = ut.pil2np(vertices, show)

		# 
		vertex_list = []
		for i in range(len(polygon_s)):
			vertex = Image.new('P', size_s, color = 0)
			draw = ImageDraw.Draw(vertex)
			draw.point([polygon_s[i]], fill = 255)
			vertex = ut.pil2np(vertex, show)
			vertex_list.append(vertex)
		vertex_list.append(np.zeros(img_size_s, dtype = np.float32))
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

	def getImage(self, batch_size):
		result = []
		while len(result) < batch_size:
			building = random.sample(self.getBuildingList(), 1)
			res = getBuildingAerialImage(building[0])
			if res:
				result.append(res)
		return result

if __name__ == '__main__':
	if False:
		obj = BuildingListConstructor(range_vertices = (4, 12))
		obj.batchAddBuildingList(
			lon = 8.4200,
			lat = 47.4600,
			lon_step = 0.0221822,
			lat_step = 0.0150000,
			lon_num = 10,
			lat_num = 10,
		)
		obj.saveBuildingList('../Dataset/buildingList.npy')
		obj.printBuildingList()
	else:
		obj = BuildingListConstructor(range_vertices = (4, 10), filename = '../Dataset/buildingList.npy')
		building = BuildingImageDownloader('../Dataset/GoogleMapAPIKey.txt')

