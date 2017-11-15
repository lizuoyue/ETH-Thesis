import numpy as np
import lxml.etree as ET
import os, math, requests, scipy.misc
from PIL import Image, ImageDraw
import random, time
import io, binascii

TILE_SIZE = 256
GOOGLE_MAP_KEY = 'AIzaSyAR7IQKtdUg3X0wmpdIoES5lcGwGqtlL7o'

# Maps Utilities ============================================================

def lonLatToWorld(lon, lat):
	# Truncating to 0.9999 effectively limits latitude to 89.189. This is
	# about a third of a tile past the edge of the world tile.
	siny = math.sin(float(lat) * math.pi / 180.0);
	siny = min(max(siny, -0.9999), 0.9999)
	return \
		TILE_SIZE * (0.5 + float(lon) / 360.0), \
		TILE_SIZE * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)) \

def lonLatToPixel(lon, lat, z):
	scale = 2 ** z
	wx, wy = lonLatToWorld(lon, lat)
	return math.floor(wx * scale), math.floor(wy * scale)

def lonLatToTile(lon, lat, z):
	scale = 2 ** z
	wx, wy = lonLatToWorld(lon, lat)
	return math.floor(wx * scale / TILE_SIZE), math.floor(wy * scale / TILE_SIZE)

def pixelToLonLat(px, py, z):
	# Return the longitude of the left boundary of a pixel
	# and the latitude of the upper boundary of a pixel
	scale = 2 ** z
	lon = (float(px) / float(scale) / TILE_SIZE - 0.5) * 360.0
	temp = math.exp((0.5 - float(py) / float(scale) / TILE_SIZE) * 4 * math.pi)
	lat = math.asin((temp - 1.0) / (temp + 1.0)) / math.pi * 180
	return lon, lat

class BoundingBox(object):

	def __init__(self, lon, lat, zoom, scale, size):
		self.center_lon = lon
		self.center_lat = lat
		self.size = size
		self.z = zoom + scale - 1
		self.center_px, self.center_py = lonLatToPixel(lon, lat, self.z)
		self.left, self.up = pixelToLonLat(self.center_px - size / 2, self.center_py - size / 2, self.z)
		self.right, self.down = pixelToLonLat(self.center_px + size / 2, self.center_py + size / 2, self.z)
		# print(self.up, self.right, self.down, self.left)

	def lonLatToRelativePixel(self, lon, lat):
		# Zero-based
		px, py = lonLatToPixel(lon, lat, self.z)
		return int(px - self.center_px + self.size / 2), int(py - self.center_py + self.size / 2)

def getBuildingAerialImage(building, filename):
	min_lon, max_lon = 200, -200
	min_lat, max_lat = 100, -100
	for lon, lat in building:
		min_lon = min(lon, min_lon)
		min_lat = min(lat, min_lat)
		max_lon = max(lon, max_lon)
		max_lat = max(lat, max_lat)
	a, b = lonLatToPixel(min_lon, max_lat, 20)
	c, d = lonLatToPixel(max_lon, min_lat, 20)
	if c - a > 200 or d - b > 200:
		return
	c_lon = (min_lon + max_lon) / 2
	c_lat = (min_lat + max_lat) / 2
	while True:
		try:
			data = requests.get(
				'https://maps.googleapis.com/maps/api/staticmap?' 						+ \
				'maptype=%s&' 			% 'satellite' 									+ \
				'center=%.7lf,%.7lf&' 	% (c_lat, c_lon) 							+ \
				'zoom=%d&' 				% 19 									+ \
				'size=%dx%d&' 			% (300, 300)	+ \
				'scale=%d&' 			% 2 									+ \
				'format=%s&' 			% 'png32' 										+ \
				'key=%s' 				% GOOGLE_MAP_KEY 								  \
			).content
			time.sleep(0.2)
			break
		except:
			print('Try again.')
			pass
	img = np.array(Image.open(io.BytesIO(data)))
	beg = int((img.shape[0] - 224) / 2)
	end = img.shape[0] - beg
	scipy.misc.imsave(filename, img[beg: end, beg: end, 0: ])

	bbox = BoundingBox(c_lon, c_lat, 19, 2, 224)
	polygon = []
	polygon_s = []
	for lon, lat in building:
		px, py = bbox.lonLatToRelativePixel(lon, lat)
		polygon.append((px, py))
		polygon_s.append((int(px / 8), int(py / 8)))
	img = Image.open(filename)
	mask = Image.new('RGBA', img.size, color = (255, 255, 255, 0))
	draw = ImageDraw.Draw(mask)
	draw.polygon(polygon, fill = (255, 0, 0, 128), outline = (255, 0, 0, 128))
	merge = Image.alpha_composite(img, mask)
	# img.show()
	# mask.show()
	# merge.show()
	# merge.save(filename.replace('.png', '-z.png'))
	# mask.save(filename.replace('.png', '-m.png'))
	img_size_s = (28, 28)

	boundary = Image.new('P', img_size_s, color = 0)
	draw = ImageDraw.Draw(boundary)
	draw.polygon(polygon_s, fill = 0, outline = 255)
	# boundary.show()

	vertices = Image.new('P', img_size_s, color = 0)
	draw = ImageDraw.Draw(vertices)
	draw.point(polygon_s, fill = 255)
	# vertices.show()

	polygon_img = []
	for vertex in polygon_s:
		single = Image.new('P', img_size_s, color = 0)
		draw = ImageDraw.Draw(single)
		draw.point([vertex], fill = 255)
		# single.show()
		polygon_img.append(np.array(single) / 255.0)
	polygon_img.append(np.array(Image.new('P', img_size_s, color = 0)) / 255.0)
	return img, mask, merge, np.array(boundary)/255.0, np.array(vertices)/255.0, np.array(polygon_img)

class BuildingListConstructor(object):
	def __init__(self, range_vertices, filename = None):
		self.building = {}
		self.range_vertices = range_vertices
		if filename:
			self.loadBuildingList(filename)
		return

	def saveBuildingList(self, filename):
		np.save(filename, self.building)
		return

	def printBuildingList(self):
		print(self.building)
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
		print(len(self.building))
		return

	def getBuildingList(self):
		return [self.building[item] for item in self.building]

	def resetBuildingList(self):
		self.building = {}
		return

	def batchAddBuildingList(self, lon, lat, lon_step, lat_step, lon_num, lat_num):
		for i in range(lon_num):
			for j in range(lat_num):
				print(i, j)
				self.addBuildingList(
					left  = lon + lon_step * i - lon_step / 2,
					down  = lat - lat_step * j - lat_step / 2,
					right = lon + lon_step * i + lon_step / 2,
					up    = lat - lat_step * j + lat_step / 2,
				)
				print(len(self.building))
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
				print('Try again.')
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
					if 'building' in d and d['building'] == 'yes':
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

	def getImage(self, batch_size):
		result = []
		while len(result) < batch_size:
			building = random.sample(self.getBuildingList(), 1)
			res = getBuildingAerialImage(building[0], 'wolegequ.png')
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
		obj.saveBuildingList('./buildingList.npy')
		obj.printBuildingList()
	else:
		obj = BuildingListConstructor(range_vertices = (4, 12), filename = './buildingList.npy')


