import numpy as np
import lxml.etree as ET
import os, math, requests, scipy.misc
from PIL import Image, ImageDraw

# Global Constants ============================================================

CITY = 'Zurich'
TILE_SIZE = 256
DOWNLOAD_SIZE = 640
FILE_PATH = '../Dataset'
GOOGLE_MAP_KEY = 'AIzaSyAR7IQKtdUg3X0wmpdIoES5lcGwGqtlL7o'
CITY_DICT = {
	# Longitude first, latitude second
	'Zurich': {
		'center': (8.5386261, 47.3930324),
		'step': (0.0030, 0.0020),
		'xrange': (-5, 5),
		'yrange': (-5, 5),
		'zoom': 17,
		'scale': 2,
		'size': 600
	},
	'London': {
		'center': (8.5402048, 47.3779211),
		'step': (0.0015, 0.0010),
		'xrange': (-2, 3),
		'yrange': (-2, 3),
		'zoom': 19,
		'scale': 2,
		'size': 1200
	}
}

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

# Image Downloader ============================================================

class AerialImageWithCorner(object):

	def __init__(self, lon, lat, zoom, scale, size, fid):
		self.lon = lon
		self.lat = lat
		self.zoom = zoom
		self.scale = scale
		self.size = size
		self.z = self.zoom + self.scale - 1
		if not os.path.exists(FILE_PATH): 
			os.makedirs(FILE_PATH)
		if not os.path.exists(FILE_PATH + '/osm'): 
			os.makedirs(FILE_PATH + '/osm')
		self.filenamePNG = FILE_PATH + '/%d-i.png' % fid
		self.filenameOSM = FILE_PATH + '/osm/%d.osm' % fid
		self.filenameMEG = FILE_PATH + '/%d-m.png' % fid
		self.filenameCLS = FILE_PATH + '/%d-c.png' % fid
		self.bbox = BoundingBox(lon = lon, lat = lat, zoom = zoom, scale = scale, size = size)

	def downloadAerialImage(self):
		f = open(self.filenamePNG, 'wb')
		img = requests.get(
			'https://maps.googleapis.com/maps/api/staticmap?' 						+ \
			'maptype=%s&' 			% 'satellite' 									+ \
			'center=%.7lf,%.7lf&' 	% (self.lat, self.lon) 							+ \
			'zoom=%d&' 				% self.zoom 									+ \
			'size=%dx%d&' 			% (DOWNLOAD_SIZE, DOWNLOAD_SIZE)	+ \
			'scale=%d&' 			% self.scale 									+ \
			'format=%s&' 			% 'png32' 										+ \
			'key=%s' 				% GOOGLE_MAP_KEY 								  \
		).content
		f.write(img)
		f.close()
		img = scipy.misc.imread(self.filenamePNG)
		beg = (img.shape[0] - self.size) / 2
		end = img.shape[0] - beg
		scipy.misc.imsave(self.filenamePNG, img[beg: end, beg: end, 0: ])

	def downloadOpenStreetMap(self):
		f = open(self.filenameOSM, 'w')
		osm = requests.get(
			'http://www.openstreetmap.org/api/0.6/map?bbox=' + \
			'%.7lf%%2C%.7lf%%2C%.7lf%%2C%.7lf' % \
			(self.bbox.left, self.bbox.down, self.bbox.right, self.bbox.up)
		).content
		f.write(osm)
		f.close()
		f = open(self.filenameOSM, 'r')
		text = f.readlines()
		text = ''.join(text[1:])
		f.close()
		osm = ET.fromstring(text)
		node = {}
		for item in osm:
			if item.tag == 'node':
				id_str = item.attrib.get('id')
				lon = item.attrib.get('lon')
				lat = item.attrib.get('lat')
				if id_str and lon and lat:
					node[id_str] = (float(lon), float(lat))
		polygons = []
		for item in osm:
			if item.tag == 'way':
				node_list = []
				for sub_item in item:
					if sub_item.tag == 'nd':
						ref = sub_item.attrib.get('ref')
						if ref:
							node_list.append(node[ref])
				d = {}
				for sub_item in item:
					if sub_item.tag == 'tag':
						k = sub_item.attrib.get('k')
						v = sub_item.attrib.get('v')
						if k and v:
							d[k] = v
				if 'building' in d or d.get('tourism') == 'hotel': # and d['building'] == 'yes':
					# flag = []
					# for coo in node_list:
					# 	col, row = self.bbox.lonLatToRelativePixel(lon = coo[0], lat = coo[1])
					# 	if col >= 0 and col < self.size and row >= 0 and row < self.size:
					# 		flag.append(1)
					# 	else:
					# 		flag.append(0)
					# if sum(flag) == 0:
					# 	continue
					# node_list = node_list[0: len(node_list) - 1]
					polygon = []
					for i, coo in enumerate(node_list):
						col, row = self.bbox.lonLatToRelativePixel(lon = coo[0], lat = coo[1])
						# if col >= 0 and col < self.size and row >= 0 and row < self.size:
						polygon.append((col, row))
						# else:
						# 	flag = False
						# 	break
					polygons.append(polygon)
		img = Image.open(self.filenamePNG)
		mask = Image.new('RGBA', img.size, color = (255, 255, 255, 0))
		draw = ImageDraw.Draw(mask)
		for polygon in polygons:
			draw.polygon(polygon, fill = (255, 0, 0, 128), outline = (255, 0, 0, 128))
		Image.alpha_composite(img, mask).save(self.filenameMEG)
		mask.save(self.filenameCLS)

if __name__ == '__main__':
	c_lon, c_lat = CITY_DICT[CITY]['center']
	x_step, y_step = CITY_DICT[CITY]['step']
	zoom = CITY_DICT[CITY]['zoom']
	scale = CITY_DICT[CITY]['scale']
	size = CITY_DICT[CITY]['size']
	xrg = CITY_DICT[CITY]['xrange']
	yrg = CITY_DICT[CITY]['yrange']
	fid = 0
	for i in range(xrg[0], xrg[1]):
		for j in range(yrg[0], yrg[1]):
			fid += 1
			lon = c_lon + x_step * i
			lat = c_lat + y_step * j
			obj = AerialImageWithCorner(lon, lat, zoom, scale, size, fid)
			obj.downloadAerialImage()
			obj.downloadOpenStreetMap()
