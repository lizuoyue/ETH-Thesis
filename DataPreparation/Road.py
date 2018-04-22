import numpy as np
import lxml.etree as ET
import os, sys, time, requests

import Config
from UtilityGeography import *
from PIL import Image, ImageDraw

def findRoads(left, up, right, down):
	# 
	while True:
		time.sleep(1)
		try:
			osm = requests.get(
				'http://www.openstreetmap.org/api/0.6/map?bbox=' + \
				'%.7lf%%2C%.7lf%%2C%.7lf%%2C%.7lf' % (left, down, right, up)
			).content
			osm = ET.fromstring(osm)
			break
		except:
			print('Try again to get .osm file.')
			time.sleep(10)

	# 
	node = {}
	for item in osm:
		if item.tag == 'node':
			id_str = item.attrib.get('id')
			lon = item.attrib.get('lon')
			lat = item.attrib.get('lat')
			if id_str and lon and lat:
				node[int(id_str)] = (float(lon), float(lat))

	#
	res = {}
	for item in osm:
		if item.tag == 'way':
			if item.attrib.get('visible') == 'true':
				node_list = []
				d = {}
				for sub_item in item:
					if sub_item.tag == 'nd':
						ref = sub_item.attrib.get('ref')
						if ref:
							node_list.append(node[int(ref)])
						continue
					if sub_item.tag == 'tag':
						k = sub_item.attrib.get('k')
						v = sub_item.attrib.get('v')
						if k and v:
							d[k] = v
				if 'highway' in d:
					rid = int(item.attrib.get('id'))
					res[rid] = node_list
	return res


if __name__ == '__main__':
	config = Config.Config()
	city_info = config.CITY_IMAGE['Zurich']
	cclon, cclat = city_info['center']
	dx, dy = city_info['step']
	x_1, x_2 = city_info['xrange']
	y_1, y_2 = city_info['yrange']
	zoom = city_info['zoom']
	scale = city_info['scale']
	size = city_info['size']
	for nx in range(x_1, x_2):
		for ny in range(y_1, y_2):
			clon, clat = cclon + dx * nx, cclat + dy * ny
			box = BoundingBox(clon, clat, size, size, zoom, scale)
			l, u = box.relativePixelToLonLat(0, 0)
			r, d = box.relativePixelToLonLat(size, size)
			d = findRoads(l, u, r, d)
			img = Image.new('P', (size, size), color = 255)
			draw = ImageDraw.Draw(img)
			for rid in d:
				road = [box.lonLatToRelativePixel(lon, lat) for lon, lat in d[rid]]
				draw.line(road, fill = 0, width = 10)
			img.show()
			input()



