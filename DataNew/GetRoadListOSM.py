import Config
import numpy as np
import lxml.etree as ET
import os, sys, time, requests

class RoadListConstructor(object):
	def __init__(self, filename = None):
		self.road = {}
		if os.path.exists(filename):
			self.loadRoadList(filename)
		self.filename = filename
		return

	def getRoadIDListSorted(self):
		li = [item for item in self.road]
		li.sort()
		return li

	def getRoadList(self):
		return [self.road[item] for item in self.road]

	def saveRoadList(self, filename):
		np.save(filename, self.road)
		return

	def printRoadList(self, show_list = False):
		if show_list:
			print(self.road)
		print('Totally %d roads.' % len(self.road))
		return

	def loadRoadList(self, filename):
		d = np.load(filename).item()
		for rid in d:
			if rid in self.road:
				assert(len(self.road[rid]) == len(d[rid]))
			else:
				self.road[rid] = d[rid]
		self.printRoadList()
		return

	def addRoadList(self, left, up, right, down):
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
					node[int(id_str)] = (float(lon), float(lat), int(id_str))

		#
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
						if rid in self.road:
							assert(len(self.road[rid]) == len(node_list))
						else:
							self.road[rid] = node_list
		return

	def batchAddRoadList(self, city_info):
		lon, lat = city_info['center']
		dx, dy = city_info['step']
		x1, x2 = city_info['xrange']
		y1, y2 = city_info['yrange']
		for x in range(x1, x2):
			for y in range(y1, y2):
				print('Step', x, y)
				self.addRoadList(
					left  = lon + dx * x,
					up    = lat + dy * y,
					right = lon + dx * x + dx,
					down  = lat + dy * y + dy,
				)
				self.printRoadList()
				self.saveRoadList(self.filename)
		return

if __name__ == '__main__':
	assert(len(sys.argv) == 2)
	config = Config.Config()
	city_name = sys.argv[1]
	objCons = RoadListConstructor(filename = './RoadList%s.npy' % city_name)
	objCons.batchAddRoadList(config.CITY_COO[city_name])
