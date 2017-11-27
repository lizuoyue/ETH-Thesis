import numpy as np
import lxml.etree as ET
import sys, time, requests

CITY_DICT = {
	'Zurich': {
		'left-up-center': (8.4200000, 47.4600000),
		'step': (0.0221822, 0.0150000),
		'xrange': (0, 10),
		'yrange': (0, 10),
	},
	'Chicago': {
		'left-up-center': (-87.7976490, 41.960124),
		'step': (0.0134483, 0.0100000),
		'xrange': (3, 9), # 12
		'yrange': (3, 15), # 18
	}
}

class BuildingListConstructor(object):

	def __init__(self, range_vertices, filename = None):
		self.building = {}
		self.range_vertices = range_vertices
		assert(range_vertices[0] >= 3)
		assert(range_vertices[0] <= range_vertices[1])
		if filename:
			self.loadBuildingList(filename)
		return

	def getBuildingList(self):
		return [self.building[item] for item in self.building]

	def resetBuildingList(self):
		self.building = {}
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

	def addBuildingList(self, left, down, right, up):
		while True:
			time.sleep(1)
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

	def batchAddBuildingList(self, city_info):
		lon = city_info['left-up-center'][0]
		lat = city_info['left-up-center'][1]
		lon_step = city_info['step'][0]
		lat_step = city_info['step'][1]
		lon_range = city_info['xrange']
		lat_range = city_info['yrange']
		for i in range(lon_range[0], lon_range[1]):
			for j in range(lat_range[0], lat_range[1]):
				print('Step', i, j)
				self.addBuildingList(
					left  = lon + lon_step * i - lon_step / 2,
					down  = lat - lat_step * j - lat_step / 2,
					right = lon + lon_step * i + lon_step / 2,
					up    = lat - lat_step * j + lat_step / 2,
				)
				self.printBuildingListLen()
				self.saveBuildingList('./BuildingList-%s.npy' % sys.argv[1])
		return

if __name__ == '__main__':
	assert(len(sys.argv) == 2)
	city_name = sys.argv[1]
	objCons = BuildingListConstructor(range_vertices = (4, 20), filename = './BuildingList-%s.npy' % city_name)
	objCons.batchAddBuildingList(CITY_DICT[city_name])
	objCons.saveBuildingList('./BuildingList-%s.npy' % city_name)
	objCons.printBuildingList()

