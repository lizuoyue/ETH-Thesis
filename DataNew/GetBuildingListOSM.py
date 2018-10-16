import Config
import numpy as np
import lxml.etree as ET
import os, sys, time, requests

class BuildingListConstructor(object):
	def __init__(self, num_vertices_range, filename = None):
		self.building = {}
		self.num_vertices_range = num_vertices_range
		assert(num_vertices_range[0] >= 3)
		assert(num_vertices_range[0] <= num_vertices_range[1])
		if os.path.exists(filename):
			self.loadBuildingList(filename)
		self.filename = filename
		return

	def getBuildingIDListSorted(self):
		li = [item for item in self.building]
		li.sort()
		return li

	def getBuilding(self, building_id):
		return self.building[building_id]

	def getBuildingList(self):
		return [self.building[item] for item in self.building]

	def saveBuildingList(self, filename):
		np.save(filename, self.building)
		return

	def printBuildingList(self, show_list = False):
		if show_list:
			print(self.building)
		print('Totally %d buidings.' % len(self.building))
		return

	def loadBuildingList(self, filename):
		d = np.load(filename).item()
		for bid in d:
			if bid in self.building:
				assert(len(self.building[bid]) == len(d[bid]))
			else:
				building = d[bid]
				if len(building) >= self.num_vertices_range[0] and len(building) <= self.num_vertices_range[1]:
					self.building[bid] = building
		self.printBuildingList()
		return

	def addBuildingList(self, left, up, right, down):
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
		hole = {}
		for item in osm:
			if item.tag == 'node':
				id_str = item.attrib.get('id')
				lon = item.attrib.get('lon')
				lat = item.attrib.get('lat')
				if id_str and lon and lat:
					node[int(id_str)] = (float(lon), float(lat))
				continue
			if item.tag == 'relation':
				for sub_item in item:
					if sub_item.tag == 'member':
						ref = sub_item.attrib.get('ref')
						role = sub_item.attrib.get('role')
						if ref and role == 'inner':
							hole[int(ref)] = None

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
					if 'building' in d:
						node_list = node_list[: -1]
						if len(node_list) >= self.num_vertices_range[0] and len(node_list) <= self.num_vertices_range[1]:
							bid = int(item.attrib.get('id'))
							if bid in self.building:
								assert(len(self.building[bid]) == len(node_list))
							else:
								if bid not in hole:
									self.building[bid] = node_list
		return

	def batchAddBuildingList(self, city_info):
		lon, lat = city_info['center']
		dx, dy = city_info['step']
		x1, x2 = city_info['xrange']
		y1, y2 = city_info['yrange']
		for x in range(x1, x2):
			for y in range(y1, y2):
				print('Step', x, y)
				self.addBuildingList(
					left  = lon + dx * x,
					up    = lat + dy * y,
					right = lon + dx * x + dx,
					down  = lat + dy * y + dy,
				)
				self.printBuildingList()
				self.saveBuildingList(self.filename)
		return

if __name__ == '__main__':
	assert(len(sys.argv) == 2)
	config = Config.Config()
	city_name = sys.argv[1]
	objCons = BuildingListConstructor(num_vertices_range = (4, 20), filename = './BuildingList%s.npy' % city_name)
	objCons.batchAddBuildingList(config.CITY_COO[city_name])
