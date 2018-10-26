import Config
import numpy as np
import lxml.etree as ET
import os, sys, time, requests, math, traceback

config = Config.Config()

class Constructor(object):
	def __init__(self, city_name):
		self.file_b = '%sBuildingList.npy' % city_name
		self.file_r = '%sRoadList.npy' % city_name
		self.road, self.building = {}, {}
		if os.path.exists(self.file_b):
			self.loadBuildingList()
		if os.path.exists(self.file_r):
			self.loadRoadList()
		self.city_name = city_name
		return

	def getBuildingIDListSorted(self):
		li = [item for item in self.building]
		li.sort()
		return li

	def getRoadIDListSorted(self):
		li = [item for item in self.road]
		li.sort()
		return li

	def getBuildingList(self):
		return [self.building[item] for item in self.building]

	def getRoadList(self):
		return [self.road[item] for item in self.road]

	def printBuildingList(self, show_list = False):
		if show_list:
			print(self.building)
		print('Totally %d buidings.' % len(self.building))
		return

	def printRoadList(self, show_list = False):
		if show_list:
			print(self.road)
		print('Totally %d roads.' % len(self.road))
		return

	def loadRoadList(self):
		d = np.load(self.file_r).item()
		for rid in d:
			if rid in self.road:
				assert(len(self.road[rid]) == len(d[rid]))
			else:
				self.road[rid] = d[rid]
		self.printRoadList()
		return

	def loadBuildingList(self):
		d = np.load(self.file_b).item()
		for bid in d:
			if bid in self.building:
				assert(len(self.building[bid]) == len(d[bid]))
			else:
				self.building[bid] = d[bid]
		self.printBuildingList()
		return

	def getOSM(self, left, up, right, down):
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
				e1, e2, e3 = sys.exc_info()
				traceback.print_exception(e1, e2, e3)
				time.sleep(5)
		return osm

	def save(self):
		# np.save(self.file_b, self.building)
		np.save(self.file_r, self.road)
		return

	def addBuilding(self, osm):
		node = {}
		hole = set()
		maybe = set()
		for item in osm:
			if item.tag == 'node':
				id_str = item.attrib.get('id')
				lon = item.attrib.get('lon')
				lat = item.attrib.get('lat')
				if id_str and lon and lat:
					node[int(id_str)] = (float(lon), float(lat))
				continue
			if item.tag == 'relation':
				d = {}
				for sub_item in item:
					if sub_item.tag == 'tag':
						k = sub_item.attrib.get('k')
						v = sub_item.attrib.get('v')
						if k and v:
							d[k] = v
				for sub_item in item:
					if sub_item.tag == 'member':
						ref = sub_item.attrib.get('ref')
						role = sub_item.attrib.get('role')
						if ref and role == 'inner':
							hole.add(int(ref))
						if 'building' in d and d['building'] != 'no':
							if ref and role == 'outer':
								maybe.add(int(ref))
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
					bid = int(item.attrib.get('id'))
					if ('building' in d and d['building'] != 'no') or (bid in maybe):
						node_list = node_list[: -1]
						if bid in self.building:
							assert(len(self.building[bid]) == len(node_list))
						else:
							if bid not in hole:
								self.building[bid] = node_list
		return

	def addRoad(self, osm):
		node = {}
		for item in osm:
			if item.tag == 'node':
				id_str = item.attrib.get('id')
				lon = item.attrib.get('lon')
				lat = item.attrib.get('lat')
				if id_str and lon and lat:
					node[int(id_str)] = (float(lon), float(lat), int(id_str))
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
					if len(node_list) < 2:
						continue
					if 'highway' in d:
						if d['highway'] in config.OSM_HIGHWAY_BLACKLIST:
							print('Highway in blacklist.')
							continue
						if 'amenity' in d and d['amenity'] == 'parking':
							print('Parking 1.')
							continue
						if 'service' in d and (d['service'] == 'parking_aisle' or d['service'] == 'driveway'):
							print('Parking 2.')
							continue
						if 'tunnel' in d and d['tunnel'] == 'yes':
							print('Tunnel 1.')
							continue
						if 'layer' in d and len(d['layer']) >= 2 and d['layer'][0] == '-':
							print('Tunnel 2.')
							continue
						rid = int(item.attrib.get('id'))
						if rid in self.road:
							assert(len(self.road[rid]) == len(node_list))
						else:
							self.road[rid] = node_list
		return

	def batchAdd(self, city_name, city_info):
		city_poly = city_info['map_area']
		poly = np.array(city_poly)
		minlat, maxlat = poly[:, 0].min(), poly[:, 0].max()
		minlon, maxlon = poly[:, 1].min(), poly[:, 1].max()
		dx, dy = config.OSM_LON_STEP, config.OSM_LAT_STEP
		nx = math.ceil((maxlon - minlon) / dx)
		ny = math.ceil((maxlat - minlat) / dy)
		for x in range(nx):
			for y in range(ny):
				print('Step', x, nx, y, ny)
				filename = './%sOSM/%d_%d.osm' % (city_name, x, y)
				if os.path.exists(filename):
					osm = ET.parse(filename).getroot()
				else:
					osm = self.getOSM(
						left  = minlon + dx * x,
						up    = maxlat - dy * y,
						right = minlon + dx * x + dx,
						down  = maxlat - dy * y - dy,
					)
					osm_str = ET.tostring(osm, pretty_print = True)
					with open(filename, 'wb') as f:
						f.write(osm_str)
				# self.addBuilding(osm)
				self.addRoad(osm)
				self.printBuildingList()
				self.printRoadList()
				self.save()
		return

if __name__ == '__main__':
	assert(len(sys.argv) == 2)
	city_name = sys.argv[1]
	objCons = Constructor(city_name)
	if not os.path.exists(city_name + 'OSM'):
		os.popen('mkdir %sOSM' % city_name)
	objCons.batchAdd(city_name, config.CITY_INFO[city_name])
