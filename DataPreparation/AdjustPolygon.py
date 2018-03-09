import sys, glob, math

def applyAlphaShiftToPolygon(info, polygon):
	alpha, shift_i, shift_j = info
	if alpha != 1:
		polygon_i = np.array([v[1] for v in polygon], np.float32)
		polygon_j = np.array([v[0] for v in polygon], np.float32)
		c_i = (polygon_i.min() + polygon_i.max()) / 2
		c_j = (polygon_j.min() + polygon_j.max()) / 2
		polygon_i = np.array(polygon_i * alpha + c_i * (1 - alpha), np.int32)
		polygon_j = np.array(polygon_j * alpha + c_j * (1 - alpha), np.int32)
		polygon = [(polygon_j[k], polygon_i[k]) for k in range(len(polygon))]
	return [(x + shift_j, y + shift_i) for x, y in polygon]

if __name__ == '__main__':
	assert(len(sys.argv) == 3)
	city_name = sys.argv[2]
	shift_info = {}
	for file in glob.glob('../../Buildings%s/*/shift.txt' % city_name):
		lines = open(file, 'r').readlines()
		alpha, shift_i, shift_j = lines[0].strip().split()
		bid = int(file.split('/')[3])
		shift_info[bid] = (float(alpha), int(shift_i), int(shift_j))

	files = glob.glob('../../Areas%s/*/polygons.txt' % city_name)
	for i, file in enumerate(files):
		print('%d/%d' % (i, len(files)))
		# Read original polygons
		lines = open(file, 'r').readlines()
		polygons = []
		for line in lines:
			if line.strip() != '':
				if line.startswith('%'):
					_, bid = line.strip().split()
					polygons.append([int(bid)])
					if int(bid) not in shift_info:
						shift_info[int(bid)] = (1, 0, 0)
				else:
					x, y = line.strip().split()
					polygons[-1].append((int(x), int(y)))

		# Adjust shift
		polygons = [(polygon[0], applyAlphaShiftToPolygon(shift_info[polygon[0]], polygon[1: ])) for polygon in polygons]
		
		# Decide the order of vertices and remove some
		new_polygons = []
		for bid, polygon in polygons:
			s = sum([x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(polygon, polygon[1: ] + [polygon[0]])])
			if s > 0:
				polygon.reverse()
			if len(polygon) > 4:
				new_polygon = []
				for (x0, y0), (x1, y1), (x2, y2) in zip([polygon[-1]] + polygon[: -1], polygon, polygon[1: ] + [polygon[0]]):
					s = 0
					s += x0 * y1 - x1 * y0
					s += x1 * y2 - x2 * y1
					s += x2 * y0 - x0 * y2
					if math.abs(s) > 1:
						new_polygon.append((x1, y1))
			else:
				new_polygon = polygon
			new_polygons.append((bid, new_polygon))

		# Write file
		with open(file.replace('polygons.txt', 'polygons_after_shift.txt', 'w')) as f:
			for bid, polygon in polygons:
				f.write('%% %d\n' % bid)
				for v in polygon:
					f.write('%d %d\n' % v)

