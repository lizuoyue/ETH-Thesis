import glob, math

def ccw(A, B, C):
	return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
	# Return true if line segments AB and CD intersect
	return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def intersectLines( pt1, pt2, ptA, ptB ): 
	# https://www.cs.hmc.edu/ACM/lectures/intersections.html
	""" this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)
		
		returns a tuple: (xi, yi, valid, r, s), where
		(xi, yi) is the intersection
		r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
		s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
			valid == 0 if there are 0 or inf. intersections (invalid)
			valid == 1 if it has a unique intersection ON the segment	"""

	DET_TOLERANCE = 0.00000001

	# the first line is pt1 + r*(pt2-pt1)
	# in component form:
	x1, y1 = pt1;   x2, y2 = pt2
	dx1 = x2 - x1;  dy1 = y2 - y1

	# the second line is ptA + s*(ptB-ptA)
	x, y = ptA;   xB, yB = ptB;
	dx = xB - x;  dy = yB - y;

	# we need to find the (typically unique) values of r and s
	# that will satisfy
	#
	# (x1, y1) + r(dx1, dy1) = (x, y) + s(dx, dy)
	#
	# which is the same as
	#
	#	[ dx1  -dx ][ r ] = [ x-x1 ]
	#	[ dy1  -dy ][ s ] = [ y-y1 ]
	#
	# whose solution is
	#
	#	[ r ] = _1_  [  -dy   dx ] [ x-x1 ]
	#	[ s ] = DET  [ -dy1  dx1 ] [ y-y1 ]
	#
	# where DET = (-dx1 * dy + dy1 * dx)
	#
	# if DET is too small, they're parallel
	#
	DET = (-dx1 * dy + dy1 * dx)

	if math.fabs(DET) < DET_TOLERANCE: return (0,0,0,0,0)

	# now, the determinant should be OK
	DETinv = 1.0/DET

	# find the scalar amount along the "self" segment
	r = DETinv * (-dy  * (x-x1) +  dx * (y-y1))

	# find the scalar amount along the input line
	s = DETinv * (-dy1 * (x-x1) + dx1 * (y-y1))

	# return the average of the two descriptions
	xi = (x1 + r*dx1 + x + s*dx)/2.0
	yi = (y1 + r*dy1 + y + s*dy)/2.0
	return ( xi, yi, 1, r, s )

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

def distL2(A, B):
	return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)

def cutPolygon(width, height, polygon):
	corner = [(0, 0), (0, height - 1), (width - 1, height - 1), (width - 1, 0)]
	edge = [(corner[i - 1], corner[i]) for i in range(4)]
	flag = [x >= 0 and y >= 0 and x < width and y < height for x, y in polygon]
	print(flag)
	if sum(flag) == len(polygon):
		return polygon
	new_poly = []
	for i, (f0, f1) in enumerate(zip(flag, flag[1: ] + [flag[0]])):
		if f0:
			new_poly.append((0, polygon[i]))
		A, B = polygon[i], polygon[(i + 1) % len(polygon)]
		if (f0 and not f1) or (f1 and not f0):
			for C, D in edge:
				if intersect(A, B, C, D):
					break
			x, y, _, _, _ = intersectLines(A, B, C, D)
			x, y = int(math.floor(x)), int(math.floor(y))
			new_poly.append((1, (x, y)))
		if not f0 and not f1:
			cand = []
			for C, D in edge:
				if intersect(A, B, C, D):
					x, y, _, _, _ = intersectLines(A, B, C, D)
					x, y = int(math.floor(x)), int(math.floor(y))
					cand.append((distL2(A, (x, y)), (x, y)))
			cand.sort()
			for _, (x, y) in cand:
				new_poly.append((2, (x, y)))
	res = []
	for i, (flag1, (x1, y1)) in enumerate(new_poly):
		flag0, (x0, y0) = new_poly[i - 1]
		res.append((x0, y0))
		if flag0 == 1 and flag1 == 1:
			if x0 != x1 and y0 != y1:
				if x0 == 0 and y1 == 0:
					res.append(corner[0])
				elif y0 == 0 and x1 == width - 1:
					res.append(corner[3]) 
				elif x0 == width - 1 and y1 == height - 1:
					res.append(corner[2])
				elif y0 == height - 1 and x1 == 0:
					res.append(corner[1])
				else:
					print(polygon)
					print(res)
					assert(False)
		if flag0 == 2 and flag1 == 2:
			if x0 != x1 and y0 != y1:
				if x0 == 0 and y1 == 0:
					choice = corner[0]
				elif y0 == 0 and x1 == width - 1:
					choice = corner[3]
				elif x0 == width - 1 and y1 == height - 1:
					choice = corner[2]
				elif y0 == height - 1 and x1 == 0:
					choice = corner[1]
				else:
					print(polygon)
					print(res)
	return res

if __name__ == '__main__':
	# print(cutPolygon(640, 640, [(8, 3), (3, 8), (-3, 8), (-3, -3), (8, -3)]))
	# print(cutPolygon(640, 640, [(632, 637), (637, 632), (643, 632), (643, 643), (632, 643)]))
	# print(cutPolygon(640, 640, [(8, 3), (3, 8), (-3, 8), (8, -3)]))
	# print(cutPolygon(640, 640, [(632, 637), (637, 632), (643, 632), (632, 643)]))
	# print(cutPolygon(640, 640, [(300, -10), (330, -10), (330, 680), (270, 680)]))
	# print(cutPolygon(600, 600, [(185, 640), (197, 622), (209, 630), (198, 648)]))
	print(cutPolygon(600, 600, [(558, -10), (633, -93), (670, -59), (613, 3), (620, 10), (602, 30)]))




# shift_info = {}
# for file in glob.glob('../../Buildings%s/*/shift.txt' % city_name):
# 	lines = open(file, 'r').readlines()
# 	alpha, shift_i, shift_j = lines[0].strip().split()
# 	score, edge_score = lines[1].strip().split()
# 	bid = int(file.split('/')[3])
# 	shift_info[bid] = (float(alpha), int(shift_i), int(shift_j), float(score), float(edge_score))






# # Decide the order of vertices
# s = sum([x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(polygon, polygon[1: ] + [polygon[0]])])
# if s > 0:
# 	polygon.reverse()

# # Remove

# if len(polygon) > 4:
# 	for (x0, y0), (x1, y1), (x2, y2) in zip([polygon[-1]] + polygon[: -1], polygon, polygon[1: ] + [polygon[0]]):
# 		s = 0
# 		s += x0 * y1 - x1 * y0
# 		s += x1 * y2 - x2 * y1
# 		s += x2 * y0 - x0 * y2
# 		if math.abs(s) < 1:


