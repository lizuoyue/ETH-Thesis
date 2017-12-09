
	def imageAugmentation(self, img):
		img = np.array(img)
		lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
		l, a, b = cv2.split(lab)
		clahe = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (8, 8))
		cl = clahe.apply(l)
		limg = cv2.merge((cl, a, b))
		final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
		return final

	def var(self, img, polygon):
		img = np.array(img)
		color = []
		for i, j in polygon:
			color.append(img[i, j, ...])
		mean = np.mean(np.array(color), axis = 0)
		color_dist = [self.colorDist(c, mean) for c in color]
		var = sum(color_dist) / len(color_dist)
		return var

	def edge(self, img, polygon):
		img = np.array(img)
		li = []
		for i, j in polygon:
			li.append(np.sum(img[i, j, ...]))
		return sum(li)

	def shiftStep(self, img, polygon, mode):
		if mode == 'var':
			search_range = range(-1, 2)
			min_var = sys.maxsize
			for i in search_range:
				for j in search_range:
					new_polygon = [(p[0] + i, p[1] + j) for p in polygon]
					var = self.var(img, new_polygon)
					if var < min_var:
						min_var = var
						shift_i = i
						shift_j = j
		if mode == 'edge':
			search_range = range(-2, 3)
			max_edge = 0
			for i in search_range:
				for j in search_range:
					new_polygon = [(p[0] + i, p[1] + j) for p in polygon]
					edge = self.edge(img, new_polygon)
					if edge > max_edge:
						max_edge = edge
						shift_i = i
						shift_j = j
		return shift_i, shift_j

	def shift(self, img, polygon, mode):
		shift_i = 0
		shift_j = 0
		if mode == 'edge':
			step_num = 6
			img = cv2.Canny(img[..., 0: 3], 100, 200)
			mask = Image.new('P', (img.shape[0], img.shape[1]), color = 0)
			draw = ImageDraw.Draw(mask)
			draw.polygon(polygon, fill = 0, outline = 255)
			mask = np.array(mask)
		if mode == 'var':
			step_num = 8
			img = self.imageAugmentation(img[..., 0: 3])
			mask = Image.new('P', (img.shape[0], img.shape[1]), color = 0)
			draw = ImageDraw.Draw(mask)
			draw.polygon(polygon, fill = 255, outline = 255)
			mask = np.array(mask)
		polygon = []
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				if mask[i, j] > 128:
					polygon.append((i, j))
		for i in range(step_num):
			new_polygon = [(p[0] + shift_i, p[1] + shift_j) for p in polygon]
			step_i, step_j = self.shiftStep(img, new_polygon, mode)
			if step_i == 0 and step_j == 0:
				break
			shift_i += step_i
			shift_j += step_j
		return shift_i, shift_j


	def colorDist(self, c1, c2):
		s = 0.0
		for i in range(3):
			s += (c1[i] - c2[i]) ** 2
		return math.sqrt(s)





		shift_i, shift_j = self.shift(img, polygon, mode = 'var')
		print(shift_i, shift_j)
		polygon = [(p[0] + shift_j, p[1] + shift_i) for p in polygon]




		
		# 
		# boundary = Image.new('P', size, color = 0)
		# draw = ImageDraw.Draw(boundary)
		# draw.polygon(polygon, fill = 0, outline = 255)
		# draw.line(polygon + [polygon[0]], fill = 255, width = 3) # <- For thicker outline
		# for point in polygon:
			# draw.ellipse((point[0] - 1, point[1] - 1, point[0] + 1, point[1] + 1), fill = 255)
		# if save:
			# boundary.save('../%s/%d/3-b.png' % (self.city_name, building_id))
		# boundary = ut.pil2np(boundary, show)

		# 
		# vertices = Image.new('P', size, color = 0)
		# draw = ImageDraw.Draw(vertices)
		# draw.point(polygon, fill = 255)
		# for point in polygon:
			# draw.ellipse((point[0] - 1, point[1] - 1, point[0] + 1, point[1] + 1), fill = 255)
		# if save:
			# vertices.save('../%s/%d/4-v.png' % (self.city_name, building_id))
		# vertices = ut.pil2np(vertices, show)

		# 
		# vertex_list = []
		# for i in range(len(polygon_s)):
		# 	vertex = Image.new('P', size_s, color = 0)
		# 	draw = ImageDraw.Draw(vertex)
		# 	draw.point([polygon_s[i]], fill = 255)
		# 	if save:
		# 		vertex.save('../%s/%d/5-v%s.png' % (self.city_name, building_id, str(i).zfill(2)))
		# 	vertex = ut.pil2np(vertex, show)
		# 	vertex_list.append(vertex)
		# vertex_list.append(np.zeros(size_s, dtype = np.float32))
		# vertex_list = np.array(vertex_list)