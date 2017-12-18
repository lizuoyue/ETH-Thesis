import numpy as np 
import math, random, time
import io, glob, zipfile
from PIL import Image, ImageDraw, ImageFilter
# PIL.ImageDraw: 0-based, (col_idx, row_idx) if taking image as matrix

BLUR = 0.75
TILE_SIZE = 256

def plotPolygon(img_size = (224, 224), resolution = (28, 28), num_vertices = 6):

	# Set image parameters
	num_row = img_size[0]
	num_col = img_size[1]
	half_x = math.floor(num_col / 2)
	half_y = math.floor(num_row / 2)
	dec_rate = (img_size[0] / resolution[0], img_size[1] / resolution[1])

	# Set polygon parameters
	epsilon = 1.0 / num_vertices
	center_r = math.floor(min(num_row, num_col) * 0.05) # <- Decide polygon's center
	polygon_size = math.floor(min(num_row, num_col) * 0.35) # <- Decide polygon's size
	delta_angle = np.pi * 2 * epsilon
	angle = np.random.uniform(0.0, delta_angle) # <- Decide polygon's first vertex

	# Determin the center of polygon
	center_x = half_x + np.random.randint(-center_r, center_r)
	center_y = half_y + np.random.randint(-center_r, center_r)

	# Determin the polygon vertices
	polygon = []
	polygon_s = []
	for i in range(num_vertices):
		r = polygon_size * np.random.uniform(0.8, 1.1) # <- Decide polygon's size range
		px = math.floor(center_x + r * np.cos(angle))
		py = math.floor(center_y - r * np.sin(angle)) # <- Decide polygon's order (counterclockwise)
		polygon.append((px, py))
		polygon_s.append((math.floor(px / dec_rate[0]), math.floor(py / dec_rate[1])))
		angle += delta_angle * np.random.uniform(1 - epsilon, 1 + epsilon) # <- Decide polygon's vertices
	first_idx = random.choice([i for i in range(num_vertices)])
	polygon = polygon[first_idx:] + polygon[:first_idx]
	polygon_s = polygon_s[first_idx:] + polygon_s[:first_idx]

	# Draw polygon
	color = (0, 204, 255)
	org = Image.new('RGB', img_size, color = (255, 255, 255))
	draw = ImageDraw.Draw(org)
	draw.polygon(polygon, fill = color, outline = color)

	# Add noise to the orginal image
	noise = np.random.normal(0, 40, (num_row, num_col, 3))
	background = np.array(org)
	img = background + noise
	img = np.array((img - np.amin(img)) / (np.amax(img) - np.amin(img)) * 255.0, dtype = np.uint8)
	img = np.array(img / 255.0)
	# Image.fromarray(np.array(img * 255.0, dtype = np.uint8)).show()
	# time.sleep(0.25)

	# Draw boundary
	boundary = Image.new('P', resolution, color = 0)
	draw = ImageDraw.Draw(boundary)
	draw.polygon(polygon_s, fill = 0, outline = 255)
	boundary = np.array(boundary) / 255.0
	# Image.fromarray(np.array(boundary * 255.0, dtype = np.uint8)).show()
	# time.sleep(0.25)

	# Draw vertices
	vertices = Image.new('P', resolution, color = 0)
	draw = ImageDraw.Draw(vertices)
	draw.point(polygon_s, fill = 255)
	vertices = np.array(vertices) / 255.0
	# Image.fromarray(np.array(vertices * 255.0, dtype = np.uint8)).show()
	# time.sleep(0.25)

	# Draw each vertex
	vertex_list = []
	for i in range(num_vertices):
		vertex = Image.new('P', resolution, color = 0)
		draw = ImageDraw.Draw(vertex)
		draw.point([polygon_s[i]], fill = 255)
		vertex = np.array(vertex) / 255.0
		vertex_list.append(vertex)
		# Image.fromarray(np.array(vertex * 255.0, dtype = np.uint8)).show()
		# time.sleep(0.25)
	# vertex_list = np.array(vertex_list)

	# Return
	return img, boundary, vertices, vertex_list

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
	lat = math.asin((temp - 1.0) / (temp + 1.0)) / math.pi * 180.0
	return lon, lat

def norm(array):
	ma = np.amax(array)
	mi = np.amin(array)
	if ma == mi:
		return np.zeros(array.shape)
	else:
		return (array - mi) / (ma - mi)

class BoundingBox(object):
	# size: (width, height)
	def __init__(self, lon, lat, zoom, scale, size = (224, 224)):
		self.center_lon = lon
		self.center_lat = lat
		self.size = size
		self.z = zoom + scale - 1
		self.center_px, self.center_py = lonLatToPixel(lon, lat, self.z)
		self.left, self.up = pixelToLonLat(
			self.center_px - math.floor(size[0] / 2), \
			self.center_py - math.floor(size[1] / 2), \
			self.z \
		)
		self.right, self.down = pixelToLonLat(
			self.center_px + math.floor(size[0] / 2), \
			self.center_py + math.floor(size[1] / 2), \
			self.z \
		)

	def lonLatToRelativePixel(self, lon, lat):
		# 0-based
		px, py = lonLatToPixel(lon, lat, self.z)
		return math.floor(px - self.center_px + self.size[0] / 2), math.floor(py - self.center_py + self.size[1] / 2)

class DataGenerator(object):
	# num_col, num_row
	def __init__(self, fake, data_path = None, max_seq_len = None, img_size = (224, 224), resolution = None):
		if fake:
			self.fake = True
			assert(max_seq_len != None)
			self.max_seq_len = max_seq_len
			self.img_size = img_size
			self.resolution = resolution
			self.blank = np.zeros(resolution, dtype = np.float32)
		else:
			# 
			self.fake = False
			assert(max_seq_len != None)
			self.max_seq_len = max_seq_len
			self.img_size = img_size
			self.resolution = resolution
			assert(data_path.endswith('.zip'))

			# 
			self.data_path = data_path.lstrip('./').replace('.zip', '')
			self.archive = zipfile.ZipFile(data_path, 'r')
			self.building_id_set = set()
			for filename in self.archive.namelist():
				if filename.startswith('__MACOSX'):
					continue
				parts = filename.split('/')
				if len(parts) == 3:
					self.building_id_set.add(int(parts[1]))
			print('Totally %d buildings.' % len(self.building_id_set))
			
			# 
			self.good_building_id_set = set()
			self.bad_building_id_set = set()
			for bid in self.building_id_set:
				self.dispatchBuilding(bid)
			print('Totally %d good buildings.' % len(self.good_building_id_set))
			print('Totally %d bad buildings.' % len(self.bad_building_id_set))

			#
			train_prob = 0.95
			self.good_building_id_list = list(self.good_building_id_set)
			self.good_building_id_list.sort()
			random.seed(0)
			random.shuffle(self.good_building_id_list)
			random.seed()
			split = int(train_prob * len(self.good_building_id_list))
			self.id_list_train = self.good_building_id_list[:split]
			self.id_list_valid = self.good_building_id_list[split:]
			self.bad_id_list = list(self.bad_building_id_set)

			# 
			self.blank = np.zeros(resolution, dtype = np.uint8)
			self.vertex_pool = [[] for i in range(resolution[1])]
			for i in range(resolution[1]):
				for j in range(resolution[0]):
					self.vertex_pool[i].append(np.copy(self.blank))
					self.vertex_pool[i][j][i, j] = 255
					self.vertex_pool[i][j] = Image.fromarray(self.vertex_pool[i][j])
			return

	def dispatchBuilding(self, building_id, th = 0.9):
		# Set path
		building_id = str(building_id)
		path = self.data_path + '/' + building_id

		#
		lines = self.archive.read(path + '/shift.txt').decode('utf-8').split('\n')
		edge_prob, _ = lines[1].strip().split()
		edge_prob = float(edge_prob)

		#
		if edge_prob >= th:
			self.good_building_id_set.add(int(building_id))
		else:
			self.bad_building_id_set.add(int(building_id))
		return

	def blur(self, img):
		# img: PIL.Image object
		if BLUR is not None:
			img = img.convert('L').filter(ImageFilter.GaussianBlur(BLUR))
			img = np.array(img, np.float32)
			img = np.minimum(img * (1.2 / np.max(img)), 1.0)
			# Image.fromarray(np.array(img * 255.0, dtype = np.uint8)).show()
		else:
			img = np.array(img, np.float32) / 255.0
		return img

	def showImagePolygon(self, img, polygon, rotate):
		mask = Image.new('RGBA', img.size, color = (255, 255, 255, 0))
		draw = ImageDraw.Draw(mask)
		draw.polygon(polygon, fill = (255, 0, 0, 128), outline = (255, 0, 0, 128))
		merge = Image.alpha_composite(img, mask.rotate(rotate))
		merge.show()
		return

	def distL1(self, p1, p2):
		return math.fabs(p1[0] - p2[0]) + math.fabs(p1[1] - p2[1])

	def getSingleData(self, building_id):
		# Set path
		building_id = str(building_id)
		path = self.data_path + '/' + building_id

		# Rotate
		rotate = random.choice([0, 90, 180, 270])

		# Get image, polygon coordinates and shift
		img = Image.open(io.BytesIO(self.archive.read(path + '/img.png')))
		lines = self.archive.read(path + '/polygon.txt').decode('utf-8').split('\n')
		polygon = []
		for line in lines:
			if line.strip() != '':
				x, y = line.strip().split()
				polygon.append((int(x), int(y)))
		lines = self.archive.read(path + '/shift.txt').decode('utf-8').split('\n')
		shift_i, shift_j = lines[0].strip().split()
		shift_i, shift_j = int(shift_i), int(shift_j)
		polygon = [(x + shift_j, y + shift_i) for x, y in polygon]

		# Get local small patch
		pad_rate = random.random() * 0.1 + 0.1
		min_x, max_x = img.size[0], 0
		min_y, max_y = img.size[1], 0
		for x, y in polygon:
			min_x = min(x, min_x)
			min_y = min(y, min_y)
			max_x = max(x, max_x)
			max_y = max(y, max_y)
		min_x = max(min_x - math.floor(img.size[0] * pad_rate), 0)
		min_y = max(min_y - math.floor(img.size[1] * pad_rate), 0)
		max_x = min(max_x + math.floor(img.size[0] * pad_rate), img.size[0])
		max_y = min(max_y + math.floor(img.size[1] * pad_rate), img.size[1])

		# Adjust image and polygon
		img_patch = img.crop((min_x, min_y, max_x, max_y))
		patch_info = [img_patch.size[0], img_patch.size[1], rotate]
		img_patch = img_patch.resize(self.img_size, resample = Image.BICUBIC).rotate(rotate)
		# img_patch.show()
		# time.sleep(0.25)
		# img_patch_backup = img_patch
		img_patch = np.array(img_patch)[..., 0: 3] / 255.0
		x_rate = self.img_size[0] / (max_x - min_x)
		y_rate = self.img_size[1] / (max_y - min_y)
		res_x = self.resolution[0] / self.img_size[0]
		res_y = self.resolution[1] / self.img_size[1]

		polygon_patch = []
		for x, y in polygon:
			a = math.floor((x - min_x) * x_rate * res_x)
			b = math.floor((y - min_y) * y_rate * res_y)
			if not polygon_patch or self.distL1((a, b), polygon_patch[-1]) > 0:
				polygon_patch.append((a, b))

		start = random.randint(0, len(polygon_patch) - 1)
		polygon_patch = polygon_patch[start:] + polygon_patch[:start]
		# self.showImagePolygon(img_patch_backup, [(x * 4, y * 4) for x, y in polygon_patch], rotate)
		# time.sleep(0.25)

		# Draw boundary and vertices
		boundary = Image.new('P', (self.resolution[0], self.resolution[1]), color = 0)
		draw = ImageDraw.Draw(boundary)
		draw.polygon(polygon_patch, fill = 0, outline = 255)
		boundary = self.blur(boundary.rotate(rotate))
		# time.sleep(0.25)

		vertices = Image.new('P', (self.resolution[0], self.resolution[1]), color = 0)
		draw = ImageDraw.Draw(vertices)
		draw.point(polygon_patch, fill = 255)
		vertices = self.blur(vertices.rotate(rotate))
		# time.sleep(0.25)

		# Get each single vertex
		vertex_input = []
		vertex_output = []
		for i, (x, y) in enumerate(polygon_patch):
			# self.vertex_pool[int(y)][int(x)].rotate(rotate).show()
			# time.sleep(0.25)
			v = self.vertex_pool[int(y)][int(x)].rotate(rotate)
			vertex_input.append(np.array(v, dtype = np.float32) / 255.0)
			if i == 0:
				continue
			# vertex_output.append(self.blur(v))
			vertex_output.append(np.array(v, dtype = np.float32) / 255.0)
		assert(len(vertex_output) == len(vertex_input) - 1)

		# 
		while len(vertex_input) < self.max_seq_len:
			vertex_input.append(np.array(self.blank, dtype = np.float32))
		while len(vertex_output) < self.max_seq_len:
			vertex_output.append(np.array(self.blank, dtype = np.float32))
		vertex_input = np.array(vertex_input)
		vertex_output = np.array(vertex_output)

		# Get end signal
		seq_len = len(polygon_patch)
		end = [0.0 for i in range(self.max_seq_len)]
		end[seq_len - 1] = 1.0
		# seq_len = 6
		# end ? ? ? ? ? ! ? ? ? ?
		# out 1 2 3 4 5 ? ? ? ? ?
		#  in 0 1 2 3 4 5 ? ? ? ?
		end = np.array(end)

		# Return
		return img_patch, boundary, vertices, vertex_input, vertex_output, end, seq_len, patch_info

	def getDataBatch(self, batch_size, mode = None):
		# Fake
		if self.fake:
			return self.getFakeDataBatch(batch_size)

		# Real
		res = []
		if mode == 'train':
			sel = np.random.choice(len(self.id_list_train), batch_size, replace = True)
			for i in sel:
				res.append(self.getSingleData(self.id_list_train[i]))
		if mode == 'valid':
			sel = np.random.choice(len(self.id_list_valid), batch_size, replace = True)
			for i in sel:
				res.append(self.getSingleData(self.id_list_valid[i]))
		if mode == 'test':
			sel = np.random.choice(len(self.bad_id_list), batch_size, replace = True)
			for i in sel:
				res.append(self.getSingleData(self.bad_id_list[i]))
		return (np.array([item[i] for item in res]) for i in range(8))

	def getFakeDataBatch(self, batch_size):
		res = []
		num_v = np.random.choice(6, batch_size, replace = True) + 4
		for seq_len in num_v:
			img, boundary, vertices, vertex_input = plotPolygon(
				img_size = self.img_size,
				resolution = self.resolution,
				num_vertices = seq_len
			)
			while len(vertex_input) < self.max_seq_len:
				vertex_input.append(np.copy(self.blank))
			vertex_output = vertex_input[1:] + [self.blank]
			vertex_input = np.array(vertex_input)
			vertex_output = np.array(vertex_output)
			end = [0.0 for i in range(self.max_seq_len)]
			end[seq_len - 1] = 1.0
			end = np.array(end)
			res.append((img, boundary, vertices, vertex_input, vertex_output, end, seq_len, [224, 224, 0]))
		return (np.array([item[i] for item in res]) for i in range(8))

if __name__ == '__main__':
	for i in range(0):
		plotPolygon(num_vertices = 7, show = True)
	dg = DataGenerator(fake = True, data_path = '../Chicago.zip', max_seq_len = 12, resolution = (28, 28))
	img_patch, boundary, vertices, v_in, v_out, end, seq_len, patch_info = dg.getDataBatch(mode = 'train', batch_size = 1)
	print(np.sum(v_in[0,1] == v_out[0,0]))
	print(np.sum(v_in[0,2] == v_out[0,1]))

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

