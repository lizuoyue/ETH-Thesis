import numpy as np
import os, sys, glob
import math, random
from PIL import Image, ImageDraw, ImageFilter
# PIL.ImageDraw: 0-based, (col_idx, row_idx) if taking image as matrix

BLUR = True
BLUR_R = 0.75

def pil2np(image, show):
	if show:
		import matplotlib.pyplot as plt
	img = np.array(image, dtype = np.float32) / 255.0
	if len(img.shape) > 2 and img.shape[2] == 4:
		img = img[..., 0: 3]
	if show:
		plt.imshow(img)
		plt.show()
	return img

def plotPolygon(img_size = (224, 224), num_vertices = 6, show = False):

	# Set image parameters
	num_row = img_size[0]
	num_col = img_size[1]
	half_x = math.floor(num_col / 2)
	half_y = math.floor(num_row / 2)
	img_size_s = (math.floor(num_row / 8), math.floor(num_col / 8))

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
		polygon_s.append((math.floor(px / 8), math.floor(py / 8)))
		angle += delta_angle * np.random.uniform(1 - epsilon, 1 + epsilon) # <- Decide polygon's vertices
	first_idx = random.choice([i for i in range(num_vertices)])
	polygon = polygon[first_idx:] + polygon[:first_idx]
	polygon_s = polygon_s[first_idx:] + polygon_s[:first_idx]

	# Draw polygon
	color = (255, 0, 0)
	org = Image.new('RGB', img_size, color = (255, 255, 255))
	draw = ImageDraw.Draw(org)
	draw.polygon(polygon, fill = color, outline = color)

	# Add noise to the orginal image
	noise = np.random.normal(0, 40, (num_row, num_col, 3))
	background = np.array(org)
	img = background + noise
	img = np.array((img - np.amin(img)) / (np.amax(img) - np.amin(img)) * 255.0, dtype = np.uint8)
	img = Image.fromarray(img)
	img = pil2np(img, show)

	# Draw boundary
	boundary = Image.new('P', img_size_s, color = 0)
	draw = ImageDraw.Draw(boundary)
	draw.polygon(polygon_s, fill = 0, outline = 255)
	boundary = pil2np(boundary, show)

	# Draw vertices
	vertices = Image.new('P', img_size_s, color = 0)
	draw = ImageDraw.Draw(vertices)
	draw.point(polygon_s, fill = 255)
	vertices = pil2np(vertices, show)

	# Draw each vertex
	vertex_list = []
	for i in range(num_vertices):
		vertex = Image.new('P', img_size_s, color = 0)
		draw = ImageDraw.Draw(vertex)
		draw.point([polygon_s[i]], fill = 255)
		vertex = pil2np(vertex, show)
		vertex_list.append(vertex)
	vertex_list.append(np.zeros(img_size_s, dtype = np.float32))
	# vertex_list = np.array(vertex_list)

	# Return
	if show:
		print(img.shape)
		print(boundary.shape)
		print(vertices.shape)
		print(vertex_list.shape)
	return img, boundary, vertices, vertex_list

TILE_SIZE = 256

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
	# 
	def __init__(self, fake, data_path = None, train_prob = None, max_seq_len = None):
		if fake:
			self.fake = True
			assert(max_seq_len != None)
			self.max_seq_len = max_seq_len
		else:
			self.fake = False
			assert(data_path != None)
			assert(train_prob != None)
			assert(max_seq_len != None)
			self.train_prob = train_prob
			self.max_seq_len = max_seq_len
			if data_path.endswith('.tar.gz'):
				self.data_file_type = 'tar'
			else:
				self.data_file_type = 'dir'
			if self.data_file_type == 'dir':
				self.data_path = data_path
				self.id_list = os.listdir(data_path)
				if '.DS_Store' in self.id_list:
					self.id_list.remove('.DS_Store')
			if self.data_file_type == 'tar':
				self.data_path = data_path.replace('.tar.gz', '')[1:]
				self.tar = tarfile.open(data_path, 'r|gz')
				self.building_list = {}
				for filename in self.tar.getnames():
					parts = filename.split('/')
					if len(parts) == 4:
						bid = int(parts[2])
						if bid in self.building_list:
							self.building_list[bid].append(filename)
						else:
							self.building_list[bid] = [filename]
				self.id_list = [k for k in self.building_list]
			print('Totally %d buildings.' % len(self.id_list))

			# Split
			random.seed(31415927)
			random.shuffle(self.id_list)
			random.seed(random.randint(0, 31415926))
			split = int(len(self.id_list) * self.train_prob)
			self.id_list_train = self.id_list[:split]
			self.id_list_valid = self.id_list[split:]

			self.blank = np.zeros((28, 28), dtype = np.float32)
			self.vertex_pool = [[] for i in range(28)]
			for i in range(28):
				for j in range(28):
					self.vertex_pool[i].append(np.zeros((28, 28), dtype = np.float32))
					self.vertex_pool[i][j][i, j] = 1.0
			return

	def blur(self, img):
		if BLUR:
			img = img.convert('L').filter(ImageFilter.GaussianBlur(BLUR_R))
			img = np.array(img, np.float32)
			img = np.minimum(img * (1.2 / np.max(img)), 1.0)
		else:
			img = np.array(img, np.float32) / 255.0
		return img

	def getDataSingle(self, building_id):
		# Set path
		if type(building_id) == int:
			building_id = str(building_id)
		path = self.data_path + '/' + building_id

		# Get images
		if self.data_file_type == 'tar':
			f = self.tar.extractfile(path + '/0-img.png')
			img = np.array(Image.open(io.BytesIO(f.read())))[..., 0: 3] / 255.0
			f = self.tar.extractfile(path + '/3-b.png')
			boundary = self.blur(Image.open(io.BytesIO(f.read())))
			f = self.tar.extractfile(path + '/4-v.png')
			vertices = self.blur(Image.open(io.BytesIO(f.read())))
			f = self.tar.extractfile(path + '/5-v.txt')
			lines = f.readlines()
			lines = [line.decode('utf-8') for line in lines]
		if self.data_file_type == 'dir':
			img = np.array(Image.open(glob.glob(path + '/' + '0-img.png')[0]))[..., 0: 3] / 255.0
			boundary = self.blur(Image.open(glob.glob(path + '/' + '3-b.png')[0]))
			vertices = self.blur(Image.open(glob.glob(path + '/' + '4-v.png')[0]))
			f = open(path + '/' + '5-v.txt', 'r')
			lines = f.readlines()
		vertex = []
		for line in lines:
			y, x = line.strip().split()
			vertex.append(self.vertex_pool[int(x)][int(y)])
		seq_len = len(vertex)

		# 
		while len(vertex) < self.max_seq_len:
			vertex.append(self.blank)
		vertex = np.array(vertex)
		end = [0.0 for i in range(self.max_seq_len)]
		end[seq_len] = 1.0
		end = np.array(end)

		# Return
		return img, boundary, vertices, vertex, end, seq_len

	def getDataBatch(self, batch_size, mode = None):
		if self.fake:
			return self.getToyDataBatch(batch_size)
		res = []
		if mode == 'train':
			batch_size = min(len(self.id_list_train), batch_size)
			sel = np.random.choice(len(self.id_list_train), batch_size, replace = False)
			for i in sel:
				res.append(self.getDataSingle(self.id_list_train[i]))
		if mode == 'valid':
			batch_size = min(len(self.id_list_valid), batch_size)
			sel = np.random.choice(len(self.id_list_valid), batch_size, replace = False)
			for i in sel:
				res.append(self.getDataSingle(self.id_list_valid[i]))
		return (np.array([item[i] for item in res]) for i in range(6))

	def getToyDataBatch(self, batch_size):
		res = []
		num_v = np.random.choice(6, batch_size, replace = True) + 4
		for n in num_v:
			img, b, v, vertex_list = plotPolygon(num_vertices = n)
			while len(vertex_list) < self.max_seq_len:
				vertex_list.append(np.zeros((28, 28), dtype = np.float32))
			vertex_list = np.array(vertex_list)
			end = [0.0 for i in range(self.max_seq_len)]
			end[n] = 1.0
			end = np.array(end)
			res.append((img, b, v, vertex_list, end, n))
		return (np.array([item[i] for item in res]) for i in range(6))

if __name__ == '__main__':
	for i in range(1):
		plotPolygon(num_vertices = 7, show = True)



