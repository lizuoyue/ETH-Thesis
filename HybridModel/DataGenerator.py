import os, sys
if os.path.exists('../../Python-Lib/'):
	sys.path.insert(1, '../../Python-Lib')
import numpy as np
import tensorflow as tf
import math, random
import zipfile, paramiko
import io, glob, time
from PIL import Image, ImageDraw, ImageFilter
from Config import *
from UtilityBoxAnchor import *
if os.path.exists('../../Python-Lib/'):
	sys.path.insert(1, '../../Python-Lib')

config = Config()

class Logger(object):
	def __init__(self, log_dir):
		self.writer = tf.summary.FileWriter(log_dir)
		return

	def log_scalar(self, tag, value, step):
		summary = tf.Summary(value = [tf.Summary.Value(tag = tag, simple_value = value)])
		self.writer.add_summary(summary, step)
		return

	def close(self):
		self.writer.close()
		return

def normalize(li):
	s = sum(li)
	return [item / s for item in li]

def rotateBox(size, box):
	w, h = size
	x1, y1, x2, y2 = box
	return (h, w), (y1, w - x2, y2, w - x1)

def overlay(img, mask):
	"""
		both img and mask PIL.Image, rgb
	"""
	img = img.convert('RGBA')
	mask = np.array(mask, np.uint32)
	alpha = np.sum(np.array(mask, np.int32), axis = 2)
	alpha[alpha > 0] = 200
	alpha = np.expand_dims(alpha, axis = 2)
	alpha = np.concatenate((mask, alpha), axis = 2)
	alpha = Image.fromarray(np.array(alpha, np.uint8), mode = 'RGBA')
	return Image.alpha_composite(img, alpha)

class DataGenerator(object):
	def __init__(self, city_name, img_size, v_out_res, max_num_vertices):
		self.img_size = img_size
		self.v_out_res = v_out_res
		self.max_num_vertices = max_num_vertices

		# 
		self.city_name = city_name
		self.area_path = config.PATH_A % city_name
		self.archive = zipfile.ZipFile(config.PATH_B % city_name, 'r')
		self.building_path = (config.PATH_B % city_name).lstrip('./').replace('.zip', '')
		bids = set()
		for filename in self.archive.namelist():
			if filename.startswith('__MACOSX'):
				continue
			parts = filename.split('/')
			if len(parts) == 3:
				bids.add(int(parts[1]))
		bids = list(bids)
		bids.sort()

		#
		self.building_polygon, li = {}, []
		for bid in bids:
			lines = self.archive.read(self.building_path + '/%d/polygon_after_shift.txt' % bid).decode('utf-8').split('\n')
			self.building_polygon[bid] = [tuple(int(item) for item in line.split()) for line in lines if bool(line.strip())]
			self.building_polygon[bid].reverse()
			lines = self.archive.read(self.building_path + '/%d/shift.txt' % bid).decode('utf-8').split('\n')
			score, _ = lines[1].split()
			li.append((float(score), bid))
		li.sort()

		self.num_v_num_building = {}
		for item in self.building_polygon:
			l = len(self.building_polygon[item])
			if l in self.num_v_num_building:
				self.num_v_num_building[l] += 1
			else:
				self.num_v_num_building[l] = 1
		print(self.num_v_num_building)

		# 
		self.bid_train = [item[1] for item in li[: int(len(li) * config.SPLIT)]]
		self.bid_valid = [item[1] for item in li[int(len(li) * config.SPLIT): ]]
		self.bid_train_p = normalize([len(self.building_polygon[item]) for item in self.bid_train])
		self.bid_valid_p = normalize([len(self.building_polygon[item]) for item in self.bid_valid])
		print('Totally %d buildings for train.' % len(self.bid_train))
		print('Totally %d buildings for valid.' % len(self.bid_valid))

		# 
		self.blank = np.zeros(self.v_out_res, dtype = np.uint8)
		self.vertex_pool = [[] for i in range(self.v_out_res[1])]
		for i in range(self.v_out_res[1]):
			for j in range(self.v_out_res[0]):
				self.vertex_pool[i].append(np.copy(self.blank))
				self.vertex_pool[i][j][i, j] = 255
				self.vertex_pool[i][j] = Image.fromarray(self.vertex_pool[i][j])

		#
		self.anchors = generatePyramidAnchors(config.ANCHOR_SCALE, config.ANCHOR_RATIO, config.FEATURE_SHAPE, config.FEATURE_STRIDE)

		#
		self.ssh = paramiko.SSHClient()
		self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
		self.ssh.connect('cab-e81-28.ethz.ch', username = 'zoli', password = '64206960lzyLZY')
		self.sftp = self.ssh.open_sftp()

		#
		aids = self.sftp.listdir(self.area_path)
		aids.sort()
		random.seed(31415926)
		random.shuffle(aids)
		random.seed()
		self.aid_train = aids[: int(len(aids) * config.SPLIT)]
		self.aid_valid = aids[int(len(aids) * config.SPLIT): ]

		print('Totally %d areas for train.' % len(self.aid_train))
		print('Totally %d areas for valid.' % len(self.aid_valid))
		return

	def blur(self, img):
		"""
			img: PIL.Image object
		"""
		img = img.convert('L').filter(ImageFilter.GaussianBlur(config.BLUR))
		img = np.array(img, np.float32)
		img = np.minimum(img * (1.2 / np.max(img)), 1.0)
		return img

	def distL1(self, p1, p2):
		return math.fabs(p1[0] - p2[0]) + math.fabs(p1[1] - p2[1])

	def getSingleBuilding(self, bid, rotate = True):
		# Rotate
		if rotate:
			rotate = random.choice([0, 90, 180, 270])
		else:
			rotate = 0

		# Get image, polygon coordinates
		while True:
			try:
				img = Image.open(io.BytesIO(self.sftp.open(self.area_path.replace('Areas', 'Buildings') + '/%s/img.png' % bid).read()))
				break
			except:
				print('Try again.')
				self.ssh = paramiko.SSHClient()
				self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
				self.ssh.connect('cab-e81-28.ethz.ch', username = 'zoli', password = '64206960lzyLZY')
				self.sftp = self.ssh.open_sftp()
		# img = Image.open(io.BytesIO(self.archive.read(self.building_path + '/%d/img.png' % bid)))
		polygon = self.building_polygon[bid]

		# Adjust image and polygon
		org_info = [img.size[0], img.size[1], rotate]
		x_rate = self.v_out_res[0] / img.size[0]
		y_rate = self.v_out_res[1] / img.size[1]
		img = img.resize(self.img_size, resample = Image.BICUBIC).rotate(rotate)
		img = np.array(img, np.float32)[..., 0: 3] - config.COLOR_MEAN['Buildings'][self.city_name]
		polygon_s = []
		for x, y in polygon:
			a, b = math.floor(x * x_rate), math.floor(y * y_rate)
			if not polygon_s or self.distL1((a, b), polygon_s[-1]) > 0:
				polygon_s.append((a, b))
		start = random.randint(0, len(polygon_s) - 1)
		polygon_s = polygon_s[start: ] + polygon_s[: start]

		# Draw boundary and vertices
		boundary = Image.new('P', self.v_out_res, color = 0)
		draw = ImageDraw.Draw(boundary)
		draw.polygon(polygon_s, fill = 0, outline = 255)
		boundary = self.blur(boundary.rotate(rotate))

		vertices = Image.new('P', self.v_out_res, color = 0)
		draw = ImageDraw.Draw(vertices)
		draw.point(polygon_s, fill = 255)
		vertices = self.blur(vertices.rotate(rotate))

		# Get each single vertex
		vertex_input, vertex_output = [], []
		for i, (x, y) in enumerate(polygon_s):
			v = self.vertex_pool[int(y)][int(x)].rotate(rotate)
			vertex_input.append(np.array(v, dtype = np.float32) / 255.0)
			if i == 0:
				continue
			vertex_output.append(np.array(v, dtype = np.float32) / 255.0)
		assert(len(vertex_output) == len(vertex_input) - 1)

		# 
		while len(vertex_input) < self.max_num_vertices:
			vertex_input.append(np.array(self.blank, dtype = np.float32))
		while len(vertex_output) < self.max_num_vertices:
			vertex_output.append(np.array(self.blank, dtype = np.float32))
		vertex_input = np.array(vertex_input)
		vertex_output = np.array(vertex_output)

		# Get end signal
		seq_len = len(polygon_s)
		end = [0.0 for i in range(self.max_num_vertices)]
		end[seq_len - 1] = 1.0
		end = np.array(end)

		# Example:
		# seq_len = 6
		# end: ? ? ? ? ? ! X X
		# out: 1 2 3 4 5 ? X X
		#  in: 0 1 2 3 4 5 X X

		# Return
		return img, boundary, vertices, vertex_input, vertex_output, end, seq_len, org_info

	def getSingleBuildingRotate(self, bid):
		theta = random.randint(0, 359)

		# Get image, polygon coordinates
		img = Image.open(io.BytesIO(self.archive.read(self.building_path + '/%d/img.png' % bid)))
		img_rot = img.rotate(theta, resample = Image.BICUBIC, expand = True)
		img_res = img_rot.resize(self.img_size, resample = Image.BICUBIC)

		# Adjust image and polygon
		org_info = [img.size[0], img.size[1], theta]
		polygon = self.building_polygon[bid]
		polygon_rot = [(
			 np.cos(np.deg2rad(theta)) * (x - img.size[0] / 2) + np.sin(np.deg2rad(theta)) * (y - img.size[1] / 2) + img_rot.size[0] / 2,
			-np.sin(np.deg2rad(theta)) * (x - img.size[0] / 2) + np.cos(np.deg2rad(theta)) * (y - img.size[1] / 2) + img_rot.size[1] / 2
		) for x, y in polygon]
		x_rate = self.v_out_res[0] / img_rot.size[0]
		y_rate = self.v_out_res[1] / img_rot.size[1]
		img = np.array(img, np.float32)[..., 0: 3] - config.COLOR_MEAN['Buildings'][self.city_name]
		polygon_s = []
		for x, y in polygon_rot:
			a, b = math.floor(x * x_rate), math.floor(y * y_rate)
			if not polygon_s or self.distL1((a, b), polygon_s[-1]) > 0:
				polygon_s.append((a, b))
		start = random.randint(0, len(polygon_s) - 1)
		polygon_s = polygon_s[start: ] + polygon_s[: start]

		# Draw boundary and vertices
		boundary = Image.new('P', self.v_out_res, color = 0)
		draw = ImageDraw.Draw(boundary)
		draw.polygon(polygon_s, fill = 0, outline = 255)
		boundary = self.blur(boundary)

		vertices = Image.new('P', self.v_out_res, color = 0)
		draw = ImageDraw.Draw(vertices)
		draw.point(polygon_s, fill = 255)
		vertices = self.blur(vertices)

		# Get each single vertex
		vertex_input, vertex_output = [], []
		for i, (x, y) in enumerate(polygon_s):
			v = self.vertex_pool[int(y)][int(x)]
			vertex_input.append(np.array(v, dtype = np.float32) / 255.0)
			if i == 0:
				continue
			vertex_output.append(np.array(v, dtype = np.float32) / 255.0)
		assert(len(vertex_output) == len(vertex_input) - 1)

		# 
		while len(vertex_input) < self.max_num_vertices:
			vertex_input.append(np.array(self.blank, dtype = np.float32))
		while len(vertex_output) < self.max_num_vertices:
			vertex_output.append(np.array(self.blank, dtype = np.float32))
		vertex_input = np.array(vertex_input)
		vertex_output = np.array(vertex_output)

		# Get end signal
		seq_len = len(polygon_s)
		end = [0.0 for i in range(self.max_num_vertices)]
		end[seq_len - 1] = 1.0
		end = np.array(end)

		# Example:
		# seq_len = 6
		# end: ? ? ? ? ? ! X X
		# out: 1 2 3 4 5 ? X X
		#  in: 0 1 2 3 4 5 X X

		# Return
		return img, boundary, vertices, vertex_input, vertex_output, end, seq_len, org_info

	def getSingleArea(self, area_idx, rotate = True):
		# Rotate, anticlockwise
		if rotate:
			n_rotate = random.choice([0, 1, 2, 3])
		else:
			n_rotate = 0

		# 
		while True:
			try:
				org = Image.open(io.BytesIO(self.sftp.open(self.area_path + '/%s/img.png' % area_idx).read()))
				lines = self.sftp.open(self.area_path + '/%s/polygons_after_shift.txt' % area_idx).read().decode('utf-8').split('\n')
				break
			except:
				print('Try again.')
				self.ssh = paramiko.SSHClient()
				self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
				self.ssh.connect('cab-e81-28.ethz.ch', username = 'zoli', password = '64206960lzyLZY')
				self.sftp = self.ssh.open_sftp()

		org_rot = org.rotate(n_rotate * 90)
		self.area_imgs.append(org_rot)
		org_resize = org_rot.resize(config.AREA_SIZE)

		polygons = []
		for line in lines:
			if line.strip() != '':
				if line.startswith('%'):
					polygons.append([])
				else:
					x, y = line.strip().split()
					polygons[-1].append((int(x), int(y)))

		gt_boxes = []
		pad = 0.1
		for polygon in polygons:
			w, h = org.size
			p = np.array(polygon, np.int32)
			l = max(0, p[:, 0].min())
			u = max(0, p[:, 1].min())
			r = min(w, p[:, 0].max())
			d = min(h, p[:, 1].max())
			if r > l and d > u:
				for _ in range(n_rotate):
					(w, h), (l, u, r, d) = rotateBox((w, h), (l, u, r, d))
				bw, bh = r - l, d - u
				l = max(0, l - bw * pad)
				u = max(0, u - bh * pad)
				r = min(w, r + bw * pad)
				d = min(h, d + bh * pad)
				gt_boxes.append([u, l, d, r])

		if False: # <- Local test
			draw = ImageDraw.Draw(org_rot)
			for u, l, d, r in gt_boxes:
				draw.line([(l, u), (r, u), (r, d), (l, d), (l, u)], fill = (255, 0, 0, 255), width = 3)
			org_rot.show()

		if len(gt_boxes) == 0:
			gt_boxes = np.zeros((0, 4), np.int32)
		else:
			gt_boxes = np.array(gt_boxes)

		# 
		self.recover_rate = org_rot.size[0] / config.AREA_SIZE[0]
		anchor_cls = np.zeros([self.anchors.shape[0], 2], np.int32)
		rpn_match, anchor_box = buildRPNTargets(self.anchors * self.recover_rate, gt_boxes)
		anchor_cls[rpn_match == 1, 0] = 1
		anchor_cls[rpn_match == -1, 1] = 1

		#
		ret_img = np.array(org_resize, np.float32)[..., 0: 3] - config.COLOR_MEAN['Areas'][self.city_name]
		return ret_img, anchor_cls, anchor_box

	def getBuildingsBatch(self, batch_size, mode = None, idx = None):
		# Real
		res = []
		if mode == 'train':
			sel = np.random.choice(self.bid_train, batch_size, replace = True, p = self.bid_train_p)
			for bid in sel:
				res.append(self.getSingleBuilding(bid))
		if mode == 'valid':
			sel = np.random.choice(self.bid_valid, batch_size, replace = True, p = self.bid_valid_p)
			for bid in sel:
				res.append(self.getSingleBuilding(bid, rotate = False))
		if mode == 'test':
			sel = self.bid_valid[batch_size * idx: batch_size * (idx + 1)]
			for bid in sel:
				res.append(self.getSingleBuilding(bid, rotate = False))
		return [np.array([item[i] for item in res]) for i in range(8)]

	def getAreasBatch(self, batch_size, mode = None, idx = None):
		# Real
		res = []
		self.area_imgs = []
		if mode == 'train':
			sel = np.random.choice(self.aid_train, batch_size, replace = True)
			for aid in sel:
				res.append(self.getSingleArea(aid))
		if mode == 'valid':
			sel = np.random.choice(self.aid_valid, batch_size, replace = True)
			for aid in sel:
				res.append(self.getSingleArea(aid, rotate = False))
		if mode == 'test':
			sel = self.aid_valid[batch_size * idx: batch_size * (idx + 1)]
			for aid in sel:
				res.append(self.getSingleArea(aid, rotate = False))
		return [np.array([item[i] for item in res]) for i in range(3)]

	def getPatchesFromAreas(self, res):
		assert(len(res) == len(self.area_imgs))
		patches = []
		org_info = []
		for i, (org, bbox) in enumerate(zip(self.area_imgs, res)):
			img = np.array(org)
			boxes = bbox * self.recover_rate
			for j in range(boxes.shape[0]):
				y1, x1, y2, x2 = tuple(list(boxes[j]))
				h, w = y2 - y1, x2 - x1
				if h * w > 16 * 16 and y1 >= 0 and x1 >= 0 and y2 < img.shape[0] and x2 < img.shape[1]:
					# y1, x1, y2, x2 = int(max(0, y1 - h * self.pad)), int(max(0, x1 - w * self.pad)), int(min(640, y2 + h * self.pad)), int(min(640, x2 + w * self.pad))
					h, w = int(max(w, h) * 1.3), int(max(w, h) * 1.3)
					cx, cy = (x1+x2)/2, (y1+y2)/2
					y1, x1, y2, x2 = int(max(0, cy-h/2)), int(max(0, cx-w/2)), int(min(img.shape[0], cy+h/2)), int(min(img.shape[1], cx+w/2))
					if y1 < y2 and x1 < x2:
						patch = np.array(Image.fromarray(img[y1: y2, x1: x2, 0: 3]).resize(config.PATCH_SIZE, resample = Image.BICUBIC), np.float32)
						patches.append(patch - config.COLOR_MEAN['Buildings'][self.city_name])
						org_info.append([i, y1, x1, y2, x2])
		return self.area_imgs, np.array(patches), org_info		

	def recoverGlobal(self, path, img, org_info, pred_v_out, pred_box, base):
		for i, im in enumerate(img):
			boxes = pred_box[i] * self.recover_rate
			draw = ImageDraw.Draw(im)
			for j in range(boxes.shape[0]):
				y1, x1, y2, x2 = tuple(list(boxes[j]))
				h, w = y2 - y1, x2 - x1
				if h * w > 16 * 16 and y1 >= 0 and x1 >= 0 and y2 < im.size[1] and x2 < im.size[0]:
					draw.polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], outline = (0, 255, 0))
		batch_size = len(org_info)
		assert(len(org_info) == pred_v_out.shape[1])
		color_count = 0
		len_c = len(config.TABLEAU20)
		for kk in range(pred_v_out.shape[0]):
			masks = [[] for i in range(batch_size)]
			for i in range(pred_v_out.shape[1]):
				idx, y1, x1, y2, x2 = org_info[i]
				w, h = x2 - x1, y2 - y1
				polygon = []
				mask = Image.fromarray(np.zeros((img[idx].size[1], img[idx].size[0], 3), np.uint8))
				draw = ImageDraw.Draw(mask)
				flag = False
				for j in range(pred_v_out.shape[2]):
					v = pred_v_out[0, i, j]
					if v.sum() >= 0.5:
						r, c = np.unravel_index(v.argmax(), v.shape)
						polygon.append((c/config.V_OUT_RES[0]*w+x1, r/config.V_OUT_RES[1]*h+y1))
					else:
						flag = True
						break
				if flag:
					draw.polygon(polygon, fill = config.TABLEAU20[color_count % len_c], outline = config.TABLEAU20_DEEP[color_count % len_c])
					color_count += 1
					masks[idx].append(mask)
			break
		for i, (im, mask) in enumerate(zip(img, masks)):
			a = im.copy()
			for msk in mask:
				a = overlay(a, msk)
			a.save(path + '/%d_%d.png' % (base, i))

if __name__ == '__main__':
	dg = DataGenerator(
		city_name = sys.argv[1],
		img_size = config.PATCH_SIZE,
		v_out_res = config.V_OUT_RES,
		max_num_vertices = config.MAX_NUM_VERTICES,
	)
	quit()
	item1 = dg.getAreasBatch(8, mode = 'valid')
	item2 = dg.getBuildingsBatch(12, mode = 'train')
	for item in item1:
		print(item.shape)
	for item in item2:
		print(item.shape)
	for k in range(12):
		for i, item in enumerate(list(item2)):
			if i < 3:
				Image.fromarray(np.array(item[k, ...],np.uint8)).resize((256,256)).show()
				time.sleep(0.5)
			elif i < 5:
				for j in range(20):
					Image.fromarray(np.array(item[k, j, ...],np.uint8)).resize((256,256)).show()
					print(j)
					time.sleep(0.5)
			else:
				print(item[k])
		input()
	a, b, c = dg.getPatchesFromAreas(
		[np.array([[100, 100, 200, 200]]),np.array([[100, 100, 200, 200]]),np.array([[100, 100, 200, 200]]),np.array([[100, 100, 200, 200]])]
	)
	print(len(a))
	print(b.shape)
	print(len(c))


