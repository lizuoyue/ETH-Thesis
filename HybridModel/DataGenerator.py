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

def rotateBox(size, box):
	w, h = size
	x1, y1, x2, y2 = box
	return (h, w), (y1, w - x2, y2, w - x1)

class DataGenerator(object):
	def __init__(self, building_path, area_path, img_size, v_out_res, max_num_vertices):
		self.img_size = img_size
		self.v_out_res = v_out_res
		self.max_num_vertices = max_num_vertices

		self.shift_info = {}

		# 
		self.area_path = area_path
		self.archive = zipfile.ZipFile(building_path, 'r')
		self.building_path = building_path.lstrip('./').replace('.zip', '')
		bids = set()
		for filename in self.archive.namelist():
			if filename.startswith('__MACOSX'):
				continue
			parts = filename.split('/')
			if len(parts) == 3:
				bids.add(int(parts[1]))
		bids = list(bids)
		bids.sort()
		print('Totally %d buildings.' % len(bids))

		#
		self.building_polygon, li = {}, []
		for bid in bids:
			#
			lines = self.archive.read(self.building_path + '/%d/shift.txt' % bid).decode('utf-8').split('\n')
			alpha, shift_i, shift_j = lines[0].strip().split()
			self.shift_info[bid] = (float(alpha), int(shift_i), int(shift_j))
			score, edge_score = lines[1].strip().split()
			li.append((score, bid))
			lines = self.archive.read(self.building_path + '/%d/polygon.txt' % bid).decode('utf-8').split('\n')
			polygon = []
			for line in lines:
				if line.strip() != '':
					x, y = line.strip().split()
					polygon.append((int(x), int(y)))
			self.building_polygon[bid] = applyAlphaShiftToPolygon(self.shift_info[bid], polygon)
		li.sort()
		split = int(len(li) * 0.75)

		# 
		self.good_bids = [item[1] for item in li[: split]]
		self.bids_test = [item[1] for item in li[split: ]]
		print('Totally %d good buildings.' % len(self.good_bids))
		print('Totally %d bad buildings.' % len(self.bids_test))

		#
		self.good_bids.sort()
		random.seed(31415926)
		random.shuffle(self.good_bids)
		random.seed()
		self.bid_train = self.good_bids[: split]
		self.bid_valid = self.good_bids[split: ]

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
		aids = self.sftp.listdir(area_path)
		split = int(len(aids) * 0.8)
		print('Totally %d areas.' % len(aids))
		aids.sort()
		random.seed(31415927)
		random.shuffle(aids)
		random.seed()
		self.aid_train = aids[: split]
		self.aid_valid = aids[split: ]
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

	def getSingleBuilding(self, bid):
		# Rotate
		rotate = random.choice([0, 90, 180, 270])

		# Get image, polygon coordinates and shift
		img = Image.open(io.BytesIO(self.archive.read(self.building_path + '/%d/img.png' % bid)))
		polygon = self.building_polygon[bid]

		# Adjust image and polygon
		org_info = (img.size, rotate)
		x_rate = self.v_out_res[0] / img.size[0]
		y_rate = self.v_out_res[1] / img.size[1]
		img = img.resize(self.img_size, resample = Image.BICUBIC).rotate(rotate)
		img = np.array(img)[..., 0: 3] / 255.0
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
		return img, boundary, vertices, vertex_input, vertex_output, end, seq_len

	def getSingleArea(self, area_idx):
		"""
		# with open(self.building_list[building_idx] + 'shift.txt', 'w') as f:
		# 	f.write('%.2lf %d %d\n' % (alphas[idx], shift_i, shift_j))
		# 	f.write('%lf %lf\n' % (score, edge_score))
		"""

		# Rotate, anticlockwise
		n_rotate = 0#random.choice([0, 1, 2, 3])

		# 
		while True:
			try:
				img = Image.open(io.BytesIO(self.sftp.open(self.area_path + '/%s/img.png' % area_idx).read()))
				self.area_imgs.append(img)
				lines = self.sftp.open(self.area_path + '/%s/polygons.txt' % area_idx).read().decode('utf-8').split('\n')
				break
			except:
				print('Try again.')

		img = img.rotate(n_rotate * 90)
		num_anchors = self.anchors.shape[0]
		org = np.array(img)[..., 0: 3] / 255.0
		img = img.resize(config.AREA_SIZE)

		polygons = []
		for line in lines:
			if line.strip() != '':
				if line.startswith('%'):
					_, bid = line.strip().split()
					polygons.append([int(bid)])
				else:
					x, y = line.strip().split()
					polygons[-1].append((int(x), int(y)))
		for polygon in polygons:
			if polygon[0] not in self.shift_info:
				self.shift_info[polygon[0]] = (1, 0, 0)
		polygons = [applyAlphaShiftToPolygon(self.shift_info[polygon[0]], polygon[1: ]) for polygon in polygons]

		gt_boxes = []
		pad = 0.1
		for polygon in polygons:
			h, w = org.shape[0], org.shape[1]
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
		if len(gt_boxes) == 0:
			gt_boxes = np.zeros((0, 4), np.int32)
		else:
			gt_boxes = np.array(gt_boxes)

		# 
		self.recover_rate = org.shape[0] / config.AREA_SIZE[0]
		anchor_cls = np.zeros([num_anchors, 2], np.int32)
		rpn_match, anchor_box = buildRPNTargets(self.anchors * self.recover_rate, gt_boxes)
		anchor_cls[rpn_match == 1, 0] = 1
		anchor_cls[rpn_match == -1, 1] = 1

		#
		return np.array(img)[..., 0: 3] / 255.0, anchor_cls, anchor_box

	def getBuildingsBatch(self, batch_size, mode = None):
		# Real
		res = []
		if mode == 'train':
			sel = np.random.choice(len(self.bid_train), batch_size, replace = True)
			for i in sel:
				res.append(self.getSingleBuilding(self.bid_train[i]))
		if mode == 'valid':
			sel = np.random.choice(len(self.bid_valid), batch_size, replace = True)
			for i in sel:
				res.append(self.getSingleBuilding(self.bid_valid[i]))
		if mode == 'test':
			sel = np.random.choice(len(self.bid_test), batch_size, replace = True)
			for i in sel:
				res.append(self.getSingleBuilding(self.bid_test[i]))
		return (np.array([item[i] for item in res]) for i in range(7))

	def getAreasBatch(self, batch_size, mode = None):
		# Real
		res = []
		self.area_imgs = []
		if mode == 'train':
			sel = np.random.choice(len(self.aid_train), batch_size, replace = True)
			for i in sel:
				res.append(self.getSingleArea(self.aid_train[i]))
		if mode == 'valid':
			sel = np.random.choice(len(self.aid_valid), batch_size, replace = True)
			for i in sel:
				res.append(self.getSingleArea(self.aid_valid[i]))
		return (np.array([item[i] for item in res]) for i in range(3))

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
				if h * w > 24 * 24 and y1 >= 0 and x1 >= 0 and y2 < img.shape[0] and x2 < img.shape[1]:
					# y1, x1, y2, x2 = int(max(0, y1 - h * self.pad)), int(max(0, x1 - w * self.pad)), int(min(640, y2 + h * self.pad)), int(min(640, x2 + w * self.pad))
					h, w = max(w, h) + 30, max(w, h) + 30
					cx, cy = (x1+x2)/2, (y1+y2)/2
					y1, x1, y2, x2 = int(max(0, cy-h/2)), int(max(0, cx-w/2)), int(min(img.shape[0], cy+h/2)), int(min(img.shape[1], cx+w/2))
					if y1 < y2 and x1 < x2:
						patches.append(np.array(Image.fromarray(img[y1: y2, x1: x2, 0: 3]).resize(config.PATCH_SIZE, resample = Image.BICUBIC))/255.0)
						org_info.append([i, y1, x1, y2, x2])
		return self.area_imgs, np.array(patches), org_info

	def recover(self, path, imgs, res, base):
		for i, img in enumerate(imgs):
			a = img.copy()
			boxes = res[i] * self.recover_rate
			draw = ImageDraw.Draw(a)
			for j in range(boxes.shape[0]):
				y1, x1, y2, x2 = tuple(list(boxes[j]))
				h, w = y2 - y1, x2 - x1
				if h * w > 24 * 24 and y1 >= 0 and x1 >= 0 and y2 < a.size[1] and x2 < a.size[0]:
					draw.polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], outline = (255, 0, 0))
			a.save(path + '/box_%d_%d.png' % (base % 100, i))

	def recoverGlobal(self, path, img, org_info, pred_v_out, base):
		batch_size = len(org_info)
		assert(len(org_info) == pred_v_out.shape[1])
		for kk in range(pred_v_out.shape[0]):
			polygons = [[] for i in range(batch_size)]
			for i in range(pred_v_out.shape[1]):
				idx, y1, x1, y2, x2 = org_info[i]
				w, h = x2 - x1, y2 - y1
				draw = ImageDraw.Draw(img[idx])
				for j in range(pred_v_out.shape[2]):
					v = pred_v_out[kk, i, j]
					if v.sum() >= 0.5:
						r, c = np.unravel_index(v.argmax(), v.shape)
						polygons[i].append((c/config.V_OUT_RES[0]*w+x1, r/config.V_OUT_RES[1]*h+y1))
					else:
						polygons[i].append(polygons[i][0])
						break
				draw.line(polygons[i], fill = config.TABLEAU20[kk], width = 2)
			break
		for i, im in enumerate(img):
			im.save(path + '/%d_%d.png' % (base % 100, i))

if __name__ == '__main__':
	city = 'Zurich'
	dg = DataGenerator(
		building_path = '../../Buildings%s.zip' % city,
		area_path = '/local/lizuoyue/Areas%s' % city,
		img_size = config.PATCH_SIZE,
		v_out_res = config.V_OUT_RES,
		max_num_vertices = config.MAX_NUM_VERTICES,
	)
	item1 = dg.getAreasBatch(4, mode = 'train')
	item2 = dg.getBuildingsBatch(12, mode = 'train')
	for item in item1:
		print(item.shape)
	for item in item2:
		print(item.shape)
	for k in range(12):
		for i, item in enumerate(item2):
			if i < 3:
				Image.fromarray(np.array(item[k, ...]*255,np.uint8)).resize((256,256)).show()
				time.sleep(0.5)
			elif i < 5:
				for j in range(20):
					Image.fromarray(np.array(item[k, j, ...]*255,np.uint8)).resize((256,256)).show()
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


