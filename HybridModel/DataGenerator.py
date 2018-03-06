import numpy as np
import math, random
import zipfile, paramiko
import os, io, sys, glob, time
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


class DataGenerator(object):
	def __init__(self, building_path, area_path, img_size, v_out_res, max_num_vertices):
		self.img_size = img_size
		self.v_out_res = v_out_res
		self.max_num_vertices = max_num_vertices

		# 
		self.area_path = area_path
		self.archive = zipfile.ZipFile(building_path, 'r')
		self.building_path = building_path.lstrip('./').replace('.zip', '')
		bids = []
		for filename in self.archive.namelist():
			if filename.startswith('__MACOSX'):
				continue
			parts = filename.split('/')
			if len(parts) == 3:
				bids.append(int(parts[1]))
		print('Totally %d buildings.' % len(bids))

		#
		li = []
		for bid in bids:
			lines = self.archive.read(self.building_path + '/%d/shift.txt' % bid).decode('utf-8').split('\n')
			score, _ = lines[1].strip().split()
			li.append((score, bid))
		li.sort()
		split = int(len(li) * 0.9)
		
		# 
		self.good_bids = [item[1] for item in li[: split]]
		self.bids_test = [item[1] for item in li[split: ]]
		print('Totally %d good buildings.' % len(self.good_bids))
		print('Totally %d bad buildings.' % len(self.bad_bids))

		#
		self.good_bids.sort()
		random.seed(31415926)
		random.shuffle(self.good_building_id_list)
		random.seed()
		self.bid_train = self.good_bids[: split]
		self.bid_valid = self.good_bids[split: ]

		# 
		self.blank = np.zeros(resolution, dtype = np.uint8)
		self.vertex_pool = [[] for i in range(resolution[1])]
		for i in range(resolution[1]):
			for j in range(resolution[0]):
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

		# with open(self.building_list[building_idx] + 'shift.txt', 'w') as f:
		# 	f.write('%.2lf %d %d\n' % (alphas[idx], shift_i, shift_j))
		# 	f.write('%lf %lf\n' % (score, edge_score))

		aids = self.sftp.listdir(area_path)
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
		lines = self.archive.read(self.building_path + '/%d/polygon.txt' % bid).decode('utf-8').split('\n')
		polygon = []
		for line in lines:
			if line.strip() != '':
				x, y = line.strip().split()
				polygon.append((int(x), int(y)))
		lines = self.archive.read(self.building_path + '/%d/shift.txt' % bid).decode('utf-8').split('\n')
		alpha, shift_i, shift_j = lines[0].strip().split()
		alpha, shift_i, shift_j = float(alpha), int(shift_i), int(shift_j)
		if alpha > 1.01:
			polygon_i = np.array([v[1] for v in polygon], np.float32)
			polygon_j = np.array([v[0] for v in polygon], np.float32)
			c_i = (polygon_i.min() + polygon_i.max()) / 2
			c_j = (polygon_j.min() + polygon_j.max()) / 2
			polygon_i = polygon_i * alpha + c_i * (1 - alpha)
			polygon_j = polygon_j * alpha + c_j * (1 - alpha)
			polygon_i = np.array(polygon_i, np.int32)
			polygon_j = np.array(polygon_j, np.int32)
			polygon = [(polygon_j[k], polygon_i[k]) for k in range(len(polygon))]
		polygon = [(x + shift_j, y + shift_i) for x, y in polygon]

		# Adjust image and polygon
		org_info = (img.size, rotate)
		img_patch = img.resize(self.img_size, resample = Image.BICUBIC).rotate(rotate)
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
		while len(vertex_input) < self.max_num_vertices:
			vertex_input.append(np.array(self.blank, dtype = np.float32))
		while len(vertex_output) < self.max_num_vertices:
			vertex_output.append(np.array(self.blank, dtype = np.float32))
		vertex_input = np.array(vertex_input)
		vertex_output = np.array(vertex_output)

		# Get end signal
		seq_len = len(polygon_patch)
		end = [0.0 for i in range(self.max_num_vertices)]
		end[seq_len - 1] = 1.0
		end = np.array(end)

		# Example:
		# seq_len = 6
		# end ? ? ? ? ? ! X X
		# out 1 2 3 4 5 ? X X
		#  in 0 1 2 3 4 5 X X

		# Return
		return img_patch, boundary, vertices, vertex_input, vertex_output, end, seq_len

	def getSingleArea(self, area_idx):
		# Set path
		path = self.area_path + '/' + area_idx

		# Rotate
		n_rotate = 0 # random.choice([0, 1, 2, 3])

		# 
		while True:
			try:
				img = Image.open(io.BytesIO(self.sftp.open(path + '/img.png').read()))
				self.area_imgs.append(img)
				# img = img.resize((256, 256), resample = Image.BICUBIC)
				lines = self.sftp.open(path + '/polygons.txt').read().decode('utf-8').split('\n')
				break
			except:
				print('Try again.')

		img = img.rotate(n_rotate * 90)
		num_anchors = self.anchors.shape[0]

		org = np.array(img)[..., 0: 3] / 255.0

		polygons = []
		for line in lines:
			if line.strip() != '':
				if line.strip() == '%':
					polygons.append([])
				else:
					x, y = line.strip().split()
					polygons[-1].append((int(x), int(y)))

		gt_boxes = []
		pad = 0
		for polygon in polygons:
			w, h = (640, 640)
			p = np.array(polygon, np.int32)
			l = max(0, p[:, 0].min())
			u = max(0, p[:, 1].min())
			r = min(w, p[:, 0].max())
			d = min(h, p[:, 1].max())
			if r > l and d > u:
				# for _ in range(n_rotate):
				# 	(w, h), (l, u, r, d) = rotate((w, h), (l, u, r, d))
				gt_boxes.append([u - pad, l - pad, d + pad, r + pad])
		if len(gt_boxes) == 0:
			gt_boxes = np.zeros((0, 4), np.int32)
		else:
			gt_boxes = np.array(gt_boxes)

		anchor_cls = np.zeros([num_anchors, 2], np.int32)
		rpn_match, anchor_box = buildRPNTargets(self.anchors * 2.5, gt_boxes)
		anchor_cls[rpn_match == 1, 0] = 1
		anchor_cls[rpn_match == -1, 1] = 1

		return org, anchor_cls, anchor_box

	def getBuildingsBatch(self, batch_size, mode = None):
		# Real
		res = []
		if mode == 'train':
			sel = np.random.choice(len(self.id_list_train), batch_size, replace = True)
			for i in sel:
				res.append(self.getSingleBuilding(self.id_list_train[i]))
		if mode == 'valid':
			sel = np.random.choice(len(self.id_list_valid), batch_size, replace = True)
			for i in sel:
				res.append(self.getSingleBuilding(self.id_list_valid[i]))
		if mode == 'test':
			sel = np.random.choice(len(self.bad_id_list), batch_size, replace = True)
			for i in sel:
				res.append(self.getSingleBuilding(self.bad_id_list[i]))
		return (np.array([item[i] for item in res]) for i in range(7))

	def getAreasBatch(self, batch_size, mode = None):
		# Real
		res = []
		self.area_imgs = []
		if mode == 'train':
			sel = np.random.choice(len(self.idx_list_train), batch_size, replace = True)
			for i in sel:
				res.append(self.getSingleArea(self.idx_list_train[i]))
		if mode == 'valid':
			sel = np.random.choice(len(self.idx_list_valid), batch_size, replace = True)
			for i in sel:
				res.append(self.getSingleArea(self.idx_list_valid[i]))
		return (np.array([item[i] for item in res]) for i in range(3))

	def getPatchesFromAreas(self, res):
		self.pad = 0.3
		assert(len(res) == len(self.area_imgs))
		patches = []
		org_info = []
		for i, (org, bbox) in enumerate(zip(self.area_imgs, res)):
			img = np.array(org)
			boxes = bbox * 2.5
			for j in range(boxes.shape[0]):
				y1, x1, y2, x2 = tuple(list(boxes[j]))
				h, w = y2 - y1, x2 - x1
				y1, x1, y2, x2 = int(max(0, y1 - h * self.pad)), int(max(0, x1 - w * self.pad)), int(min(640, y2 + h * self.pad)), int(min(640, x2 + w * self.pad))
				if y1 < y2 and x1 < x2:
					patches.append(np.array(Image.fromarray(img[y1: y2, x1: x2, 0: 3]).resize((224, 224), resample = Image.BICUBIC))/255.0)
					org_info.append([i, y1, x1, y2, x2])
		return self.area_imgs, np.array(patches), org_info

	def recover(self, path, imgs, res, base):
		for i, img in enumerate(imgs):
			a = img.copy()
			boxes = res[i] * 2.5
			draw = ImageDraw.Draw(a)
			# f = open(path + '/_%s.txt' % i, 'w')
			for j in range(boxes.shape[0]):
				u, l, d, r = tuple(list(boxes[j]))
				if (r - l) * (d - u) > 24 * 24:
					draw.polygon([(l, u), (r, u), (r, d), (l, d)], outline = (255, 0, 0))
				# f.write('%d %d %d %d\n' % (u, l, d, r))
			# f.close()
			a.save(path + '/box_%d_%d.png' % (base, i))

	def recoverGlobal(self, path, img, org_info, pred_v_out, base):
		# Sequence length and polygon
		# batch_size = len(org_info)
		# assert(len(org_info) == pred_v_out.shape[0])
		# polygons = [[] for i in range(batch_size)]
		# for i in range(pred_v_out.shape[0]):
		# 	idx, y1, x1, y2, x2 = org_info[i]
		# 	w, h = x2 - x1, y2 - y1
		# 	draw = ImageDraw.Draw(img[idx])
		# 	for j in range(pred_v_out.shape[1]):
		# 		v = pred_v_out[i, j]
		# 		if v.sum() >= 0.5:
		# 			r, c = np.unravel_index(v.argmax(), v.shape)
		# 			polygons[i].append((c/28*w+x1, r/28*h+y1))
		# 		else:
		# 			polygons[i].append(polygons[i][0])
		# 			break
		# 	draw.line(polygons[i], fill = (255, 0, 0), width = 3)
		# for i, im in enumerate(img):
		# 	im.save(path + '/___%s.png' % i)
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
						polygons[i].append((c/28*w+x1, r/28*h+y1))
					else:
						polygons[i].append(polygons[i][0])
						break
				draw.line(polygons[i], fill = tableau20[kk], width = 2)
			break
		for i, im in enumerate(img):
			im.save(path + '/%d_%d.png' % (base, i))

if __name__ == '__main__':
	dg = DataGenerator(building_path = '../../Chicago.zip', area_path = '/local/lizuoyue/Chicago_Area', max_num_vertices = 24, img_size = (224, 224), resolution = (28, 28))
	item1 = dg.getAreasBatch(4, mode = 'train')
	item2 = dg.getBuildingsBatch(12, mode = 'train')
	for item in item1:
		print(item.shape)
	for item in item2:
		print(item.shape)
	a, b, c = dg.getPatchesFromAreas([np.array([[100, 100, 200, 200]]),np.array([[100, 100, 200, 200]]),np.array([[100, 100, 200, 200]]),np.array([[100, 100, 200, 200]])])
	print(len(a))
	print(b.shape)
	print(len(c))


