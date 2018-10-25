import numpy as np
import os, sys, glob, math, time, random, cv2
from Config import *
from UtilityBoxAnchor import *
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from PIL import Image, ImageDraw, ImageFilter

config = Config()

def setValidNum(ann):
	num = int(len(ann['segmentation'][0]) / 2)
	if num >= 3 and num <= config.MAX_NUM_VERTICES and ann['area'] >= 96:
		return num
	else:
		return 0

def normalize(li):
	s = sum(li)
	return [item / s for item in li]

def rotateBox(size, box):
	w, h = size
	x1, y1, x2, y2 = box
	return (h, w), (y1, w - x2, y2, w - x1)

def rotatePolygon(size, polygon):
	w, h = size
	return (h, w), [(y, w - 1 - x) for x, y in polygon]

def overlay(img, mask):
	"""
		both img and mask PIL.Image, rgb
	"""
	img = img.convert('RGBA')
	mask = np.array(mask, np.uint32)
	alpha = np.sum(np.array(mask, np.int32), axis = 2)
	alpha[alpha > 0] = 160
	alpha = np.expand_dims(alpha, axis = 2)
	alpha = np.concatenate((mask, alpha), axis = 2)
	alpha = Image.fromarray(np.array(alpha, np.uint8), mode = 'RGBA')
	return Image.alpha_composite(img, alpha)

class DataGenerator(object):
	def __init__(self, city_name, img_size, v_out_res, max_num_vertices, mode = 'train'):
		assert(mode in ['train', 'val', 'test'])
		self.city_name = city_name
		self.img_size = img_size
		self.v_out_res = v_out_res
		self.max_num_vertices = max_num_vertices

		self.TRAIN_ANNOTATIONS_PATH = config.PATH[city_name]['ann-train']
		self.VAL_ANNOTATIONS_PATH   = config.PATH[city_name]['ann-val']
		self.TEST_ANNOTATIONS_PATH  = config.PATH[city_name]['ann-test']
		self.TRAIN_IMAGES_DIRECTORY = config.PATH[city_name]['img-train']
		self.VAL_IMAGES_DIRECTORY   = config.PATH[city_name]['img-val']

		self.TEST_CURRENT = 0
		self.TEST_FLAG = True
		self.TEST_RESULT = []

		if mode == 'test':
			self.TEST_IMAGES_DIRECTORY = config.PATH[city_name]['img-test']
			if self.TEST_ANNOTATIONS_PATH is None:
				self.TEST_IMAGE_IDS = list(range(len(glob.glob(self.TEST_IMAGES_DIRECTORY + '/*'))))
			else:
				self.coco_test = COCO(self.TEST_ANNOTATIONS_PATH)
				self.TEST_IMAGE_IDS = list(self.coco_test.getImgIds(catIds = self.coco_train.getCatIds()))
		if mode == 'val':
			self.coco_valid = COCO(self.VAL_ANNOTATIONS_PATH)
			self.TEST_IMAGES_DIRECTORY = config.PATH[city_name]['img-val']
			self.TEST_IMAGE_IDS = list(self.coco_valid.getImgIds(catIds = self.coco_train.getCatIds()))
		if mode == 'train':
			self.coco_train = COCO(self.TRAIN_ANNOTATIONS_PATH)
			self.coco_valid = COCO(self.VAL_ANNOTATIONS_PATH)
			self.train_img_ids = self.coco_train.getImgIds(catIds = self.coco_train.getCatIds())
			self.train_ann_ids = self.coco_train.getAnnIds(catIds = self.coco_train.getCatIds())
			self.valid_img_ids = self.coco_valid.getImgIds(catIds = self.coco_valid.getCatIds())
			self.valid_ann_ids = self.coco_valid.getAnnIds(catIds = self.coco_valid.getCatIds())

			train_anns = self.coco_train.loadAnns(self.train_ann_ids)
			valid_anns = self.coco_valid.loadAnns(self.valid_ann_ids)
			self.num_v_num_building = {}
			for ann in train_anns + valid_anns:
				l = int(len(ann['segmentation'][0]) / 2)
				if l in self.num_v_num_building:
					self.num_v_num_building[l] += 1
				else:
					self.num_v_num_building[l] = 1
			print(self.num_v_num_building)

			# 
			self.train_ann_p = normalize([setValidNum(ann) for ann in train_anns])
			self.valid_ann_p = normalize([setValidNum(ann) for ann in valid_anns])

			print('Totally %d buildings for train.' % len(self.train_ann_ids))
			print('Totally %d buildings for valid.' % len(self.valid_ann_ids))
			print('Totally %d areas for train.' % len(self.train_img_ids))
			print('Totally %d areas for valid.' % len(self.valid_img_ids))

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
		return

	def area(self, polygon):
		s = sum([x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(polygon, polygon[1: ] + [polygon[0]])])
		return s / 2.0

	def polygon2bbox(self, polygon):
		x, y = polygon[0::2], polygon[1::2]
		x0, y0, x1, y1 = int(np.round(min(x))), int(np.round(min(y))), int(np.round(max(x))) + 1, int(np.round(max(y))) + 1
		return [(int(np.round(xx)) - x0, int(np.round(yy)) - y0) for xx, yy in zip(x, y)], (x0, y0, x1, y1)

	def distL1(self, p1, p2):
		return math.fabs(p1[0] - p2[0]) + math.fabs(p1[1] - p2[1])

	def removeColinear(self, polygon):
		flag = [polygon[i-1] != polygon[i] for i in range(len(polygon))]
		polygon = [v for i, v in enumerate(polygon) if flag[i]]
		flag = []
		for i in range(len(polygon)):
			temp_poly = [polygon[i - 1], polygon[i], polygon[(i + 1) % len(polygon)]]
			flag.append(abs(self.area(temp_poly)) > 1e-6)
		return [v for i, v in enumerate(polygon) if flag[i]]

	def getSingleBuilding(self, mode, ann_id, rotate):
		rotate_deg = rotate * 90
		if mode == 'train':
			annotation = self.coco_train.loadAnns([ann_id])[0]
			img_info = self.coco_train.loadImgs(annotation['image_id'])[0]
			image_path = os.path.join(self.TRAIN_IMAGES_DIRECTORY, img_info['file_name'])
		if mode == 'valid':
			annotation = self.coco_valid.loadAnns([ann_id])[0]
			img_info = self.coco_valid.loadImgs(annotation['image_id'])[0]
			image_path = os.path.join(self.VAL_IMAGES_DIRECTORY, img_info['file_name'])

		img = Image.open(image_path)
		org_w, org_h = img.size

		# Rotate image
		img = np.array(img.rotate(rotate_deg))
		img_h, img_w = float(img.shape[0] - 1), float(img.shape[1] - 1)

		# Rotate polygon
		polygon = annotation['segmentation'][0]
		polygon = [(int(round(x)), int(round(y))) for x, y in zip(polygon[0::2], polygon[1::2])]
		polygon = [(min(max(x, 0), org_w - 1), min(max(y, 0), org_h - 1)) for x, y in polygon]
		w, h = org_w, org_h
		for _ in range(rotate):
			(w, h), polygon = rotatePolygon((w, h), polygon)
		poly_np = np.array(polygon)
		x0, y0, x1, y1 = poly_np[:, 0].min(), poly_np[:, 1].min(), poly_np[:, 0].max() + 1, poly_np[:, 1].max() + 1
		x0_old, y0_old, x1_old, y1_old, = x0, y0, x1, y1

		# Padding
		if True:
			x0 = max(0, x0 - random.randint(0, 20))
			x1 = min(img.shape[1], x1 + random.randint(0, 20))
			y0 = max(0, y0 - random.randint(0, 20))
			y1 = min(img.shape[0], y1 + random.randint(0, 20))

		img_patch = img[y0: y1, x0: x1, ...]
		polygon = [(x - x0, y - y0) for x, y in polygon]
		if self.area(polygon) > 0:
			polygon.reverse()
		patch_h, patch_w = img_patch.shape[0: 2]

		# Adjust image and polygon
		crop_info = [self.to_batch_idx[annotation['image_id']], y0 / img_h, x0 / img_w, y1 / img_h, x1 / img_w]
		x_rate = self.v_out_res[0] / patch_w
		y_rate = self.v_out_res[1] / patch_h
		img_patch = Image.fromarray(img_patch).resize(self.img_size, resample = Image.BICUBIC)
		img_patch = np.array(img_patch, np.float32)[..., 0: 3]
		polygon_s = []
		for x, y in polygon:
			a, b = int(math.floor(x * x_rate)), int(math.floor(y * y_rate))
			if not polygon_s or self.distL1((a, b), polygon_s[-1]) > 0:
				polygon_s.append((a, b))
		polygon_s = self.removeColinear(polygon_s)
		if len(polygon_s) <= 2:
			return None

		# Draw boundary and vertices
		boundary = Image.new('P', self.v_out_res, color = 0)
		draw = ImageDraw.Draw(boundary)
		draw.polygon(polygon_s, fill = 0, outline = 255)
		boundary = np.array(boundary) / 255.0

		vertices = Image.new('P', self.v_out_res, color = 0)
		draw = ImageDraw.Draw(vertices)
		draw.point(polygon_s, fill = 255)
		vertices = np.array(vertices) / 255.0

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
		return img_patch, boundary, vertices, vertex_input, vertex_output, end, seq_len, crop_info

	def getSingleArea(self, mode, img_id, rotate):
		# Rotate, anticlockwise
		rotate_deg = rotate * 90
		if mode == 'train':
			img_info = self.coco_train.loadImgs([img_id])[0]
			image_path = os.path.join(self.TRAIN_IMAGES_DIRECTORY, img_info['file_name'])
			annotations = self.coco_train.loadAnns(self.coco_train.getAnnIds(imgIds = img_info['id']))
		if mode == 'valid':
			img_info = self.coco_valid.loadImgs([img_id])[0]
			image_path = os.path.join(self.VAL_IMAGES_DIRECTORY, img_info['file_name'])
			annotations = self.coco_valid.loadAnns(self.coco_valid.getAnnIds(imgIds = img_info['id']))
		if mode == 'test':
			image_path = os.path.join(self.TEST_IMAGES_DIRECTORY, str(img_id).zfill(12) + '.jpg')

		img = Image.open(image_path)
		org_w, org_h = img.size
		img = img.rotate(rotate_deg)
		self.recover_rate = (img.size[0] / config.AREA_SIZE[0], img.size[1] / config.AREA_SIZE[1])
		self.area_imgs.append(img)
		ret_img = np.array(img.resize(config.AREA_SIZE), np.float32)[..., 0: 3]

		if mode == 'test':
			return ret_img

		gt_boxes = []
		rx, ry = self.recover_rate
		for annotation in annotations:
			w, h = org_w, org_h
			_, (l, u, r, d) = self.polygon2bbox(annotation['segmentation'][0])
			for _ in range(rotate):
				(w, h), (l, u, r, d) = rotateBox((w, h), (l, u, r, d))
			gt_boxes.append([u / rx, l / ry, d / rx, r / ry])

		if False: # <- Local test
			draw = ImageDraw.Draw(img)
			for uu, ll, dd, rr in gt_boxes:
				l, u, r, d = ll * rx, uu * ry, rr * rx, dd * ry
				draw.line([(l, u), (r, u), (r, d), (l, d), (l, u)], fill = (255, 0, 0, 255), width = 1)
			img.save('%s.png' % str(img_id).zfill(12))

		if len(gt_boxes) == 0:
			gt_boxes = np.zeros((0, 4), np.int32)
		else:
			gt_boxes = np.array(gt_boxes)

		# 
		anchor_cls = np.zeros([self.anchors.shape[0], 2], np.int32)
		rpn_match, anchor_box = buildFPNTargets(self.anchors, gt_boxes)
		anchor_cls[rpn_match == 1, 0] = 1
		anchor_cls[rpn_match == -1, 1] = 1

		#
		return ret_img, anchor_cls, anchor_box

	def getBuildingsBatch(self, batch_size, mode = None):
		# Real
		res = []
		if mode == 'train':
			anns = self.coco_train.loadAnns(self.anns_choose_from)
			anns_p = normalize([setValidNum(ann) for ann in anns])
			while len(res) < batch_size:
				ann_id = np.random.choice(self.anns_choose_from, 1, replace = True, p = anns_p)
				ann = self.getSingleBuilding('train', int(ann_id), rotate = self.rotate)
				if ann is not None:
					res.append(ann)
		if mode == 'valid':
			anns = self.coco_valid.loadAnns(self.anns_choose_from)
			anns_p = normalize([setValidNum(ann) for ann in anns])
			while len(res) < batch_size:
				ann_id = np.random.choice(self.anns_choose_from, 1, replace = True, p = anns_p)
				ann = self.getSingleBuilding('valid', int(ann_id), rotate = self.rotate)
				if ann is not None:
					res.append(ann)
		return [np.array([item[i] for item in res]) for i in range(8)]

	def getAreasBatch(self, batch_size, mode = None):
		# Real
		res = []
		self.rotate = random.choice([0, 1, 2, 3])
		self.area_imgs = []
		self.anns_choose_from = []
		self.to_batch_idx = {}
		if mode == 'train':
			while True:
				sel = np.random.choice(self.train_img_ids, batch_size, replace = True)
				self.anns_choose_from = self.coco_train.getAnnIds(imgIds = sel)
				if len(self.anns_choose_from) > 0:
					break
			for i, img_id in enumerate(sel):
				res.append(self.getSingleArea('train', img_id, rotate = self.rotate))
				self.to_batch_idx[img_id] = i
		if mode == 'valid':
			while True:
				sel = np.random.choice(self.valid_img_ids, batch_size, replace = True)
				self.anns_choose_from = self.coco_valid.getAnnIds(imgIds = sel)
				if len(self.anns_choose_from) > 0:
					break
			for i, img_id in enumerate(sel):
				res.append(self.getSingleArea('valid', img_id, rotate = self.rotate))
				self.to_batch_idx[img_id] = i
		if mode == 'test':
			self.rotate = 0
			if self.TEST_FLAG:
				beg = self.TEST_CURRENT
				end = min(len(self.TEST_IMAGE_IDS), beg + batch_size)
				self.TEST_CURRENT_IDS = self.TEST_IMAGE_IDS[beg: end]
				if end - beg < batch_size:
					self.TEST_CURRENT_IDS.extend(self.TEST_IMAGE_IDS[0: (batch_size - end + beg)])
				if end == len(self.TEST_IMAGE_IDS):
					self.TEST_FLAG = False
					self.TEST_TRUE_IDS = self.TEST_IMAGE_IDS[beg: end]
				else:
					self.TEST_CURRENT = end
				for img_id in self.TEST_CURRENT_IDS:
					res.append(self.getSingleArea('test', img_id, rotate = 0))
				return np.array(res)
			else:
				return None
		print(sel, self.rotate)
		return [np.array([item[i] for item in res]) for i in range(3)]

	def getPatchesFromAreas(self, pred_score, pred_box):
		assert(len(pred_box) == len(self.area_imgs))
		rx, ry = self.recover_rate
		crop_info = []
		patch_info = []
		box_info = []
		for i, (im, score, boxes) in enumerate(zip(self.area_imgs, pred_score, pred_box)):
			img = np.array(im, np.uint8)[..., 0: 3]
			assert(score.shape[0] == boxes.shape[0])
			for j in range(boxes.shape[0]):
				y1, x1, y2, x2 = tuple(list(boxes[j]))
				x1, x2, y1, y2 = max(0, x1 * rx), min(img.shape[1], x2 * rx), max(0, y1 * ry), min(img.shape[0], y2 * ry)
				h, w = y2 - y1, x2 - x1
				if h > 0 and w > 0:
					if h * w > config.MIN_BBOX_AREA:
						nh, nw = int(h * 1.2), int(w * 1.2)
						cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
						ny1, nx1, ny2, nx2 = int(max(0, cy - nh / 2)), int(max(0, cx - nw / 2)), int(min(img.shape[0], cy + nh / 2)), int(min(img.shape[1], cx + nw / 2))
						if ny1 < ny2 and nx1 < nx2:
							img_h, img_w = img.shape[0] - 1, img.shape[1] - 1
							crop_info.append([i, ny1 / img_h, nx1 / img_w, ny2 / img_h, nx2 / img_w])
							patch_info.append((i, ny1, nx1, ny2, nx2))
						else:
							patch_info.append(None)
					else:
						patch_info.append(None)
					box_info.append((i, y1, x1, y2, x2, score[j]))
		return np.array(crop_info, np.float32), patch_info, box_info

	def recoverBoxPolygon(self, patch_info, box_info, pred_v_out, mode, visualize = True, path = None, batch_idx = None):
		#
		if visualize:
			assert(path != None)
			assert(batch_idx != None)
		assert(len(patch_info) == len(box_info))
		valid_patch_info = [(i, item) for i, item in enumerate(patch_info) if item]
		assert(len(valid_patch_info) == pred_v_out.shape[1])

		#
		res_ann = []
		img_idx = []
		for idx, y1, x1, y2, x2, score in box_info:
			res_ann.append({
				'category_id': 100,
				'bbox': [x1, y1, x2 - x1, y2 - y1],
				'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]],
				'score': score
			})
			if mode == 'test':
				res_ann[-1]['image_id'] = self.TEST_CURRENT_IDS[idx]
			img_idx.append(idx)

		# pred_v_out: [beam width, batch_size, max_len, ...]
		batch_size = len(self.area_imgs)
		for i in range(pred_v_out.shape[1]):
			ann_idx, (idx, y1, x1, y2, x2) = valid_patch_info[i]
			w, h = x2 - x1, y2 - y1
			polygon, flag = [], False
			for j in range(pred_v_out.shape[2]):
				v = pred_v_out[0, i, j]
				e = 1 - v.sum()
				if v.max() > e:
					vv = cv2.resize(v, tuple(config.PATCH_SIZE), interpolation = cv2.INTER_CUBIC)
					r, c = np.unravel_index(vv.argmax(), vv.shape)
					x, y = c / config.PATCH_SIZE[0] * w + x1, r / config.PATCH_SIZE[1] * h + y1
					polygon.extend([x, y])
				else:
					flag = True
					break
			if flag and len(polygon) >= 6:
				res_ann[ann_idx]['segmentation'] = [polygon]

		if mode == 'test':
			self.TEST_RESULT.extend(res_ann)
			if not self.TEST_FLAG:
				while self.TEST_RESULT[-1]['image_id'] not in self.TEST_TRUE_IDS:
					self.TEST_RESULT.pop()

		if not visualize:
			return

		bbox_mask = [Image.new('RGB', (im.size[1], im.size[0]), color = (0, 0, 0)) for im in self.area_imgs]
		bbox_mask_draw = [ImageDraw.Draw(mask) for mask in bbox_mask]
		for idx, y1, x1, y2, x2, score in box_info:
			bbox_mask_draw[idx].polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], outline = (0, 227, 0))

		color_count, len_c = 0, len(config.TABLEAU20)
		ins_mask = [[] for i in range(batch_size)]
		for idx, ann in zip(img_idx, res_ann):
			polygon = ann['segmentation'][0]
			polygon = [(x, y) for x, y in zip(polygon[0::2], polygon[1::2])]
			mask = Image.new('RGB', (self.area_imgs[idx].size[1], self.area_imgs[idx].size[0]), color = (0, 0, 0))
			draw = ImageDraw.Draw(mask)
			draw.polygon(polygon, fill = config.TABLEAU20[color_count % len_c])
			draw.line(polygon + [polygon[0]], fill = config.TABLEAU20_DEEP[color_count % len_c], width = 2)
			color_count += 1
			ins_mask[idx].append(mask)

		for i, (im, masks) in enumerate(zip(self.area_imgs, ins_mask)):
			img = im.copy()
			for mask in masks:
				img = overlay(img, mask)
			img = overlay(img, bbox_mask[i])
			img.save(path + '/%d_%d.png' % (batch_idx, i))

		return

if __name__ == '__main__':
	np.random.seed(8888)

	dg = DataGenerator(
		city_name = sys.argv[1],
		img_size = config.PATCH_SIZE,
		v_out_res = config.V_OUT_RES,
		max_num_vertices = config.MAX_NUM_VERTICES,
	)

	for n in range(10000):
		print(n)
		item1 = dg.getAreasBatch(4, mode = 'train')
		item2 = dg.getBuildingsBatch(12, mode = 'train')
		item3 = dg.getAreasBatch(4, mode = 'valid')
		item4 = dg.getBuildingsBatch(12, mode = 'valid')
	quit()

	for k in range(12):
		for i, item in enumerate(list(item2)):
			if i < 1:
				Image.fromarray(np.array(item[k, ...], np.uint8)).save('%d-%d.png' % (k, i))
			elif i < 3:
				Image.fromarray(np.array(item[k, ...] * 255.0, np.uint8)).save('%d-%d.png' % (k, i))
			elif i < 5:
				for j in range(config.MAX_NUM_VERTICES):
					Image.fromarray(np.array(item[k, j, ...] * 255.0, np.uint8)).save('%d-%d-%d.png' % (k, i, j))
			else:
				print(item[k])


