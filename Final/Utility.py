import numpy as np
import math, random
import os, io, sys, glob
import time, zipfile, paramiko
from PIL import Image, ImageDraw, ImageFilter
if os.path.exists('../../Python-Lib/'):
	sys.path.insert(1, '../../Python-Lib')

BLUR = 0.75
ANCHOR_SCALE   = [16, 32, 64, 128]
ANCHOR_RATIO   = [0.25, 0.5, 1, 2, 4]
FEATURE_SHAPE  = [[64, 64], [32, 32], [16, 16], [8, 8]]
FEATURE_STRIDE = [4, 8, 16, 32]

############################################################
#  Bounding Boxes
############################################################

def computeIoU(box, boxes, box_area, boxes_area):
	"""Calculates IoU of the given box with the array of the given boxes.
	box: 1D vector [y1, x1, y2, x2]
	boxes: [boxes_count, (y1, x1, y2, x2)]
	box_area: float. the area of 'box'
	boxes_area: array of length boxes_count.

	Note: the areas are passed in rather than calculated here for
		  efficency. Calculate once in the caller to avoid duplicate work.
	"""
	# Calculate intersection areas
	y1 = np.maximum(box[0], boxes[:, 0])
	y2 = np.minimum(box[2], boxes[:, 2])
	x1 = np.maximum(box[1], boxes[:, 1])
	x2 = np.minimum(box[3], boxes[:, 3])
	intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
	union = box_area + boxes_area[:] - intersection[:]
	return intersection / union

def computeOverlaps(boxes1, boxes2):
	"""Computes IoU overlaps between two sets of boxes.
	boxes1, boxes2: [N, (y1, x1, y2, x2)].

	For better performance, pass the largest set first and the smaller second.
	"""
	# Areas of anchors and GT boxes
	area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
	area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

	# Compute overlaps to generate matrix [boxes1 count, boxes2 count]
	# Each cell contains the IoU value.
	overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
	for i in range(overlaps.shape[1]):
		box2 = boxes2[i]
		overlaps[:, i] = computeIoU(box2, boxes1, area2[i], area1)
	return overlaps

def nonMaxSuppression(boxes, scores, threshold):
	"""Performs non-maximum supression and returns indicies of kept boxes.
	boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
	scores: 1-D array of box scores.
	threshold: Float. IoU threshold to use for filtering.
	"""
	assert boxes.shape[0] > 0
	if boxes.dtype.kind != "f":
		boxes = boxes.astype(np.float32)

	# Compute box areas
	y1 = boxes[:, 0]
	x1 = boxes[:, 1]
	y2 = boxes[:, 2]
	x2 = boxes[:, 3]
	area = (y2 - y1) * (x2 - x1)

	# Get indicies of boxes sorted by scores (highest first)
	ixs = scores.argsort()[::-1]

	pick = []
	while len(ixs) > 0:
		# Pick top box and add its index to the list
		i = ixs[0]
		pick.append(i)
		# Compute IoU of the picked box with the rest
		iou = computeIoU(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
		# Identify boxes with IoU over the threshold. This
		# returns indicies into ixs[1:], so add 1 to get
		# indicies into ixs.
		remove_ixs = np.where(iou > threshold)[0] + 1
		# Remove indicies of the picked and overlapped boxes.
		ixs = np.delete(ixs, remove_ixs)
		ixs = np.delete(ixs, 0)
	return np.array(pick, dtype=np.int32)

def applyBoxesDeltas(boxes, deltas):
	"""Applies the given deltas to the given boxes.
	boxes: [N, (y1, x1, y2, x2)]. Note that (y2, x2) is outside the box.
	deltas: [N, (dy, dx, log(dh), log(dw))]
	"""
	deltas *= np.array([0.1, 0.1, 0.2, 0.2])
	boxes = boxes.astype(np.float32)
	# Convert to y, x, h, w
	height = boxes[:, 2] - boxes[:, 0]
	width = boxes[:, 3] - boxes[:, 1]
	center_y = boxes[:, 0] + 0.5 * height
	center_x = boxes[:, 1] + 0.5 * width
	# Apply deltas
	center_y += deltas[:, 0] * height
	center_x += deltas[:, 1] * width
	height *= np.exp(deltas[:, 2])
	width *= np.exp(deltas[:, 3])
	# Convert back to y1, x1, y2, x2
	y1 = center_y - 0.5 * height
	x1 = center_x - 0.5 * width
	y2 = y1 + height
	x2 = x1 + width
	return np.stack([y1, x1, y2, x2], axis=1)

def boxRefinement(box, gt_box):
	"""Compute refinement needed to transform box to gt_box.
	box and gt_box are [N, (y1, x1, y2, x2)]. (y2, x2) is
	assumed to be outside the box.
	"""
	box = box.astype(np.float32)
	gt_box = gt_box.astype(np.float32)

	height = box[:, 2] - box[:, 0]
	width = box[:, 3] - box[:, 1]
	center_y = box[:, 0] + 0.5 * height
	center_x = box[:, 1] + 0.5 * width

	gt_height = gt_box[:, 2] - gt_box[:, 0]
	gt_width = gt_box[:, 3] - gt_box[:, 1]
	gt_center_y = gt_box[:, 0] + 0.5 * gt_height
	gt_center_x = gt_box[:, 1] + 0.5 * gt_width

	dy = (gt_center_y - center_y) / height
	dx = (gt_center_x - center_x) / width
	dh = np.log(gt_height / height)
	dw = np.log(gt_width / width)

	return np.stack([dy, dx, dh, dw], axis=1) / np.array([0.1, 0.1, 0.2, 0.2])

############################################################
#  Anchors
############################################################

def generateAnchors(scales, ratios, shape, feature_stride, anchor_stride):
	"""
	scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
	ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
	shape: [height, width] spatial shape of the feature map over which
			to generate anchors.
	feature_stride: Stride of the feature map relative to the image in pixels.
	anchor_stride: Stride of anchors on the feature map. For example, if the
		value is 2 then generate anchors for every other feature map pixel.
	"""
	# Get all combinations of scales and ratios
	scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
	scales = scales.flatten()
	ratios = ratios.flatten()

	# Enumerate heights and widths from scales and ratios
	heights = scales / np.sqrt(ratios)
	widths = scales * np.sqrt(ratios)

	# Enumerate shifts in feature space
	shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
	shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
	shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

	# Enumerate combinations of shifts, widths, and heights
	box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
	box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

	# Reshape to get a list of (y, x) and a list of (h, w)
	box_centers = np.stack(
		[box_centers_y + feature_stride / 2, box_centers_x+ feature_stride / 2], axis=2).reshape([-1, 2])
	box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

	# Convert to corner coordinates (y1, x1, y2, x2)
	boxes = np.concatenate([box_centers - 0.5 * box_sizes,
							box_centers + 0.5 * box_sizes], axis=1)
	return np.array(boxes, dtype=np.float32)

def generatePyramidAnchors(scales, ratios, feature_shapes, feature_strides,
							 anchor_stride):
	"""Generate anchors at different levels of a feature pyramid. Each scale
	is associated with a level of the pyramid, but each ratio is used in
	all levels of the pyramid.

	Returns:
	anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
		with the same order of the given scales. So, anchors of scale[0] come
		first, then anchors of scale[1], and so on.
	"""
	# Anchors
	# [anchor_count, (y1, x1, y2, x2)]
	anchors = []
	for i in range(len(scales)):
		anchors.append(generateAnchors(scales[i], ratios, feature_shapes[i],
										feature_strides[i], anchor_stride))
	return np.concatenate(anchors, axis=0)

def buildRPNTargets(anchors, gt_boxes):
	"""Given the anchors and GT boxes, compute overlaps and identify positive
	anchors and deltas to refine them to match their corresponding GT boxes.

	anchors: [num_anchors, (y1, x1, y2, x2)]
	gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

	Returns:
	rpn_match: [N] (int32) matches between anchors and GT boxes.
			   1 = positive anchor, -1 = negative anchor, 0 = neutral
	rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
	"""
	# RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
	rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
	# RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
	rpn_bbox = np.zeros((anchors.shape[0], 4), dtype = np.float32)
	if gt_boxes.shape[0] == 0:
		return rpn_match, rpn_bbox

	# Compute overlaps [num_anchors, num_gt_boxes]
	overlaps = computeOverlaps(anchors, gt_boxes)

	# Match anchors to GT Boxes
	# If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
	# If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
	# Neutral anchors are those that don't match the conditions above,
	# and they don't influence the loss function.
	# However, don't keep any GT box unmatched (rare, but happens). Instead,
	# match it to the closest anchor (even if its max IoU is < 0.3).
	#
	# 1. Set negative anchors first. They get overwritten below if a GT box is
	# matched to them. Skip boxes in crowd areas.
	anchor_iou_argmax = np.argmax(overlaps, axis=1)
	anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
	rpn_match[(anchor_iou_max < 0.2)] = -1
	# 2. Set an anchor for each GT box (regardless of IoU value).
	# TODO: If multiple anchors have the same IoU match all of them
	gt_iou_argmax = np.argmax(overlaps, axis=0)
	rpn_match[gt_iou_argmax] = 1
	# 3. Set anchors with high overlap as positive.
	rpn_match[anchor_iou_max >= 0.5] = 1

	# Subsample to balance positive and negative anchors
	# Don't let positives be more than half the anchors
	ids = np.where(rpn_match == 1)[0]
	extra = len(ids) - (256 // 2)
	if extra > 0:
		# Reset the extra ones to neutral
		ids = np.random.choice(ids, extra, replace=False)
		rpn_match[ids] = 0
	# Same for negative proposals
	ids = np.where(rpn_match == -1)[0]
	extra = len(ids) - (256 - np.sum(rpn_match == 1))
	if extra > 0:
		# Rest the extra ones to neutral
		ids = np.random.choice(ids, extra, replace=False)
		rpn_match[ids] = 0

	# For positive anchors, compute shift and scale needed to transform them
	# to match the corresponding GT boxes.
	ids = np.where(rpn_match == 1)[0]
	# TODO: use box_refinment() rather than duplicating the code here
	for i, a in zip(ids, anchors[ids]):
		# # Closest gt box (it might have IoU < 0.7)
		gt = gt_boxes[anchor_iou_argmax[i]]
		rpn_bbox[i,:] = boxRefinement(np.array([a]), np.array([gt]))
	return rpn_match, rpn_bbox

def polygons2bboxes(polygons):
	"""
		polygons: list of [(x, y), ..., (x, y)]
	"""
	bboxes = []
	for polygon in polygons:
		p = np.array(polygon)
		x1 = p[:, 0].min()
		y1 = p[:, 1].min()
		x2 = p[:, 0].max()
		y2 = p[:, 1].max()
		bboxes.append([y1, x1, y2, x2])
	return np.array(bboxes)

def img2patches(img, bboxes, polygons, show = False):
	"""
		img: [nrow, ncol, 3]
		bboxes: [N, 4]
	"""
	pad = 0.1
	resize = (256, 256)
	resolution = (32, 32)

	res = []
	for i in range(bboxes.shape[0]):
		y1, x1, y2, x2 = bboxes[i]
		w, h = x2 - x1, y2 - y1
		delta_w, delta_h = int(w * pad), int(h * pad)
		x1 = max(0, int(x1 - delta_w))
		y1 = max(0, int(y1 - delta_h))
		x2 = min(img.shape[1], int(x2 + delta_w))
		y2 = min(img.shape[0], int(y2 + delta_h))

		patch = Image.fromarray(np.array(img[y1: y2, x1: x2, :] * 255, np.uint8)).resize(resize)
		polygon = [((x - x1) * 256 / (x2 - x1), (y - y1) * 256 / (y2 - y1)) for x, y in polygons[i]]
		polygon_s = [(x / 8, y / 8) for x, y in polygon]
		if show:
			draw = ImageDraw.Draw(patch)
			draw.polygon(polygon, outline = (255, 0, 0))
			patch.show()
			time.sleep(0.25)
		patch = np.array(patch) / 255.0

		boundary = Image.new('P', resolution, color = 0)
		draw = ImageDraw.Draw(boundary)
		draw.polygon(polygon_s, fill = 0, outline = 255)
		if show:
			boundary.show()
			time.sleep(0.25)
		boundary = np.array(boundary) / 255.0

		vertices = Image.new('P', resolution, color = 0)
		draw = ImageDraw.Draw(vertices)
		draw.point(polygon_s, fill = 255)
		if show:
			vertices.show()
			time.sleep(0.25)
		vertices = np.array(vertices) / 255.0

		v_in = []
		seq_len = len(polygon)
		for i in range(seq_len):
			vertex = Image.new('P', resolution, color = 0)
			draw = ImageDraw.Draw(vertex)
			draw.point([polygon_s[i]], fill = 255)
			v_in.append(np.array(vertex) / 255.0)
			if show:
				vertex.show()
				time.sleep(0.25)
		while len(v_in) < 10:
			v_in.append(np.zeros(resolution, np.float32))
		v_out = v_in[1: ] + [np.zeros(resolution, np.float32)]
		v_in = np.array(v_in)
		v_out = np.array(v_out)

		end = [0.0 for i in range(10)]
		end[seq_len - 1] = 1.0
		end = np.array(end)

		res.append((patch, boundary, vertices, v_in, v_out, end, seq_len))
	return tuple([np.array([item[i] for item in res]) for i in range(7)])

#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################

class DataGenerator(object):
	# num_col, num_row
	def __init__(self, building_path = None, area_path = None, max_seq_len = None, img_size = (224, 224), resolution = (28, 28)):
		assert(max_seq_len != None)
		self.max_seq_len = max_seq_len
		self.img_size = img_size
		self.resolution = resolution
		assert(building_path.endswith('.zip'))

		# 
		self.building_path = building_path.lstrip('./').replace('.zip', '')
		self.area_path = area_path
		self.archive = zipfile.ZipFile(building_path, 'r')
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

		with open('./AreaIdxList.txt', 'r') as f:
			self.area_idx_list = eval(f.read())
		print('Totally %d areas.' % len(self.area_idx_list))
		random.seed(0)
		random.shuffle(self.area_idx_list)
		random.seed()
		#
		split = int(train_prob * len(self.area_idx_list))
		self.idx_list_train = self.area_idx_list[:split]
		self.idx_list_valid = self.area_idx_list[split:]

		self.ssh = paramiko.SSHClient()
		self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
		self.ssh.connect('cab-e81-28.ethz.ch', username = 'zoli', password = '64206960lzyLZY')
		self.sftp = self.ssh.open_sftp()

		self.anchors = generatePyramidAnchors(ANCHOR_SCALE, ANCHOR_RATIO, FEATURE_SHAPE, FEATURE_STRIDE, 1)
		return

	def dispatchBuilding(self, building_id, th = 0.9):
		# Set path
		building_id = str(building_id)
		path = self.building_path + '/' + building_id

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

	def getSingleBuilding(self, building_id):
		# Set path
		building_id = str(building_id)
		path = self.building_path + '/' + building_id

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

	def recover(self, path, imgs, res):
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
			a.save(path + '/_%s.png' % i)

	def recoverGlobal(self, path, img, org_info, pred_v_out):
		# Sequence length and polygon
		batch_size = len(org_info)
		assert(len(org_info) == pred_v_out.shape[0])
		polygons = [[] for i in range(batch_size)]
		for i in range(pred_v_out.shape[0]):
			idx, y1, x1, y2, x2 = org_info[i]
			w, h = x2 - x1, y2 - y1
			draw = ImageDraw.Draw(img[idx])
			for j in range(pred_v_out.shape[1]):
				v = pred_v_out[i, j]
				if v.sum() >= 0.5:
					r, c = np.unravel_index(v.argmax(), v.shape)
					polygons[i].append((c/28*w+x1, r/28*h+y1))
				else:
					polygons[i].append(polygons[i][0])
					break
			draw.line(polygons[i], fill = (255, 0, 0), width = 3)
		for i, im in enumerate(img):
			im.save(path + '/___%s.png' % i)

if __name__ == '__main__':
	dg = DataGenerator(building_path = '../../Chicago.zip', area_path = '/local/lizuoyue/Chicago_Area', max_seq_len = 24, img_size = (224, 224), resolution = (28, 28))
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


