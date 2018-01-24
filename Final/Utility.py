import numpy as np
import math, random
import os, io, sys, glob
import time, zipfile, paramiko
from PIL import Image, ImageDraw, ImageFilter
if os.path.exists('../Python-Lib/'):
	sys.path.insert(1, '../Python-Lib')

ANCHOR_SCALE   = [16, 32, 64, 128]
ANCHOR_RATIO   = [1.0 / 3, 1.0 / 2, 1, 2, 3]
FEATURE_SHAPE  = [[64, 64], [32, 32], [16, 16], [8, 8]]
FEATURE_STRIDE = [4, 8, 16, 32]

def randomPolygon(img_size, num_vertices, center, max_radius):
	epsilon = 1.0 / num_vertices
	delta_angle = np.pi * 2 * epsilon
	angle = np.random.uniform(0.0, delta_angle)
	polygon = [] # counterclockwise
	for i in range(num_vertices):
		if np.random.randint(0, 2):
			r = max_radius * np.random.uniform(0.8, 1.0)
		else:
			r = max_radius * np.random.uniform(0.3, 0.5)
		px = math.floor(center[0] + r * np.cos(angle))
		py = math.floor(center[1] - r * np.sin(angle))
		px = min(max(0, px), img_size[0])
		py = min(max(0, py), img_size[1])
		polygon.append((px, py))
		angle += delta_angle * np.random.uniform(1 - epsilon, 1 + epsilon)
	first_idx = random.choice([i for i in range(num_vertices)])
	polygon = polygon[first_idx:] + polygon[:first_idx]
	return polygon

def plotPolygons(img_size = (640, 640), num_polygons = 6, show = False):
	num_row = img_size[1]
	num_col = img_size[0]
	half_x = math.floor(num_col / 2)
	half_y = math.floor(num_row / 2)

	if num_polygons != 0:
		epsilon = 1.0 / num_polygons
	else:
		epsilon = 0
	center_r = math.floor(min(num_row, num_col) * 0.05) # <- Decide track's center
	track_size = math.floor(min(num_row, num_col) * 0.35) # <- Decide track's size
	delta_angle = np.pi * 2 * epsilon
	angle = np.random.uniform(0.0, delta_angle) # <- Decide polygon's first vertex
	center_x = half_x + np.random.randint(-center_r, center_r)
	center_y = half_y + np.random.randint(-center_r, center_r)

	polygons = []
	for i in range(num_polygons):
		r = track_size * np.random.uniform(0.8, 1.0)
		px = math.floor(center_x + r * np.cos(angle))
		py = math.floor(center_y - r * np.sin(angle))
		num_vertices = np.random.randint(4, 8)
		max_radius = 35 + num_vertices * 6
		polygons.append(randomPolygon(img_size, num_vertices, [px, py], max_radius))
		angle += delta_angle * np.random.uniform(1 - epsilon, 1 + epsilon)

	# Draw polygon
	org = Image.new('RGB', img_size, color = (255, 255, 255))
	draw = ImageDraw.Draw(org)
	for polygon in polygons:
		color = tuple([np.random.randint(80, 200) for i in range(3)])
		draw.polygon(polygon, fill = color, outline = color)

	# Add noise to the orginal image
	noise = np.random.normal(0, 40, (num_row, num_col, 3))
	background = np.array(org)
	background[0, 0] = (0, 127, 255)
	img = background + noise
	img = np.array((img - np.amin(img)) / (np.amax(img) - np.amin(img)) * 255.0, dtype = np.uint8)
	img = np.array(img / 255.0)
	if show:
		Image.fromarray(np.array(img * 255.0, dtype = np.uint8)).show()
	return img, polygons

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
	rpn_match[(anchor_iou_max < 0.3)] = -1
	# 2. Set an anchor for each GT box (regardless of IoU value).
	# TODO: If multiple anchors have the same IoU match all of them
	gt_iou_argmax = np.argmax(overlaps, axis=0)
	rpn_match[gt_iou_argmax] = 1
	# 3. Set anchors with high overlap as positive.
	rpn_match[anchor_iou_max >= 0.7] = 1

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

class DataGenerator(object):
	def __init__(self):
		self.anchors = generatePyramidAnchors(ANCHOR_SCALE, ANCHOR_RATIO, FEATURE_SHAPE, FEATURE_STRIDE, 1)
		return

	def getFakeDataBatch(self, batch_size):
		res_a = []
		res_b = []
		num_p = np.random.choice(8, batch_size, replace = True) + 1
		for num in num_p:
			#
			img, polygons = plotPolygons(num_polygons = num)
			bboxes = polygons2bboxes(polygons)
			rpn_match, anchor_box = buildRPNTargets(self.anchors, bboxes)
			anchor_cls = np.zeros([self.anchors.shape[0], 2], np.int32)
			anchor_cls[rpn_match == 1, 0] = 1
			anchor_cls[rpn_match == -1, 1] = 1

			#
			res_a.append((img, anchor_cls, anchor_box))
			res_b.append(img2patches(img, bboxes, polygons))
		res_a = tuple([np.array([item[i] for item in res_a]) for i in range(3)])
		res_b = tuple([np.concatenate([item[i] for item in res_b], 0) for i in range(7)])
		return res_a + res_b, num_p
		

	# def recover(self, path, idx, img, res):
	# 	for i in range(img.shape[0]):
	# 		boxes = res[i]
	# 		org = Image.fromarray(np.array(img[i] * 255.0, dtype = np.uint8))
	# 		draw = ImageDraw.Draw(org)
	# 		f = open(path + '/%s.txt' % idx[i], 'w')
	# 		for j in range(boxes.shape[0]):
	# 			u, l, d, r = tuple(list(boxes[j, :]))
	# 			if (r - l) * (d - u) > 24*24:
	# 				draw.polygon([(l, u), (r, u), (r, d), (l, d)], outline = (255, 0, 0))
	# 				f.write('%d %d %d %d\n' % (u, l, d, r))
	# 		f.close()
	# 		org.save(path + '/%s.png' % idx[i])


if __name__ == '__main__':
	for i in range(0):
		plotPolygons(num_polygons = i, show = True)
	dg = DataGenerator()
	item, num_p = dg.getFakeDataBatch(4)
	print(num_p)
	for i in range(10):
		print(item[i].shape)



