import json, os, time
from PIL import Image, ImageDraw
import numpy as np

IMG_SIZE = 256
IMG_SIZE_2 = 128
SEARCH_R = 4
RANGE = {'x4':[
	(-1,  0),
	(-1, -1), (-2,  0), (-1,  1),
	(-1, -2), (-2, -1), (-3,  0), (-2,  1), (-1,  2),
	(-1, -3), (-2, -2), (-3, -1), (-4,  0), (-3,  1), (-2,  2), (-1,  3)
]}
RANGE['x3'] = RANGE['x4'][:9]
RANGE['x2'] = RANGE['x3'][:4]
RANGE['x1'] = RANGE['x2'][:1]
for k in list(RANGE.keys()):
	RANGE[k.replace('x', 'y')] = [(item[1], item[0]) for item in RANGE[k]]


dts = json.load(open('../../EvalR/predictions_roadtracer-dalabgpu_vgg16_test.json'))
gts = json.load(open('../../EvalR/roadtracer_test.json'))
gt_by_id = {}
for gt in gts['images']:
	gt_by_id[gt['id']] = gt


dt_by_city_tile = {}
for dt in dts:
	gt = gt_by_id[dt['image_id']]
	l, u, _, _ = gt['tile_crop_lurd']
	city_tile = gt['tile_file'].replace('_sat.png', '')
	if city_tile not in dt_by_city_tile:
		dt_by_city_tile[city_tile] = [[], [], [], []]
	graph_idx = int((l % IMG_SIZE) / IMG_SIZE_2) * 2 + int((u % IMG_SIZE) / IMG_SIZE_2)
	dt_by_city_tile[city_tile][graph_idx].append((l, u, dt))
for k, v in dt_by_city_tile.items():
	for item in v:
		item.sort()


def add_to_res(res, dts):
	for l, u, dt in dts:
		ll, uu = round(l / 256 * 28), round(u / 256 * 28)
		r2a = []
		for x, y, _ in dt['vertices']:
			v = (x + ll, y + uu)
			idx = len(res['v'])
			res['vd'][v] = idx
			res['v'].append(v)
			r2a.append(idx)
			if x == 0 or x == 1:
				for dx, dy in RANGE['x%d' % SEARCH_R]:
					p = (ll + dx, y + uu + dy)
					if p in res['vd']:
						res['e'].append((res['vd'][p], idx, 2))
						res['e'].append((idx, res['vd'][p], 2))
			if y == 0 or y == 1:
				for dx, dy in RANGE['y%d' % SEARCH_R]:
					p = (x + ll + dx, uu + dy)
					if p in res['vd']:
						res['e'].append((res['vd'][p], idx, 2))
						res['e'].append((idx, res['vd'][p], 2))
		for a, b, s in dt['edges']:
			if s >= 0.5:
				res['e'].append((r2a[a], r2a[b], s))
	return


result = {}
for city_tile, graphs in dt_by_city_tile.items():
	res = [{'v': [], 'e': [], 'vd': {}} for _ in range(4)]
	for sub_res, graph in zip(res, graphs):
		add_to_res(sub_res, graph)
	result[city_tile] = res

############ Now we have totally 4 graphs

def l2dist(a, b):
	diff = np.array(a) - np.array(b)
	return np.sqrt(np.dot(diff, diff))

def show_graph(graph):
	v = graph['v']
	e = graph['e']
	img = Image.new('RGB', (round(28 * 4096 / 256), round(28 * 4096 / 256)), color = (0, 0, 0))
	draw = ImageDraw.Draw(img)
	for s, t, score in e:
		if score <= 1:
			draw.line(v[s] + v[t], width = 1, fill = (255, 255, 255))
		elif score <= 2:
			draw.line(v[s] + v[t], width = 1, fill = (255, 0, 0))
		elif score <= 3:
			draw.line(v[s] + v[t], width = 1, fill = (0, 255, 0))
	img.show()
	return



def get_crossing(s1, s2):
	xa, ya = s1[0][0], s1[0][1]
	xb, yb = s1[1][0], s1[1][1]
	xc, yc = s2[0][0], s2[0][1]
	xd, yd = s2[1][0], s2[1][1]
	a = np.matrix([
		[xb - xa, -(xd - xc)],
		[yb - ya, -(yd - yc)]
	])
	delta = np.linalg.det(a)
	if np.fabs(delta) < 1e-6:
		return None
	c = np.matrix([
		[xc - xa, -(xd - xc)],
		[yc - ya, -(yd - yc)]
	])
	d = np.matrix([
		[xb - xa, xc - xa],
		[yb - ya, yc - ya]
	])
	lamb = np.linalg.det(c) / delta
	miu = np.linalg.det(d) / delta
	if lamb <= 1 and lamb >= 0 and miu >= 0 and miu <= 1:
		x = xc + miu * (xd - xc)
		y = yc + miu * (yd - yc)
		return (x, y)
	else:
		return None

def which_most(li):
	d = {}
	for item in li:
		if item in d:
			d[item] += 1
		else:
			d[item] = 1
	l = [(v, k) for k, v in d.items()]
	return max(l)[1]

from scipy import spatial

def merge_graph(g_ms, g_add):
	# Init
	size = (round(28 * 4096 / 256), round(28 * 4096 / 256))
	img_master = np.zeros(size, np.int32)
	for e_idx, (s, t, _) in enumerate(g_ms['e']):
		img = Image.new('P', size, color = 0)
		draw = ImageDraw.Draw(img)
		draw.line(g_ms['v'][s] + g_ms['v'][t], width = 1, fill = 1)
		img_master = np.maximum(img_master, np.array(img) * (e_idx + 1))
	img_master -= 1
	kd_tree = spatial.KDTree(g_ms['v'])

	# Add
	# 1. 先加点。若主图中有某点和新点距离很近，则直接连起来。
	for add_v in g_add['v']:
		if add_v not in g_ms['vd']:
			add_idx = len(g_ms['v'])
			g_ms['vd'][add_v] = add_idx
			g_ms['v'].append(add_v)
			indices = kd_tree.query_ball_point(add_v, 3)
			for idx in indices:
				g_ms['e'].append((idx, add_idx, 3))
				g_ms['e'].append((add_idx, idx, 3))

	# 2. 再加边。若新边能覆盖大部分白色，则直接加边，两端相连；若不能覆盖大部分白色，则计算交点。
	for s, t, score in g_add['e']:
		x1, y1 = g_add['v'][s]
		x2, y2 = g_add['v'][t]
		g_ms['e'].append((g_ms['vd'][(x1, y1)], g_ms['vd'][(x2, y2)], score))
		g_ms['e'].append((g_ms['vd'][(x2, y2)], g_ms['vd'][(x1, y1)], score))

		# l, u, r, d = min(x1, x2), min(y1, y2), max(x1, x2) + 1, max(y1, y2) + 1
		# l, u, r, d = max(l, 0), max(u, 0), min(r, size[0]), min(d, size[1])
		# img_ms = img_master[u: d, l: r]
		# img = Image.new('P', (r - l, d - u), color = 0)
		# draw = ImageDraw.Draw(img)
		# draw.line([x1 - l, y1 - u, x2 - l, y2 - u], width = 1, fill = 1)
		# img = np.array(img)
		# edge_indices = img_ms[img > 0.5]
		# if (edge_indices >= 0).mean() > 0.7:
		if True:
			_, idx = kd_tree.query((x1, y1))
			g_ms['e'].append((idx, g_ms['vd'][(x1, y1)], 3))
			g_ms['e'].append((g_ms['vd'][(x1, y1)], idx, 3))
			_, idx = kd_tree.query((x2, y2))
			g_ms['e'].append((idx, g_ms['vd'][(x2, y2)], 3))
			g_ms['e'].append((g_ms['vd'][(x2, y2)], idx, 3))
		# else:
		# 	edge_indices = [item for item in list(edge_indices) if item >= 0]
		# 	if not edge_indices:
		# 		continue
		# 	for e_idx in [which_most(edge_indices)]:
		# 		ms, mt, _ = g_ms['e'][e_idx]
		# 		cross = get_crossing((g_ms['v'][ms], g_ms['v'][mt]), ((x1, y1), (x2, y2)))
		# 		if cross:
		# 			if cross not in g_ms['vd']:
		# 				add_idx = len(g_ms['v'])
		# 				g_ms['vd'][cross] = add_idx
		# 				g_ms['v'].append(cross)
		# 			else:
		# 				add_idx = g_ms['vd'][cross]
		# 			g_ms['e'].append((add_idx, g_ms['vd'][(x1, y1)], 3))
		# 			g_ms['e'].append((add_idx, g_ms['vd'][(x2, y2)], 3))
		# 			g_ms['e'].append((add_idx, ms, 3))
		# 			g_ms['e'].append((add_idx, mt, 3))
		# 			g_ms['e'].append((g_ms['vd'][(x1, y1)], add_idx, 3))
		# 			g_ms['e'].append((g_ms['vd'][(x2, y2)], add_idx, 3))
		# 			g_ms['e'].append((ms, add_idx, 3))
		# 			g_ms['e'].append((mt, add_idx, 3))

	d = {}
	for s, t, score in g_ms['e']:
		if (s, t) in d:
			d[(s, t)] = min(d[(s, t)], score)
		else:
			d[(s, t)] = score
	g_ms['e'] = [k + (v,) for k, v in d.items()]
	return


os.popen('mkdir out_graph_1')
for city_tile, graphs in result.items():
	new_graph = graphs[0]
	for i in range(1, 4):
		merge_graph(new_graph, graphs[i])

	print(city_tile)
	v = new_graph['v']
	e = new_graph['e']

	with open('./out_graph_1/%s.out.graph' % city_tile, 'w') as f:
		for item in v:
			f.write('%d %d\n' % item)
		f.write('\n')
		for s, t, score in e:
			f.write('%d %d %.4lf\n' % (s, t, score))


