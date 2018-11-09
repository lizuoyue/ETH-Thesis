import os, glob
import numpy as np
from PIL import Image, ImageDraw

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

def read_graph(filename):
	with open(filename) as f:
		lines = f.readlines()
	for i, line in enumerate(lines):
		if line == '\n':
			sep = i
			break
	v = lines[:sep]
	e = lines[sep+1:]
	v = [tuple(int(item) for item in line.strip().split()) for line in v]
	e = [tuple(float(item) for item in line.strip().split()) for line in e]
	e = [(int(item[0]), int(item[1]), item[2]) for item in e]
	return {'v': v, 'e': e}

def save_graph(filename, v, e):
	with open(filename, 'w') as f:
		for vv in v:
			f.write('%d %d\n' % vv)
		f.write('\n')
		for ee in e:
			f.write('%d %d\n' % ee)
	return


def get_e_val(bias, graph):
	bx, by = bias
	bx *= 448
	by *= 448
	res = []
	v = [(x + bx, y + by) for x, y in graph['v']]
	for s, t, score in graph['e']:
		res.append((v[s], v[t], score))
	return res


def merge_4(city_d):
	# g0 g2
	# g1 g3
	e_val = []
	for bias, graph in city_d.items():
		e_val.extend(get_e_val(bias, graph))

	vd = set()
	for v1, v2, _ in e_val:
		vd.add(v1)
		vd.add(v2)
	v_li = sorted(list(vd))
	vd = {v: k for k, v in enumerate(v_li)}

	for x, y in vd:
		if x == 448 or x == 449:
			for dx, dy in RANGE['x%d' % SEARCH_R]:
				p = (448 + dx, y + dy)
				if p in vd:
					e_val.append((p, (x, y), 2.0))
					e_val.append(((x, y), p, 2.0))
		if x == 447 or x == 446:
			for dx, dy in RANGE['x%d' % SEARCH_R]:
				p = (447 - dx, y + dy)
				if p in vd:
					e_val.append((p, (x, y), 2.0))
					e_val.append(((x, y), p, 2.0))
		if y == 448 or y == 449:
			for dx, dy in RANGE['y%d' % SEARCH_R]:
				p = (x + dx, 448 + dy)
				if p in vd:
					e_val.append((p, (x, y), 2.0))
					e_val.append(((x, y), p, 2.0))
		if y == 447 or y == 446:
			for dx, dy in RANGE['y%d' % SEARCH_R]:
				p = (x + dx, 447 - dy)
				if p in vd:
					e_val.append((p, (x, y), 2.0))
					e_val.append(((x, y), p, 2.0))

	e_d = {}
	for v1, v2, score in e_val:
		idx1 = (vd[v1], vd[v2])
		idx2 = (vd[v2], vd[v1])
		if idx1 in e_d:
			e_d[idx1] = min(e_d[idx1], score)
		else:
			e_d[idx1] = score
		if idx2 in e_d:
			e_d[idx2] = min(e_d[idx2], score)
		else:
			e_d[idx2] = score
	e_idx = sorted([k + (v,) for k, v in e_d.items()])
	return {'v': v_li, 'e': e_idx, 'vd': vd}



if __name__ == '__main__':
	os.popen('mkdir out_graph_2')
	files = glob.glob('./out_graph_1/*.out.graph')
	d = {}
	for file in files:
		city, bx, by = file.split('/')[-1].replace('.out.graph', '').split('_')
		bx, by = int(bx), int(by)
		if city in d:
			d[city][(bx, by)] = read_graph(file)
		else:
			d[city] = {(bx, by): read_graph(file)}
	for city in d:
		print(city)
		bx, by = sorted(list(d[city].keys()))[0]
		d[city] = merge_4(d[city])
		v = d[city]['v']
		e = d[city]['e']

		if False:
			img = Image.new('RGB', (448 * 2, 448 * 2), color = (0, 0, 0))
			draw = ImageDraw.Draw(img)

			for s, t, score in e:
				x1, y1 = v[s][0] - bx * 448, v[s][1] - by * 448
				x2, y2 = v[t][0] - bx * 448, v[t][1] - by * 448
				if score <= 1.5 and score >= 0.7:
					draw.line([x1, y1, x2, y2], width = 1, fill = (255, 255, 255))
				elif score <= 2.5:
					draw.line([x1, y1, x2, y2], width = 1, fill = (255, 0, 0))
				elif score <= 3.5:
					draw.line([x1, y1, x2, y2], width = 1, fill = (0, 255, 0))
			img.show()
			input()

		for th in np.arange(0.5, 1.0, 0.05):
			with open('./out_graph_2/%s_%.2lf.out.graph' % (city, th), 'w') as f:
				for item in v:
					x, y = round(item[0] * (256 / 28)), round(item[1] * (256 / 28))
					f.write('%d %d\n' % (x, y))
				f.write('\n')
				for s, t, score in e:
					if score >= th:
						f.write('%d %d\n' % (s, t))





