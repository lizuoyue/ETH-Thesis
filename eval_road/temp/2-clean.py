import os, glob
import numpy as np

def colinear_angle(p0, p1, p2):
	# angle of p0
	def l2dist(a, b):
		diff = np.array(a) - np.array(b)
		return np.sqrt(np.dot(diff, diff))
	a, b, c = l2dist(p0, p1), l2dist(p0, p2), l2dist(p1, p2)
	cos_C = (a * a + b * b - c * c) / (2 * a * b)
	return cos_C < -0.996 # cos(174.8736°)

def graph_process(v, e):
	# 1. Remove duplicate
	graph = [(v[s], v[t]) for s, t in e]
	graph = [item for item in graph if item[0] != item[1]]

	# 1. Remove duplicate
	v_val, e_val = set(), set()
	for item in graph:
		v_val.add(item[0])
		v_val.add(item[1])
		e_val.add(item)
		e_val.add((item[1], item[0]))
	v_val = list(v_val)
	e_val = list(e_val)
	v_val2idx = {v: k for k, v in enumerate(v_val)}
	e_idx = [(v_val2idx[s], v_val2idx[t]) for s, t in e_val]

	# 2. Get v to be removed
	nb = [[] for _ in range(len(v_val))]
	for s, t in e_idx:
		nb[s].append(t)
	v_rm = []
	for vid, (v, vnb) in enumerate(zip(v_val, nb)):
		if len(vnb) == 2:
			v0, v1 = v_val[vnb[0]], v_val[vnb[1]]
			if colinear_angle(v, v0, v1):
				v_rm.append(vid)
	v_rm_set = set(v_rm)

	# 3. Get e to be added
	e_add = []
	visited = [False for _ in range(len(v_val))]
	for vid in v_rm_set:
		if not visited[vid]:
			visited[vid] = True
			assert(len(nb[vid]) == 2)
			res = []
			for nvid_iter in nb[vid]:
				nvid = int(nvid_iter)
				while nvid in v_rm_set:
					visited[nvid] = True
					v1, v2 = nb[nvid]
					assert((v1 in v_rm_set and visited[v1]) + (v2 in v_rm_set and visited[v2]) == 1)
					if v1 in v_rm_set and visited[v1]:
						nvid = v2
					else:
						nvid = v1
				res.append(nvid)
			assert(len(res) == 2)
			e_add.append((res[0], res[1]))
			e_add.append((res[1], res[0]))

	# 4. Remove v and add e
	e_idx = [(s, t) for s, t in e_idx if s not in v_rm_set and t not in v_rm_set]
	e_idx.extend(e_add)

	return v_val, e_idx

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
	e = [tuple(int(item) for item in line.strip().split()) for line in e]
	return v, e

def save_graph(filename, v, e):
	with open(filename, 'w') as f:
		for vv in v:
			f.write('%d %d\n' % vv)
		f.write('\n')
		for ee in e:
			f.write('%d %d\n' % ee)
	return

if __name__ == '__main__':
	os.popen('mkdir out_graph_clean_1')
	files = glob.glob('out_graph/*0.5*.graph')
	for file in files:
		v, e = read_graph(file)
		v, e = graph_process(v, e)
		save_graph(file.replace('out_graph', 'out_graph_clean_1'), v, e)




