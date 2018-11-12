import numpy as np
import os, glob, random
from PIL import Image, ImageDraw

from scipy import spatial

cities = [
	'boston', 'chicago', 'amsterdam', 'denver',
	'la', 'montreal', 'paris', 'pittsburgh',
	'saltlakecity', 'tokyo', 'toronto', 'vancouver',
	'new york', 'kansas city', 'san diego'
]

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
	nb = [set() for item in v]
	for s, t in e:
		nb[s].add(t)
		nb[t].add(s)
	return v, e, nb

def save_graph(filename, v, e):
	with open(filename, 'w') as f:
		for vv in v:
			f.write('%d %d\n' % vv)
		f.write('\n')
		for ee in e:
			f.write('%d %d\n' % ee)
	return

def l2_dist(v1, v2):
	diff = np.array(v1) - np.array(v2)
	return np.sqrt(np.dot(diff, diff))

if __name__ == '__main__':
	files = glob.glob('out_ours_cu/*') + glob.glob('out_ours_uc/*')
	for file in files:
		print(file)
		v, e, nb = read_graph(file)
		invalid_idx = set()
		for idx, vnb in enumerate(nb):
			if len(vnb) == 1 and l2_dist(v[idx], v[list(vnb)[0]]) <= 64:
				invalid_idx.add(idx)

		e_rec = [(v[s], v[t]) for s, t in e if s not in invalid_idx and t not in invalid_idx]
		v_rec = set()
		for s, t in e_rec:
			v_rec.add(s)
			v_rec.add(t)
		v_val2idx = {}
		new_v = list(v_rec)
		for k, val in enumerate(new_v):
			v_val2idx[val] = k
		new_e = [(v_val2idx[s], v_val2idx[t]) for s, t in e_rec]

		save_graph(file, new_v, new_e)





