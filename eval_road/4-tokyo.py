import numpy as np
import os, glob, random
from PIL import Image, ImageDraw

from scipy import spatial

class UnionFind:
	"""Weighted quick-union with path compression.
	The original Java implementation is introduced at
	https://www.cs.princeton.edu/~rs/AlgsDS07/01UnionFind.pdf
	>>> uf = UnionFind(10)
	>>> for (p, q) in [(3, 4), (4, 9), (8, 0), (2, 3), (5, 6), (5, 9),
	...                (7, 3), (4, 8), (6, 1)]:
	...  uf.union(p, q)
	>>> uf._id
	[8, 3, 3, 3, 3, 3, 3, 3, 3, 3]
	>>> uf.find(0, 1)
	True
	>>> uf._id
	[3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
	"""

	def __init__(self, n):
		self._id = list(range(n))
		self._sz = [1] * n

	def _root(self, i):
		j = i
		while (j != self._id[j]):
			self._id[j] = self._id[self._id[j]]
			j = self._id[j]
		return j

	def find(self, p, q):
		return self._root(p) == self._root(q)
	
	def union(self, p, q):
		i = self._root(p)
		j = self._root(q)
		if i == j:
			return
		if (self._sz[i] < self._sz[j]):
			self._id[i] = j
			self._sz[j] += self._sz[i]
		else:
			self._id[j] = i
			self._sz[i] += self._sz[j]

cities = ['tokyo']

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
	os.popen('mkdir out_graph_big_cu_extra')
	files = []
	for city in cities:
		files.extend(glob.glob('out_graph_big_clean_uf/%s*' % city))
	for file in files:
		print(file)
		v, e = read_graph(file)

		uf = UnionFind(len(v))
		for s, t in e:
			uf.union(s, t)

		dv = {uf_id: set() for uf_id in set(uf._id)}
		de = {uf_id: [] for uf_id in set(uf._id)}
		for s, t in e:
			assert(uf.find(s, t))
			de[uf._root(s)].append((s, t))
			dv[uf._root(s)].add(s)
			dv[uf._root(s)].add(t)

		# print(sorted([len(dv[uf_id]) for uf_id in dv], reverse = True))
		valid_uf_id = set([uf_id for uf_id in dv if len(dv[uf_id]) >= 5])

		kd_tree = spatial.KDTree(v)
		pairs = list(kd_tree.query_pairs(r = 128))
		# print(len(pairs))
		# continue
		# pairs = [(i, j) for i, j in pairs if uf._root(i) != uf._root(j)]
		# pairs = [(i, j) for i, j in pairs if uf._root(i) in valid_uf_id and  uf._root(j) in valid_uf_id]
		pairs.extend([(j, i) for i, j in pairs])

		e_rec = []
		for uf_id in valid_uf_id:
			e_rec.extend(de[uf_id])
		e_rec.extend(pairs)
		e_rec = [(v[s], v[t]) for s, t in e_rec]
		v_rec = set()
		for s, t in e_rec:
			v_rec.add(s)
			v_rec.add(t)
		v_val2idx = {}
		new_v = list(v_rec)
		for k, val in enumerate(new_v):
			v_val2idx[val] = k
		new_e = [(v_val2idx[s], v_val2idx[t]) for s, t in e_rec]

		save_graph(file.replace('out_graph_big_clean_uf', 'out_graph_big_cu_extra'), new_v, new_e)





