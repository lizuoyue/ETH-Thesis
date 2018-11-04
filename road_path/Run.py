import os, sys, time

while True:
	nvidia = os.popen('nvidia-smi').read()
	li = nvidia.split('\n')
	res = []
	for i in range(3, len(li) + 1):
		if li[-i].startswith('|==='):
			break
		res.append(int(li[-i][5]))
	print(78)
	quit()
