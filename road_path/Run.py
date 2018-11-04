import os, sys, time

while True:
	nvidia = os.popen('nvidia-smi').read()
	li = nvidia.split('\n')
	for i in range(3, len(li) + 1):
		print(int(li[-i][5]))
		if li[-i].startswith('|==='):
			break
	quit()
