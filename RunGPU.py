import os, re, time

def do():
	model_list = os.listdir('./tmp')
	model_list.sort()
	num = re.findall('model\-(\d+)\.', model_list[-1])[0]
	command = 'bsub -n 20 -R \"rusage[mem=4500,ngpus_excl_p=8]\" python3 PolygonRNN.py %s' % num
	os.system(command)
	return

if __name__ == '__main__':
	do()
	last_time = int(time.time())
	while True:
		try:
			time.sleep(100)
		except:
			pass
		current_time = int(time.time())
		if current_time - last_time > 14700:
			last_time = current_time
			do()
