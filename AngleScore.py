import numpy as np
import math

res_num = 28 * 28
v_out_res = [28]
angle_score = np.ones((res_num * res_num, res_num + 1), np.float32)
for i in range(res_num):
	a = np.array([math.floor(i / v_out_res[0]), i % v_out_res[0]])
	for j in range(res_num):
		print(i, j)
		if i == j:
			continue
		b = np.array([math.floor(j / v_out_res[0]), j % v_out_res[0]])
		ab = b - a
		norm_ab = np.linalg.norm(ab)
		for k in range(res_num):
			if k == j:
				angle_score[i * res_num + j, k] = 0
				continue
			c = np.array([math.floor(k / v_out_res[0]), k % v_out_res[0]])
			bc = c - b
			norm_bc = np.linalg.norm(bc)
			angle_score[i * res_num + j, k] = np.sqrt(np.maximum(1.0 - np.matmul(ab, bc) / norm_ab / norm_bc ** 2, 0.0))
np.save('./Angle_Score.npy', angle_score)