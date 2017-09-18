# Imports

import numpy as np
import tensorflow as tf


current_0 = 5.0
current_ud = 4.0
current_mean = (current_ud + current_0)/2
current_diff = (current_0 - current_ud)

def load_data(file_list,M):
	data_file = 'data_tmp/current_'+str(file_list[0])+'.dat'
	load_data = np.genfromtxt(data_file, delimiter='  ', defaultfmt='%e')
	full_data = np.array([load_data])

	for i in range(2,M):
		data_string = 'data_tmp/current_'+str(file_list[i])+'.dat'
		load_data = np.genfromtxt(data_string, delimiter='  ', defaultfmt='%e')
		renorm_data = (load_data - current_mean)/current_diff
		full_data = np.append(full_data, [renorm_data], axis=0)


	return np.float32(full_data)

