# Imports

import numpy as np
import os, glob

def create_file_list():
	file_list = []
	for file_name in glob.glob("data_tmp/params_*.dat"):
		#print(file_name)
		file_tmp = file_name.split("_")
		file_tmp = file_tmp[2]
		file_tmp = file_tmp.split(".")
		file_list.append(file_tmp[0])
	return file_list

