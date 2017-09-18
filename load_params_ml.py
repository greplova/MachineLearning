# Imports

import numpy as np
import tensorflow as tf

def load_params(file_list,Np,M):
	data_file = 'data_tmp/params_'+str(file_list[0])+'.dat'
	load_data = np.genfromtxt(data_file, delimiter='  ', defaultfmt='%e')

	gamma_down = load_data[0]
	gamma_up = load_data[1]
	omega = load_data[2]


	gamma_downs = np.linspace(1,6,num=Np)
	gamma_ups = np.linspace(1,6,num=Np)
	omegas = np.linspace(4,10,num=Np)

	gamma_down_index = np.argmin(abs(gamma_down-gamma_downs))
	gamma_downs_onehot = np.full(Np,0)
	gamma_downs_onehot[gamma_down_index] = 1

	gamma_up_index = np.argmin(abs(gamma_up-gamma_ups))
	gamma_ups_onehot = np.full(Np,0)
	gamma_ups_onehot[gamma_up_index] = 1

	omega_index = np.argmin(abs(omega-omegas))
	omegas_onehot = np.full(Np,0)
	omegas_onehot[omega_index] = 1

	full_data=np.array([[gamma_downs_onehot,gamma_ups_onehot, omegas_onehot]])

	for i in range(2,M):
		data_string = 'data_tmp/params_'+str(file_list[i])+'.dat'
		load_data = np.genfromtxt(data_string, delimiter='  ', defaultfmt='%e')

		gamma_down = load_data[0]
		gamma_up = load_data[1]
		omega = load_data[2]


		gamma_downs = np.linspace(1,6,num=Np)
		gamma_ups = np.linspace(1,6,num=Np)
		omegas = np.linspace(4,10,num=Np)

		gamma_down_index = np.argmin(abs(gamma_down-gamma_downs))
		gamma_downs_onehot = np.full(Np,0)
		gamma_downs_onehot[gamma_down_index] = 1

		gamma_up_index = np.argmin(abs(gamma_up-gamma_ups))
		gamma_ups_onehot = np.full(Np,0)
		gamma_ups_onehot[gamma_up_index] = 1

		omega_index = np.argmin(abs(omega-omegas))
		omegas_onehot = np.full(Np,0)
		omegas_onehot[omega_index] = 1
	
		onehot_data = np.array([gamma_downs_onehot,gamma_ups_onehot, omegas_onehot])

		full_data = np.append(full_data, [onehot_data], axis=0)

	#full_data = np.transpose(full_data,axes=(0,2,1))

	return np.float32(full_data)
