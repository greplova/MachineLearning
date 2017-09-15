# Imports
import numpy as np
import tensorflow as tf
import random as random

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

from load_data_ml import load_data
from load_params_ml import load_params
from filelist_ml import create_file_list

import sys
import subprocess


#start session, first two lines are to suppress warnings:
#https://github.com/tensorflow/tensorflow/issues/7778
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.logging.set_verbosity(tf.logging.INFO)


#We start by defining the model for 1D CNN
def onedim_model(x, labels, mode):
	#for the input layer we reshape the current
	input_layer = tf.reshape(x, [-1, N, 1])

	#Make 1d convolution:
	#filters is set to 32 -- this is arbritrary
	#Kernel size, Nk should be tau/dt
	#with tau the time-scale of dynamics
	#such that this essentially is integrating over
	#at least one jump
	#see: https://www.tensorflow.org/api_docs/python/tf/layers/conv1d
	conv1 = tf.layers.conv1d(
		inputs=input_layer,
		filters=16,
		kernel_size=[Nk],
		padding="same",
		activation=tf.nn.relu)
	#output_size = N - Nk + 1 #if padding = valid
	output_size = N



	#Pooling Layer
	#Nm is the length ver which we average
	#see: https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling1d
	pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=[Nm], strides=Nm)


	#Now we can add more layers

	#Conv layer no 2
	#I doubled the filters like in the mnist_deep.py
	conv2 = tf.layers.conv1d(
	inputs=pool1,
	filters=32, 
	kernel_size=[Nk],
	padding="same",
	activation=tf.nn.relu)
	#output_size2 = output_size/2 - Nk + 1 #if padding = valid
	output_size2 = output_size/Nm

	#Pooling layer no 2
	pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=[Nm], strides=Nm)


	#Afterwards we flatten
	pool1_flat = tf.reshape(pool2, [-1, (output_size2/Nm) * 32])

	# Dense Layer
	# Densely connected layer with 1024 neurons
	# Input Tensor Shape: [batch_size, (N/Nm) * 32]
	# Output Tensor Shape: [batch_size, 1024]
	dense = tf.layers.dense(inputs=pool1_flat, units=1024, activation=tf.nn.relu)

	# Add dropout operation; 0.6 probability that element will be kept
	dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

	# Logits layer
	# Input Tensor Shape: [batch_size, 1024]
	# Output Tensor Shape: [batch_size, Np, 3]
	logits = tf.reshape(tf.layers.dense(inputs=dropout, units=Np*3),[-1,3,Np])

	#These things are really just copy paste from here:
	# https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/layers/cnn_mnist.py
	loss = None
	train_op = None


	# Calculate Loss function
	# This is really the function that we want to minimize
	# and here it is the cross entropy
	# since cross entropy = 0 means everything is correctly predicted
	if mode != learn.ModeKeys.INFER:
		loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

	# Configure the Training Op
	# This defines the which optimizer to use and which settings
	# SGD is gradient descent optimizer
	if mode == learn.ModeKeys.TRAIN:
		train_op = tf.contrib.layers.optimize_loss(loss=loss, global_step=tf.contrib.framework.get_global_step(), learning_rate=0.000001, optimizer="Adam")


	# This uses a built in function to calculate accuracy
	calculated_accuracy = tf.metrics.accuracy(tf.argmax(input=labels, axis=2), tf.argmax(input=logits, axis=2))
	
	# The weasel function:
	# calculate the mean_squared_error of outputs vs. labels
	distance_squared = tf.abs(tf.argmax(input=labels, axis=2) - tf.argmax(input=logits, axis=2))
	error = tf.reduce_mean(distance_squared,axis=1)	
	
	# introduce minimum distance for which we will accept the result (in a tensor form)
	# threshold = tf.constant(1,shape=error.get_shape(),dtype=tf.float32)
	
	# now we compare error and threshold element-wise
	# overlap=tf.less(tf.cast(error,tf.float32), threshold)

	#assign floats and calculate accuracy
	# new_accuracy=tf.reduce_mean(tf.cast(overlap,tf.float32))
	
	# Calculate the estimated parameters:
	# make tensor of params:
	tf_omegas = tf.reshape(tf.constant(all_params,dtype=tf.float32),[3, Np])
	# calculate probabilities:
	probabilities = tf.nn.softmax(logits,dim=-1)

	# make the inner product 
	estimated_params = tf.reduce_sum(probabilities * tf_omegas,axis=2)

	# calculate input params from labels:
	true_params = tf.reduce_sum(labels * tf_omegas,axis=2)
	
	# Average distance in MHz:
	distance_params = tf.reduce_mean(tf.abs(estimated_params - true_params))

	# Distance in MHz for each trajectory:
	distance = tf.reduce_mean(tf.abs(estimated_params - true_params), axis=1)
	
	# Threshold for distance = 0.5 MHz
	threshold_distance = tf.constant(0.5,shape=distance.get_shape(),dtype=tf.float32)
	
	# now we compare error and threshold element-wise
	overlap_distance = tf.less(tf.cast(distance,tf.float32), threshold_distance)

	#assign floats and calculate accuracy
	distance_accuracy=tf.reduce_mean(tf.cast(overlap_distance,tf.float32))
	
	distance_summary = tf.summary.scalar(name="distance_training", tensor=distance_params)
	histogram_summary = tf.summary.histogram(name="histogram_summary", values=distance)

	histogram_eval = tf.summary.histogram(name="histogram_eval", values=distance)
	
	eval_metric_ops = { "tensorflow_accuracy": calculated_accuracy, "average_distance": distance_params,  "distance_accuracy": distance_accuracy }
	
	# Generate Predictions
	# this is the result of the neural network
	predictions = {
		"classes": tf.argmax(input=logits, axis=1),
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}


	# Return a ModelFnOps object
	return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)





#N is size of current
N = 50000
Nm = 2
Np = 6
Nk = 9

M = 1000
Meval = 1000


# parameter arrays:
# to be used for making prediction of parameters
gamma_downs = np.linspace(1,6,num=Np)
gamma_ups = np.linspace(1,6,num=Np)
omegas = np.linspace(4,10,num=Np)

all_params = np.asarray([gamma_downs, gamma_ups, omegas])


def convert_to_int(x):
	return int(x)
convert_array_to_int = np.vectorize(convert_to_int)

# make folder for temporary data
data_dir = "data_tmp/"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# input function defined as in:
# https://www.tensorflow.org/versions/master/tutorials/kernel_methods
def get_input_fn(rseed, batch_size):
	def _input_fn():
		# delete all files in data_dir
		# https://stackoverflow.com/questions/185936/delete-folder-contents-in-python
		for the_file in os.listdir(data_dir):
		    file_path = os.path.join(data_dir, the_file)
		    try:
			if os.path.isfile(file_path):
			    os.unlink(file_path)
		    except Exception as e:
			print(e)

		# run the matlab command
		# https://stackoverflow.com/questions/325463/launch-a-shell-command-with-in-a-python-script-wait-for-the-termination-and-ret
		cmd = ['./load_and_unpack', str(rseed+1)]
		p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
		p.wait()

		#Create the list of number where data exist
		print("File List preparation")
		file_list = create_file_list()
		file_list = convert_array_to_int(file_list)
		#M = len(file_list)


		np.random.shuffle(file_list)

		#print("LOADING PARAMETERS!!!")
		#params: [M,Np,3] tensor
		#Np is the grid size for each parameter
		#3 is the number of params
		params = load_params(file_list,Np,batch_size)

		#print("LOADING CURRENTS!!!")
		#current: [M,N] tensor,
		#M time-traces
		#N point (total time T = N dt)
		#reshape seems important here
		currents = load_data(file_list,batch_size)
		
		data = tf.constant(currents)
		labels = tf.constant(params)
		print(data)
		print(labels)
		
		return data, labels
	return _input_fn

eval_input_fn = get_input_fn(1, batch_size=Meval)


# A validation monitor, to see how well we are doing
# https://www.tensorflow.org/get_started/monitors 
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    input_fn=eval_input_fn,
    eval_steps=1,
    every_n_steps=50)



histogram_monitor = tf.contrib.learn.monitors.SummarySaver(
    summary_op="histogram_summary",
    save_steps=10,
    output_dir="sparse_model_batches_2/histogram_summary")




distance_monitor = tf.contrib.learn.monitors.SummarySaver(
    summary_op="distance_training",
    save_steps=10,
    output_dir="sparse_model_batches_2/distance_training")



# Create the Estimator
dot_classifier = learn.Estimator(
	model_fn=onedim_model,
	model_dir="sparse_model_batches_2",
	config=tf.contrib.learn.RunConfig(num_cores=28,save_checkpoints_steps=50,save_checkpoints_secs=None,keep_checkpoint_max=1))



print("START TRAINING!")
for step in range(200):
	# Train the model
	train_input_fn = get_input_fn(step, batch_size=M)
	dot_classifier.fit(input_fn=train_input_fn
		,steps = 250
		,monitors = [histogram_monitor, validation_monitor, distance_monitor]
		)

	
print("TRAINING FINISHED!")


print("START EVALUATING!")
# Evaluate the model and print results
eval_results = dot_classifier.evaluate(
	input_fn=eval_input_fn,
	steps=1)

print(eval_results)
