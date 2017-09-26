import tensorflow as tf
import numpy as np
import glob, os


# This is the path where the model is saved
# it can be a relative path, if script is in the same folder that contain the model data
inpath = 'sparse_model_batches_noisy/'


#####################################
#First we export the evaluation data#
#####################################

# First we create a list to save the steps with data
steps_list_eval = []

# First loop is over all the event files in the path
for event_file in glob.glob(inpath+'events*'):
    # Then we loop over all the events in the event file
    for e in tf.train.summary_iterator(event_file):
        # Then we loop over each value stored for each event
        for v in e.summary.value:
            # Now if the value is the histogram_eval then
            if v.tag == 'histogram_eval':
                # we append the step number to the list
                steps_list_eval.append(e.step)
                # We open a files for writing
                f = open('histogram_data_files_noisy/histogram_eval_'+str(e.step)+'.dat', 'w')
                # Loop over all buckets in the histogram
                for n in range(len(v.histo.bucket)-1):
                    # Write the histogram values to the file
                    f.write(str(v.histo.bucket_limit[n])+', '+str(v.histo.bucket[n])+'\n')
                # Remeber to always close the file
                f.close()

# Write a file with all the step numbers
f = open('histogram_data_files_noisy/histogram_eval_steps.dat', 'w')
for n in range(len(steps_list_eval)):
    f.write(str(steps_list_eval[n])+'\n')
f.close()

#############################
#Now we export training data#
#############################


# First we create the step list
steps_list_train = []

# Now we do the same loops
# The training summaries is saved in a different path, so we add 'histogram_summary/'
for event_file in glob.glob(inpath+'histogram_summary/events*'):
    for e in tf.train.summary_iterator(event_file):
        for v in e.summary.value:
            if v.tag == 'histogram_summary':
                # Appending the step number
                steps_list_train.append(e.step)
                # Opening file for writing
                f = open('histogram_data_files_noisy/histogram_training_'+str(e.step)+'.dat', 'w')
                for n in range(len(v.histo.bucket)-1):
                    f.write(str(v.histo.bucket_limit[n])+', '+str(v.histo.bucket[n])+'\n')
                # Remeber to always close the file
                f.close()


# Write a file with all the step numbers
f = open('histogram_data_files_noisy/histogram_training_steps.dat', 'w')
for n in range(len(steps_list_train)):
    f.write(str(steps_list_train[n])+'\n')
f.close()
