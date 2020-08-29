"""
Python script to train a neural network model of the model error
run 'python train.py -h' to see the usage
"""

# Load the python modules.
import os, sys
# Insert the common folder in the path (to be able to load python module)
sys.path.insert(0,os.path.join(os.path.pardir,'common'))
from toolbox import load_config, path
import argparse
import numpy as np
import random as rn

from sklearn.model_selection import ParameterGrid
from l2s_utils import default_param, buildmodel
import tensorflow.keras.backend as K
import tensorflow as tf
# Parse the commande line
parser = argparse.ArgumentParser()
parser.add_argument('--paths',nargs='?',
	default='config/paths.yml',
	help='Configuration file containing the data directory')
parser.add_argument('--params',nargs='?',
	default='config/ref.yml',
	help='Configuration file containing the template names of the files to load/save')
args = parser.parse_args()

# Load the config files
paths = load_config(args.paths)
params = load_config(args.params)
template = params['template']
parnn = params['parnn']

#Paths where to save the results
savedir = path(os.path.join(paths['rootdir'],params['savedir']))
weightsdir = path(os.path.join(paths['rootdir'],params['weightsdir']))


#File of the trainingset
file_train = os.path.join(savedir,template['train'])

#File template containing the weight
file_weights = os.path.join(weightsdir,template['weights'])

#File template to save the history
file_history = os.path.join(savedir, template['history'])

used_paramter = { 'p', 'std_o', 'dtObs', 'std_m' ,'N','T','seed','Nfil_train'}

#List of values for the used paramters (should in a list)
lparam = {k:params.get(k,[default_param[k]]) for k in used_paramter}

#Sequence of all the combination of the parameters
seq_param = ParameterGrid(lparam)

#Sequence to ignore at the beginning of the training set (DA spinup & filter borders)
burn = params['burn']
nn = len(seq_param)
for i, dparam in enumerate(seq_param):
	print('Training {}/{}:'.format(i+1,nn),dparam)
	# Load the dataset
	data = np.load(file_train.format(**dparam))
	xx = data['x'][burn:]
	yy = data['y'][burn:]

	ival = params['ival']
	itrain = np.min((params['maxtrain'], xx.shape[0] + np.abs(ival) - 1))

	# Training is taken at the begininng of the time series, Vaidation at the end
	xx_train, yy_train = xx[:itrain], yy[:itrain]
	xx_val, yy_val = xx[ival:], yy[ival:]

	# Define the NN model
	K.clear_session()

	# Inialize random generation numbers
	np.random.seed(parnn['seed'])
	rn.seed(parnn['seed'])
	os.environ['PYTHONHASHSEED']=str(parnn['seed'])
	tf.random.set_seed(parnn['seed'])

	model = buildmodel(params['archi'],reg=parnn['reg'],batchlayer=parnn['batch_layer'])
	model.compile(loss='mse', optimizer=parnn['optimizer'])
	verbose = parnn.get('verbose', 1)

	#Train the NN
	history = model.fit(xx_train, yy_train,
		epochs=parnn['epochs'],
		batch_size=parnn['batch_size'],
		validation_data=(xx_val, yy_val),
		verbose=verbose)

	# SAVE HISTORY
	np.savez(file_history.format(**dparam),
		loss=history.history['loss'],
		val_loss=history.history['val_loss'])

	# SAVE WEIGHTS
	model.save_weights(file_weights.format(**dparam))