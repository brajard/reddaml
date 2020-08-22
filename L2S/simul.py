"""
Python script to run a simulation (incuding ensemble simulation) of a model (including truncated model)
run 'python simul.py -h' to see the usage
"""

# Load the python modules.
import os, sys
# Insert the common folder in the path (to be able to load python module)
sys.path.insert(0,os.path.join(os.path.pardir,'common'))
from toolbox import load_config, path
import argparse
import numpy as np
from sklearn.model_selection import ParameterGrid
from dapper import Chronology
from l2s_utils import default_param
import importlib
from tqdm import tqdm


# Parse the commande line
parser = argparse.ArgumentParser()
parser.add_argument('--paths',nargs='?',
	default='config/paths.yml',
	help='Configuration file containing the data directory')
parser.add_argument('--params',nargs='?',
	default='config/ref.yml',
	help='Configuration file containing the template names of the files to load/save')
parser.add_argument('--model',nargs='?',
	default='config/model_trunc.yml',
	help='Setup of the model')
args = parser.parse_args()

# Load the config files
paths = load_config(args.paths)
params = load_config(args.params)
model = load_config(args.model)
template = params['template']

#Paths where to save the results
#initfile is used to load (if restart=True) or save (if restart=False) the initial state
savedir = path(os.path.join(paths['rootdir'],params['savedir']))
simulfile = os.path.join(savedir,template[model['name']])
initfile = os.path.join(savedir,template['initname'])


#Check if init on restart or with spinup

if model['restart']:
	print('Starting from template file {}'.format(initfile))
	T_spinup = 0.

else:
	assert 'spinup' in model,\
	'Starting with spinup, please specity spinup time in {}'.format(args.configfile)
	T_spinup = float(model['spinup'])
	print('Starting after {} spinup time'.format(T_spinup))

#Parmaeters to be set for a simulation
used_paramter = {'N','T','seed'}

#List of values for the used paramters (should in a list)
lparam = {k:params.get(k,[default_param[k]]) for k in used_paramter}

#Sequence of all the combination of the parameters
seq_param = ParameterGrid(lparam)

#Import the model (in DAPPER HMM format)
model_module = __import__(model['model_module'])

if model['type'] == 'hybrid':
	HMM_trunc = getattr(model_module,model['model_name'])
	archi = model['archi']
	parnn = model['parnn']
	from l2s_utils import build_HMM_resolv
	HMM = build_HMM_resolv(archi, reg=parnn['reg'], batchlayer=parnn['batch_layer'], trunc=HMM_trunc)
	print("--> run a hybrid model")
else:
	HMM = getattr(model_module,model['model_name'])
	print("--> run a purely physical model")


#Loop over all the experiments
for dparam in seq_param:
	T = dparam['T']
	N = dparam['N']
	seed = dparam['seed']

	#Define chrono (spinup + simulation)
	HMM.t = Chronology(dkObs=1,T=T+T_spinup,dt=model['dt'])

	#First index of the simulation after spinup (=0 if initialized from an initial file)
	idx0 = int(T_spinup / HMM.t.dt)

	#Init the random seed
	np.random.seed(seed)

	if model['restart']: #TODO: NOT TESTED

		X0 = np.load(initfile.format(**dparam))

		if X0.ndim == 1:
			#If there is one member, make sure to have the first dimension==1
			assert N == 1, 'restart file {} has only one member({} are required)'.format(initfile.format(**dparam),N)
			X0 = X0[np.newaxis, :]
		#Crop the spatial dimension if needed
		X0 = X0[:,:HMM.Dyn.M]

		#Check if the dimension of the init == the dimension of the state
		assert X0.shape[1] == HMM.Dyn.M

		#Check if there the size of the initial ensemble fit the required size
		assert X0.shape[0] == N, 'restart file {} has only {} members({} are required)'.format(initfile.format(**dparam), X0.shape[0],
			N)

	else: #Used default initialisation of the HMM object
		X0 = HMM.X0.sample(N)

	#Dyn: dynamical model, t: chronology
	Dyn, chrono = HMM.Dyn, HMM.t

	#Initiaze to simulation to zeros (number of time step, size of the ensemble, size of the state)
	#Only work for 1D states vectors
	xx = np.zeros((chrono.K+1, N, Dyn.M))

	#Initial time step
	xx[0] = X0

	print('Size of the full dataset:{}'.format(xx[idx0:].shape[0]))

	for k, kObs, t, dt in tqdm(chrono.ticker):
		xx[k] = Dyn(xx[k - 1], t - dt, dt)

	#Save the simulations
	print("File saved:")

	np.savez(simulfile.format(**dparam), xx=xx[idx0:], T=T, dt=HMM.t.dt)
	print("->",simulfile.format(**dparam))
	if not model['restart']:
		np.save(initfile.format(**dparam), xx[idx0])
		print("->",initfile.format(**dparam))