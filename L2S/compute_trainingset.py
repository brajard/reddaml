"""
Python script to compute a training set for a machine learning method. If the training is computed from observation, a DA algorithm is applied.
run 'python compute_trainingset.py -h' to see the usage
"""


# Load the python modules.
import os, sys
import argparse
import numpy as np
from tqdm import tqdm
from l2s_utils import default_param, Observator
from sklearn.model_selection import ParameterGrid
from dapper import EnKS, EnKF_N, with_recursion, print_averages, Chronology, GaussRV

# Insert the common folder in the path (to be able to load python module)
sys.path.insert(0,os.path.join(os.path.pardir,'common'))

from toolbox import load_config, path, my_lowfilter

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

#Full path of files
savedir = path(os.path.join(paths['rootdir'],params['savedir']))

#File of the true simulations (used to produce observations)
file_truth = os.path.join(savedir,template['truth'])

#File of the trainingset
file_train = os.path.join(savedir,template['train'])

#Parameters to be set for a simulation
if params['obstype'] == 'perfect':
	used_paramter = {'N','T','seed','dtObs'}
	print('--> use perfect observations (no DA)')
else:
	used_paramter = { 'p', 'std_o', 'dtObs', 'std_m' ,'N','T','seed','Nfil_train'}
	damethod = params['damethod']
	paramda = params['paramda']
	# Output of the DA
	file_da = os.path.join(savedir, template['da'])
	print('--> use noisy/sparse observations')

#List of values for the used paramters (should in a list)
lparam = {k:params.get(k,[default_param[k]]) for k in used_paramter}

#Sequence of all the combination of the parameters
seq_param = ParameterGrid(lparam)

#Load the model
dt = model['dt']
model_module = __import__(model['model_module'])

if model['type'] == 'physical':

	HMM_trunc = getattr(model_module, model['model_name'])
	trunc_model = with_recursion(HMM_trunc.Dyn.model)


else:
	raise NotImplementedError("Only physical is available as model type")

nn = len(seq_param)
for i, dparam in enumerate(seq_param):
	print('Experiment {}/{}:'.format(i+1,nn),dparam)
	#Load True simulation
	data_truth = np.load(file_truth.format(**dparam))
	xx = data_truth['xx']
	dt_full = data_truth['dt']
	T = data_truth['T']
	dtObs = dparam['dtObs']


	#Modify chrono
	HMM_trunc.t = Chronology(T=T, dt=dt, dtObs=dtObs)
	assert HMM_trunc.t.T == T, 'chrono inconsitent, T={}, chrono.T={}'.format(T, HMM_trunc.t.T)
	dk = int(HMM_trunc.t.dt / dt_full)

	#Subsample truth
	xxnU = xx[::dk,0,:HMM_trunc.Dyn.M]


	if params['obstype'] == 'perfect':
		xx_train = xxnU[HMM_trunc.t.kkObs]

	else:
		p = dparam['p']
		std_o = dparam['std_o']
		std_m = dparam['std_m']

	# Modify observator
		obs = Observator(t=HMM_trunc.t, p=p, std_o=std_o, m=HMM_trunc.Dyn.M, sample='random')
		HMM_trunc.Obs = obs.h_operator()

	# Modify model noise
		sigma_m = 0.06 / np.sqrt(HMM_trunc.t.dt)  # because it is multiplied later on in DAPPER
		HMM_trunc.Dyn.noise = GaussRV(C=sigma_m)

		#Set the DA algorithm

		if damethod == 'EnKF_N':
			configda = EnKF_N(**paramda,liveplotting=False)
		elif damethod == 'EnKS':
			configda = EnKS(**paramda, liveplotting=False)
		else:
			raise NotImplementedError('DA method '+damethod+ ' is not implemented')
		np.random.seed(params['daseed'])
		Obs, chrono = HMM_trunc.Obs, HMM_trunc.t
		yy = np.zeros((chrono.KObs+1, Obs.M))
		xxobs = np.zeros((chrono.KObs + 1, Obs.M))
		for k, kObs, t, dt_ in tqdm(chrono.ticker, desc='Generate Obs'):
			if kObs is not None:
				yy[kObs] = Obs(xxnU[k], t) + Obs.noise.sample(1)
				xxobs[kObs] = Obs(xxnU[k], t)
		stats = configda.assimilate(HMM_trunc, xxnU, yy)

		np.savez(file_da.format(**dparam), mua=stats.mu.a, muf=stats.mu.f, mus=stats.mu.s, vara=stats.var.a,
			varf=stats.var.f, vars=stats.var.s, infl=stats.infl)

		# Stats
		avrgs = stats.average_in_time()
		print_averages(configda, avrgs, [], ['rmse_a', 'rmv_a'])

		xx_train = stats.mu.a
		if dparam['Nfil_train']>1:
			print('Filtering data with size',dparam['Nfil_train'])
			xx_train = my_lowfilter(xx_train, dparam['Nfil_train'])

	# Compute traininset

	#Input of the dataset
	xx_in = xx_train[:-1]

	# Estimation of the true value after dtObs MTU
	xx_out = xx_train[1:]

	# Truncated value after dtObs MTU
	xx_trunc = trunc_model(xx_in, HMM_trunc.t.dkObs, np.nan, dt)[-1]

	# Estimation of the model error
	delta = (xx_out - xx_trunc) / dtObs

	#Save the training file
	np.savez(file_train.format(**dparam), x=xx_in, y=delta, dtObs=dtObs)
