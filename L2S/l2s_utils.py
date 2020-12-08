# Plot L96d
import numpy as np
import matplotlib.pyplot as plt
import copy

from mpl_toolkits.axes_grid1 import make_axes_locatable
from dapper.mods.LorenzUV.lorenz95 import HMM_full, HMM_trunc as HMM_trunc_dapper
from dapper.mods.LorenzUV.lorenz95 import LUV

from dapper import with_rk4, ens_compatible, Id_mat, GaussRV, Operator

# Default value of paramter (overwrited using the configuration files)
default_param = {'p': 36, 'std_o': 1, 'dtObs': 0.05, 'dt':0.01, 'Nfil_train':1, 'N': 20, 'seed': 10, 'T':15.,
                 'std_m':0.06}


def plot_L96_2D(xx,xxpred,tt,labels,vmin=None,vmax=None,vdelta=None):
	"""
	Plot a comparison between two L96 simulations
	"""
	if vmin is None:
		vmin,vmax = np.nanmin(xx),np.nanmax(xx)
	if vdelta is None:
		vdelta = np.nanmax(np.abs(xxpred-xx))
	m = xx.shape[1]
	tmin = tt[0]
	tmax = tt[-1]
	fig,ax = plt.subplots(nrows=3,sharex='all')

	divider = [make_axes_locatable(a) for a in ax]

	cax = dict()
	for i in range(3):
		cax [i] = divider[i].append_axes('right', size='5%', pad=0.05)

	delta= dict()
	delta[0] = ax[0].imshow(xx.T,vmin=vmin,vmax =vmax,cmap=plt.get_cmap('viridis'),extent=[tmin,tmax,0,m],aspect='auto')
	delta[1] = ax[1].imshow(xxpred.T,vmin=vmin,vmax=vmax,cmap=plt.get_cmap('viridis'),extent=[tmin,tmax,0,m],aspect='auto')
	delta[2] = ax[2].imshow(xxpred.T- xx.T,cmap=plt.get_cmap('bwr'),
		extent=[tmin,tmax,0,m],aspect='auto',vmin=-vdelta,vmax=vdelta)
	ax[0].set_ylabel(labels[0])
	ax[1].set_ylabel(labels[1])
	ax[2].set_ylabel(labels[1][:2] + ' - ' + labels[0][:2] )
	for i in delta:
		fig.colorbar(delta[i],cax=cax[i],orientation='vertical')
	return fig, ax

def other():
    print()

# trunc HMM (no param)
HMM_trunc = copy.deepcopy(HMM_trunc_dapper)
HMM_trunc.Dyn.model = with_rk4(LUV.dxdt_trunc, autonom=True)

import numpy as np
from dapper import ens_compatible, Id_mat, GaussRV, Operator


#TODO: allow list of index instead of maxind
class Observator:
	"""This class handles sparse observations. It allows to create a dapper-compatible observation operators
	"""
	def __init__(self,  t, m, max_ind=None, std_o=1, p=None, sample='random', seed_obs=2, prec=10000):
		"""
		Input arguments:
		t: dapper chronoolgy of the experiment
		m: size of the state
		max_ind: maximum index of observations locations
		std_o: standard deviation of observation noise
		p: number of observation at observational time
		sample: 'random': draw randomly p observation at each time step, 'regular': regularly sample the observation
		seed_obs: seed for the random generator
		prec: fixed precision on time step (t*1000 should be int) in order to have integer as dictionnary keys.
		"""
		self.t_ = t
		if max_ind is None:
			max_ind = m
		if p is None:
			p = max_ind
		self.m = m
		self.max_ind = max_ind
		self.std_o = std_o
		self.p = p
		self.sample = sample
		self.seed_obs = seed_obs
		self.prec = prec  # Precision on time stamp (*1000 should be int)
		self.tinds = dict()
		self.compute_tinds()

	def compute_tinds(self):
		self.tinds = dict()
		save_state = np.random.get_state()
		np.random.seed(self.seed_obs)
		for k, KObs, t_, dt in self.t.ticker:
			if KObs is not None:
				if self.sample == 'random':
					self.tinds[int(self.prec*t_)] = np.sort(np.random.choice(self.max_ind, size=self.p, replace=False))
				elif self.sample == 'regular':
					self.tinds[int(self.prec*t_)] = np.linspace(0, self.max_ind, self.p, endpoint=False, dtype=np.int)
		np.random.set_state(save_state)

	def def_hmod(self):
		@ens_compatible
		def hmod(ensemble, t):
			return ensemble[self.tinds[int(self.prec*t)]]
		return hmod

	def h_dict(self):
		h = {'M': self.p,
			'model': self.def_hmod(),
			'jacob': Id_mat(self.p),
			'noise': GaussRV(C=self.std_o * np.eye(self.p))}
		return h

	def h_operator(self):
		h = self.h_dict()
		return Operator(**h)
    
	@property
	def t( self ):
		return self.t_

	@t.setter
	def t( self, value ):
		self.t_ = value
		self.compute_tinds()

from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Layer
from tensorflow.keras import regularizers
from tensorflow.keras import Model
from  tensorflow.keras import backend as K

def buildmodel(archi, m=36, reg=0.0487, batchlayer=1):
	inputs = Input(shape=(m,1))
	border = int(np.sum(np.array([kern//2 for nfil,kern,activ in archi])))
	x = Periodic1DPadding(padding_size=border)(inputs)
	x = BatchNormalization()(x)
	for i, (nfil, kern, activ) in enumerate(archi):
		if i == batchlayer:
			x = BatchNormalization()(x)
		x = Conv1D(nfil, kern, activation=activ)(x)
		#x = BatchNormalization()(x)
	output= Conv1D(1,1,activation='linear', kernel_regularizer=regularizers.l2(reg))(x)
	return Model(inputs,output)

def build_HMM_resolv(archi, m=36, reg=0.0487, batchlayer=1, weightfile=None, trunc=None):
	"""
	Build a hybrid model combining a physical core (trunc) and NN part
	:param archi: architecture
	:param m: size of the state
	:param reg: regularization parameter of the model
	:param batchlayer: position of the batch layer in the architecture
	:param weightfile: (optional) file containing the weight model
	:param trunc: truncated model in the dapper format
	:return: hybrid model in the dapper format
	"""
	if trunc is None:
		trunc = HMM_trunc
	HMM_resolv = copy.deepcopy(trunc)
	model_nn = buildmodel(archi, m=m, reg=reg, batchlayer=batchlayer)
	if weightfile is not None:
		model_nn.load_weights(weightfile)
	def step(x0,t0,dt):
		physical_step = with_rk4(LUV.dxdt_trunc, autonom=True)
		ml_step = model_nn.predict
		output = physical_step(x0,t0,dt) + dt*ml_step(x0[...,np.newaxis]).squeeze()
		return output
	HMM_resolv.Dyn.model = step
	HMM_resolv.Dyn.nn = model_nn
	return HMM_resolv


class Periodic1DPadding(Layer):
	"""Add a periodic padding to the output

	# Arguments
		padding_size: tuple giving the padding size (left, right)

	# Output Shape
		input_shape+left+right
	"""


	def __init__ (self, padding_size, **kwargs):
		super(Periodic1DPadding, self).__init__(**kwargs)
		if isinstance(padding_size, int):
			padding_size = (padding_size, padding_size)
		self.padding_size = tuple(padding_size)

	def compute_output_shape( self, input_shape ):
		space = input_shape[1:-1]
		if len(space) != 1:
			raise ValueError ('Input shape should be 1D with channel at last')
		new_dim = space[0] + np.sum(self.padding_size)
		return (input_shape[0],new_dim,input_shape[-1])



	def build( self , input_shape):
		super(Periodic1DPadding,self).build(input_shape)

	def call( self, inputs ):
		vleft, vright = self.padding_size
		leftborder = inputs[:, -vleft:, :]
		rigthborder = inputs[:, :vright, :]
		return K.concatenate([leftborder, inputs, rigthborder], axis=-2)
