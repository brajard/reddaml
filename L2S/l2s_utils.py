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
	return fig

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


