# Plot L96d
import numpy as np
import matplotlib.pyplot as plt
import copy

from mpl_toolkits.axes_grid1 import make_axes_locatable
from dapper.mods.LorenzUV.lorenz95 import HMM_full, HMM_trunc as HMM_trunc_dapper
from dapper.mods.LorenzUV.lorenz95 import LUV

from dapper import with_rk4

# Default value of paramter (overwrited using the configuration files)
default_param = {'p': 36, 'std_o': 1, 'dtObs': 0.05, 'dt':0.01, 'Nfil_train':1, 'N': 20, 'seed': 10, 'T':15.}


def plot_L96_2D(xx,xxpred,tt,labels,vmin=None,vmax=None,vdelta=None):
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

