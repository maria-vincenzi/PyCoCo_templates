import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from scipy import interpolate
from scipy import integrate
import matplotlib.cm as cm
import scipy
import scipy.optimize as opt
from scipy.optimize import minimize
from itertools import cycle
from scipy.interpolate import griddata


import george
from george.kernels import Matern32Kernel


import sys
sys.path.insert(0, '/Users/mariavincenzi/PhD/pycoco_2/')
import pycoco_general_info as PyCoCo_info

mycmap = plt.cm.viridis
mycmap.set_under('r')


def prepare_grid(snname, GP2DIM_Class):
	lcfit = GP2DIM_Class.open_LCfit_file()
	GP2DIM_Class.get_filter_LC()
	xa, ya, grid_nt, griderr_nt = GP2DIM_Class.grid_all_spectraltimeseries()
	xa_ext, ya_ext, grid_ext, griderr_ext = GP2DIM_Class.extend_grid_all_spectraltimeseries()
	
	raw_numbers = grid_ext.values
	raw_numbers_err = griderr_ext.values
	off_xa = xa_ext
	off_ya = ya_ext
	return (raw_numbers, raw_numbers_err, off_xa, off_ya, grid_ext.columns.values)


def transform2LOG_reshape(GP2DIM_Class, raw_numbers, raw_numbers_err,  off_xa, off_ya):
	
	# Trasform to LOG
	data_lin = (raw_numbers.T.reshape(raw_numbers.shape[0]*raw_numbers.shape[1]))
	data_lin_err = (raw_numbers_err.T.reshape(raw_numbers_err.shape[0]*raw_numbers_err.shape[1]))
	
	data = np.copy(data_lin)
	data[data_lin<=0.] = np.nan
	
	#if LOG:
	data = np.log(data)
	data_err = (data_lin_err/data_lin)
	#else:
	#	data_err = np.copy(data_lin_err)
	
	data_err[data_lin<=0.] = np.nan
	
	offset = np.min(data[~np.isnan(data)])
	scale_factor = np.median(data[~np.isnan(data)] - offset)
	
	data_scaled = ((data - offset)/scale_factor)
	data_error_scaled = (data_err)/scale_factor#/((data - offset))

	## Reshape the grid to feed the 2dGP
	x = []
	for i in range(raw_numbers.shape[1]):
		x = np.concatenate([x, np.arange(0,raw_numbers.shape[0],1)])
	y = []
	for i in range(raw_numbers.shape[1]):
		y = np.concatenate([y, np.ones(raw_numbers.shape[0])*i])
	
	resh_wls = []
	for i in range(raw_numbers.shape[1]):
		resh_wls = np.concatenate([resh_wls, off_xa])
	
	resh_mjd = []
	for i in off_ya:
		resh_mjd = np.concatenate([resh_mjd, np.ones(len(off_xa))*i])
	
	NOT_Isnan = (~np.isnan(data_scaled))&(~np.isnan(data_error_scaled))
	Isnan = (np.isnan(data_scaled))|(np.isnan(data_error_scaled))
	
	x1_data = resh_wls[NOT_Isnan]
	x2_data = resh_mjd[NOT_Isnan]
	
	x_tuble_nonan = np.array([i for i in zip(x1_data, x2_data)])
	y_data_nonan = np.copy(data_scaled[NOT_Isnan])
	y_data_nonan_err = np.copy(data_error_scaled[NOT_Isnan])
			
	norm1 = 11000.#max(x1_data)
	offset2 = min(x2_data)
	norm2 = max(x2_data-offset2)
	
	x1_data_norm = (x1_data/norm1)
	x2_data_norm = (x2_data-offset2)/norm2
	
	GP2DIM_Class.grid_norm_info = {'offset':offset, 'scale_factor':scale_factor,
									 'norm1':norm1, 'norm2':norm2, 'offset2':offset2}
	return (y_data_nonan, y_data_nonan_err, x1_data_norm, x2_data_norm)


def make_plots(GP2DIM_Class, y_data_nonan, y_data_nonan_err, x1_data_norm, x2_data_norm):	
	fig=plt.figure(1, figsize=(10,3))

	plt.subplot(121)
	plt.xlabel('MJD')
	plt.ylabel('wls')
	plt.title('Training Data')
	
	plt.grid(True)
	
	plt.scatter(GP2DIM_Class.grid_norm_info['norm2']*x2_data_norm, 
		GP2DIM_Class.grid_norm_info['norm1']*x1_data_norm, marker='s', s=9,c=y_data_nonan)
	plt.colorbar(label='Flux rescaled')
	#plt.savefig('gaussian_processes_2d_training_data.png', bbox_inches='tight')
	
	plt.subplot(122)
	plt.xlabel('MJD')
	plt.ylabel('wls')
	#plt.xlim(x1_min,x1_max)
	#plt.ylim(x2_min,x2_max)
	plt.title('Training Data ERRORS')
	
	plt.grid(True)
		
	plt.scatter(GP2DIM_Class.grid_norm_info['norm2']*x2_data_norm, 
		GP2DIM_Class.grid_norm_info['norm1']*x1_data_norm,  marker='s', s=9, c=np.log10(y_data_nonan_err))
	plt.colorbar(label='Err Flux rescaled')
	plt.show()
	fig.savefig(GP2DIM_Class.save_plot_path+'/data_for2d_interpolation.pdf', bbox_inches='tight')
	plt.close(fig)
	
def setPRIOR(GP2DIM_Class, type_):

	norm1 = GP2DIM_Class.grid_norm_info['norm1']
	norm2 = GP2DIM_Class.grid_norm_info['norm2']
	offset = GP2DIM_Class.grid_norm_info['offset']
	offset2 = GP2DIM_Class.grid_norm_info['offset2']
	scale_factor = GP2DIM_Class.grid_norm_info['scale_factor']

	if type_ in ['II', 'IIn', 'IIP', 'IIL']:
		PRIOR = '/Users/mariavincenzi/PhD/pycoco_2/ipython_notebook test/prior_Hrich.txt'
	else:
		PRIOR = '/Users/mariavincenzi/PhD/pycoco_2/ipython_notebook test/prior_SE.txt'
	wls_prior, phase_prior, color_prior = np.genfromtxt(PRIOR, delimiter=',', unpack=True)
	wls_prior_norm = wls_prior/norm1
	
	#DATALC_PATH+'/results_template/%s/fitted_phot_%s.dat'%(snname,snname)

	original_fit = pd.read_csv(GP2DIM_Class.path_fit_phot ,delimiter='\t')
	original_fit.dropna(subset=['Bessell_V'], inplace=True)
	Vflux = original_fit['Bessell_V'].values
	if 'Bessell_B' in original_fit.columns: BVflux = original_fit['Bessell_V'].values + original_fit['Bessell_B'].values
	else: BVflux = original_fit['Bessell_V'].values + original_fit['swift_B'].values
	peak = (original_fit.MJD.values[np.argmax(BVflux[~np.isnan(BVflux)])])
	phase_prior_norm = ((phase_prior+int(peak))-offset2)/norm2
	
	reshaped_color_prior = color_prior.reshape(len(np.unique(wls_prior)),len(np.unique(phase_prior)))
	Vflux_phase = np.interp(np.unique(phase_prior), original_fit.MJD-peak,Vflux)
	flux_prior = (reshaped_color_prior*Vflux_phase)
	flux_prior_transform = (np.log(flux_prior)-offset)/scale_factor
	points = np.array([tup for tup in zip(wls_prior_norm, phase_prior_norm)])
	values = (flux_prior_transform).reshape(len(np.unique(phase_prior))*len(np.unique(wls_prior)))
	return points, values

def run_2DGP_GRID(GP2DIM_Class, y_data_nonan, y_data_nonan_err, x1_data_norm, x2_data_norm,\
		kernel_wls_scale, kernel_time_scale, extrap_mjds, prior=False, points=np.nan, values=np.nan):
	
	""" ## for NUV extention:   extrap_mjds = grid_ext_columns
	## for spectra augmentation: 
	extrap_mjds = grid_ext.columns.values
	 if (len(extrap_mjds)>200):
		 extrap_mjds = grid_ext.columns.values[:200]
	 if (max(extrap_mjds-min(extrap_mjds))>200):
		 extrap_mjds = extrap_mjds[extrap_mjds-min(extrap_mjds)<200]
	 
	 tot_iteration = int(len(extrap_mjds)/slot_size+1)
	 print (tot_iteration)"""

	# TRAINING: X, y, terr
	norm1 = GP2DIM_Class.grid_norm_info['norm1']

	if prior:
		from george.modeling import Model

		class Model_2dim(Model):
			parameter_names = ()
			def get_value(self, t):
				points_eval = np.array([tup for tup in zip(t[:,0], t[:,1])])
				grid_z1 = griddata(points, values, points_eval, method='nearest')
				grid_z1[np.isnan(grid_z1)] = 0.
				plt.plot(t[:,0]*norm1, grid_z1, '-b', label='PRIOR')
				#if grid_z1.shape[0]>1000: plt.show()
				return grid_z1
    	
		mean_model = Model_2dim()

	X = np.vstack((x1_data_norm, x2_data_norm)).T
	y = y_data_nonan
	yerr = y_data_nonan_err
	
	kernel_mix = Matern32Kernel([kernel_wls_scale, kernel_time_scale], ndim=2)
	kernel2dim = np.var(y)*kernel_mix #+ 0.3*np.var(y)*kernel2*kernel1
	
	if prior: gp = george.GP(kernel2dim, mean=mean_model)  #, fit_mean=True, fit_white_noise=True)
	else:  gp = george.GP(kernel2dim)

	gp.compute(X, yerr)
		
	wls_normed_range = np.sort(np.concatenate(( np.arange(1600.,3000., 40),
											  np.arange(3000.,9000., 10),
											  np.arange(9000.,10350., 40))))/GP2DIM_Class.grid_norm_info['norm1']

	mu_fill_resh = []
	std_fill_resh = []
	
	# check that extrapolation is not over 200 new spectra
	#if (len(extrap_mjds)>200):
	#	 extrap_mjds = grid_ext.columns.values[:200]
	#if (max(extrap_mjds-min(extrap_mjds))>200):
	#	 extrap_mjds = extrap_mjds[extrap_mjds-min(extrap_mjds)<200]
	 
	slot_size = 3
	tot_iteration = int(len(extrap_mjds)/slot_size+1)

	for j in range(int(len(extrap_mjds)/slot_size+1)):
		mjd_normed_range = ((extrap_mjds[j*slot_size:(j+1)*slot_size])-GP2DIM_Class.grid_norm_info['offset2'])/GP2DIM_Class.grid_norm_info['norm2']
		x1_fill = []#np.random.permutation(np.linspace(0,1., N))
		x2_fill = []#np.random.permutation(np.linspace(0,1., N))
		for i in wls_normed_range:
			for k in mjd_normed_range:
				x1_fill.append(i)
				x2_fill.append(k)
		
		x1_fill=np.array(x1_fill) 
		x2_fill=np.array(x2_fill)
		
		X_fill = np.vstack((x1_fill, x2_fill)).T	
		if GP2DIM_Class.verbose: 
			print (j, 'of', int(len(extrap_mjds)/slot_size+1))
		frac_tot_iteration = int(20.*(j+1)/tot_iteration)
		#print('[','*'*frac_tot_iteration,' '*(20-frac_tot_iteration),']' + ' %i of %i'%(slot_size*(j+1),slot_size*tot_iteration)+' spec extrapolated', end='\r')

		mu_iter, cov_iter = (gp.predict(y, X_fill, return_cov=True))
		std_iter = np.sqrt(np.diag(cov_iter))
		
		plt.plot(x1_fill*norm1, mu_iter, '-k', label='PREDICTION')
		#plt.grid()
		#plt.legend()
		plt.show()

		mu_resh_iter = mu_iter.reshape(len(wls_normed_range), len(mjd_normed_range))
		std_resh_iter = std_iter.reshape(len(wls_normed_range), len(mjd_normed_range))

		if mu_fill_resh==[]:
			mu_fill_resh = np.copy(mu_resh_iter)
			std_fill_resh = np.copy(std_resh_iter)
		else:
			mu_fill_resh = np.concatenate([mu_fill_resh, mu_resh_iter], axis=1)
			std_fill_resh = np.concatenate([std_fill_resh, std_resh_iter], axis=1)

	print('[','*'*frac_tot_iteration,' '*(20-frac_tot_iteration),']' + '%i of %i'%(slot_size*(j+1),slot_size*tot_iteration)+'spec extrapolated')
	mu_fill = mu_fill_resh.reshape(len(wls_normed_range)*len(extrap_mjds))
	std_fill = std_fill_resh.reshape(len(wls_normed_range)*len(extrap_mjds))

	mjd_normed_range = (extrap_mjds-GP2DIM_Class.grid_norm_info['offset2'])/GP2DIM_Class.grid_norm_info['norm2']
	
	x1_fill = []#np.random.permutation(np.linspace(0,1., N))
	x2_fill = []#np.random.permutation(np.linspace(0,1., N))
	for i in wls_normed_range:
		for k in mjd_normed_range:
			x1_fill.append(i)
			x2_fill.append(k)
	
	x1_fill=np.array(x1_fill) 
	x2_fill=np.array(x2_fill)
	
	print ('EXTENDING SPECTRA BETWEEN:')
	print ('WLS:', min(x1_fill*GP2DIM_Class.grid_norm_info['norm1']), max(x1_fill*GP2DIM_Class.grid_norm_info['norm1']))
	print ('MJD:', min(x2_fill*GP2DIM_Class.grid_norm_info['norm2']), max(x2_fill*GP2DIM_Class.grid_norm_info['norm2']))

	return (x1_fill, x2_fill, mu_fill, std_fill)


def make_results_plots(GP2DIM_Class, x1_fill, x2_fill, mu_fill, std_fill):
	norm1 = GP2DIM_Class.grid_norm_info['norm1']
	norm2 = GP2DIM_Class.grid_norm_info['norm2']
	offset = GP2DIM_Class.grid_norm_info['offset']
	offset2 = GP2DIM_Class.grid_norm_info['offset2']
	scale_factor = GP2DIM_Class.grid_norm_info['scale_factor']

	#plt.scatter(norm2*x2_fill, norm1*x1_fill, marker='.', c=mu_fill, alpha=1., 
	#		vmin=0., cmap = mycmap)
	##plt.scatter(x2_data_norm, x1_data_norm, marker='s', c=y_data)
	##plt.scatter(x2_data_norm, x1_data_norm, marker='s', c=y_data)
	#plt.xlabel('MJD')
	#plt.ylabel('wls')
	#plt.colorbar()
	
	# PLOT xWLS LC and check how smooth the time variation in each single wls is:
	fit_wls = (np.unique(x1_fill)[::10])
	len_wls = len(fit_wls)
	color=cycle(plt.cm.gnuplot(np.linspace(0.05,0.95,len_wls)))
	
	fig = plt.figure(figsize=(10,6))
	plt.subplot(221)
	plt.title('from %.1f to %.1f'%(min(fit_wls[:int(len_wls/4)]*norm1),max(fit_wls[:int(len_wls/4)]*norm1)))
	for i in fit_wls[:int(len_wls/4)]:
		mask = x1_fill==i
		plt.plot((x2_fill[mask])*norm2+offset2, np.exp(mu_fill[mask]*scale_factor + offset), 
				 lw=3, color=next(color), label='%i'%(i*norm1))
	plt.yscale('log')
	plt.subplot(222)
	plt.title('from %.1f to %.1f'%(min(fit_wls[int(len_wls/4):2*int(len_wls/4)]*norm1),max(fit_wls[int(len_wls/4):2*int(len_wls/4)]*norm1)))
	for i in fit_wls[int(len_wls/4):2*int(len_wls/4)]:
		mask = x1_fill==i
		plt.plot((x2_fill[mask])*norm2+offset2, np.exp(mu_fill[mask]*scale_factor + offset), 
				 lw=3, color=next(color), label='%i'%(i*norm1))
	plt.yscale('log')
	plt.subplot(223)
	plt.title('from %.1f to %.1f'%(min(fit_wls[2*int(len_wls/4):3*int(len_wls/4)]*norm1),max(fit_wls[2*int(len_wls/4):3*int(len_wls/4)]*norm1)))
	for i in fit_wls[2*int(len_wls/4):3*int(len_wls/4)]:
		mask = x1_fill==i
		plt.plot((x2_fill[mask])*norm2+offset2, np.exp(mu_fill[mask]*scale_factor + offset), 
				 lw=3, color=next(color), label='%i'%(i*norm1))
	plt.yscale('log')
	plt.subplot(224)
	plt.title('from %.1f to %.1f'%(min(fit_wls[3*int(len_wls/4):int(len_wls)]*norm1),max(fit_wls[3*int(len_wls/4):int(len_wls)]*norm1)))
	for i in fit_wls[3*int(len_wls/4):int(len_wls)]:
	
		mask = x1_fill==i
		plt.plot((x2_fill[mask])*norm2+offset2, np.exp(mu_fill[mask]*scale_factor + offset), 
				 lw=3, color=next(color), label='%i'%(i*norm1))
	plt.yscale('log')
	plt.show()


def transform_back_andPlot(GP2DIM_Class, x1_fill, x2_fill, mu_fill, std_fill, y_data_nonan):

	norm1 = GP2DIM_Class.grid_norm_info['norm1']
	norm2 = GP2DIM_Class.grid_norm_info['norm2']
	offset = GP2DIM_Class.grid_norm_info['offset']
	offset2 = GP2DIM_Class.grid_norm_info['offset2']
	scale_factor = GP2DIM_Class.grid_norm_info['scale_factor']

	#if LOG:
	mu_fill_conv = np.exp(mu_fill*scale_factor + offset)
	std_fill_conv = np.abs( scale_factor*mu_fill_conv *std_fill )
	
	y_data_conv = np.exp(y_data_nonan*scale_factor + offset)
	#else:
	#	mu_fill_conv = (mu_fill*scale_factor + offset)
	#	std_fill_conv = np.abs( scale_factor*std_fill )
	#
	#	y_data_conv =(y_data_nonan*scale_factor + offset)
		
	fig = plt.figure(1, figsize=(10,3))
	plt.subplot(121)
	plt.scatter(norm2*x2_fill, norm1*x1_fill, marker='s', s=10,  c=mu_fill_conv, alpha=1., 
				vmin=0., cmap = mycmap)
	#plt.scatter(x2_data_norm, x1_data_norm, marker='s', c=y_data)
	#plt.scatter(x2_data_norm, x1_data_norm, marker='s', c=y_data)
	plt.xlabel('MJD')
	plt.ylabel('wls')
	plt.colorbar()
	
	plt.subplot(122)
	plt.scatter(norm2*x2_fill, norm1*x1_fill, marker='s', s=10,  c=std_fill_conv, alpha=1., 
				vmin=0., cmap = mycmap)
	#plt.scatter(x2_data_norm, x1_data_norm, marker='s', c=y_data)
	#plt.scatter(x2_data_norm, x1_data_norm, marker='s', c=y_data)
	plt.xlabel('MJD')
	plt.ylabel('wls')
	plt.colorbar()
	plt.show()
	fig.savefig(GP2DIM_Class.save_plot_path+'/2d_surface.png', bbox_inches='tight')
	plt.close(fig)
	
	max_val = np.max(y_data_conv)
	med_val = np.median(y_data_conv)
	
	#fig = plt.figure(1, figsize=(8,4))
	#spec_mjd_list = GP2DIM_Class.get_spec_mjd()
	#scale = (max_val-med_val)/5.
	#a=0
	#mangled_original_list = GP2DIM_Class.mangledspec_list
	#
	#for j in range(len(GP2DIM_Class.get_spec_mjd())):
	#	mj = spec_mjd_list[j]
	#	spec_file_original = GP2DIM_Class.load_mangledfile(mangled_original_list[j])
	#	a +=1
	#	#mask = x2_fill==(mj-offset2)/norm2
	#	#plt.plot(x1_fill[mask]*norm1, mu_fill_conv[mask], label='%i'%(mj-offset2), lw=0.8, color='r')
	#	#plt.plot(off_xa, grid_ext[mj]+(a-1)*scale, label='Raw spec %i'%(mj-offset2), lw=1.8, color='k')
	#	plt.plot(spec_file_original['wls'], spec_file_original['flux']+(a-1)*scale,
	#			 label='Raw spec %i'%(mj-offset2), lw=1.0, color='k')
	#for b in GP2DIM_Class.avail_filters:
	#	wls, T = GP2DIM_Class.get_filt_transmission(b)
	#	plt.plot(wls, 0.5*T*max_val/max(T), linestyle='-', lw=2, color=PyCoCo_info.color_dict[b])
	#plt.xlim(1600,11000)
	#plt.title(GP2DIM_Class.snname)
	#plt.xlabel('Wavelength')
	#plt.ylabel('Calibrated Flux + offset')
	#fig.savefig(GP2DIM_Class.save_plot_path+'/to_be_extended_spec1.pdf', bbox_inches='tight')
	#plt.show()
	#plt.close(fig)
	
	fig = plt.figure(1, figsize=(8,5))

	spec_mjd_list = GP2DIM_Class.get_spec_mjd()
	scale = (max_val-med_val)/5.
	a=0
	for j in range(len(GP2DIM_Class.get_spec_mjd())):
		mj = spec_mjd_list[j]
		a +=1
		mask = x2_fill==(mj-offset2)/norm2
		plt.plot(x1_fill[mask]*norm1, mu_fill[mask]+(a-1)*scale, 
				 label='Extrapolated %i'%(mj-offset2), lw=0.8, color='r')
		plt.fill_between(x1_fill[mask]*norm1, (mu_fill[mask]-std_fill[mask])+(a-1)*scale , 
				 (mu_fill[mask]+std_fill[mask])+(a-1)*scale , facecolor='r', alpha=0.3)
	
	plt.xlim(1600,11000)
	for b in GP2DIM_Class.avail_filters:
		plt.vlines((GP2DIM_Class.lam_eff(b)), 0, 1., linestyle='--', lw=4, label=b, color=PyCoCo_info.color_dict[b])

	plt.title(GP2DIM_Class.snname)
	plt.xlabel('Wavelength')
	plt.ylabel('Calibrated Flux + offset')
	fig.savefig(GP2DIM_Class.save_plot_path+'/extended_spec_LOG_SPACE.pdf', bbox_inches='tight')
	plt.show()
	plt.close(fig)
	
	fig = plt.figure(1, figsize=(14,6))
	plt.rc('font', family='serif')
	plt.rc('xtick', labelsize=13)
	plt.rc('ytick', labelsize=13)
	
	spec_mjd_list = GP2DIM_Class.get_spec_mjd()
	scale = (max_val-med_val)/5.
	a=0
	for j in range(len(GP2DIM_Class.get_spec_mjd())):
		mj = spec_mjd_list[j]
		a +=1
		mask = x2_fill==(mj-offset2)/norm2
		plt.plot(x1_fill[mask]*norm1, mu_fill_conv[mask]+(a-1)*scale, 
				 label='Extrapolated %i'%(mj-offset2), lw=0.8, color='r')
		plt.fill_between(x1_fill[mask]*norm1, (mu_fill_conv[mask]-std_fill_conv[mask])+(a-1)*scale , 
				 (mu_fill_conv[mask]+std_fill_conv[mask])+(a-1)*scale , facecolor='r', alpha=0.3)
	a=0	
	mangled_original_list = GP2DIM_Class.mangledspec_list
	
	for j in range(len(GP2DIM_Class.get_spec_mjd())):
		mj = spec_mjd_list[j]
		spec_file_original = GP2DIM_Class.load_mangledfile(mangled_original_list[j])
		a +=1
		mask = x2_fill==(mj-offset2)/norm2
		#plt.plot(x1_fill[mask]*norm1, mu_fill_conv[mask], label='%i'%(mj-offset2), lw=0.8, color='r')
		#plt.plot(off_xa, grid_ext[mj]+(a-1)*scale, label='Raw spec %i'%(mj-offset2), lw=1.8, color='k')
		plt.plot(spec_file_original['wls'], spec_file_original['flux']+(a-1)*scale,
				 label='Raw spec %i'%(mj-offset2), lw=1, color='k')
	
	plt.xlim(1600,11000)
	
	#for b in GP2DIM_Class.avail_filters:
	#	wls, T = GP2DIM_Class.get_filt_transmission(b)
	#	plt.plot(wls, 0.5*T*max_val/max(T), linestyle='-', lw=4, color=PyCoCo_info.color_dict[b])
	plt.title(GP2DIM_Class.snname)
	plt.xlabel('Wavelength')
	plt.ylabel('Calibrated Flux + offset')
	fig.savefig(GP2DIM_Class.save_plot_path+'/extended_spec.pdf', bbox_inches='tight')
	plt.show()
	plt.close(fig)
	
	return (mu_fill_conv, std_fill_conv, y_data_conv)


# list_mjds_tot = grid_ext_columns
# list_mjds_tot = extrap_mjds


def save_plots_files(GP2DIM_Class, list_mjds_tot, y_data_conv, x1_fill, x2_fill, mu_fill_conv, std_fill_conv):
	results_directory = GP2DIM_Class.main_path+'/results_template/%s/'%GP2DIM_Class.snname
	norm1 = GP2DIM_Class.grid_norm_info['norm1']
	norm2 = GP2DIM_Class.grid_norm_info['norm2']
	offset = GP2DIM_Class.grid_norm_info['offset']
	offset2 = GP2DIM_Class.grid_norm_info['offset2']
	scale_factor = GP2DIM_Class.grid_norm_info['scale_factor']

	fig = plt.figure(1, figsize=(11,8))
	
	max_val = np.max(y_data_conv)
	med_val = np.median(y_data_conv)

	scale = (max_val-med_val)/5.
	#list_mjds_tot = grid_ext_columns
	
	list_mjds_spec = np.array(GP2DIM_Class.get_spec_mjd())
	list_mjds_spec_file = np.array(GP2DIM_Class.mangledspec_list)
#	if snname=='iPTF13bvn':
#		list_mjds_spec_special = np.array([56463.43, 56468.42, 56473.39, 56476.31, 56478.35,
#					   56481.34, 56483.36, 56486.33, 56488.35, 56493.29])

	min_mjd = min(list_mjds_tot)
	a=0
	for j in range(len(list_mjds_tot)):
		mj = list_mjds_tot[j]
		mask = x2_fill==(mj-offset2)/norm2
		wls = x1_fill[mask]*norm1
		smooth_ext_spec = mu_fill_conv[mask]
		smooth_ext_spec_err = std_fill_conv[mask]

		a = a-1

		if (mj in list_mjds_spec)&(GP2DIM_Class.mode=='extend_spectra'):
			file = list_mjds_spec_file[np.where(list_mjds_spec==mj)[0]][0]
			spec_orig = GP2DIM_Class.load_mangledfile(file)
			UV_mask = (wls<min(spec_orig['wls']))
			IR_mask = (wls>max(spec_orig['wls']))
			ext_spec_wls = np.concatenate((wls[UV_mask], 
										   spec_orig['wls'], 
										   wls[IR_mask]))
			ext_spec_flx = np.concatenate((smooth_ext_spec[UV_mask], 
										   spec_orig['flux'], 
										   smooth_ext_spec[IR_mask]))
			ext_spec_flx_err = np.concatenate((smooth_ext_spec_err[UV_mask], 
											   spec_orig['fluxerr'], 
											   smooth_ext_spec_err[IR_mask]))

			plt.plot(spec_orig['wls'], spec_orig['flux']+(a+1)*scale, lw=1, color='b')
			plt.plot(ext_spec_wls, ext_spec_flx+(a+1)*scale, lw=0.6, color='k', linestyle='--')
			plt.fill_between(ext_spec_wls, 
							 (ext_spec_flx-ext_spec_flx_err)+(a+1)*scale, 
							 (ext_spec_flx+ext_spec_flx_err)+(a+1)*scale, 
							 alpha=0.3, facecolor='k')

			plt.text(ext_spec_wls[0], (a+1)*scale, '%.2f'%(mj-min_mjd))
			# write the file
			fout = open(results_directory+'TwoDextended_spectra'+'/%.2f_spec_extended.txt'%mj, 'w')
			fout.write('#wls\tflux\tfluxerr\n')
			for w,f,ferr in zip(ext_spec_wls, ext_spec_flx,ext_spec_flx_err):
				fout.write('%E\t%E\t%E\n'%(w,f,ferr))
			fout.close()

		elif (mj not in list_mjds_spec)&(GP2DIM_Class.mode=='extrapolate_spectra'):
			plt.plot(wls, smooth_ext_spec+(a+1)*scale, label='Extrapolated %i'%(mj-offset2), lw=0.8, color='r')
			plt.fill_between(wls, (smooth_ext_spec-smooth_ext_spec_err)+(a+1)*scale,
							 (smooth_ext_spec+smooth_ext_spec_err)+(a+1)*scale,
							 alpha=0.3, facecolor='r')
			plt.text(wls[0], (a+1)*scale, '%.2f'%(mj-min_mjd))
			fout = open(results_directory+'TwoDextended_spectra'+'/%.2f_spec_extended_FL.txt'%mj, 'w')
			fout.write('#wls\tflux\tfluxerr\n')
			for w,f,ferr in zip(wls, smooth_ext_spec, smooth_ext_spec_err):
				fout.write('%E\t%E\t%E\n'%(w,f,ferr))
			fout.close()

	plt.xlabel('Wavelength')
	plt.ylabel('Calibrated Flux + offset')
	plt.title(GP2DIM_Class.snname)
	#fig.savefig(save_plot_path+'/extended_spec_plusFL.pdf', bbox_inches='tight')
	plt.show()
	plt.close(fig)
