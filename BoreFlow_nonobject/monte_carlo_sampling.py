# Numerical packages
import numpy as np

# Local packages
from BoreFlow import heat_flow_functions


# ----------------------------------------------------------------

def subsample_depth_series(x, suffix_root, monte_carlo_subsample_type, monte_carlo_subsample_factor, *args):
	if monte_carlo_subsample_factor == None:
		subsampled_list = [x, suffix_root]
		for arg in args: subsampled_list.append(arg)
	else:
		if monte_carlo_subsample_type == 'regular':
			# Randomly select start index between 0 and monte_carlo_subsample_factor-1
			start_index = np.random.randint(0, high=monte_carlo_subsample_factor)
			# Select value at indices separated by monte_carlo_subsample_factor
			x_subsampled = x[start_index::monte_carlo_subsample_factor]
			subsampled_list = [x_subsampled, suffix_root + '_ssreg' + str(monte_carlo_subsample_factor)]
			for arg in args: subsampled_list.append(arg[start_index::monte_carlo_subsample_factor])
		elif monte_carlo_subsample_type == 'random':
			# Randomly draw (n / monte_carlo_subsample_factor) indices from the indices of 1D array x, where n is size of x. Sort drawn indices into ascending order
			subsample_indices = np.sort(np.random.choice(np.arange(0, np.size(x), 1), replace=False, size=int((np.size(x)/monte_carlo_subsample_factor))))
			x_subsampled = x[subsample_indices]
			subsampled_list = [x_subsampled, suffix_root + '_ssrnd' + str(monte_carlo_subsample_factor)]
			for arg in args: subsampled_list.append(arg[subsample_indices])
	return(tuple(subsampled_list))

# ----------------------------------------------------------------

def perturb_values(sampling_distribution, *args):
	if len(args) % 2 != 0:
		print('missing uncertainties in function perturb_values() - exiting')
		exit()
	perturbed_list = []
	for argindex in range(len(args[0::2])):
		if len(args[2*argindex]) != len(args[2*argindex+1]):
			print('arrays are not same length in function perturb_values() - exiting')
			exit()
		else:
			if sampling_distribution == 'normal' or sampling_distribution == 'in_situ_normal':
				perturbed_list.append(np.random.normal(args[2*argindex], args[2*argindex+1]))
			elif sampling_distribution == 'uniform':
				perturbed_list.append(np.random.uniform(args[2*argindex], args[2*argindex+1]))
			else:
				print('sampling_distribution not specified in function perturb_values() - exiting')
	if len(perturbed_list) > 1:
		perturbed_list = tuple(perturbed_list)
	else:
		perturbed_list = np.array(perturbed_list)[0,:]
	return(perturbed_list)

# ----------------------------------------------------------------

def perturb_T(T_suffix, monte_carlo_T_subsample_type, monte_carlo_T_subsample_factor, zT_m_cut, T_cut, zT_error_m_cut, T_error_cut, monte_carlo_T_option):
	zT_m_subsampled, T_suffix, T_subsampled, zT_error_m_subsampled, T_error_subsampled = subsample_depth_series(zT_m_cut, T_suffix, monte_carlo_T_subsample_type, monte_carlo_T_subsample_factor, T_cut, zT_error_m_cut, T_error_cut)
	T_suffix_subsampled = T_suffix
	if monte_carlo_T_option == 'yes':
		### Perturb depths and values of temperature measurements under assumptions that errors are normally distributed
		zT_m_input, T_input = perturb_values('normal', zT_m_subsampled, zT_error_m_subsampled, T_subsampled, T_error_subsampled)
		# Make sure depths are monotonically increasing
		# TODO It's probably not correct to impose monotonically increasing depths - will have sampling of distribution
		zT_m_input = np.sort(zT_m_input)
		# Set uncertainties on perturbed values to zero
		zT_error_m_input, T_error_input = None, None
		T_suffix = T_suffix + '_mcnorm'
	else:
		zT_m_input, zT_error_m_input, T_input, T_error_input = zT_m_subsampled, zT_error_m_subsampled, T_subsampled, T_error_subsampled
	return(zT_m_subsampled, T_subsampled, zT_error_m_subsampled, T_error_subsampled, T_suffix_subsampled, zT_m_input, zT_error_m_input, T_input, T_error_input, T_suffix)

# ----------------------------------------------------------------

def perturb_k(layer_zmid_m, k_suffix, monte_carlo_in_situ_k_subsample_type, monte_carlo_in_situ_k_subsample_factor, layer_zmid_error_m, layer_z0_m, layer_z1_m, layer_mean_k, layer_z0_error_m, layer_z1_error_m, layer_mean_k_error, layer_min_k, layer_max_k, monte_carlo_k_option, k_low_clip, k_distribution):
	
	### Subsample in situ conductivity measurements (monte_carlo_in_situ_k_subsample_factor_list is set to [None] if conductivity input is not in situ)
	layer_zmid_m_subsampled, k_suffix, layer_zmid_error_m_subsampled, layer_z0_m_subsampled, layer_z1_m_subsampled, layer_mean_k_subsampled, layer_z0_error_m_subsampled, layer_z1_error_m_subsampled, layer_mean_k_error_subsampled, layer_min_k_subsampled, layer_max_k_subsampled = subsample_depth_series(layer_zmid_m, k_suffix, monte_carlo_in_situ_k_subsample_type, monte_carlo_in_situ_k_subsample_factor, layer_zmid_error_m, layer_z0_m, layer_z1_m, layer_mean_k, layer_z0_error_m, layer_z1_error_m, layer_mean_k_error, layer_min_k, layer_max_k)
	### Make arrays for plotting
	# layer_zk_m_subsampled_plot = np.concatenate(np.column_stack((layer_z0_m_subsampled, layer_z1_m_subsampled)))
	# layer_mean_k_subsampled_plot = np.concatenate(np.column_stack((layer_mean_k_subsampled, layer_mean_k_subsampled))).astype('float64')
	# layer_mean_k_error_subsampled_plot = np.concatenate(np.column_stack((layer_mean_k_error_subsampled, layer_mean_k_error_subsampled))).astype('float64')
	# layer_min_k_subsampled_plot = np.concatenate(np.column_stack((layer_min_k_subsampled, layer_min_k_subsampled))).astype('float64')
	# layer_max_k_subsampled_plot = np.concatenate(np.column_stack((layer_max_k_subsampled, layer_max_k_subsampled))).astype('float64')
	k_suffix_subsampled = k_suffix
	
	### Recalculate layer boundaries for subsampled in situ conductivities
	# print('monte_carlo_in_situ_k_subsample_factor', monte_carlo_in_situ_k_subsample_factor)
	if monte_carlo_in_situ_k_subsample_factor is not None and k_distribution == 'in_situ_normal':
		layer_z0_m_subsampled, layer_z0_error_m_subsampled, layer_z1_m_subsampled, layer_z1_error_m_subsampled = heat_flow_functions.calculate_conductivity_midpoint_depths(layer_zmid_m_subsampled, layer_zmid_error_m_subsampled)
	
	if monte_carlo_k_option == 'yes':
		### Perturb layer midpoints
		if k_distribution == 'in_situ_normal':
			### For in situ conductivities, perturb layer midpoints assuming that errors are normally distributed
			layer_zmid_m_input = perturb_values('normal', layer_zmid_m_subsampled, layer_zmid_error_m_subsampled)
			### Recalculate layer boundaries for perturbed layer midpoints
			layer_z0_m_input, layer_z0_error_m_input, layer_z1_m_input, layer_z1_error_m_input = heat_flow_functions.calculate_conductivity_midpoint_depths(layer_zmid_m_input, np.full(np.size(layer_zmid_m_input), np.nan))
		else:
			### For layered descriptions of conductivity, perturb layer boundaries
			layer_z1_m_input = perturb_values('normal', layer_z1_m_subsampled, layer_z1_error_m_subsampled)
			# Make sure depths are monotonically increasing
			# TODO It's probably not correct to impose monotonically increasing depths - will have sampling of distribution
			layer_z1_m_input = np.sort(layer_z1_m_input)
			layer_z0_m_input = np.append(0, layer_z1_m_input[0:-1])
			# ### Calculate midpoints from layer boundaries
# 			layer_zmid_m_input, layer_zmid_error_m_input, unused, unused = heat_flow_functions.calculate_conductivity_midpoint_depths(np.append(0, layer_z1_m_input), np.full(np.size(layer_z1_m_input)+1, np.nan))
			layer_zmid_m_input = np.full(np.size(layer_z1_m_input), np.nan)
			
		### Perturb values of thermal conductivity using either normal or uniform distribution
		if k_distribution == 'normal' or k_distribution == 'in_situ_normal':
			layer_k_input = perturb_values(k_distribution, layer_mean_k_subsampled, layer_mean_k_error_subsampled)
			k_suffix = k_suffix + '_mcnorm'
		elif k_distribution == 'uniform':
			layer_k_input = perturb_values(k_distribution, layer_min_k_subsampled, layer_max_k_subsampled)
			k_suffix = k_suffix + '_mcuni'
		# Make sure conductivity values are not negative
		layer_k_input = np.clip(layer_k_input, k_low_clip, 1e99)
		# Set uncertainties on perturbed values to NaN
		layer_zmid_error_m_input, layer_z0_error_m_input, layer_z1_error_m_input, layer_k_error_input, layer_min_k_input, layer_max_k_input = None, None, None, None, None, None
	else:
		layer_zmid_m_input, layer_zmid_error_m_input, layer_z0_m_input, layer_z0_error_m_input, layer_z1_m_input, layer_z1_error_m_input, layer_k_input, layer_k_error_input, layer_min_k_input, layer_max_k_input = layer_zmid_m_subsampled, layer_zmid_error_m_subsampled, layer_z0_m_subsampled, layer_z0_error_m_subsampled, layer_z1_m_subsampled, layer_z1_error_m_subsampled, layer_mean_k_subsampled, layer_mean_k_error_subsampled, layer_min_k_subsampled, layer_max_k_subsampled
	### Calculate layer thicknesses
	layer_delta_z_m_input, layer_delta_z_error_m_input = heat_flow_functions.calculate_layer_thicknesses(layer_z0_m_input, layer_z1_m_input, layer_z0_error_m_input, layer_z1_error_m_input)
	### Make arrays for plotting
	layer_zk_m_input_plot = np.concatenate(np.column_stack((layer_z0_m_input, layer_z1_m_input)))
	layer_k_input_plot = np.concatenate(np.column_stack((layer_k_input, layer_k_input))).astype('float64')
	if monte_carlo_k_option is not None:
		layer_k_error_input_plot, layer_min_k_input_plot, layer_max_k_input_plot = None, None, None
	else:
		layer_k_error_input_plot = np.concatenate(np.column_stack((layer_k_error_input, layer_k_error_input))).astype('float64')
		layer_min_k_input_plot = np.concatenate(np.column_stack((layer_min_k_input, layer_min_k_input))).astype('float64')
		layer_max_k_input_plot = np.concatenate(np.column_stack((layer_max_k_input, layer_max_k_input))).astype('float64')
	
	return(layer_zmid_m_subsampled, layer_zmid_error_m_subsampled, layer_z0_m_subsampled, layer_z0_error_m_subsampled, layer_z1_m_subsampled, layer_z1_error_m_subsampled, layer_mean_k_subsampled, layer_mean_k_error_subsampled, layer_min_k_subsampled, layer_max_k_subsampled, k_suffix_subsampled, layer_zmid_m_input, layer_zmid_error_m_input, layer_z0_m_input, layer_z0_error_m_input, layer_z1_m_input, layer_z1_error_m_input, layer_k_input, layer_k_error_input, layer_min_k_input, layer_max_k_input, layer_delta_z_m_input, layer_delta_z_error_m_input, layer_zk_m_input_plot, layer_k_input_plot, layer_k_error_input_plot, layer_min_k_input_plot, layer_max_k_input_plot, k_suffix)
	
	
	

# ----------------------------------------------------------------

def perturb_climate(monte_carlo_Tsurf_option, palaeoclimate_sigma_t_dist, palaeoclimate_t0_seconds, palaeoclimate_sigma_t0_seconds, palaeoclimate_t1_seconds, palaeoclimate_sigma_t1_seconds, palaeoclimate_sigma_deltaTs_dist, palaeoclimate_deltaTs, palaeoclimate_sigma_deltaTs, recent_climtemp_sigma_year_smoothing_cutoff_dist, recent_climtemp_t0_seconds_smoothing_cutoff, recent_climtemp_sigma_t0_seconds_smoothing_cutoff, recent_climtemp_t1_seconds_smoothing_cutoff, recent_climtemp_sigma_t1_seconds_smoothing_cutoff, recent_climtemp_sigma_deltaTs_smoothed_dist, recent_climtemp_deltaTs_smoothed_cut, recent_climtemp_sigma_deltaTs_smoothed_cut, palaeoclimate_suffix_mc, recent_climtemp_suffix_mc):
	if monte_carlo_Tsurf_option == 'yes':
		# TODO Add uniform distribution
		### Perturb palaeoclimate
		if palaeoclimate_sigma_t_dist == 'normal':
			palaeoclimate_t0_seconds_input = perturb_values('normal', palaeoclimate_t0_seconds, palaeoclimate_sigma_t0_seconds)
			# Make sure times are monotonically decreasing
			palaeoclimate_t0_seconds_input = np.flip(np.sort(palaeoclimate_t0_seconds_input))
			# Make sure most recent time is not negative
			palaeoclimate_t0_seconds_input = np.clip(palaeoclimate_t0_seconds_input, a_min=palaeoclimate_t1_seconds[-1], a_max=None)
			palaeoclimate_t1_seconds_input = np.hstack([palaeoclimate_t0_seconds_input[1::], palaeoclimate_t1_seconds[-1]])
			palaeoclimate_suffix_mc = palaeoclimate_suffix_mc + "_tmcnorm"
		if palaeoclimate_sigma_deltaTs_dist == 'normal':
			palaeoclimate_deltaTs_input = perturb_values('normal', palaeoclimate_deltaTs, palaeoclimate_sigma_deltaTs)
			palaeoclimate_suffix_mc = palaeoclimate_suffix_mc + "_deltaTmcnorm"
		### Perturb recent climate history
		if recent_climtemp_sigma_year_smoothing_cutoff_dist == 'normal':
			recent_climtemp_t0_seconds_input = perturb_values('normal', recent_climtemp_t0_seconds_smoothing_cutoff, recent_climtemp_sigma_t0_seconds_smoothing_cutoff)
			# Make sure times are monotonically decreasing
			recent_climtemp_t0_seconds_input = np.flip(np.sort(recent_climtemp_t0_seconds_input))
			# Make sure most recent time is not negative
			recent_climtemp_t0_seconds_input = np.clip(recent_climtemp_t0_seconds_input, a_min=recent_climtemp_t1_seconds_smoothing_cutoff[-1], a_max=None)
			recent_climtemp_t1_seconds_input = np.hstack([recent_climtemp_t0_seconds_input[1::], recent_climtemp_t1_seconds_smoothing_cutoff[-1]])
			recent_climtemp_suffix_mc = recent_climtemp_suffix_mc + "_tmcnorm"
		if recent_climtemp_sigma_deltaTs_smoothed_dist == 'normal':
			recent_climtemp_deltaTs_input = perturb_values('normal', recent_climtemp_deltaTs_smoothed_cut, recent_climtemp_sigma_deltaTs_smoothed_cut)
			recent_climtemp_suffix_mc = recent_climtemp_suffix_mc + "_deltaTmcnorm"
		# Set uncertainties on perturbed values to zero
		palaeoclimate_sigma_t0_seconds_input, palaeoclimate_sigma_t1_seconds_input, palaeoclimate_sigma_deltaTs_input, recent_climtemp_sigma_t0_seconds_input, recent_climtemp_sigma_t1_seconds_input, recent_climtemp_sigma_deltaTs_input = None, None, None, None, None, None
	else:
		palaeoclimate_t0_seconds_input, palaeoclimate_sigma_t0_seconds_input, palaeoclimate_t1_seconds_input, palaeoclimate_sigma_t1_seconds_input, palaeoclimate_deltaTs_input, palaeoclimate_sigma_deltaTs_input, recent_climtemp_t0_seconds_input, recent_climtemp_sigma_t0_seconds_input, recent_climtemp_t1_seconds_input, recent_climtemp_sigma_t1_seconds_input, recent_climtemp_deltaTs_input, recent_climtemp_sigma_deltaTs_input = palaeoclimate_t0_seconds, palaeoclimate_sigma_t0_seconds, palaeoclimate_t1_seconds, palaeoclimate_sigma_t1_seconds, palaeoclimate_deltaTs, palaeoclimate_sigma_deltaTs, recent_climtemp_t0_seconds_smoothing_cutoff, recent_climtemp_sigma_t0_seconds_smoothing_cutoff, recent_climtemp_t1_seconds_smoothing_cutoff, recent_climtemp_sigma_t1_seconds_smoothing_cutoff, recent_climtemp_deltaTs_smoothed_cut, recent_climtemp_sigma_deltaTs_smoothed_cut
	return(palaeoclimate_t0_seconds_input, palaeoclimate_sigma_t0_seconds_input, palaeoclimate_t1_seconds_input, palaeoclimate_sigma_t1_seconds_input, palaeoclimate_deltaTs_input, palaeoclimate_sigma_deltaTs_input, recent_climtemp_t0_seconds_input, recent_climtemp_sigma_t0_seconds_input, recent_climtemp_t1_seconds_input, recent_climtemp_sigma_t1_seconds_input, recent_climtemp_deltaTs_input, recent_climtemp_sigma_deltaTs_input, palaeoclimate_suffix_mc, recent_climtemp_suffix_mc)

# ----------------------------------------------------------------

def set_up_monte_carlo_options(monte_carlo_dict, strat_interp_name, monte_carlo_option):
	if monte_carlo_option is None:
		monte_carlo_nsim_list, monte_carlo_T_subsample_factor_list, monte_carlo_in_situ_k_subsample_factor_list, monte_carlo_T_option, monte_carlo_k_option, monte_carlo_Tsurf_option = [1], [None], [None], 'no', 'no', 'no'
	elif monte_carlo_option == 'all':
		monte_carlo_nsim_list, monte_carlo_T_subsample_factor_list, monte_carlo_in_situ_k_subsample_factor_list, monte_carlo_T_option, monte_carlo_k_option, monte_carlo_Tsurf_option = monte_carlo_dict['monte_carlo_nsim_list'], monte_carlo_dict['monte_carlo_T_subsample_factor_list'], monte_carlo_dict['monte_carlo_in_situ_k_subsample_factor_list'], 'yes', 'yes', 'yes'
	elif monte_carlo_option == 'T':
		monte_carlo_nsim_list, monte_carlo_T_subsample_factor_list, monte_carlo_in_situ_k_subsample_factor_list, monte_carlo_T_option, monte_carlo_k_option, monte_carlo_Tsurf_option = monte_carlo_dict['monte_carlo_nsim_list'], monte_carlo_dict['monte_carlo_T_subsample_factor_list'], [None], 'yes', 'no', 'no'
	elif monte_carlo_option == 'k':
		monte_carlo_nsim_list, monte_carlo_T_subsample_factor_list, monte_carlo_in_situ_k_subsample_factor_list, monte_carlo_T_option, monte_carlo_k_option, monte_carlo_Tsurf_option = monte_carlo_dict['monte_carlo_nsim_list'], [None], monte_carlo_dict['monte_carlo_in_situ_k_subsample_factor_list'], 'no', 'yes', 'no'
	elif monte_carlo_option == 'climate':
		monte_carlo_nsim_list, monte_carlo_T_subsample_factor_list, monte_carlo_in_situ_k_subsample_factor_list, monte_carlo_T_option, monte_carlo_k_option, monte_carlo_Tsurf_option = monte_carlo_dict['monte_carlo_nsim_list'], [None], [None], 'no', 'no', 'yes'
	### Do not do Monte Carlo subsampling of conductivity unless values are in situ measurements
	if strat_interp_name != 'in_situ_conds':
		monte_carlo_in_situ_k_subsample_factor_list = [None]
	return(monte_carlo_nsim_list, monte_carlo_T_subsample_factor_list, monte_carlo_in_situ_k_subsample_factor_list, monte_carlo_T_option, monte_carlo_k_option, monte_carlo_Tsurf_option)

# ----------------------------------------------------------------

def set_up_monte_carlo_output_arrays(monte_carlo_nsim, zT_m, zk_interp_number_steps, cc_interp_number_steps):
	# Arrays for whole borehole # TODO Make into dictionary ### TODO Need to add subsampling
	zT_m_input_mc_all = np.full([monte_carlo_nsim, np.size(zT_m)], np.nan)
	T_input_mc_all = np.full([monte_carlo_nsim, np.size(zT_m)], np.nan)
	T_input_cc_mc_all = np.full([monte_carlo_nsim, np.size(zT_m)], np.nan)
	# Arrays for climatologies
	pc_deltaTs_input_plot_mc_all = np.full([monte_carlo_nsim, cc_interp_number_steps], np.nan)
	pc_t_input_plot_mc_all = np.full([monte_carlo_nsim, cc_interp_number_steps], np.nan)
	rc_deltaTs_input_plot_mc_all = np.full([monte_carlo_nsim, cc_interp_number_steps], np.nan)
	rc_t_input_plot_mc_all = np.full([monte_carlo_nsim, cc_interp_number_steps], np.nan)
	# TODO Need to add subsampling
	# Arrays for plotting of conductivity after Monte Carlo analysis
	# zk_interp_step_m = np.max(layer_z_plot)/zk_interp_number_steps
	layer_zk_m_input_plot_mc_all = np.full([monte_carlo_nsim, zk_interp_number_steps], np.nan)
	layer_k_input_plot_mc_all = np.full([monte_carlo_nsim, zk_interp_number_steps], np.nan)
	
	# layer_k_error_input_plot_mc_all = np.zeros([monte_carlo_nsim, np.size(layer_z_plot)])
	# layer_min_k_input_plot_mc_all = np.zeros([monte_carlo_nsim, np.size(layer_z_plot)])
	# layer_max_k_input_plot_mc_all = np.zeros([monte_carlo_nsim, np.size(layer_z_plot)])

	# zk_m_input_mc_all = np.zeros([monte_carlo_nsim, np.size(zk_m)])
	#                                                 k_input_mc_all = np.zeros([monte_carlo_nsim, np.size(zk_m)])
	#                                                 layer_zmid_m_input, layer_zmid_error_m_input, layer_z0_m_input, layer_z0_error_m_input, layer_z1_m_input, layer_z1_error_m_input, layer_k_input, layer_k_error_input, layer_min_k_input, layer_max_k_input, layer_delta_z_m_input, layer_delta_z_error_m_input, k_suffix_mc
	
	### MC outputs for uncorrected temperatures
	whole_bullard_R1_downward_mc_all = np.full([monte_carlo_nsim, np.size(zT_m)], np.nan)
	whole_bullard_sigma_R1_downward_mc_all = np.full([monte_carlo_nsim, np.size(zT_m)], np.nan)
	whole_bullard_R1_upward_mc_all = np.full([monte_carlo_nsim, np.size(zT_m)], np.nan)
	whole_bullard_sigma_R1_upward_mc_all = np.full([monte_carlo_nsim, np.size(zT_m)], np.nan)
	whole_bullard_q_downward_TvR_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_sigma_q_downward_TvR_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_c_downward_TvR_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_sigma_c_downward_TvR_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_q_downward_RvT_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_sigma_q_downward_RvT_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_invc_downward_RvT_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_sigma_invc_downward_RvT_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_q_upward_TvR_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_sigma_q_upward_TvR_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_c_upward_TvR_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_sigma_c_upward_TvR_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_q_upward_RvT_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_sigma_q_upward_RvT_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_invc_upward_RvT_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_sigma_invc_upward_RvT_mc_all = np.full(monte_carlo_nsim, np.nan)
	
	### MC outputs for climate-corrected temperatures
	whole_bullard_R1_downward_cc_mc_all = np.full([monte_carlo_nsim, np.size(zT_m)], np.nan)
	whole_bullard_sigma_R1_downward_cc_mc_all = np.full([monte_carlo_nsim, np.size(zT_m)], np.nan)
	whole_bullard_R1_upward_cc_mc_all = np.full([monte_carlo_nsim, np.size(zT_m)], np.nan)
	whole_bullard_sigma_R1_upward_cc_mc_all = np.full([monte_carlo_nsim, np.size(zT_m)], np.nan)
	whole_bullard_q_downward_TvR_cc_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_sigma_q_downward_TvR_cc_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_c_downward_TvR_cc_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_sigma_c_downward_TvR_cc_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_q_downward_RvT_cc_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_sigma_q_downward_RvT_cc_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_invc_downward_RvT_cc_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_sigma_invc_downward_RvT_cc_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_q_upward_TvR_cc_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_sigma_q_upward_TvR_cc_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_c_upward_TvR_cc_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_sigma_c_upward_TvR_cc_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_q_upward_RvT_cc_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_sigma_q_upward_RvT_cc_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_invc_upward_RvT_cc_mc_all = np.full(monte_carlo_nsim, np.nan)
	whole_bullard_sigma_invc_upward_RvT_cc_mc_all = np.full(monte_carlo_nsim, np.nan)
	
	return(zT_m_input_mc_all, T_input_mc_all, T_input_cc_mc_all, zk_interp_number_steps, layer_zk_m_input_plot_mc_all, layer_k_input_plot_mc_all, whole_bullard_R1_downward_mc_all, whole_bullard_sigma_R1_downward_mc_all, whole_bullard_R1_upward_mc_all, whole_bullard_sigma_R1_upward_mc_all, whole_bullard_q_downward_TvR_mc_all, whole_bullard_sigma_q_downward_TvR_mc_all, whole_bullard_c_downward_TvR_mc_all, whole_bullard_sigma_c_downward_TvR_mc_all, whole_bullard_q_downward_RvT_mc_all, whole_bullard_sigma_q_downward_RvT_mc_all, whole_bullard_invc_downward_RvT_mc_all, whole_bullard_sigma_invc_downward_RvT_mc_all, whole_bullard_q_upward_TvR_mc_all, whole_bullard_sigma_q_upward_TvR_mc_all, whole_bullard_c_upward_TvR_mc_all, whole_bullard_sigma_c_upward_TvR_mc_all, whole_bullard_q_upward_RvT_mc_all, whole_bullard_sigma_q_upward_RvT_mc_all, whole_bullard_invc_upward_RvT_mc_all, whole_bullard_sigma_invc_upward_RvT_mc_all, whole_bullard_R1_downward_cc_mc_all, whole_bullard_sigma_R1_downward_cc_mc_all, whole_bullard_R1_upward_cc_mc_all, whole_bullard_sigma_R1_upward_cc_mc_all, whole_bullard_q_downward_TvR_cc_mc_all, whole_bullard_sigma_q_downward_TvR_cc_mc_all, whole_bullard_c_downward_TvR_cc_mc_all, whole_bullard_sigma_c_downward_TvR_cc_mc_all, whole_bullard_q_downward_RvT_cc_mc_all, whole_bullard_sigma_q_downward_RvT_cc_mc_all, whole_bullard_invc_downward_RvT_cc_mc_all, whole_bullard_sigma_invc_downward_RvT_cc_mc_all, whole_bullard_q_upward_TvR_cc_mc_all, whole_bullard_sigma_q_upward_TvR_cc_mc_all, whole_bullard_c_upward_TvR_cc_mc_all, whole_bullard_sigma_c_upward_TvR_cc_mc_all, whole_bullard_q_upward_RvT_cc_mc_all, whole_bullard_sigma_q_upward_RvT_cc_mc_all, whole_bullard_invc_upward_RvT_cc_mc_all, whole_bullard_sigma_invc_upward_RvT_cc_mc_all, pc_deltaTs_input_plot_mc_all, pc_t_input_plot_mc_all, rc_deltaTs_input_plot_mc_all, rc_t_input_plot_mc_all)
