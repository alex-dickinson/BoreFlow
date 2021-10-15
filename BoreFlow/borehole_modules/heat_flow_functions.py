import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from borehole_modules import general_python_functions


def trim_conductivities(z_k_f, sigma_z_k_f, k_f, sigma_k_f, z_T_max_f):
	# Trim z_k and k values that are deeper than the deepest meaured temperature 
	mask = z_k_f <= z_T_max_f
	z_k_f = z_k_f[mask]
	sigma_z_k_f = sigma_z_k_f[mask]
	k_f = k_f[mask]
	sigma_k_f = sigma_k_f[mask]
	return(z_k_f, sigma_z_k_f, k_f, sigma_k_f)

# ----------------------------------------------------------------

# Estimate heat flow using constant conductivity, k
def heat_flow_from_constant_conductivity(k_f, sigma_k_f, dTdz_f, sigma_dTdz_f):
	q_f = k_f * dTdz_f
	# Estimate uncertainty in q by combining uncertainties in k and dTdz
	sigma_q_f = q_f * np.power((np.power((sigma_k_f / k_f), 2) + np.power((sigma_dTdz_f / dTdz_f), 2)),0.5)
	return(q_f, sigma_q_f)

# ----------------------------------------------------------------

def estimate_depth_mean(delta_z_f, k_f):
	k_mean_z_f = np.zeros(delta_z_f.size)
	stderr_k_mean_z_f = np.zeros(delta_z_f.size)
	k_hmean_z_f = np.zeros(delta_z_f.size)
	
	for i_f in range(delta_z_f.size):
		# Arithmetic mean weighted by depth of each layer
		k_mean_z_f[i_f] = (np.sum(delta_z_f[:i_f+1] * k_f[:i_f+1]) / np.sum(delta_z_f[:i_f+1]))
		stderr_k_mean_z_f[i_f] = (1 / np.power( (np.sum(delta_z_f[:i_f+1])), 0.5))
		
		# Harmonic mean weighted by depth of each layer
		k_hmean_z_f[i_f] = 1 / (np.sum(delta_z_f[:i_f+1] / k_f[:i_f+1]) / np.sum(delta_z_f[:i_f+1]))
	
	# TODO Estimating error on this mean - not correct to use a formula to estimate variance (e.g. Norris (1940)) as not forming estimate based on values drawn from same sample (i.e. same lithology). Instead forming estimate of mean effect on conductivity of all lithologies above a certain depth. Estimate error using uncertainties in k and delta_z
	sigma_k_hmean_z_f = np.full(k_hmean_z_f.size, 1)
	
	return(k_mean_z_f, stderr_k_mean_z_f, k_hmean_z_f, sigma_k_hmean_z_f)

# ----------------------------------------------------------------

def calculate_conductivity_midpoint_depths(z_k_f, sigma_z_k_f):
	# Estimate midpoint depths between adjacent conductivity measurements
	z_k_lower_bounds_f = z_k_f - np.diff(np.hstack([[0.0], z_k_f])) / 2
	z_k_lower_bounds_f[0] = 0
	z_k_upper_bounds_f = np.hstack(([z_k_lower_bounds_f[1::], z_k_f[-1]]))
	### TODO Check errors
	sigma_z_k_lower_bounds_f = np.hstack([0, 0.5*np.power( np.power(sigma_z_k_f[:-1], 2) + np.power(sigma_z_k_f[1::], 2), 0.5)])
	sigma_z_k_upper_bounds_f = np.hstack((sigma_z_k_lower_bounds_f[1::], sigma_z_k_f[-1]))
	return(z_k_lower_bounds_f, sigma_z_k_lower_bounds_f, z_k_upper_bounds_f, sigma_z_k_upper_bounds_f)

# ----------------------------------------------------------------

def calculate_layer_thicknesses(z0_f, z1_f, z0_error_f, z1_error_f):
	delta_z_f = z1_f - z0_f
	if z0_error_f is not None and z1_error_f is not None:
		delta_z_error_f = np.power( np.power(z0_error_f, 2) + np.power(z1_error_f, 2), 0.5)
	else:
		delta_z_error_f = None
	return(delta_z_f, delta_z_error_f)

# ----------------------------------------------------------------

def measure_depths_from_bottom_hole(z1_f, z1_error_f, z0_f, z0_error_f, k_f, k_error_f, zT_f, zT_error_f, T_f, T_error_f):
	# Convert depths measured from surface downwards into heights measured from lowest depth upwards
	# z1 are lower boundaries of layers; z0 are upper boundaries
	z0_rev_f = z1_f[-1] - np.flip(z0_f)
	z0_error_rev_f = np.flip(z0_error_f)
	z1_rev_f = z1_f[-1] - np.flip(z1_f)
	z1_error_rev_f = np.flip(z1_error_f)
	# Convert sense of conductivity values
	k_rev_f = np.flip(k_f)
	k_error_rev_f = np.flip(k_error_f)
	# Convert depths of temperature measurements
	zT_rev_f = zT_f[-1] - np.flip(zT_f)
	zT_error_rev_f = np.flip(zT_error_f)
	T_rev_f = np.flip(T_f)
	T_error_rev_f = np.flip(T_error_f)
	return(z1_rev_f, z1_error_rev_f, z0_rev_f, z0_error_rev_f, k_rev_f, k_error_rev_f, zT_rev_f, zT_error_rev_f, T_rev_f, T_error_rev_f)

# ----------------------------------------------------------------

def calculate_thermal_resistance_same_depths(z_f, sigma_z_f, k_f, sigma_k_f):
	# Assumes conductivites and temperatures are measured at same depths
	delta_z_f = np.diff(np.hstack([[0.0], z_f]))
	sigma_delta_z_f = np.zeros(len(z_f))
	
	for i_f in range(np.size(z_f)):
		if i_f != 0:
			sigma_delta_z_f[i_f] = np.power( ( np.power(sigma_z_f[i_f], 2) + np.power(sigma_z_f[i_f-1], 2) ), 0.5)
	sigma_delta_z_f[0] = sigma_delta_z_f[1]
	
	R_f = (delta_z_f / k_f).cumsum()
	
	sigma_R_f_delta_z_term = (np.power((sigma_delta_z_f / k_f), 2)).cumsum()
	sigma_R_f_delta_k_term = (np.power((sigma_k_f * delta_z_f / np.power(k_f, 2)), 2)).cumsum()
	sigma_R_f = np.power( (sigma_R_f_delta_z_term + sigma_R_f_delta_k_term), 0.5)
	
	return(delta_z_f, sigma_delta_z_f, R_f, sigma_R_f)

# ----------------------------------------------------------------

def calculate_thermal_resistance_temperature_depths(z_k_lower_bounds_f, sigma_z_k_lower_bounds_f, z_k_upper_bounds_f, sigma_z_k_upper_bounds_f, z_k_thicknesses_f, sigma_z_k_thicknesses_f, z_T_f, sigma_z_T_f, k_f, sigma_k_f):
	
	# Calculate thermal resistance at depths of measured temperatures. Start at top of hole and work downwards. Input layers must start at surface (i.e. first value of z_k_upper_bounds_f must equal 0 m)
	R_z_T_downward_f = np.zeros(z_T_f.size)
	sigma_R_z_T_downward_f = np.zeros(z_T_f.size)
	for i_f in range(z_T_f.size):
		# Calculate thicknesses of each unit down to depth of ith temperature measurement
		if z_k_lower_bounds_f[0] > z_T_f[i_f]:
			z_k_lower_bounds_index_f = 0
			z_k_T_thicknesses = z_T_f[i_f]
			k_f_array = k_f[:z_k_lower_bounds_index_f+1]
			if sigma_z_T_f is None:
				sigma_z_k_T_thicknesses = None
			else:
				sigma_z_k_T_thicknesses = sigma_z_T_f[i_f]
			if sigma_k_f is None:
				sigma_k_f_array = None
			else:
				sigma_k_f_array = sigma_k_f[:z_k_lower_bounds_index_f+1]
		else:
			z_k_lower_bounds_index_f = np.max(np.where(z_k_lower_bounds_f <= z_T_f[i_f]))
			if z_T_f[i_f] == z_k_lower_bounds_f[z_k_lower_bounds_index_f]:
				z_k_T_thicknesses = z_k_thicknesses_f[:z_k_lower_bounds_index_f+1]
				k_f_array = k_f[:z_k_lower_bounds_index_f+1]
				if sigma_z_T_f is None:
					sigma_z_k_T_thicknesses = None
				else:
					sigma_z_k_T_thicknesses = sigma_z_k_thicknesses_f[:z_k_lower_bounds_index_f+1]
				if sigma_k_f is None:
					sigma_k_f_array = None
				else:
					sigma_k_f_array = sigma_k_f[:z_k_lower_bounds_index_f+1]
			else:
				z_k_T_thicknesses = np.hstack(( z_k_thicknesses_f[:z_k_lower_bounds_index_f+1], z_T_f[i_f] - z_k_lower_bounds_f[z_k_lower_bounds_index_f]))
				k_f_array = k_f[:z_k_lower_bounds_index_f+2]
				if sigma_z_T_f is None:
					sigma_z_k_T_thicknesses = None
				else:
					sigma_z_k_T_thicknesses = np.hstack(( sigma_z_k_thicknesses_f[:z_k_lower_bounds_index_f+1], np.power( ( np.power(sigma_z_T_f[i_f], 2) + np.power( sigma_z_k_lower_bounds_f[z_k_lower_bounds_index_f], 2) ), 0.5) ))
				if sigma_k_f is None:
					sigma_k_f_array = None
				else:
					sigma_k_f_array = sigma_k_f[:z_k_lower_bounds_index_f+2]
		if z_T_f[i_f] > z_k_lower_bounds_f[-1]:
			k_max_z_k = k_f[-1]
			k_f_array = np.hstack(( k_f_array, np.full((np.size(z_k_T_thicknesses) - np.size(k_f_array)), k_f[-1]) ))
			if sigma_k_f is None:
				sigma_k_f_array = None
			else:
				sigma_k_f_array = np.hstack(( sigma_k_f_array, np.full((np.size(z_k_T_thicknesses) - np.size(sigma_k_f_array)), sigma_k_f[-1]) ))
		R_z_T_downward_f[i_f] = (z_k_T_thicknesses / k_f_array).cumsum()[-1]
		if sigma_z_k_T_thicknesses is not None:
			sigma_R_z_T_downward_f_delta_z_term = (np.power((sigma_z_k_T_thicknesses / k_f_array), 2)).cumsum()
		else:
			sigma_R_z_T_downward_f_delta_z_term = None
		if sigma_k_f_array is not None:
			sigma_R_z_T_downward_f_delta_k_term = (np.power((sigma_k_f_array * z_k_T_thicknesses / np.power(k_f_array, 2)), 2)).cumsum()
		else:
			sigma_R_z_T_downward_f_delta_k_term = None
		if sigma_R_z_T_downward_f_delta_z_term is not None and sigma_R_z_T_downward_f_delta_k_term is not None:
			sigma_R_z_T_downward_f[i_f] = np.power( (sigma_R_z_T_downward_f_delta_z_term + sigma_R_z_T_downward_f_delta_k_term), 0.5)[-1]
		elif sigma_R_z_T_downward_f_delta_z_term is None and sigma_R_z_T_downward_f_delta_k_term is not None:
			sigma_R_z_T_downward_f[i_f] = sigma_R_z_T_downward_f_delta_k_term
		elif sigma_R_z_T_downward_f_delta_z_term is not None and sigma_R_z_T_downward_f_delta_k_term is None:
			sigma_R_z_T_downward_f[i_f] = sigma_R_z_T_downward_f_delta_z_term
		elif sigma_R_z_T_downward_f_delta_z_term is None and sigma_R_z_T_downward_f_delta_k_term is None:
			sigma_R_z_T_downward_f = None
	return(R_z_T_downward_f, sigma_R_z_T_downward_f)

# ----------------------------------------------------------------

def make_constant_depth_intervals(z, z_int):
	z_intervals = np.arange(0, np.max(z)+z_int, z_int)
	z0_int = z_intervals[0:-1]
	z1_int = z_intervals[1::]
	number_intervals = np.size(z0_int)
	z_int_plot = np.concatenate(np.column_stack((z0_int, z1_int)))
	return(number_intervals, z0_int, z1_int, z_int_plot)

# ----------------------------------------------------------------
	
def mean_layer_properties(z0_int_f, z1_int_f, z_prop, sigma_z_prop, prop, sigma_prop):
	prop_int_indexed = np.zeros([np.size(z_prop), 5])
	prop_int_indexed[:,0] = z_prop
	prop_int_indexed[:,1] = sigma_z_prop
	prop_int_indexed[:,2] = prop
	prop_int_indexed[:,3] = sigma_prop
	
	mean_prop_int = np.zeros(np.size(z0_int_f))
	std_mean_prop_int = np.zeros(np.size(z0_int_f))
	weighted_mean_prop_int = np.zeros(np.size(z0_int_f))
	std_weighted_mean_prop_int = np.zeros(np.size(z0_int_f))
	hmean_prop_int = np.zeros(np.size(z0_int_f))
	sigma_hmean_prop_int = np.zeros(np.size(z0_int_f))
	gmean_prop_int = np.zeros(np.size(z0_int_f))
	sdfactor_gmean_prop_int = np.zeros(np.size(z0_int_f))
	lower_bound_gmean_prop_int = np.zeros(np.size(z0_int_f))
	upper_bound_gmean_prop_int = np.zeros(np.size(z0_int_f))
	gmean_minus_prop_int = np.zeros(np.size(z0_int_f))
	gmean_plus_prop_int = np.zeros(np.size(z0_int_f))
	
	# Assumes normal errors in measurements and that measurements (and their errors) are independent. Not likely to be true.
	for i in range(np.size(z0_int_f)):
		prop_int_indexed[(prop_int_indexed[:,0] >= z0_int_f[i]) & (prop_int_indexed[:,0] < z1_int_f[i]),4] = i
		prop_int = prop[(z_prop >= z0_int_f[i]) & (z_prop < z1_int_f[i])]
		sigma_prop_int = sigma_prop[(z_prop >= z0_int_f[i]) & (z_prop < z1_int_f[i])]
		### Calculate weighted arithmetic mean where weights are given by 1 / (variance squared) i.e. 1 / (sigma_prop_int squared)
		mean_prop_int[i], std_mean_prop_int[i], weighted_mean_prop_int[i], std_weighted_mean_prop_int[i] = general_python_functions.unweighted_weighted_mean(prop_int, np.power(sigma_prop_int, -2))
		hmean_prop_int[i], sigma_hmean_prop_int[i] = general_python_functions.harmonic_mean(prop_int, sigma_prop_int)
		gmean_prop_int[i], sdfactor_gmean_prop_int[i], lower_bound_gmean_prop_int[i], upper_bound_gmean_prop_int[i], gmean_minus_prop_int[i], gmean_plus_prop_int[i] = general_python_functions.geometric_mean(prop_int, sigma_prop_int)
	
	mean_prop_int_plot = np.concatenate(np.column_stack((mean_prop_int, mean_prop_int)))
	std_mean_prop_int_plot = np.concatenate(np.column_stack((std_mean_prop_int, std_mean_prop_int)))
	weighted_mean_prop_int_plot = np.concatenate(np.column_stack((weighted_mean_prop_int, weighted_mean_prop_int)))
	std_weighted_mean_prop_int_plot = np.concatenate(np.column_stack((std_weighted_mean_prop_int, std_weighted_mean_prop_int)))
	hmean_prop_int_plot = np.concatenate(np.column_stack((hmean_prop_int, hmean_prop_int)))
	sigma_hmean_prop_int_plot = np.concatenate(np.column_stack((sigma_hmean_prop_int, sigma_hmean_prop_int)))
	gmean_prop_int_plot = np.concatenate(np.column_stack((gmean_prop_int, gmean_prop_int)))
	sdfactor_gmean_prop_int_plot = np.concatenate(np.column_stack((sdfactor_gmean_prop_int, sdfactor_gmean_prop_int)))
	lower_bound_gmean_prop_int_plot = np.concatenate(np.column_stack((lower_bound_gmean_prop_int, lower_bound_gmean_prop_int)))
	upper_bound_gmean_prop_int_plot = np.concatenate(np.column_stack((upper_bound_gmean_prop_int, upper_bound_gmean_prop_int)))
	gmean_minus_prop_int_plot = np.concatenate(np.column_stack((gmean_minus_prop_int, gmean_minus_prop_int)))
	gmean_plus_prop_int_plot = np.concatenate(np.column_stack((gmean_plus_prop_int, gmean_plus_prop_int)))
	
	return(prop_int_indexed, mean_prop_int, std_mean_prop_int, weighted_mean_prop_int, std_weighted_mean_prop_int, hmean_prop_int, sigma_hmean_prop_int, gmean_prop_int, sdfactor_gmean_prop_int, lower_bound_gmean_prop_int, upper_bound_gmean_prop_int, gmean_minus_prop_int, gmean_plus_prop_int, mean_prop_int_plot, std_mean_prop_int_plot, weighted_mean_prop_int_plot, std_weighted_mean_prop_int_plot, hmean_prop_int_plot, sigma_hmean_prop_int_plot, gmean_prop_int_plot, sdfactor_gmean_prop_int_plot, lower_bound_gmean_prop_int_plot, upper_bound_gmean_prop_int_plot, gmean_minus_prop_int_plot, gmean_plus_prop_int_plot)

# ----------------------------------------------------------------

def estimate_q_bullard_whole_borehole(R_f, sigma_R_f, T_f, sigma_T_f, bullard_sense_option, R_T_fit_option):
	if R_T_fit_option == 'T_versus_R':
		# Fit straight line to T versus R using least squares method
		if sigma_T_f is None:
			sigma_T_f = np.ones(np.size(T_f))
		q, sigma_q, c, sigma_c, cov_q_c, r_q_c, dof, q_c_chi_sq, q_c_reduced_chi_sq, q_c_chi_sq_sig_gamma, q_c_chi_sq_sig_ppf = general_python_functions.fit_straight_line_variable_errors_direct_expressions(R_f, T_f, sigma_T_f)
		invq, sigma_invq, invc, sigma_invc, cov_q_invc, r_q_invc, q_invc_chi_sq, q_invc_reduced_chi_sq, q_invc_chi_sq_sig_gamma, q_invc_chi_sq_sig_ppf = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
		sigma_T_f = None
		
	elif R_T_fit_option == 'R_versus_T':
		# Fit straight line to R versus T using least squares method
		if sigma_R_f is None:
			sigma_R_f = np.ones(np.size(T_f))
		invq, sigma_invq, invc, sigma_invc, cov_q_invc, r_q_invc, dof, q_invc_chi_sq, q_invc_reduced_chi_sq, q_invc_chi_sq_sig_gamma, q_invc_chi_sq_sig_ppf = general_python_functions.fit_straight_line_variable_errors_direct_expressions(T_f, R_f, sigma_R_f)
		q = 1 / invq
		sigma_q = sigma_invq / np.power(invq, 2)
		c, sigma_c, cov_q_c, r_q_c, q_c_chi_sq, q_c_reduced_chi_sq, q_c_chi_sq_sig_gamma, q_c_chi_sq_sig_ppf = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
		sigma_T_f = None
	
	Q = q*1e3
	sigma_Q = sigma_q*1e3
	q_round, sigma_q_round = general_python_functions.round_to_sf(q, sigma_q)
	Q_round, sigma_Q_round = general_python_functions.round_to_sf(Q, sigma_Q)
	c_round, sigma_c_round = general_python_functions.round_to_sf(c, sigma_c)
	
	if bullard_sense_option == 'upward':
		q = -1*q
		Q = -1*Q
		q_round = -1*q_round
		Q_round = -1*Q_round
	
	# print(bullard_sense_option, R_T_fit_option)
	# print(Q_round, sigma_Q_round)
	# print(c, sigma_c)
	
		
	# TODO Add chi-squared measures etc
	
	return(q, sigma_q, q_round, sigma_q_round, Q, sigma_Q, Q_round, sigma_Q_round, c, sigma_c, c_round, sigma_c_round, invq, sigma_invq, invc, sigma_invc)
	
# ----------------------------------------------------------------
	
def estimate_heat_flows_from_k_layers_and_T_depths_bullard_plot(zk0_m, zk0_error_m, zk1_m, zk1_error_m, k, k_error, zTk_m, zTk_error_m, T_k, T_k_error):
	
	# Set up files for writing out data
	# T_versus_R_downward_outfile = outfile_root + "_TvR_downward"
	# R_versus_T_downward_outfile = outfile_root + "_RvT_downward"
	# T_versus_R_upward_outfile = outfile_root + "_TvR_upward"
	# R_versus_T_upward_outfile = outfile_root + "_RvT_upward"
	
	### Estimate heat flow using thermal resistances and temperatures measured downwards from top of borehole
	# Calculate layer thicknesses. All layers must start from surface (i.e. first value of zk0_m is 0 m)
	delta_zk_m, delta_zk_error_m = calculate_layer_thicknesses(zk0_m, zk1_m, zk0_error_m, zk1_error_m)
	# Calculate thermal resistance at depths of temperature measurements (in this case these depths are the same as the base of thermal conductivity layers)
	### TODO Return values of conductivity at these depths
	R1_downward, sigma_R1_downward = calculate_thermal_resistance_temperature_depths(zk1_m, zk1_error_m, zk0_m, zk0_error_m, delta_zk_m, delta_zk_error_m, zTk_m, zTk_error_m, k, k_error)
	# Estimate heat flow using Bullard plot of T versus R and save to file
	bullard_sense_option = 'downward'
	R_T_fit_option = 'T_versus_R'
	q_downward_TvR, sigma_q_downward_TvR, q_round_downward_TvR, sigma_q_round_downward_TvR, Q_downward_TvR, sigma_Q_downward_TvR, Q_round_downward_TvR, sigma_Q_round_downward_TvR, c_downward_TvR, sigma_c_downward_TvR, c_round_downward_TvR, sigma_c_round_downward_TvR, invq_downward_TvR, sigma_invq_downward_TvR, invc_downward_TvR, sigma_invc_downward_TvR = estimate_q_bullard_whole_borehole(R1_downward, sigma_R1_downward, T_k, T_k_error, bullard_sense_option, R_T_fit_option)
	# ftemp = open(T_versus_R_downward_outfile + '.csv', 'w')
# 	ftemp.write("### description, bullard_sense_option, z_top, z_base, R_T_fit_option, q, sigma_q, q_round, sigma_q_round, Q, sigma_Q, Q_round, sigma_Q_round, c, sigma_c, c_round, sigma_c_round, invq, sigma_invq, invc, sigma_invc\n")
# 	ftemp.write("### values, %s, %f, %f, %s, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n" % (bullard_sense_option, zTk_m[0], zTk_m[-1], R_T_fit_option, q, sigma_q, q_round, sigma_q_round, Q, sigma_Q, Q_round, sigma_Q_round, c, sigma_c, c_round, sigma_c_round, invq, sigma_invq, invc, sigma_invc))
# 	dtemp = {'zTk_m':zTk_m, 'zTk_error_m':zTk_error_m, 'T':T_k, 'T_error':T_k_error, 'R1_downward':R1_downward, 'sigma_R1_downward':sigma_R1_downward}
# 	ftemp_df = pd.DataFrame(data=dtemp)
# 	ftemp_df.to_csv(ftemp, index=False, header=True)
# 	ftemp.close()
	# Estimate heat flow using Bullard plot of R versus T and save to file
	R_T_fit_option = 'R_versus_T'
	q_downward_RvT, sigma_q_downward_RvT, q_round_downward_RvT, sigma_q_round_downward_RvT, Q_downward_RvT, sigma_Q_downward_RvT, Q_round_downward_RvT, sigma_Q_round_downward_RvT, c_downward_RvT, sigma_c_downward_RvT, c_round_downward_RvT, sigma_c_round_downward_RvT, invq_downward_RvT, sigma_invq_downward_RvT, invc_downward_RvT, sigma_invc_downward_RvT = estimate_q_bullard_whole_borehole(R1_downward, sigma_R1_downward, T_k, T_k_error, bullard_sense_option, R_T_fit_option)
	# ftemp = open(R_versus_T_downward_outfile + '.csv', 'w')
# 	ftemp.write("### description, bullard_sense_option, z_top, z_base, R_T_fit_option, q, sigma_q, q_round, sigma_q_round, Q, sigma_Q, Q_round, sigma_Q_round, c, sigma_c, c_round, sigma_c_round, invq, sigma_invq, invc, sigma_invc\n")
# 	ftemp.write("### values, %s, %f, %f, %s, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n" % (bullard_sense_option, zTk_m[0], zTk_m[-1], R_T_fit_option, q, sigma_q, q_round, sigma_q_round, Q, sigma_Q, Q_round, sigma_Q_round, c, sigma_c, c_round, sigma_c_round, invq, sigma_invq, invc, sigma_invc))
# 	dtemp = {'zTk_m':zTk_m, 'zTk_error_m':zTk_error_m, 'T':T_k, 'T_error':T_k_error, 'R1_downward':R1_downward, 'sigma_R1_downward':sigma_R1_downward}
# 	ftemp_df = pd.DataFrame(data=dtemp)
# 	ftemp_df.to_csv(ftemp, index=False, header=True)
# 	ftemp.close()
	
	### Estimate heat flow using thermal resistances and temperatures measured upwards from base of borehole
	# Reverse sense of layer boundaries, temperature measurements and conductivity values. zk1_m_rev and zk0_m_rev still describe the same boundaries, but zk1_m_rev now starts from 0
	zk1_m_rev, zk1_error_m_rev, zk0_m_rev, zk0_error_m_rev, k_rev, k_error_rev, zTk_m_rev, zTk_error_m_rev, T_k_rev, T_k_error_rev = measure_depths_from_bottom_hole(zk1_m, zk1_error_m, zk0_m, zk0_error_m, k, k_error, zTk_m, zTk_error_m, T_k, T_k_error)
	# Calculate layer thicknesses. All layers must start from surface (i.e. first value of zk0_m is 0 m)
	delta_zk_m_rev, delta_zk_error_m_rev = calculate_layer_thicknesses(zk1_m_rev, zk0_m_rev, zk1_error_m_rev, zk0_error_m_rev)
	# Calculate thermal resistance at depths of temperature measurements
	R1_upward, sigma_R1_upward = calculate_thermal_resistance_temperature_depths(zk0_m_rev, zk0_error_m_rev, zk1_m_rev, zk1_error_m_rev, delta_zk_m_rev, delta_zk_error_m_rev, zTk_m_rev, zTk_error_m_rev, k_rev, k_error_rev)
	# Estimate heat flow using Bullard plot of T versus R and save to file
	bullard_sense_option = 'upward'
	R_T_fit_option = 'T_versus_R'
	q_upward_TvR, sigma_q_upward_TvR, q_round_upward_TvR, sigma_q_round_upward_TvR, Q_upward_TvR, sigma_Q_upward_TvR, Q_round_upward_TvR, sigma_Q_round_upward_TvR, c_upward_TvR, sigma_c_upward_TvR, c_round_upward_TvR, sigma_c_round_upward_TvR, invq_upward_TvR, sigma_invq_upward_TvR, invc_upward_TvR, sigma_invc_upward_TvR = estimate_q_bullard_whole_borehole(R1_upward, sigma_R1_upward, T_k_rev, T_k_error_rev, bullard_sense_option, R_T_fit_option)
	# ftemp = open(T_versus_R_upward_outfile + '.csv', 'w')
# 	ftemp.write("### description, bullard_sense_option, z_top, z_base, R_T_fit_option, q, sigma_q, q_round, sigma_q_round, Q, sigma_Q, Q_round, sigma_Q_round, c, sigma_c, c_round, sigma_c_round, invq, sigma_invq, invc, sigma_invc\n")
# 	ftemp.write("### values, %s, %f, %f, %s, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n" % (bullard_sense_option, zTk_m[0], zTk_m[-1], R_T_fit_option, q, sigma_q, q_round, sigma_q_round, Q, sigma_Q, Q_round, sigma_Q_round, c, sigma_c, c_round, sigma_c_round, invq, sigma_invq, invc, sigma_invc))
# 	dtemp = {'zTk_m':zTk_m, 'zTk_error_m':zTk_error_m, 'T':T_k, 'T_error':T_k_error, 'R1_downward':R1_downward, 'sigma_R1_downward':sigma_R1_upward}
# 	ftemp_df = pd.DataFrame(data=dtemp)
# 	ftemp_df.to_csv(ftemp, index=False, header=True)
# 	ftemp.close()
	# Estimate heat flow using Bullard plot of R versus T and save to file
	R_T_fit_option = 'R_versus_T'
	q_upward_RvT, sigma_q_upward_RvT, q_round_upward_RvT, sigma_q_round_upward_RvT, Q_upward_RvT, sigma_Q_upward_RvT, Q_round_upward_RvT, sigma_Q_round_upward_RvT, c_upward_RvT, sigma_c_upward_RvT, c_round_upward_RvT, sigma_c_round_upward_RvT, invq_upward_RvT, sigma_invq_upward_RvT, invc_upward_RvT, sigma_invc_upward_RvT = estimate_q_bullard_whole_borehole(R1_upward, sigma_R1_upward, T_k_rev, T_k_error_rev, bullard_sense_option, R_T_fit_option)
	# ftemp = open(R_versus_T_upward_outfile + '.csv', 'w')
# 	ftemp.write("### description, bullard_sense_option, z_top, z_base, R_T_fit_option, q, sigma_q, q_round, sigma_q_round, Q, sigma_Q, Q_round, sigma_Q_round, c, sigma_c, c_round, sigma_c_round, invq, sigma_invq, invc, sigma_invc\n")
# 	ftemp.write("### values, %s, %f, %f, %s, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n" % (bullard_sense_option, zTk_m[0], zTk_m[-1], R_T_fit_option, q, sigma_q, q_round, sigma_q_round, Q, sigma_Q, Q_round, sigma_Q_round, c, sigma_c, c_round, sigma_c_round, invq, sigma_invq, invc, sigma_invc))
# 	dtemp = {'zTk_m':zTk_m, 'zTk_error_m':zTk_error_m, 'T':T_k, 'T_error':T_k_error, 'R1_downward':R1_downward, 'sigma_R1_downward':sigma_R1_upward}
# 	ftemp_df = pd.DataFrame(data=dtemp)
# 	ftemp_df.to_csv(ftemp, index=False, header=True)
# 	ftemp.close()
	
	return(R1_downward, sigma_R1_downward, R1_upward, sigma_R1_upward, q_downward_TvR, sigma_q_downward_TvR, q_round_downward_TvR, sigma_q_round_downward_TvR, Q_downward_TvR, sigma_Q_downward_TvR, Q_round_downward_TvR, sigma_Q_round_downward_TvR, c_downward_TvR, sigma_c_downward_TvR, c_round_downward_TvR, sigma_c_round_downward_TvR, q_downward_RvT, sigma_q_downward_RvT, q_round_downward_RvT, sigma_q_round_downward_RvT, Q_downward_RvT, sigma_Q_downward_RvT, Q_round_downward_RvT, sigma_Q_round_downward_RvT, invq_downward_RvT, sigma_invq_downward_RvT, invc_downward_RvT, sigma_invc_downward_RvT, q_upward_TvR, sigma_q_upward_TvR, q_round_upward_TvR, sigma_q_round_upward_TvR, Q_upward_TvR, sigma_Q_upward_TvR, Q_round_upward_TvR, sigma_Q_round_upward_TvR, c_upward_TvR, sigma_c_upward_TvR, c_round_upward_TvR, sigma_c_round_upward_TvR, q_upward_RvT, sigma_q_upward_RvT, q_round_upward_RvT, sigma_q_round_upward_RvT, Q_upward_RvT, sigma_Q_upward_RvT, Q_round_upward_RvT, sigma_Q_round_upward_RvT, invq_upward_RvT, sigma_invq_upward_RvT, invc_upward_RvT, sigma_invc_upward_RvT)
	
# ----------------------------------------------------------------

def estimate_heat_flows_interval_method(zT_f, zT_f_error, mean_k_f, std_mean_k_f, hmean_k_f, sigma_hmean_k_f, gmean_k_f, sigma_gmean_k_f, T_f, T_error_f, outfile):
	
	# Estimate dT/dz
	dTdz, sigma_dTdz, T0, sigma_T0, cov_dTdz_T0, r_dTdz_T0, dof, dTdz_T0_chi_sq, dTdz_T0_reduced_chi_sq, dTdz_T0_chi_sq_sig_gamma, dTdz_T0_chi_sq_sig_ppf = general_python_functions.fit_straight_line_variable_errors_direct_expressions(zT_f, T_f, T_error_f)
	
	# print(dTdz, sigma_dTdz, "dTdz, sigma_dTdz")
	
	# Estimate heat flow using interval method:
	# Arithmetic mean
	q_int_mean, sigma_q_int_mean = heat_flow_from_constant_conductivity(mean_k_f, std_mean_k_f, dTdz, sigma_dTdz)
	# Harmonic mean - TODO How to properly incorporate unequal errors in harmonic mean? For now using std_mean_k_int. Can't assume Gaussian
	q_int_hmean, sigma_q_int_hmean = heat_flow_from_constant_conductivity(hmean_k_f, std_mean_k_f, dTdz, sigma_dTdz)
	# Geometric mean - TODO How to properly incorporate unequal errors in geometric mean? For now using std_mean_k_int. Can't assume Gaussian
	q_int_gmean, sigma_q_int_gmean = heat_flow_from_constant_conductivity(gmean_k_f, std_mean_k_f, dTdz, sigma_dTdz)


	# Round results to appropriate number of significant figures
	dTdz_round, sigma_dTdz_round = general_python_functions.round_to_sf(dTdz, sigma_dTdz)
	T0_round, sigma_T0_round = general_python_functions.round_to_sf(T0, sigma_T0)
	q_int_mean_round, sigma_q_int_mean_round = general_python_functions.round_to_sf(q_int_mean, sigma_q_int_mean)
	Q_int_mean_round, sigma_Q_int_mean_round = general_python_functions.round_to_sf(1e3*q_int_mean, 1e3*sigma_q_int_mean)
	q_int_gmean_round, sigma_q_int_gmean_round = general_python_functions.round_to_sf(q_int_gmean, sigma_q_int_gmean)
	Q_int_gmean_round, sigma_Q_int_gmean_round = general_python_functions.round_to_sf(1e3*q_int_gmean, 1e3*sigma_q_int_gmean)
	q_int_hmean_round, sigma_q_int_hmean_round = general_python_functions.round_to_sf(q_int_hmean, sigma_q_int_hmean)
	Q_int_hmean_round, sigma_Q_int_hmean_round = general_python_functions.round_to_sf(1e3*q_int_hmean, 1e3*sigma_q_int_hmean)
	
	# print(Q_int_mean_round, sigma_Q_int_mean_round)
	# print(Q_int_gmean_round, sigma_Q_int_gmean_round)
	# print(Q_int_hmean_round, sigma_Q_int_hmean_round)
	
	return()

# ----------------------------------------------------------------

def estimate_mean_conductivities(layer_k_input, layer_k_error_input, layer_min_k_input, layer_max_k_input, layer_delta_z_m_input, layer_delta_z_error_m_input, k_suffix_mc, k_distribution, monte_carlo_k_option):
	### TODO Assume that all conductivity above top of UKOGL well tops is drift deposits (use mudrock conductivity?)
	if monte_carlo_k_option == 'yes':
		### Calculate mean conductivity for whole borehole
		# Calculate unweighted and weighted arithmetic mean where weights are given by 1 / (variance squared) i.e. 1 / (sigma_prop_int squared)
		mean_k_all, std_mean_k_all, sigma_weighted_mean_k_all, stdev_sigma_weighted_mean_k_all = np.mean(layer_k_input), None, np.mean(layer_k_input), None
		# Calculate unweighted harmonic mean
		hmean_k_all, sigma_hmean_k_all = general_python_functions.unweighted_harmonic_mean(layer_k_input)
		sigma_hmean_k_all = None
		# Calculate unweighted geometric mean
		gmean_k_all, sdfactor_gmean_k_all, lower_bound_gmean_k_all, upper_bound_gmean_k_all, gmean_minus_k_all, gmean_plus_k_all = general_python_functions.unweighted_geometric_mean(layer_k_input)
		sdfactor_gmean_k_all, lower_bound_gmean_k_all, upper_bound_gmean_k_all, gmean_minus_k_all, gmean_plus_k_all = None, None, None, None, None
	else:
		### Analysis for case in which conductivity errors are assumed to follow normal distribution
		if k_distribution == 'normal' or k_distribution == 'in_situ_normal':
			# TODO Use these mean conductivities to estimate heat flows based on bottom-hole temperatures
			### Calculate mean conductivity for whole borehole
			# Calculate unweighted and weighted arithmetic mean where weights are given by 1 / (variance squared) i.e. 1 / (sigma_prop_int squared)
			mean_k_all, std_mean_k_all, sigma_weighted_mean_k_all, stdev_sigma_weighted_mean_k_all = general_python_functions.unweighted_weighted_arithmetic_mean(layer_k_input, np.power(layer_k_error_input, -2))
			# Calculate unweighted harmonic mean
			hmean_k_all, sigma_hmean_k_all = general_python_functions.unweighted_harmonic_mean(layer_k_input)
			# Calculate unweighted geometric mean
			gmean_k_all, sdfactor_gmean_k_all, lower_bound_gmean_k_all, upper_bound_gmean_k_all, gmean_minus_k_all, gmean_plus_k_all = general_python_functions.unweighted_geometric_mean(layer_k_input)
		else: # TODO Sort out for uniform distribution
			mean_k_all, std_mean_k_all, sigma_weighted_mean_k_all, stdev_sigma_weighted_mean_k_all, hmean_k_all, sigma_hmean_k_all, gmean_k_all, sdfactor_gmean_k_all, lower_bound_gmean_k_all, upper_bound_gmean_k_all, gmean_minus_k_all, gmean_plus_k_all = None, None, None, None, None, None, None, None, None, None, None, None
	
	### TODO Calculate depth-weighted means (arithmetic, geometric, harmonic) of conductivity for whole borehole
	### i.e. each value is weighted by the depth interval in which it is found
	mean_k_all, std_mean_k_all, depth_weighted_mean_k_all, stdev_depth_weighted_mean_k_all = general_python_functions.unweighted_weighted_arithmetic_mean(layer_k_input, layer_delta_z_m_input)
	
	### Calculate depth-weighted means (arithmetic, geometric, harmonic) of thermal conductivity above base of each layer (i.e. z1_m)
	# TODO Sort out proper estimation of uncertainty on k_hmean_z. Add geometric mean
	depth_weighted_mean_k_z1, sigma_depth_weighted_mean_k_z1, depth_weighted_hmean_k_z1, sigma_depth_weighted_hmean_k_z1 = estimate_depth_mean(layer_delta_z_m_input, layer_k_input)
	
	return(mean_k_all, std_mean_k_all, sigma_weighted_mean_k_all, stdev_sigma_weighted_mean_k_all, hmean_k_all, sigma_hmean_k_all, gmean_k_all, sdfactor_gmean_k_all, lower_bound_gmean_k_all, upper_bound_gmean_k_all, gmean_minus_k_all, gmean_plus_k_all, depth_weighted_mean_k_all, stdev_depth_weighted_mean_k_all, depth_weighted_mean_k_z1, sigma_depth_weighted_mean_k_z1, depth_weighted_hmean_k_z1, sigma_depth_weighted_hmean_k_z1)
	
# ----------------------------------------------------------------

def estimate_mc_heat_flow_distribution_statistics(q_array, sigma_q_array, c_array):
	# Unweighted averages
	q_array_mean = np.nanmean(q_array)
	q_array_stdev = np.nanstd(q_array)
	c_median = np.nanmedian(c_array)
	Q_array_mean_round, Q_array_stdev_round = general_python_functions.round_to_sf(1e3*q_array_mean, 1e3*q_array_stdev)
	q_array_P10 = np.percentile(q_array, 10)
	q_array_P50 = np.percentile(q_array, 50)
	q_array_P90 = np.percentile(q_array, 90)
	
	# TODO Add weighted averages
	
	return(q_array_mean, q_array_stdev, c_median, Q_array_mean_round, Q_array_stdev_round, q_array_P10, q_array_P50, q_array_P90)
		
	
	
	
	
	
	
	
	
	
	
	
	
	