import numpy as np
from scipy.special import erfc


# TODO Add error estimation
def halfspace_climate_correction(kappa, t0, t1, Ts, z):
	return(Ts*(erfc(z/(2.0*np.sqrt(kappa*t1))) - erfc(z/(2.0*np.sqrt(kappa*t0)))))





def run_climatic_corrections(monte_carlo_k_option, monte_carlo_T_option, monte_carlo_Tsurf_option, climatic_correction_type, rho_halfspace, sigma_rho_halfspace, cp_halfspace, sigma_cp_halfspace, k_halfspace, sigma_k_halfspace, zT, sigma_zT, layer_z0, layer_sigma_z0, layer_z1, layer_sigma_z1, k_distribution, layer_k, layer_sigma_k, layer_min_k, layer_max_k, t0_seconds, sigma_t0_seconds, t1_seconds, sigma_t1_seconds, deltaTs, sigma_deltaTs):
	
	### Correction for assumption of halfspace of constant thermal diffusivity
	if climatic_correction_type == 'cst_borehole_conductivity' or climatic_correction_type == 'cst_specified_conductivity':
		kappa_halfspace = k_halfspace / (cp_halfspace*rho_halfspace)
		deltaT_out = np.zeros(np.size(zT))
		sigma_deltaT_out = np.zeros(np.size(zT))
		for j in range(t0_seconds.size):
			deltaT_out += halfspace_climate_correction(kappa_halfspace, t0_seconds[j], t1_seconds[j], deltaTs[j], zT)
	if monte_carlo_k_option == 'yes' or monte_carlo_T_option == 'yes' or monte_carlo_Tsurf_option == 'yes':
		sigma_deltaT_out = None
	
	# TODO Add different options for climatic correction
	
	
	return(deltaT_out, sigma_deltaT_out)