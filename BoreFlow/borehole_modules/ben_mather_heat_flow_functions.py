import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import csv
import os
from scipy.interpolate import UnivariateSpline
from scipy import odr
from scipy.special import erfc
from scipy import optimize


# All functions here are imported from Ben Mather notebook (some have been modified) - see https://github.com/brmather/palaeoclimate_bullard_plot

def trim_borehole(z_T_f, sigma_z_T_f, T_f, sigma_T_f, z_k_f, sigma_z_k_f, k_f, sigma_k_f):
	
#     z, sigma_z, T, sigma_T   = np.loadtxt(temperatures_file, delimiter=',', unpack=True, skiprows=1, usecols=(0,1,2,3), dtype=str)
#     z_k, sigma_z_k, k, sigma_k = np.loadtxt(conductivities_file, delimiter=',', unpack=True, skiprows=1, usecols=(0,1,2,3), dtype=str)
    
#     print(np.size(z), np.size(T), np.size(sigma_T), np.size(z_k), np.size(k), np.size(sigma_k))
        
#     def trim_columns(*args):
#         trim_args = [None]*len(args)
#         for a, arg in enumerate(args):
#             trim_args[a] = arg[arg != ''].astype(float)
#         return trim_args
            
#     # trim columns
#     z, sigma_z, T, sigma_T, z_k, sigma_z_k, k, sigma_k = trim_columns(z, T, sigma_T, z_k, k, sigma_k)
    
	# Trim z_k and k values that are deeper than the deepest meaured temperature 
	mask = z_k_f <= z_T_f.max()
	z_k_f = z_k_f[mask]
	sigma_z_k_f = sigma_z_k_f[mask]
	k_f = k_f[mask]
	sigma_k_f = sigma_k_f[mask]

	return(z_T_f, sigma_z_T_f, T_f, sigma_T_f, z_k_f, sigma_z_k_f, k_f, sigma_k_f)

#----------------------------------------------------------------------------------------

def thermal_resistance(z, k, sigma_z, sigma_k):
	delta_z = np.diff(np.hstack([[0.0], z]))
	R = np.cumsum(delta_z/k)
	invk = delta_z/k
	dinvk = np.sqrt((sigma_k/k)**2 + (sigma_z/delta_z)**2) * invk
	sigma_R = np.sqrt(np.cumsum(dinvk**2))
	return(R, sigma_R)

#----------------------------------------------------------------------------------------

def thermal_resistance_no_errors(z, k):
	delta_z = np.diff(np.hstack([[0.0], z]))
	R = np.cumsum(delta_z/k)
	return(R)

#----------------------------------------------------------------------------------------

def least_squares(R, T):
	A = np.vstack([R, np.ones(len(R))]).T
	m, c = np.linalg.lstsq(A, T, rcond=None)[0]
	return m, c

#----------------------------------------------------------------------------------------

def linear_func(p, x):
    m, c = p
    return m*x + c

#----------------------------------------------------------------------------------------

# def climate_correction_period(kappa, t0, t1, Tk, z):
# 	# plt.plot(Tk*(erfc(z/(2.0*np.sqrt(kappa*t1))) - erfc(z/(2.0*np.sqrt(kappa*t0)))), z)
# 	# plt.show()
# 	# print(Tk*(erfc(z/(2.0*np.sqrt(kappa*t1))) - erfc(z/(2.0*np.sqrt(kappa*t0)))))
# 	return Tk*(erfc(z/(2.0*np.sqrt(kappa*t1))) - erfc(z/(2.0*np.sqrt(kappa*t0))))

#----------------------------------------------------------------------------------------

def odr_model(func, R, T, sigma_R, sigma_T):
    model = odr.Model(func)
    data = odr.RealData(R, T, sx=sigma_R, sy=sigma_T)
    
    A = np.vstack([R, np.ones(len(R))]).T
    m, c = np.linalg.lstsq(A, T_k)[0]
    x0 = [m, c]
    
    reg = odr.ODR(data, model, beta0=x0)
    out = reg.run()
    return out

#----------------------------------------------------------------------------------------

# def MC_simulation(nsim, t0, t1, Tk, z, k, cp=800.0, rho=2700.0):
#     delT = np.zeros((nsim, z.size))
#     k_mean = k.mean()
#     for i in range(0, nsim):
#         t0_random = np.clip(t0, 1e-6, 1e99) #np.random.normal(t0, sigma_t0)
#         t1_random = np.clip(t1, 1e-6, 1e99) #np.random.normal(t1, sigma_t1)
#         Tk_random = np.random.normal(Tk[0], Tk[1])
#         k_random = np.clip(np.random.normal(k_mean, 0.25*k_mean), 0.001, 1e99)
#         kappa = k_random/(cp*rho)
#         delT[i] = climate_correction_period(kappa, t0_random, t1_random, Tk_random, z)
#     return np.mean(delT, axis=0), np.std(delT, axis=0)

#----------------------------------------------------------------------------------------

def spline_interp_temp(z_k_f, sigma_z_k_f, k_f, sigma_k_f, z_T_f, sigma_z_T_f, T_f, sigma_T_f):
	# Spline interpolation of temperature to depths at which have thermal conductivity data.
	if len(z_T_f) <= 3:
		sigma_z_T_k_f = sigma_z_T_f
		T_k_f = T_f
		sigma_T_k_f = sigma_T_f
	else:
		spl_f = UnivariateSpline(z_T_f, sigma_z_T_f, s=0) 
		sigma_z_T_k_f = spl_f(z_k_f) # Bit meaningless to interpolate sigma values
		spl_f = UnivariateSpline(z_T_f, T_f, s=0) # I think really want to interpolate R? Have set s=0 to fit values perfectly. Previously s was default value, which smoothed spline too much.
		T_k_f = spl_f(z_k_f)
		spl_f = UnivariateSpline(z_T_f, sigma_T_f, s=0)
		sigma_T_k_f = spl_f(z_k_f) # Bit meaningless to interpolate sigma values

	# Estimate errors in spline interpolation depths using Ben Mather approach
	sigma_z_k_f = np.zeros_like(z_k_f)
	for i in range(0, len(z_k_f)):
		sigma_z_k_f[i] = np.abs(z_k_f[i] - z_T_f).min()
	# TODO How to estimate errors in interpolated temperatures
	
	return(T_k_f, sigma_T_k_f, sigma_z_k_f)








