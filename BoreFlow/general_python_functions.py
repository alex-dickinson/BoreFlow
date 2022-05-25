import os

import numpy as np
from scipy import optimize
from scipy import special
from scipy import stats
import math

import matplotlib.pyplot as plt

# ----------------------------------------------------------------
### FUNCTIONS FOR CREATING NEW DIRECTORIES

def set_up_directory(check_path):
	if not os.path.exists(check_path):
		os.makedirs(check_path)
	return()

# ----------------------------------------------------------------
### UNIT CONVERSIONS

def y2s(t):
	"years to seconds"
	return t*3.15567*1e7
def s2y(t):
	return t/3.15567/1e7

# ----------------------------------------------------------------

# Smooth data using a boxcar filter
def smooth_data_boxcar(xdata, ydata, smoothing_length):
	if smoothing_length % 2 == 1:
		ydata_smoothed = np.convolve(ydata, np.ones((smoothing_length,))/smoothing_length, mode='valid')
		xdata_smoothed_cutoff = xdata[int(np.floor(smoothing_length/2)):int(np.ceil(-1*smoothing_length/2))]
	elif smoothing_length % 2 == 0:
		ydata_smoothed = np.convolve(ydata, np.ones((smoothing_length,))/smoothing_length, mode='valid')
		xdata_smoothed_cutoff = xdata[int(smoothing_length/2 - 1):int(-1*smoothing_length/2)]
	return(xdata_smoothed_cutoff, ydata_smoothed)

# ----------------------------------------------------------------

# Sort arrays based on monotonic increasing of first array
def sort_arrays(sort_array, *args):
	indices = np.argsort(sort_array)
	sorted_array = sort_array[indices]
	sorted_list = [sorted_array]
	for argindex in range(len(args)):
		if args[argindex].ndim == 1:
			if len(args[argindex]) != len(sort_array):
				print('arrays are not same length in function sort_arrays() - exiting')
				exit()
			else:
				sorted_list.append(args[argindex][indices])
		elif args[argindex].ndim == 2:
			if len(args[argindex][:,0]) != len(sort_array):
				print('arrays are not same length in function sort_arrays() - exiting')
				exit()
			else:
				sorted_list.append(args[argindex][indices,:])
	if len(sorted_list) > 1:
		sorted_list = tuple(sorted_list)
	else:
		sorted_list = np.array(sorted_list)[0,:]
	return(sorted_list)

# ----------------------------------------------------------------

# Round values to number of significant figures determined by uncertainty
def round_to_sf(x, sigma_x): 
	if x == 0 or sigma_x == 0 or math.isnan(x) == True or math.isnan(sigma_x) == True:
		x_round = x
		sigma_x_round = sigma_x
	else:
		sigma_x_power = -int(np.floor(np.log10(abs(sigma_x))))
		sigma_x_first_digit = round(sigma_x, sigma_x_power) * np.power(10, float(sigma_x_power))
		
		if sigma_x_first_digit < 3:
			n = 2
		else:
			n = 1
		
		sigma_x_round = round(sigma_x, sigma_x_power + (n - 1)) 
		x_round = round(x, sigma_x_power + (n - 1))
		
		if sigma_x_round % 1 == 0:
			x_round = int(x_round)
			sigma_x_round = int(sigma_x_round)
		
	return(x_round, sigma_x_round)

# ----------------------------------------------------------------
# ----------------------------------------------------------------

### Estimate different means

def unweighted_weighted_arithmetic_mean(x, weight_x):
	if np.size(x) > 1:
		# Unweighted mean
		mean = np.mean(x)
		std_mean = np.std(x)
		# Weighted mean
		weighted_mean = np.sum(x * weight_x) / np.sum(weight_x)
		# TODO Sort out estimate of error on weighted mean. This expression assumes unit input variances.
		stderr_weighted_mean = (1 / np.power( (np.sum(weight_x)), 0.5))
		stdev_weighted_mean = np.power(np.size(x), 0.5) * stderr_weighted_mean
		# std_weighted_mean = (np.power((1 / np.sum(1 / np.power(sigma_x, 2))), 0.5))
	else:
		mean, std_mean, weighted_mean, stdev_weighted_mean = x, np.nan, x, np.nan
	return(mean, std_mean, weighted_mean, stdev_weighted_mean)

def unweighted_harmonic_mean(x):
	if np.size(x) > 1:
	# Unweighted harmonic mean
		hmean = stats.hmean(x)
		std_hmean = np.power( np.mean( np.power( ((1/x) - hmean), 2) ), 0.5)
		std_hmean = np.std(1/x)
		# TODO Check this in Norris (1940)
		sigma_hmean = np.power((1/hmean), 2) * np.std(1/x) / np.power((np.size(x)-1), 0.5)
	else:
		hmean, sigma_hmean = x, np.nan
	return(hmean, sigma_hmean)

def unweighted_geometric_mean(x):
	if np.size(x) > 1:
		# Unweighted geometric mean
		gmean = stats.gmean(x)
		sdfactor_gmean = stats.gstd(x)
		lower_bound_gmean = gmean / sdfactor_gmean
		upper_bound_gmean = gmean * sdfactor_gmean
		gmean_minus = gmean - lower_bound_gmean
		gmean_plus = upper_bound_gmean - gmean
	else:
		gmean, sdfactor_gmean, lower_bound_gmean, upper_bound_gmean, gmean_minus, gmean_plus = x, np.nan, x, x, 0, 0
	return(gmean, sdfactor_gmean, lower_bound_gmean, upper_bound_gmean, gmean_minus, gmean_plus)
	
	

# ----------------------------------------------------------------
# ----------------------------------------------------------------

### Different functions for straight line fitting
### Straight line fitting using least squares method. Returns both fitted gradient and fitted intercept, together with errors estimated from covariance matrix.
# Function for fitting straight line to data with errors in y but not x. This method assumes constant errors on y.
# def fit_straight_line_constant_errors_y(xdata, ydata, sigma_y, x0):
#     def func(params, R, ydata):
#         return (ydata - np.dot(R, params))
#
#     ### Organise x data into response matrix format.
#     R = np.column_stack((np.ones(np.size(xdata)), xdata))
#
#     line = optimize.leastsq(func, x0, args=(R,ydata), full_output=1)
#     c = line[0][0]
#     sigma_c = sigma_y*np.sqrt(line[1][0,0])
#     m = line[0][1]
#     sigma_m = sigma_y*np.sqrt(line[1][1,1])
#
#     return(m, sigma_m, c, sigma_c)

# ----------------------------------------------------------------

# Function for fitting straight line to data with errors in y but not x. Error on different values of y is not assumed to be constant. Assumes that errors on y are independent
# def fit_straight_line_variable_errors_y_curve_fit(xdata, ydata, sigma_y):
#     def func(xdata, m, c):
#         return (m*xdata + c)
#
# #     plt.errorbar(xdata, ydata, yerr=sigma_y)
#
#     popt, pcov = optimize.curve_fit(func, xdata, ydata)
#
# #     sigma=sigma_y, absolute_sigma=False
#
#     return(popt, pcov)

# ----------------------------------------------------------------

# Direct expressions for fitting line to data with non-constant errors in y. See pages 661-666 of Numerical Recipes (Second Edition).
# Wikipedia article on standard error: In regression analysis, the term "standard error" refers either to the square root of the reduced chi-squared statistic
def fit_straight_line_variable_errors_direct_expressions(xdata, ydata, sigma_y):
	S = np.sum(1. / np.power(sigma_y, 2.))
	Sx = np.sum(xdata / np.power(sigma_y, 2.))
	Sy = np.sum(ydata / np.power(sigma_y, 2.))
	Sxx = np.sum(np.power(xdata, 2.) / np.power(sigma_y, 2.))
	Sxy = np.sum(xdata * ydata / np.power(sigma_y, 2.))
	Delta = S * Sxx - np.power(Sx, 2) 
	c = (Sxx * Sy - Sx * Sxy) / Delta
	m = (S * Sxy - Sx * Sy) / Delta
	# Estimate uncertainties on gradient and intercept, their covariance and the correlation coefficient for their uncertainties
	sigma_c = np.power( (Sxx / Delta), 0.5)
	sigma_m = np.power( (S / Delta), 0.5)
	cov_m_c = -Sx / Delta
	r_mc = -Sx / np.power((S*Sxx), 0.5)
	# Check goodness of fit
	dof = np.size(xdata) - 2
	chi_sq = np.sum(np.power(((ydata[:] - (m*xdata[:] + c)) / sigma_y[:]), 2.))
	reduced_chi_sq = chi_sq / dof
	# Compute incomplete gamma function (see equation 6.2.1 in Numerical Recipes Edition 2)
	chi_sq_sig_gamma = special.gammainc((dof-2)/2, chi_sq/2)
	chi_sq_sig_ppf = stats.chi2.ppf(0.95, dof)
	return(m, sigma_m, c, sigma_c, cov_m_c, r_mc, dof, chi_sq, reduced_chi_sq, chi_sq_sig_gamma, chi_sq_sig_ppf)

def gradient_uncertainty(xdata, ydata):
	### Calculate root-mean-squared error in straight line.
	residual_squared = np.power((xdata[:] - ydata[:]), 2.)
	sigma_estimate = np.power((np.sum(np.power(residual_squared, 2.))/(np.size(residual_squared) - 2.)), 0.5)
	return(residual_squared, sigma_estimate)

# --------------
# Create empty class
class EmptyClass:
	pass
	
