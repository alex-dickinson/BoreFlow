import numpy as np
import pandas as pd
# import dataclasses
import matplotlib.pyplot as plt


from BoreFlow import general_python_functions


def call_function():
	print('hello')

# Functions to be used with different data types - put into new file
def cut_top_depth(z_m_uncut, cut_top_m, suffix_root, *args):
	print('cut_top_depth')
	if cut_top_m == None:
		cut_list = [z_m_uncut, suffix_root]
		for arg in args: cut_list.append(arg)
	else:
		cut_indices = np.where(z_m_uncut > cut_top_m)[0]
		cut_list = [z_m_uncut[cut_indices], str(suffix_root) + 'ct' + str(cut_top_m) + 'm']
		for arg in args: cut_list.append(arg[cut_indices])
	return(tuple(cut_list))


def select_bottomhole(z_m_whole, bottomhole_option, suffix_root, cut_top_m, *args):
	if bottomhole_option == None:
		bottomhole_list = [z_m_whole, suffix_root, cut_top_m]
		for arg in args: bottomhole_list.append(arg)
	elif bottomhole_option == 'deepest':
		bottomhole_list = [z_m_whole[-1], str(suffix_root) + 'bthdpst' + str(z_m_whole[-1]) + 'm', None]
		for arg in args: bottomhole_list.append(arg[-1])
	elif type(bottomhole_option) == int or type(bottomhole_option) == float:
		bottomhole_index = (np.abs(z_m_whole - bottomhole_option)).argmin()
		bottomhole_list = [z_m_whole[bottomhole_index], str(suffix_root) + 'bthspec' + str(bottomhole_option) + 'm' + str(z_m_whole[bottomhole_index]) + 'm', None]
		for arg in args: bottomhole_list.append(arg[bottomhole_index])
	return(tuple(bottomhole_list))


def subsample_depth_series(x, suffix_root, subsample_type, subsample_factor, *args):
	if subsample_factor == None:
		subsampled_list = [x, str(suffix_root)]
		for arg in args: subsampled_list.append(arg)
	else:
		if subsample_type == 'regular':
			# Randomly select start index between 0 and subsample_factor-1
			start_index = np.random.randint(0, high=subsample_factor)
			# Select value at indices separated by subsample_factor
			x_subsampled = x[start_index::subsample_factor]
			subsampled_list = [x_subsampled, str(suffix_root) + '_ssreg' + str(subsample_factor)]
			for arg in args: subsampled_list.append(arg[start_index::subsample_factor])
		elif subsample_type == 'random':
			# Randomly draw (n / subsample_factor) indices from the indices of 1D array x, where n is size of x. Sort drawn indices into ascending order
			subsample_indices = np.sort(np.random.choice(np.arange(0, np.size(x), 1), replace=False, size=int((np.size(x)/subsample_factor))))
			x_subsampled = x[subsample_indices]
			subsampled_list = [x_subsampled, str(suffix_root) + '_ssrnd' + str(subsample_factor)]
			for arg in args: subsampled_list.append(arg[subsample_indices])
	return(tuple(subsampled_list))


def perturb_values(suffix_root, sampling_distribution, *args):
	if len(args) % 2 != 0:
		print('missing uncertainties in function perturb_values() - exiting')
		exit()
	for argindex in range(len(args[0::2])):
		if len(args[2*argindex]) != len(args[2*argindex+1]):
			print('arrays are not same length in function perturb_values() - exiting')
			exit()
		else:
			if sampling_distribution == 'normal' or sampling_distribution == 'in_situ_normal':
				if argindex == 0: perturbed_list = [str(suffix_root) + '_pertnorm']
				perturbed_list.append(np.random.normal(args[2*argindex], args[2*argindex+1]))
			elif sampling_distribution == 'uniform':
				if argindex == 0: perturbed_list = [str(suffix_root) + '_pertuni']
				perturbed_list.append(np.random.uniform(args[2*argindex], args[2*argindex+1]))
			else:
				print('sampling_distribution not specified in function perturb_values() - exiting')
	if len(perturbed_list) > 1:
		perturbed_list = tuple(perturbed_list)
	else:
		perturbed_list = np.array(perturbed_list)[0,:]
	return(perturbed_list)



# -------------------

class BoreHole:
	def __init__(self, cfg):
		self.config = cfg['borehole']
		self.data = general_python_functions.EmptyClass()
		self.data.borehole = general_python_functions.EmptyClass()
	
	def load_temperature_measurements(self, insuffix):
		temperatures_file = self.config['filenames']['input_temperature_file']
		if self.config['filenames']['temperatures_filetype'] == "excel":
			# Load Excel spreadsheet
			temperatures_df = pd.read_excel(temperatures_file + ".xlsx", usecols=["depth(m)","depth_quoted_error(m)","depth_assigned_error(m)","temperature(degrees_C)","temperature_quoted_error(degrees_C)","temperature_assigned_error(degrees_C)","use?"])
		# Remove rows that are not to be used and rename columns. Make sure NaN values are properly represented
		temperatures_df = temperatures_df.drop(index=temperatures_df[temperatures_df["use?"] == "n"].index).drop(columns=["use?"]).rename(columns={"depth(m)":"z_m", "depth_quoted_error(m)":'z_quoted_error_m', "depth_assigned_error(m)":'z_assigned_error_m', "temperature(degrees_C)":'T', "temperature_quoted_error(degrees_C)":"T_quoted_error", "temperature_assigned_error(degrees_C)":"T_assigned_error"}).fillna(value=np.nan)
		# Check that all temperature depths are monotonically increasing
		skip_borehole = "no"
		if temperatures_df['z_m'].is_monotonic_increasing == False:
			print("Temperature depths not monotonically increasing - skipping")
			skip_borehole='yes'
			
		setattr(self.data.borehole, insuffix, general_python_functions.EmptyClass())
		x = getattr(self.data.borehole, insuffix)
		# self.data.borehole.
		# self.data.borehole.temperatures_df = temperatures_df
		x.T_suffix = insuffix
		
		### Read temperature dataframe into individual numpy arrays for ease of use ###
		errors_option = self.config['errors_option']
		x.z = np.array(temperatures_df['z_m'].copy(deep=True))
		x.T = np.array(temperatures_df['T'].copy(deep=True))
		if errors_option == 'quoted':
			x.z_error = np.array(temperatures_df['z_quoted_error_m'].copy(deep=True))
			x.T_error = np.array(temperatures_df['T_quoted_error'].copy(deep=True))
		elif errors_option == 'assigned':
			x.z_error = np.array(temperatures_df['z_assigned_error_m'].copy(deep=True))
			x.T_error = np.array(temperatures_df['T_assigned_error'].copy(deep=True))
		
		outsuffix = insuffix
		return(outsuffix)
	
	
	def cut_top_temperature_measurements(self, insuffix):
		x = getattr(self.data.borehole, insuffix)
		
		T_cut_top_m = self.config['temperature_preprocessing']['T_cut_top_m']
		z_cut, outsuffix_cut, T_cut, z_error_cut, T_error_cut = cut_top_depth(x.z, T_cut_top_m, insuffix, x.T, x.z_error, x.T_error)
		
		setattr(self.data.borehole, outsuffix, general_python_functions.EmptyClass())
		y = getattr(self.data.borehole, outsuffix)
		y.z, y.z_error, y.T, y.T_error = z_cut, z_error_cut, T_cut, T_error_cut
		
		return(outsuffix_cut)
		
		
	def select_bottomhole_temperature(self):
		self.data.borehole.T_bottomhole_option = self.config['temperature_preprocessing']['T_bottomhole_option']
		self.data.borehole.zT_m_bottomhole, self.data.borehole.T_bottomhole_suffix, self.data.borehole.T_cut_top_m, self.data.borehole.T_bottomhole, self.data.borehole.zT_error_m_bottomhole, self.data.borehole.T_error_bottomhole = select_bottomhole(self.data.borehole.zT_m, self.data.borehole.T_bottomhole_option, self.data.borehole.T_suffix, self.data.borehole.T_cut_top_m, self.data.borehole.T, self.data.borehole.zT_error_m, self.data.borehole.T_error)
	
	
	def subsample_temperature(self, insuffix):
		x = getattr(self.data.borehole, insuffix)
		
		T_subsample_factor = self.config['temperature_preprocessing']['T_subsample_factor']
		T_subsample_type = self.config['temperature_preprocessing']['T_subsample_type']
		z_subsampled, outsuffix_subsampled, T_subsampled, z_error_subsampled, T_error_subsampled = subsample_depth_series(x.z, insuffix, T_subsample_type, T_subsample_factor, x.T, x.z_error, x.T_error)
		
		setattr(self.data.borehole, outsuffix_subsampled, general_python_functions.EmptyClass())
		y = getattr(self.data.borehole, outsuffix_subsampled)
		y.z, y.z_error, y.T, y.T_error = z_subsampled, z_error_subsampled, T_subsampled, T_error_subsampled
		
		return(outsuffix_subsampled)


	def perturb_temperature(self, insuffix):
		x = getattr(self.data.borehole, insuffix)
		
		T_errors_dist = self.config['temperature_preprocessing']['T_errors_dist']
		
		### Perturb depths and values of temperature measurements within error distribution
		outsuffix_perturbed, z_perturbed, T_perturbed = perturb_values(insuffix, T_errors_dist, x.z, x.z_error, x.T, x.T_error)
		# Sort perturbed values so that depths are monotonically increasing
		z_perturbed_sorted, T_perturbed_sorted = general_python_functions.sort_arrays(z_perturbed, T_perturbed)
		# Set uncertainties on perturbed values to zero
		z_error_perturbed, T_error_perturbed = None, None
		
		setattr(self.data.borehole, outsuffix_perturbed, general_python_functions.EmptyClass())
		y = getattr(self.data.borehole, outsuffix_perturbed)
		y.z, y.z_error, y.T, y.T_error = z_perturbed_sorted, z_error_perturbed, T_perturbed_sorted, T_error_perturbed
		
		return(outsuffix_perturbed)
	
	
	### TEMPORARY ---------
	# Temporary
	def plot_temperature(self):
		print('plotting')
		plt.plot(self.data.borehole.T, self.data.borehole.zT_m, 'x')
	
	def test_value_creation(self, suffix):
		
		setattr(self.data.borehole, suffix, general_python_functions.EmptyClass())
		
		xd = getattr(self.data.borehole, 'bigboy')
		xd.temp = np.array([1,2,3,4])
		
		
		
		# self.data.borehole.suffix = general_python_functions.EmptyClass()
# 		self.data.borehole.suffix.testdata = np.linspace(1,10,1)
		
		
		
		

		
		
		
		