import numpy as np
import pandas as pd
# import dataclasses
import matplotlib.pyplot as plt


from BoreFlow import general_python_functions
from BoreFlow import plotting_functions


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


def select_bottomhole(z_m_whole, bottomhole_option, suffix_root, *args):
	if bottomhole_option == None:
		bottomhole_list = [z_m_whole, suffix_root]
		for arg in args: bottomhole_list.append(arg)
	elif bottomhole_option == 'deepest':
		bottomhole_list = [z_m_whole[-1], str(suffix_root) + 'bthdpst' + str(z_m_whole[-1]) + 'm']
		for arg in args: bottomhole_list.append(arg[-1])
	elif type(bottomhole_option) == int or type(bottomhole_option) == float:
		bottomhole_index = (np.abs(z_m_whole - bottomhole_option)).argmin()
		bottomhole_list = [z_m_whole[bottomhole_index], str(suffix_root) + 'bthspec' + str(bottomhole_option) + 'm' + str(z_m_whole[bottomhole_index]) + 'm']
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
		
		setattr(self.data.borehole, outsuffix_cut, general_python_functions.EmptyClass())
		y = getattr(self.data.borehole, outsuffix_cut)
		y.z, y.z_error, y.T, y.T_error = z_cut, z_error_cut, T_cut, T_error_cut
		
		return(outsuffix_cut)
		
		
	def select_bottomhole_temperature(self, insuffix):
		x = getattr(self.data.borehole, insuffix)

		T_bottomhole_option = self.config['temperature_preprocessing']['T_bottomhole_option']

		z_bottomhole, outsuffix_bottomhole, T_bottomhole, z_error_bottomhole, T_error_bottomhole = select_bottomhole(x.z, T_bottomhole_option, insuffix, x.T, x.z_error, x.T_error)

		setattr(self.data.borehole, outsuffix_bottomhole, general_python_functions.EmptyClass())
		y = getattr(self.data.borehole, outsuffix_bottomhole)
		y.z, y.z_error, y.T, y.T_error = z_bottomhole, z_error_bottomhole, T_bottomhole, T_error_bottomhole

		return(outsuffix_bottomhole)
	
	
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
	def build_plots(self, plot_depth_limits, **kwargs):

		print(kwargs)




		
		# def build_plots(self, max_depth_m_plot, k_distribution, plot_lith_fill_dict, plot_lith_fill_dict_keyword, strat_interp_lith_k_calcs_df, T_plotting_dict, k_plotting_dict, res_plotting_dict, bullard_TvR_plotting_dict, bullard_RvT_plotting_dict, pc_plotting_dict, rc_plotting_dict, qhist_plotting_dict, T, figure_name):
		
		# Set up figure limits
		min_depth_m_plot = plot_depth_limits[0]
		max_depth_m_plot = plot_depth_limits[1]



		# TODO Move this to plotting_functions.py
		if 'T_plotting_dict' in kwargs:
			T_plotting_dict = kwargs['T_plotting_dict']
			if isinstance(T_plotting_dict['limits']['Tmin'], str) == True:
				Tlim_min = (getattr(self.data.borehole, T_plotting_dict['limits']['Tmin']).T)
				min_T = np.min(Tlim_min) - 0.05 * np.ptp(Tlim_min)
			elif isinstance(T_plotting_dict['limits']['Tmin'], float) == True or isinstance(T_plotting_dict['limits']['Tmin'], int) == True:
				min_T = T_plotting_dict['limits']['Tmin']
			else:
				print("minimum temp for plotting not specified - exiting")
				return
			if isinstance(T_plotting_dict['limits']['Tmax'], str) == True:
				Tlim_max = (getattr(self.data.borehole, T_plotting_dict['limits']['Tmax']).T)
				max_T = np.max(Tlim_max) + 0.05 * np.ptp(Tlim_max)
			elif isinstance(T_plotting_dict['limits']['Tmax'], float) == True or isinstance(T_plotting_dict['limits']['Tmax'], int) == True:
				max_T = T_plotting_dict['limits']['Tmax']
			else:
				print("maximum temp for plotting not specified - exiting")
				return
			
			print(min_T, max_T)
		
		



			# min_T=np.min(T) - 0.05 * np.ptp(T)
			# max_T=np.max(T) + 0.05 * np.ptp(T)

		



		
		min_k=0
		max_k=7
			
		# if 'layer_k_plot' in k_plotting_dict['line0']:
		# 	print('layer_k_plot')
		# 	min_k = np.min(k_plotting_dict['line0']['layer_k_plot']) - 0.05 * np.ptp(k_plotting_dict['line0']['layer_k_plot'])
		# 	max_k = np.max(k_plotting_dict['line0']['layer_k_plot']) + 0.05 * np.ptp(k_plotting_dict['line0']['layer_k_plot'])
		# elif 'min_k' in k_plotting_dict['line0'] and 'max_k' in k_plotting_dict['line0']:
		# 	print('min_k')
		# 	min_k = np.min(k_plotting_dict['line0']['min_k']) - 0.05 * (np.max(k_plotting_dict['line0']['max_k']) - np.min(k_plotting_dict['line0']['min_k']))
		# 	max_k = np.max(k_plotting_dict['line0']['max_k']) + 0.05 * (np.max(k_plotting_dict['line0']['max_k']) - np.min(k_plotting_dict['line0']['min_k']))
		# print(min_k, max_k)
		
		min_R=0
		max_R=300
		# if res_plotting_dict != None:
	# 		min_R = np.min(res_plotting_dict['line0']['R_plot']) - 0.05 * np.ptp(res_plotting_dict['line0']['R_plot'])
	# 		max_R = np.max(res_plotting_dict['line0']['R_plot']) + 0.05 * np.ptp(res_plotting_dict['line0']['R_plot'])
	# 	print(min_R, max_R, "min_R, max_R")
		
		min_Q = 25
		max_Q = 90
		

		
		
		# if axes != 'temp_only':
		# 	z_int_plot = np.concatenate(np.column_stack((strat_interp_lith_k_calcs_df['z0_m'], strat_interp_lith_k_calcs_df['z1_m'])))
		
		# Set up plotting preferences - define in YAML
		axis_setup_dict = {
			'figure_width':7.25,
			'figure_height':10,
			'init_xpos':0,
			'init_ypos':10,
			'figure_label_inset':0.125
		}
		
		axis_position_dict = {
			'temp':{'width':2, 'height':4, 'hspace_after':2.5, 'vspace_after':0},
			'strat':{'width':0.25, 'height':4, 'hspace_after':0.5, 'vspace_after':0},
			'cond':{'width':2, 'height':4, 'hspace_after':2.25, 'vspace_after':0},
			'res':{'width':2, 'height':4, 'hspace_after':-5.25, 'vspace_after':-2.25},
			'pc':{'width':3.5, 'height':2, 'hspace_after':3.75, 'vspace_after':0},
			'rc':{'width':3.5, 'height':2, 'hspace_after':4.75, 'vspace_after':3.25},
			'bullard':{'width':3.5, 'height':3, 'hspace_after':0, 'vspace_after':-3.25},
			'qhist':{'width':3.5, 'height':3, 'hspace_after':0, 'vspace_after':0}
			# 'row_spacings':[0.25,1], 'column_spacing':0.25
		}
		
		# # Set up plotting preferences
		# axis_setup_dict = {
		# 	'init_xpos':0,
		# 	'init_ypos':10,
		# 	'temp':{'width':2, 'height':4, 'hspace_after':0.5, 'vspace_after':0},
		# 	'strat':{'width':0.25, 'height':4, 'hspace_after':0.25, 'vspace_after':0},
		# 	'cond':{'width':2, 'height':4, 'hspace_after':0.5, 'vspace_after':0},
		# 	'res':{'width':2, 'height':4, 'hspace_after':-3.75, 'vspace_after':-4.25},
		# 	'pc':{'width':3.5, 'height':4, 'hspace_after':0.25, 'vspace_after':0},
		# 	'rc':{'width':3.5, 'height':4, 'hspace_after':-3.75, 'vspace_after':-3},
		# 	'bullard':{'width':3.5, 'height':4, 'hspace_after':0.25, 'vspace_after':0},
		# 	'figure_label_inset':0.125,
		# 	# 'row_spacings':[0.25,1], 'column_spacing':0.25
		# }
		# axis_position_dict = {
		# 	'row1':{'panels':['temp','column_space','column_space','strat','column_space','cond','column_space','res'], 'height':4, 'below_spacing'},
		# 	'row2':{'panels':['pc','column_space','rc'], 'height':2},
		# 	'row3':{'panels':['bullard','column_space','bullard'], 'height':2},
		# 	#'row2':['bullard','column_space','bullard']
		# }
		plot_format_dict = plotting_functions.set_up_plot_formatting_constant_height(axis_setup_dict, axis_position_dict)
		
		# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
		plt.rcParams['xtick.bottom'] = False
		plt.rcParams['xtick.labelbottom'] = False
		plt.rcParams['xtick.top'] = True
		plt.rcParams['xtick.labeltop'] = True
		
		fig = plt.figure(figsize=(plot_format_dict['figure_width'], plot_format_dict['figure_height']), dpi=100)
		
		# Plot temperature
		if 'T_plotting_dict' in kwargs:
			# TODO Move this to plotting_functions.py
			plot_format_dict_local = plot_format_dict['fractional_pos_dict']['axis_temp']
			ax00 = plotting_functions.set_up_temperature_axes(fig, plot_format_dict_local, min_depth_m_plot, max_depth_m_plot, min_T, max_T)
			spl_temp_option_f = 'no'
			# plot_temperatures(max_depth_m_plot, zT_m, T, zT_error_m, T_error, ax00)
			if T_plotting_dict['number_lines'] > 0:
				for line_number in range(T_plotting_dict['number_lines']):
					tempdict = T_plotting_dict['line' + str(line_number)]
					tempdict['T_plot'] = getattr(self.data.borehole, tempdict['suffix']).T
					tempdict['T_error_plot'] = getattr(self.data.borehole, tempdict['suffix']).T_error
					tempdict['z_plot'] = getattr(self.data.borehole, tempdict['suffix']).z
					tempdict['z_error_plot'] = getattr(self.data.borehole, tempdict['suffix']).z_error
					plotting_functions.plot_temperatures(tempdict, ax00, plot_format_dict_local, plot_format_dict)
		
		# Plot stratigraphy
		if 'strat_plotting_dict' in kwargs:
			if k_distribution != 'in_situ_normal':
				### TODO temp
				strat_label_option = 'no'
				plot_format_dict_local = plot_format_dict['fractional_pos_dict']['axis_strat']
				ax01 = fig.add_axes([plot_format_dict_local['frac_xpos'], plot_format_dict_local['frac_ypos'], plot_format_dict_local['panel_frac_width'], plot_format_dict_local['panel_frac_height']], ylabel=None)
				ax01.set_ylim(top=min_depth_m_plot, bottom=max_depth_m_plot)
				ax01.set_xlim(left=0, right=1)
				ax01.yaxis.set_ticks_position('both')
				ax01.xaxis.set_ticks_position('none')
				ax01.xaxis.set_minor_locator(MultipleLocator(5))
				ax01.yaxis.set_minor_locator(MultipleLocator(25))
				ax01.yaxis.set_major_locator(MultipleLocator(50))
				ax01.yaxis.set_major_formatter(FormatStrFormatter(''))
				ax01.xaxis.set_major_formatter(FormatStrFormatter(''))
				plot_stratigraphy(plot_lith_fill_dict, plot_lith_fill_dict_keyword, strat_interp_lith_k_calcs_df, z_int_plot, plot_format_dict, strat_label_option, ax01)
			
			# Plot conductivity
		if 'k_plotting_dict' in kwargs:
			plot_format_dict_local = plot_format_dict['fractional_pos_dict']['axis_cond']
			ax02 = fig.add_axes([plot_format_dict_local['frac_xpos'], plot_format_dict_local['frac_ypos'], plot_format_dict_local['panel_frac_width'], plot_format_dict_local['panel_frac_height']], xlabel=r'$k$ / W m$^{-1}$ K$^{-1}$', ylabel=None)
			ax02.set_ylim(top=min_depth_m_plot, bottom=max_depth_m_plot)
			ax02.set_xlim(left=min_k, right=max_k)
			ax02.yaxis.set_ticks_position('both')
			ax02.xaxis.set_ticks_position('both')
			ax02.xaxis.set_minor_locator(MultipleLocator(1))
			ax02.xaxis.set_major_locator(MultipleLocator(2))
			ax02.yaxis.set_minor_locator(MultipleLocator(25))
			ax02.yaxis.set_major_locator(MultipleLocator(50))
			ax02.yaxis.set_major_formatter(FormatStrFormatter(''))
			# ax02.xaxis.set_major_formatter(FormatStrFormatter(''))
			ax02.xaxis.set_label_position('top') 
			if k_plotting_dict['number_lines'] > 0:
				for line_number in range(k_plotting_dict['number_lines']):
					tempdict = k_plotting_dict['line' + str(line_number)]
					plot_conductivity(k_distribution, tempdict, ax02, plot_format_dict_local, plot_format_dict)
			
			# Plot palaeoclimatic records
		if 'pc_plotting_dict' in kwargs:
			plot_format_dict_local = plot_format_dict['fractional_pos_dict']['axis_pc']
			ax10 = fig.add_axes([plot_format_dict_local['frac_xpos'], plot_format_dict_local['frac_ypos'], plot_format_dict_local['panel_frac_width'], plot_format_dict_local['panel_frac_height']])
			ax10.set_ylabel(r'$\Delta T_s$ / $^{\circ}$C')
			ax10.set_xlabel('Time before present / ka')
			ax10.set_xlim(left=0, right=150)
			ax10.xaxis.set_label_position('bottom')
			# ax10.set_ylim(top=0, bottom=max_depth_m_plot)
		# 	ax10.set_xlim(left=min_k, right=max_k)
			ax10.xaxis.tick_bottom()
			ax10.yaxis.set_ticks_position('both')
			ax10.xaxis.set_ticks_position('both')
			ax10.xaxis.set_minor_locator(MultipleLocator(12.5))
			ax10.xaxis.set_major_locator(MultipleLocator(25))
			ax10.yaxis.set_minor_locator(MultipleLocator(2.5))
			ax10.yaxis.set_major_locator(MultipleLocator(5))
		# 	ax10.yaxis.set_major_formatter(FormatStrFormatter(''))
		# 	# ax10.xaxis.set_major_formatter(FormatStrFormatter(''))
		# 	ax10.xaxis.set_label_position('top')
			plot_palaeoclimate_subplot(ax10, plot_format_dict_local, pc_plotting_dict, plot_format_dict)
			
			
			# Plot recent climate records
		if 'rc_plotting_dict' in kwargs:
			plot_format_dict_local = plot_format_dict['fractional_pos_dict']['axis_rc']
			ax11 = fig.add_axes([plot_format_dict_local['frac_xpos'], plot_format_dict_local['frac_ypos'], plot_format_dict_local['panel_frac_width'], plot_format_dict_local['panel_frac_height']])
			ax11.xaxis.set_label_position('bottom')
			ax11.yaxis.set_label_position('right')
			# ax11.set_ylim(top=0, bottom=max_depth_m_plot)
		# 	ax11.set_xlim(left=min_k, right=max_k)
			ax11.yaxis.tick_right()
			ax11.xaxis.tick_bottom()
			ax11.yaxis.set_ticks_position('both')
			ax11.xaxis.set_ticks_position('both')
			ax11.set_ylabel(r'$\Delta T_s$ / $^{\circ}$C')
			ax11.set_xlabel('Time before borehole drilled / years')
			ax11.xaxis.set_minor_locator(MultipleLocator(12.5))
			ax11.xaxis.set_major_locator(MultipleLocator(25))
			ax11.yaxis.set_minor_locator(MultipleLocator(0.25))
			ax11.yaxis.set_major_locator(MultipleLocator(0.5))
		# 	ax11.yaxis.set_major_formatter(FormatStrFormatter(''))
		# 	# ax11.xaxis.set_major_formatter(FormatStrFormatter(''))
		# 	ax11.xaxis.set_label_position('top')
			plot_recent_climate_subplot(ax11, plot_format_dict_local, rc_plotting_dict, plot_format_dict)
		
		
		# Plot thermal resistivity if specified
		if 'res_plotting_dict' in kwargs:
			plot_format_dict_local = plot_format_dict['fractional_pos_dict']['axis_res']
			ax03 = fig.add_axes([plot_format_dict_local['frac_xpos'], plot_format_dict_local['frac_ypos'], plot_format_dict_local['panel_frac_width'], plot_format_dict_local['panel_frac_height']], xlabel=r'$R$ / W$^{-1}$ K m$^2$', ylabel=r'$z$ / m')
			ax03.set_ylim(top=min_depth_m_plot, bottom=max_depth_m_plot)
			ax03.set_xlim(left=min_R, right=max_R)
			ax03.yaxis.tick_right()
			ax03.yaxis.set_ticks_position('both')
			ax03.xaxis.set_ticks_position('both')
			ax03.xaxis.set_minor_locator(MultipleLocator(50))
			ax03.xaxis.set_major_locator(MultipleLocator(100))
			ax03.yaxis.set_minor_locator(MultipleLocator(25))
			ax03.yaxis.set_major_locator(MultipleLocator(50))
			# ax03.yaxis.set_major_formatter(FormatStrFormatter(''))
			# ax03.xaxis.set_major_formatter(FormatStrFormatter(''))
			ax03.xaxis.set_label_position('top')
			ax03.yaxis.set_label_position('right')
			if res_plotting_dict['number_lines'] > 0:
				for line_number in range(res_plotting_dict['number_lines']):
					tempdict = res_plotting_dict['line' + str(line_number)]
					plot_resistivity(tempdict, ax03, plot_format_dict_local, plot_format_dict)
		
		# Plot Bullard plots if specified
		# Plot T versus R
		if 'bullard_TvR_plotting_dict' in kwargs:
			plot_format_dict_local = plot_format_dict['fractional_pos_dict']['axis_bullard']
			ax20 = fig.add_axes([plot_format_dict_local['frac_xpos'], plot_format_dict_local['frac_ypos'], plot_format_dict_local['panel_frac_width'], plot_format_dict_local['panel_frac_height']], xlabel=r'$R$ / W$^{-1}$ K m$^2$', ylabel=r'$T$ / $^{\circ}$C')
			ax20.set_ylim(top=max_T, bottom=min_T)
			ax20.set_xlim(left=min_R, right=max_R)
			ax20.xaxis.tick_top()
			ax20.yaxis.tick_right()
			ax20.yaxis.set_ticks_position('both')
			ax20.xaxis.set_ticks_position('both')
			ax20.xaxis.set_minor_locator(MultipleLocator(50))
			ax20.xaxis.set_major_locator(MultipleLocator(100))
			ax20.yaxis.set_minor_locator(MultipleLocator(2.5))
			ax20.yaxis.set_major_locator(MultipleLocator(5))
			# ax20.yaxis.set_major_formatter(FormatStrFormatter(''))
			# ax20.xaxis.set_major_formatter(FormatStrFormatter(''))
			ax20.xaxis.set_label_position('top')
			ax20.yaxis.set_label_position('right')
			if bullard_TvR_plotting_dict['number_lines'] > 0:
				for line_number in range(bullard_TvR_plotting_dict['number_lines']):
					tempdict = bullard_TvR_plotting_dict['line' + str(line_number)]
					plot_bullard(tempdict, ax20, plot_format_dict_local, plot_format_dict)
		
		# Plot R versus T
		if 'bullard_RvT_plotting_dict' in kwargs:
			plot_format_dict_local = plot_format_dict['fractional_pos_dict']['axis_bullard']
			ax21 = fig.add_axes([plot_format_dict_local['frac_xpos'], plot_format_dict_local['frac_ypos'], plot_format_dict_local['panel_frac_width'], plot_format_dict_local['panel_frac_height']], xlabel=r'$T$ / $^{\circ}$C', ylabel=r'$R$ / W$^{-1}$ K m$^2$')
			ax21.set_ylim(top=max_R, bottom=min_R)
			ax21.set_xlim(left=min_T, right=max_T)
			ax21.xaxis.tick_bottom()
			ax21.yaxis.tick_right()
			ax21.yaxis.set_ticks_position('both')
			ax21.xaxis.set_ticks_position('both')
			ax21.xaxis.set_minor_locator(MultipleLocator(2.5))
			ax21.xaxis.set_major_locator(MultipleLocator(5))
			ax21.yaxis.set_minor_locator(MultipleLocator(50))
			ax21.yaxis.set_major_locator(MultipleLocator(100))
			# ax21.yaxis.set_major_formatter(FormatStrFormatter(''))
			# ax21.xaxis.set_major_formatter(FormatStrFormatter(''))
			ax21.xaxis.set_label_position('bottom')
			ax21.yaxis.set_label_position('right')
			if bullard_RvT_plotting_dict['number_lines'] > 0:
				for line_number in range(bullard_RvT_plotting_dict['number_lines']):
					tempdict = bullard_RvT_plotting_dict['line' + str(line_number)]
					plot_bullard(tempdict, ax21, plot_format_dict_local, plot_format_dict)
		
		# Plot histograms of heat-flow estimates
		if 'qhist_plotting_dict' in kwargs:
			plot_format_dict_local = plot_format_dict['fractional_pos_dict']['axis_qhist']
			ax22 = fig.add_axes([plot_format_dict_local['frac_xpos'], plot_format_dict_local['frac_ypos'], plot_format_dict_local['panel_frac_width'], plot_format_dict_local['panel_frac_height']], xlabel="$q_s \ / \ \mathrm{mW \, m^{-2}}$", ylabel='Probability Density')
			# ax22.set_ylim(top=max_R, bottom=min_R)
			# ax22.set_xlim(left=min_q, right=max_q)
			ax22.xaxis.tick_bottom()
			ax22.yaxis.tick_right()
			ax22.yaxis.set_ticks_position('both')
			ax22.xaxis.set_ticks_position('both')
			# ax22.xaxis.set_minor_locator(MultipleLocator(2.5))
			# ax22.xaxis.set_major_locator(MultipleLocator(5))
			# ax22.yaxis.set_minor_locator(MultipleLocator(50))
			# ax22.yaxis.set_major_locator(MultipleLocator(100))
			# ax22.yaxis.set_major_formatter(FormatStrFormatter(''))
			# ax22.xaxis.set_major_formatter(FormatStrFormatter(''))
			ax22.xaxis.set_label_position('bottom')
			ax22.yaxis.set_label_position('right')
			if qhist_plotting_dict['number_hists'] > 0:
				for hist_number in range(qhist_plotting_dict['number_hists']):
					print('hist'+str(hist_number))
					tempdict = qhist_plotting_dict['hist' + str(hist_number)]
					plot_heat_flow_histograms(tempdict, ax22, plot_format_dict_local, plot_format_dict)
		
		if kwargs['figure_name'] != None:
			print('saving')
			fig.savefig(kwargs['figure_name'] + ".jpg", dpi=300, bbox_inches='tight', transparent=True)
			plt.show()
			plt.close(fig)


	def plot_temperature(self):
		print('plotting')
		plt.plot(self.data.borehole.T, self.data.borehole.zT_m, 'x')
	
	def test_value_creation(self, suffix):
		
		setattr(self.data.borehole, suffix, general_python_functions.EmptyClass())
		
		xd = getattr(self.data.borehole, 'bigboy')
		xd.temp = np.array([1,2,3,4])
		
		
		
		# self.data.borehole.suffix = general_python_functions.EmptyClass()
# 		self.data.borehole.suffix.testdata = np.linspace(1,10,1)
		
		
		
		

		
		
		
		