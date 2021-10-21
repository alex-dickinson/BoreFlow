# System packages
from shutil import copyfile

# Numerical packages
import numpy as np
import pandas as pd

# Formatting packages
import pprint

# Local packages
from BoreFlow import general_python_functions
from BoreFlow import heat_flow_functions

# ----------------------------------------------------------------

### FUNCTIONS FOR LOADING SURFACE TEMPERATURE HISTORIES ###

def load_palaeoclimate(palaeoclimate_csv):
	# TODO Include uncertainties in t0 and t1
	palaeoclimate = np.loadtxt(palaeoclimate_csv, delimiter=',', skiprows=1)
	palaeoclimate_t0_seconds = general_python_functions.y2s(palaeoclimate[:,1]*1e3) # Start of period of temperature palaeoclimate_deltaTs in seconds
	palaeoclimate_t1_seconds = general_python_functions.y2s(palaeoclimate[:,0]*1e3) # End of period of temperature palaeoclimate_deltaTs in seconds
	palaeoclimate_t1_seconds[0] = 3e9 # Set to non-zero to prevent division by zero
	palaeoclimate_deltaTs = palaeoclimate[:,2] # Palaeoclimatic temperature
	palaeoclimate_sigma_deltaTs = palaeoclimate[:,3] # Uncertainty in Tc
	# Flip arrays so that oldest values (i.e. largest t0) are given first
	palaeoclimate_t0_seconds, palaeoclimate_t1_seconds, palaeoclimate_deltaTs, palaeoclimate_sigma_deltaTs = np.flip(palaeoclimate_t0_seconds), np.flip(palaeoclimate_t1_seconds), np.flip(palaeoclimate_deltaTs), np.flip(palaeoclimate_sigma_deltaTs)
	return(palaeoclimate_t0_seconds, palaeoclimate_t1_seconds, palaeoclimate_deltaTs, palaeoclimate_sigma_deltaTs)

def load_recent_climate_history(recent_temperature_history_file):
	# TODO Add uncertainties
	recent_temperature_history_year, recent_temperature_history_deltaTs = np.loadtxt(recent_temperature_history_file, delimiter=',', skiprows=1, unpack=True, usecols=(0,8))
	return(recent_temperature_history_year, recent_temperature_history_deltaTs)

def load_surface_temperature_histories(palaeoclimate_csv, recent_climtemp_hist_csv, recent_climtemp_smoother, recent_climtemp_smoothing_length, recent_climtemp_smoothed_outfile):
	# Load palaeoclimate history
	palaeoclimate_t0_seconds, palaeoclimate_t1_seconds, palaeoclimate_deltaTs, palaeoclimate_sigma_deltaTs = load_palaeoclimate(palaeoclimate_csv)
	palaeoclimate_suffix = "_pc"

	# Load time series of recent temperature changes at surface - NASA record of global temperature change within latitude bands. British mainland approx 50 to 60 N therefore use latitude band 44N - 64N. In ninth column
	# TODO - add uncertainty
	recent_climtemp_year, recent_climtemp_deltaTs = load_recent_climate_history(recent_climtemp_hist_csv)
	recent_climtemp_suffix = "_rc"
	# Smooth temperature history
	if recent_climtemp_smoother == "boxcar":
		recent_climtemp_year_smoothing_cutoff, recent_climtemp_deltaTs_smoothed = general_python_functions.smooth_data_boxcar(recent_climtemp_year, recent_climtemp_deltaTs, recent_climtemp_smoothing_length)
		dtemp = {'recent_climtemp_year_smoothing_cutoff':recent_climtemp_year_smoothing_cutoff, 'recent_climtemp_deltaTs_smoothed':recent_climtemp_deltaTs_smoothed}
		recent_climtemp_smoothed_df = pd.DataFrame(data=dtemp)
		ftemp = open(recent_climtemp_smoothed_outfile + '.csv', 'w')
		ftemp.write('### Original data: ' + str(recent_climtemp_hist_csv) + "\n")
		ftemp.write('### Smoothed with boxcar filter of length ' + str(recent_climtemp_smoothing_length) + "\n")
		recent_climtemp_smoothed_df.to_csv(ftemp, index=False)
		ftemp.close()
		recent_climtemp_suffix = recent_climtemp_suffix + '_bcs' + str(recent_climtemp_smoothing_length)
	else:
		print("Smoothing for recent climate not specified - exiting")
		exit()
	
	return(palaeoclimate_t0_seconds, palaeoclimate_t1_seconds, palaeoclimate_deltaTs, palaeoclimate_sigma_deltaTs, recent_climtemp_year, recent_climtemp_deltaTs, recent_climtemp_year_smoothing_cutoff, recent_climtemp_deltaTs_smoothed, recent_climtemp_smoothed_df, palaeoclimate_suffix, recent_climtemp_suffix)

# ----------------------------------------------------------------

### FUNCTIONS FOR LOADING BOREHOLE DATA ###

def load_borehole_data(individual_boreholes_dir, borehole, temperatures_extension, conductivities_extension, all_borehole_overview_df, cermak1982_cond_dict, rollin1987_cond_dict, ukogl_well_tops_depth_error_m, liths_lookup_dict_df, rollin1987_unit_plot_dict, cermak1982_unit_plot_dict):
	
	### Set up file paths and names
	borehole_path = individual_boreholes_dir + "/" + borehole
	raw_data_path = borehole_path + "/raw_data/" + borehole
	figures_path = borehole_path + "/figures/" + borehole
	temperatures_file = raw_data_path + temperatures_extension # Load measured temperatures
	conductivities_file = raw_data_path + conductivities_extension # Load measured conductivities
		
	### Read metadata from master spreadsheet
	borehole_row = all_borehole_overview_df.loc[all_borehole_overview_df['file_name'] == borehole]
	borehole_year = int(borehole_row['year_temp_measurements'].values[0]) # Read year of measurements from overview file. Assume that if measurements taken long time after drilling, climate perturbation will affect the borehole shaft in the same way as it affects undisturbed rock.
	in_situ_conductivity_flag = borehole_row['conductivity?'].values[0]
	number_strat_interps = int(borehole_row["number_strat_interps"].values[0])
	borehole_name = borehole_row['name'].values[0]
	
	
	### Load temperature measurements
	# Load Excel spreadsheet
	temperatures_df = pd.read_excel(temperatures_file + ".xlsx", usecols=["depth(m)","depth_quoted_error(m)","depth_assigned_error(m)","temperature(degrees_C)","temperature_quoted_error(degrees_C)","temperature_assigned_error(degrees_C)","use?"])
	# Remove rows that are not to be used and rename columns. Make sure NaN values are properly represented
	temperatures_df = temperatures_df.drop(index=temperatures_df[temperatures_df["use?"] == "n"].index).drop(columns=["use?"]).rename(columns={"depth(m)":"z_m", "depth_quoted_error(m)":'z_quoted_error_m', "depth_assigned_error(m)":'z_assigned_error_m', "temperature(degrees_C)":'T', "temperature_quoted_error(degrees_C)":"T_quoted_error", "temperature_assigned_error(degrees_C)":"T_assigned_error"}).fillna(value=np.nan)
	# Check that all temperature depths are monotonically increasing - TODO Why not just sort by depth?
	skip_borehole = "no"
	if temperatures_df['z_m'].is_monotonic_increasing == False:
		print("Temperature depths not monotonically increasing - skipping")
		skip_borehole='yes'
	
	
	# # Add missing uncertainties to temperature values
	# temp_sigma_df = temp_sigma_df.replace(to_replace = {'depth_error(m)': "unstated", 'temperature_error(degrees_C)':"unstated"}, value = {'depth_error(m)': sigma_z_T_const, 'temperature_error(degrees_C)': sigma_T_const})
	#
	# # Divide into numpy arrays
	# z_T = np.array(temp_sigma_df['depth(m)'])
	# sigma_z_T = np.array(temp_sigma_df['depth_error(m)'])
	# T = np.array(temp_sigma_df['temperature(degrees_C)'])
	# sigma_T = np.array(temp_sigma_df['temperature_error(degrees_C)'])
	
	
	### Load conductivity values
	# Set up dictionary to record details about different conductivity profiles
	# TODO Rename this dictionary as not just conductivity from strat interps
	strat_interp_dict = {}
	strat_interp_names = []
	
	### Load in situ conductivity measurements if they exist
	if in_situ_conductivity_flag == "y":
		
		std_format_conds_directory = borehole_path + "/raw_data/std_format_conds"
		general_python_functions.set_up_directory(std_format_conds_directory)
		
		# Load Excel spreadsheet
		in_situ_conductivities_df = pd.read_excel(conductivities_file + ".xlsx", usecols=["depth(m)","depth_quoted_error(m)","depth_assigned_error(m)","conductivity(Wm-1K-1)","conductivity_quoted_error(Wm-1K-1)","conductivity_assigned_error(Wm-1K-1)","use?"], na_values='unstated')
		# Remove rows that are not to be used and rename columns. Make sure NaN values are properly represented
		in_situ_conductivities_df = in_situ_conductivities_df.drop(index=in_situ_conductivities_df[in_situ_conductivities_df["use?"] == "n"].index).drop(columns="use?").rename(columns={"depth(m)":"z_m", "depth_quoted_error(m)":'z_quoted_error_m', "depth_assigned_error(m)":'z_assigned_error_m', "conductivity(Wm-1K-1)":'k', "conductivity_quoted_error(Wm-1K-1)":"k_quoted_error", "conductivity_assigned_error(Wm-1K-1)":"k_assigned_error"}).fillna(value=np.nan)
		# Check that all conductivity depths are monotonically increasing - TODO Why not just sort by depth?
		if in_situ_conductivities_df['z_m'].is_monotonic_increasing == False:
			print("Conductivity depths not monotonically increasing - skipping")
			skip_borehole='yes'
		
		
		### Find midpoints of thermal conductivity measurements and assign layers of constant conductivity centred around each measurement depth
		# Use quoted errors on depth
		z_k_lower_bounds, sigma_z_k_quoted_lower_bounds, z_k_upper_bounds, sigma_z_k_quoted_upper_bounds = heat_flow_functions.calculate_conductivity_midpoint_depths(np.array(in_situ_conductivities_df['z_m']), np.array(in_situ_conductivities_df['z_quoted_error_m']))
		# Use assigned errors on depth
		z_k_lower_bounds, sigma_z_k_assigned_lower_bounds, z_k_upper_bounds, sigma_z_k_assigned_upper_bounds = heat_flow_functions.calculate_conductivity_midpoint_depths(np.array(in_situ_conductivities_df['z_m']), np.array(in_situ_conductivities_df['z_assigned_error_m']))
		
		### Organise into standard format for calculations
		nan_array = np.full(shape=np.size(z_k_lower_bounds), fill_value=np.nan)
		distribution_type = 'in_situ_normal'
		distribution_type_list = np.full(shape=np.size(z_k_lower_bounds), fill_value=np.nan)
		if np.size(z_k_lower_bounds) > 1:
			distribution_type_list = np.size(z_k_lower_bounds) * [distribution_type]
		else:
			distribution_type_list = distribution_type
		
		dtemp={'zk_m':in_situ_conductivities_df['z_m'], 'zk_quoted_error_m':in_situ_conductivities_df['z_quoted_error_m'], 'zk_assigned_error_m':in_situ_conductivities_df['z_assigned_error_m'], 'z0_m':z_k_lower_bounds, 'z0_quoted_error_m':sigma_z_k_quoted_lower_bounds, 'z0_assigned_error_m':sigma_z_k_assigned_lower_bounds, 'z1_m':z_k_upper_bounds, 'z1_quoted_error_m':sigma_z_k_quoted_upper_bounds, 'z1_assigned_error_m':sigma_z_k_assigned_upper_bounds, 'geological_description':nan_array, 'number_k_samples':nan_array, 'mean_k':in_situ_conductivities_df['k'], 'k_quoted_error':in_situ_conductivities_df['k_quoted_error'], 'k_assigned_error':in_situ_conductivities_df['k_assigned_error'], 'min_k':nan_array, 'max_k':nan_array, 'k_distribution':distribution_type_list}
		in_situ_conds_std_df = pd.DataFrame(data=dtemp)
		
		
		### TODO Is there any point in doing this trimming?
		# ### Trim conductivity measurements that lie beneath depth of deepest temperature measurement
		# max_z_T = z_T.max()
		# z_k_trim, sigma_z_k_trim, k_trim, sigma_k_trim  = heat_flow_functions.trim_conductivities(z_k, sigma_z_k, k, sigma_k, max_z_T)
		# z_k_trim_lower_bounds, sigma_z_k_trim_lower_bounds, z_k_trim_upper_bounds, sigma_z_k_trim_upper_bounds = heat_flow_functions.calculate_conductivity_midpoint_depths(z_k_trim, sigma_z_k_trim)
		
		
		## Add in situ conductivity to dictionary of cond profiles to analyse
		strat_interp_label = 'in_situ_conds'
		strat_interp_names.append(strat_interp_label)
		strat_interp_ext = '_in_situ_conds'
		strat_interp_lith_outfile = std_format_conds_directory + "/" + borehole + conductivities_extension + strat_interp_ext
		strat_interp_lith_cond_calcs_file = strat_interp_lith_outfile + '_calcformat'
		strat_interp_dict[strat_interp_label] = {}
		strat_interp_dict[strat_interp_label]["strat_interp_ext"] = strat_interp_ext
		strat_interp_dict[strat_interp_label]["strat_interp_lith_cond"] = {}
		strat_interp_dict[strat_interp_label]["strat_interp_lith_cond"][strat_interp_label] = {'strat_interp_lith_cond_name':strat_interp_label, 'strat_interp_lith_cond_file':strat_interp_lith_outfile, 'strat_interp_lith_cond_calcs_file':strat_interp_lith_cond_calcs_file, 'run_calcs':True, 'plot_lith_fill_dict':np.nan, 'plot_lith_fill_dict_keyword':np.nan, 'k_distribution':'in_situ_normal'}
		
		# Save to files. Normal format and calc_format are same for in situ profiles
		in_situ_conds_std_df.to_csv(strat_interp_lith_outfile + '.csv', index=False, na_rep=np.nan)
		in_situ_conds_std_df.to_csv(strat_interp_lith_cond_calcs_file + '.csv', index=False, na_rep=np.nan)
	
	else:
		in_situ_conductivities_df = 'NaN'
	
	
	
	### Read in lithologies ###
	orig_lith_files_path = borehole_path + '/raw_data/lithologies/original_liths/' + borehole
	sorted_lith_files_path = borehole_path + '/raw_data/lithologies/sorted_liths'
	general_python_functions.set_up_directory(sorted_lith_files_path)
	sorted_lith_files_path = sorted_lith_files_path + '/' + borehole
	
	for strat_interp_index in range(number_strat_interps):
		strat_interp_label_index = "strat_interp" + str(strat_interp_index+1)
		strat_interp_ext_index = "strat_interp" + str(strat_interp_index+1) + "_ext"
		strat_interp_label = borehole_row[strat_interp_label_index].values[0]
		strat_interp_ext = borehole_row[strat_interp_ext_index].values[0]
		strat_interp_lith_infile = orig_lith_files_path + strat_interp_ext
		strat_interp_lith_outfile = sorted_lith_files_path + strat_interp_ext
		strat_interp_dict[strat_interp_label] = {}
		strat_interp_names.append(strat_interp_label)
		
		if strat_interp_label == "ukogl_well_tops":
			
			ukogl_well_tops_lith_input_df = pd.read_csv(strat_interp_lith_infile + '.csv')
			
			z0 = np.array(ukogl_well_tops_lith_input_df['MD (m)'])
			z1 = z0[1::]
			z0 = z0[0:-1]
			
			# Add errors and conductivity values for near-surface drift deposits to UKOGL well top depths
			if z0[0] == 0:
				z0_assigned_error = np.hstack([[0], np.full(np.size(z0)-1, ukogl_well_tops_depth_error_m)])
			else:
				# # Add in layer at top of UKOGL formations and assume it is drift deposit
				# z0 = np.hstack([[0], z0])
				# z1 = np.hstack([[z0[0]], z1])
				# print(z0)
				# print(z1)
				z0_assigned_error = np.full(np.size(z0), ukogl_well_tops_depth_error_m)
			z1_assigned_error = np.full(np.size(z0), ukogl_well_tops_depth_error_m)
			
			formation = ukogl_well_tops_lith_input_df['Formation'].values[:-1]
			age = ukogl_well_tops_lith_input_df['Age'].values[:-1]
			detail = ukogl_well_tops_lith_input_df['Detail'].values[:-1]
			# if np.size(z0) > 1:
			# 	unstated_list = np.size(z0) * ['unstated']
			# else:
			# 	unstated_list = 'unstated'
			nan_array = np.zeros(np.size(z0))
			nan_array[:] = np.nan
			dtemp = {'zk_m':nan_array, 'zk_quoted_error_m':nan_array, 'zk_assigned_error_m':nan_array, 'z0_m':z0, 'z0_quoted_error_m':nan_array, 'z0_assigned_error_m':z0_assigned_error, 'z1_m':z1, 'z1_quoted_error_m':nan_array, 'z1_assigned_error_m':z1_assigned_error, 'source_formation':formation, 'source_age':age, 'source_notes':detail}
			ukogl_well_tops_lith_output_df = pd.DataFrame(data=dtemp)
			ukogl_well_tops_lith_output_df[['source','source_group','source_member','source_lith','source_lith_cermak1982']] = pd.DataFrame([['UKOGL_well_formation_tops', nan_array, nan_array, nan_array, nan_array]], index=ukogl_well_tops_lith_output_df.index)
			
			# For UKOGL well tops, use formation name to look up lithology
			ukogl_well_tops_lith_output_df = pd.merge(ukogl_well_tops_lith_output_df.applymap(str), liths_lookup_dict_df.applymap(str), how='left', left_on=['source_formation','source_age','source_notes'], right_on = ['ukogl_well_tops_formation','ukogl_well_tops_age','ukogl_well_tops_detail']).drop(['commented','strat_log_lith','ukogl_well_tops_formation','ukogl_well_tops_age','ukogl_well_tops_detail','ukogl_borehole','cermak1982_unit_lith_original'], axis=1).rename(columns={"cermak1982_unit_lith_expanded":"unit_lith_cermak1982"})
			
			# Define as strat_interp_lith_df and convert all 'unstated' to nan
			strat_interp_lith_df = ukogl_well_tops_lith_output_df.replace({'unstated':np.nan})
			
			
		elif strat_interp_label == "bgs_borehole_scan":
			bgs_borehole_scan_lith_df = pd.read_excel(strat_interp_lith_infile + '.xlsx', comment="%!%#").drop(['commented'], axis=1)
			bgs_borehole_scan_lith_df = bgs_borehole_scan_lith_df.rename(columns={"z0(m)":"z0_m", 'z0_quoted_error(m)':'z0_quoted_error_m', 'z0_assigned_error(m)':'z0_assigned_error_m', 'z1(m)':'z1_m', 'z1_quoted_error(m)':'z1_quoted_error_m', 'z1_assigned_error(m)':'z1_assigned_error_m'})
			
			nan_array = np.full(np.size(bgs_borehole_scan_lith_df['z0_m']), np.nan)
			
			# Add nan values to columns that would hold in situ conductivities
			bgs_borehole_scan_lith_df['zk_m'], bgs_borehole_scan_lith_df['zk_quoted_error_m'], bgs_borehole_scan_lith_df['zk_assigned_error_m']  = [nan_array, nan_array, nan_array]
			
			# First use specified unit to look up lithology
			bgs_borehole_scan_lith_df = pd.merge(bgs_borehole_scan_lith_df.applymap(str), liths_lookup_dict_df.applymap(str), how='left', left_on=['strat_log_lookup_key'], right_on = ['strat_log_lookup_key']).drop(['commented','cermak1982_unit_lith_original','ukogl_well_tops_formation','ukogl_well_tops_age','ukogl_well_tops_detail','ukogl_borehole','strat_log_lith'], axis=1).rename(columns={"cermak1982_unit_lith_expanded":"unit_lith_cermak1982"})
			
			# Second use recorded lithology to look up corresponding lithology for Cermak (1982)
			bgs_borehole_scan_lith_df = pd.merge(bgs_borehole_scan_lith_df.applymap(str), liths_lookup_dict_df[['strat_log_lith','cermak1982_unit_lith_expanded']].applymap(str), how='left', left_on=['source_lith'], right_on = ['strat_log_lith']).drop(['strat_log_lith'], axis=1).rename(columns={"cermak1982_unit_lith_expanded":"source_lith_cermak1982"})
			
			# Define as strat_interp_lith_df and convert all 'unstated' to nan
			strat_interp_lith_df = bgs_borehole_scan_lith_df.replace({'unstated':np.nan})
			
		
		
		### Assign conductivities based on formation names and lithologies
		# Assign conductivity values of Cermak (1982) 
		for lith_type in cermak1982_cond_dict.keys():
			# Assign conductivities to lithologies recorded in source report
			strat_interp_lith_df.loc[strat_interp_lith_df['source_lith_cermak1982'] == lith_type, 'source_lith_cermak1982_number_k_samples'] = cermak1982_cond_dict[lith_type]['number_samples']
			strat_interp_lith_df.loc[strat_interp_lith_df['source_lith_cermak1982'] == lith_type, 'source_lith_cermak1982_mean_k'] = cermak1982_cond_dict[lith_type]['mean_k(Wm-1K-1)']
			strat_interp_lith_df.loc[strat_interp_lith_df['source_lith_cermak1982'] == lith_type, 'source_lith_cermak1982_stdev_k'] = cermak1982_cond_dict[lith_type]['stdev_k(Wm-1K-1)']
			strat_interp_lith_df.loc[strat_interp_lith_df['source_lith_cermak1982'] == lith_type, 'source_lith_cermak1982_min_k'] = cermak1982_cond_dict[lith_type]['min_k(Wm-1K-1)']
			strat_interp_lith_df.loc[strat_interp_lith_df['source_lith_cermak1982'] == lith_type, 'source_lith_cermak1982_max_k'] = cermak1982_cond_dict[lith_type]['max_k(Wm-1K-1)']
			# Assign conductivities to lithologies based on formation names
			strat_interp_lith_df.loc[strat_interp_lith_df['unit_lith_cermak1982'] == lith_type, 'unit_lith_cermak1982_number_k_samples'] = cermak1982_cond_dict[lith_type]['number_samples']
			strat_interp_lith_df.loc[strat_interp_lith_df['unit_lith_cermak1982'] == lith_type, 'unit_lith_cermak1982_mean_k'] = cermak1982_cond_dict[lith_type]['mean_k(Wm-1K-1)']
			strat_interp_lith_df.loc[strat_interp_lith_df['source_lith_cermak1982'] == lith_type, 'unit_lith_cermak1982_stdev_k'] = cermak1982_cond_dict[lith_type]['stdev_k(Wm-1K-1)']
			strat_interp_lith_df.loc[strat_interp_lith_df['unit_lith_cermak1982'] == lith_type, 'unit_lith_cermak1982_min_k'] = cermak1982_cond_dict[lith_type]['min_k(Wm-1K-1)']
			strat_interp_lith_df.loc[strat_interp_lith_df['unit_lith_cermak1982'] == lith_type, 'unit_lith_cermak1982_max_k'] = cermak1982_cond_dict[lith_type]['max_k(Wm-1K-1)']
		
		# Assign conductivity values of Rollin (1987) to units
		for description_code in rollin1987_cond_dict.keys():
			strat_interp_lith_df.loc[strat_interp_lith_df['rollin1987_unit'] == description_code, 'rollin1987_unit_number_k_samples'] = rollin1987_cond_dict[description_code]['number_samples']
			strat_interp_lith_df.loc[strat_interp_lith_df['rollin1987_unit'] == description_code, 'rollin1987_unit_mean_k'] = rollin1987_cond_dict[description_code]['mean_k(Wm-1K-1)']
			strat_interp_lith_df.loc[strat_interp_lith_df['rollin1987_unit'] == description_code, 'rollin1987_unit_stdev_k'] = rollin1987_cond_dict[description_code]['stdev_k(Wm-1K-1)']
			strat_interp_lith_df.loc[strat_interp_lith_df['rollin1987_unit'] == description_code, 'rollin1987_unit_min_k'] = rollin1987_cond_dict[description_code]['min_k(Wm-1K-1)']
			strat_interp_lith_df.loc[strat_interp_lith_df['rollin1987_unit'] == description_code, 'rollin1987_unit_max_k'] = rollin1987_cond_dict[description_code]['max_k(Wm-1K-1)']	
		
		### Combine adjacent stratigrahic descriptions that are identical
		# Write out only rollin1987 stratigraphic units and combine adjacent units that are identical
		strat_interp_lith_rollin1987_unit_df = strat_interp_lith_df[['zk_m','zk_quoted_error_m','zk_assigned_error_m','z0_m','z0_quoted_error_m','z0_assigned_error_m','z1_m','z1_quoted_error_m','z1_assigned_error_m','rollin1987_unit','rollin1987_unit_number_k_samples','rollin1987_unit_mean_k','rollin1987_unit_stdev_k','rollin1987_unit_min_k','rollin1987_unit_max_k']].copy(deep=True)
		# Combine adjacent lithologies that are identical
		strat_interp_lith_rollin1987_unit_df = strat_interp_lith_rollin1987_unit_df.loc[strat_interp_lith_rollin1987_unit_df['rollin1987_unit'].shift(1) != strat_interp_lith_rollin1987_unit_df['rollin1987_unit']]
		# Set z1 values to correct values based on new z0 values
		strat_interp_lith_rollin1987_unit_df['z1_m'] = strat_interp_lith_rollin1987_unit_df['z0_m'].shift(-1)
		# Set well bottom to correct value
		strat_interp_lith_rollin1987_unit_df.iloc[-1, strat_interp_lith_rollin1987_unit_df.columns.get_loc('z1_m')] = strat_interp_lith_df.iloc[-1, strat_interp_lith_df.columns.get_loc('z1_m')]
		# Copy to dataframe with standardised format for calculations
		strat_interp_lith_rollin1987_unit_calcformat_df = strat_interp_lith_rollin1987_unit_df.copy(deep=True).rename(columns={'rollin1987_unit':'geological_description', 'rollin1987_unit_number_k_samples':'number_k_samples', 'rollin1987_unit_mean_k':'mean_k', 'rollin1987_unit_stdev_k':'k_quoted_error', 'rollin1987_unit_min_k':'min_k', 'rollin1987_unit_max_k':'max_k'})
		# Set quoted standard deviation of Rollin (1987) values as error to use (k_assigned_error)
		strat_interp_lith_rollin1987_unit_calcformat_df['k_assigned_error'] = strat_interp_lith_rollin1987_unit_calcformat_df['k_quoted_error']
		strat_interp_lith_rollin1987_unit_calcformat_df['k_distribution'] = 'uniform'
		# Convert all 'unstated' to nan
		strat_interp_lith_rollin1987_unit_calcformat_df = strat_interp_lith_rollin1987_unit_calcformat_df.replace({'unstated':np.nan})
		
		# Write out only cermak1982 lithologies based on source lithologies
		strat_interp_lith_source_lith_cermak1982_df = strat_interp_lith_df[['zk_m','zk_quoted_error_m','zk_assigned_error_m','z0_m','z0_quoted_error_m','z0_assigned_error_m','z1_m','z1_quoted_error_m','z1_assigned_error_m','source_lith_cermak1982','source_lith_cermak1982_number_k_samples','source_lith_cermak1982_mean_k','source_lith_cermak1982_stdev_k','source_lith_cermak1982_min_k','source_lith_cermak1982_max_k']].copy(deep=True)
		# Combine adjacent lithologies that are identical
		strat_interp_lith_source_lith_cermak1982_df = strat_interp_lith_source_lith_cermak1982_df.loc[strat_interp_lith_source_lith_cermak1982_df['source_lith_cermak1982'].shift(1) != strat_interp_lith_source_lith_cermak1982_df['source_lith_cermak1982']]
		# Set z1 values to correct values based on new z0 values
		strat_interp_lith_source_lith_cermak1982_df['z1_m'] = strat_interp_lith_source_lith_cermak1982_df['z0_m'].shift(-1)
		# Set well bottom to correct value
		strat_interp_lith_source_lith_cermak1982_df.iloc[-1, strat_interp_lith_source_lith_cermak1982_df.columns.get_loc('z1_m')] = strat_interp_lith_df.iloc[-1, strat_interp_lith_df.columns.get_loc('z1_m')]
		# Copy to dataframe with standardised format for calculations
		strat_interp_lith_source_lith_cermak1982_calcformat_df = strat_interp_lith_source_lith_cermak1982_df.copy(deep=True).rename(columns={'source_lith_cermak1982':'geological_description', 'source_lith_cermak1982_number_k_samples':'number_k_samples', 'source_lith_cermak1982_mean_k':'mean_k', 'source_lith_cermak1982_stdev_k':'k_quoted_error', 'source_lith_cermak1982_min_k':'min_k', 'source_lith_cermak1982_max_k':'max_k'})
		# Set k_assigned_error to nan (error not used for uniform distribution)
		strat_interp_lith_source_lith_cermak1982_calcformat_df['k_assigned_error'] = np.full(np.size(strat_interp_lith_source_lith_cermak1982_calcformat_df['z0_m']), np.nan)
		strat_interp_lith_source_lith_cermak1982_calcformat_df['k_distribution'] = 'uniform'
		# Convert all 'unstated' to nan
		strat_interp_lith_source_lith_cermak1982_calcformat_df = strat_interp_lith_source_lith_cermak1982_calcformat_df.replace({'unstated':np.nan})
		
		# Write out only cermak1982 lithologies based on stratigraphic units and combine adjacent lithologies that are identical
		strat_interp_lith_unit_lith_cermak1982_df = strat_interp_lith_df[['zk_m','zk_quoted_error_m','zk_assigned_error_m','z0_m','z0_quoted_error_m','z0_assigned_error_m','z1_m','z1_quoted_error_m','z1_assigned_error_m','unit_lith_cermak1982','unit_lith_cermak1982_number_k_samples','unit_lith_cermak1982_mean_k','unit_lith_cermak1982_stdev_k','unit_lith_cermak1982_min_k','unit_lith_cermak1982_max_k']].copy(deep=True)
		# Combine adjacent lithologies that are identical
		strat_interp_lith_unit_lith_cermak1982_df = strat_interp_lith_unit_lith_cermak1982_df.loc[strat_interp_lith_unit_lith_cermak1982_df['unit_lith_cermak1982'].shift(1) != strat_interp_lith_unit_lith_cermak1982_df['unit_lith_cermak1982']]
		# Set z1 values to correct values based on new z0 values
		strat_interp_lith_unit_lith_cermak1982_df['z1_m'] = strat_interp_lith_unit_lith_cermak1982_df['z0_m'].shift(-1)
		# Set well bottom to correct value
		strat_interp_lith_unit_lith_cermak1982_df.iloc[-1, strat_interp_lith_unit_lith_cermak1982_df.columns.get_loc('z1_m')] = strat_interp_lith_df.iloc[-1, strat_interp_lith_df.columns.get_loc('z1_m')]
		# Copy to dataframe with standardised format for calculations
		strat_interp_lith_unit_lith_cermak1982_calcformat_df = strat_interp_lith_unit_lith_cermak1982_df.copy(deep=True).rename(columns={'unit_lith_cermak1982':'geological_description', 'unit_lith_cermak1982_number_k_samples':'number_k_samples', 'unit_lith_cermak1982_mean_k':'mean_k', 'unit_lith_cermak1982_stdev_k':'k_quoted_error', 'unit_lith_cermak1982_min_k':'min_k', 'unit_lith_cermak1982_max_k':'max_k'})
		# Set k_assigned_error to nan (error not used for uniform distribution)
		strat_interp_lith_unit_lith_cermak1982_calcformat_df['k_assigned_error'] = np.full(np.size(strat_interp_lith_unit_lith_cermak1982_calcformat_df['z0_m']), np.nan)
		strat_interp_lith_unit_lith_cermak1982_calcformat_df['k_distribution'] = 'uniform'
		# Convert all 'unstated' to nan
		strat_interp_lith_unit_lith_cermak1982_calcformat_df = strat_interp_lith_unit_lith_cermak1982_calcformat_df.replace({'unstated':np.nan})
		
		
		# Write out data to dictionary and save to file TODO define all names in same place and pass to dict and to to_csv()
		strat_interp_dict[strat_interp_label]["strat_interp_ext"] = strat_interp_ext
		strat_interp_dict[strat_interp_label]["strat_interp_lith_cond"] = {}
		strat_interp_dict[strat_interp_label]["strat_interp_lith_cond"]['all_conds'] = {'strat_interp_lith_cond_name':strat_interp_label+'_all_conds', 'strat_interp_lith_cond_file':strat_interp_lith_outfile + '_std_format_conds', 'strat_interp_lith_cond_calcs_file':np.nan, 'run_calcs':False, 'plot_lith_fill_dict':np.nan, 'plot_lith_fill_dict_keyword':np.nan, 'k_distribution':np.nan}
		strat_interp_dict[strat_interp_label]["strat_interp_lith_cond"]["rollin1987_unit_conds"] = {'strat_interp_lith_cond_name':strat_interp_label+'_rollin1987_unit_conds', 'strat_interp_lith_cond_file':strat_interp_lith_outfile + '_rollin1987_unit_conds', 'strat_interp_lith_cond_calcs_file':strat_interp_lith_outfile + '_rollin1987_unit_conds_calcformat', 'run_calcs':True, 'plot_lith_fill_dict':rollin1987_unit_plot_dict, 'plot_lith_fill_dict_keyword':'rollin1987_unit', 'k_distribution':'normal'}
		strat_interp_dict[strat_interp_label]["strat_interp_lith_cond"]["cermak1982_unit_lith_conds"] = {'strat_interp_lith_cond_name':strat_interp_label+'_cermak1982_unit_lith_conds', 'strat_interp_lith_cond_file':strat_interp_lith_outfile + '_cermak1982_unit_lith_conds', 'strat_interp_lith_cond_calcs_file':strat_interp_lith_outfile + '_cermak1982_unit_lith_conds_calcformat', 'run_calcs':True, 'plot_lith_fill_dict':cermak1982_unit_plot_dict, 'plot_lith_fill_dict_keyword':'unit_lith_cermak1982', 'k_distribution':'uniform'}
		if strat_interp_label != 'ukogl_well_tops':
			strat_interp_dict[strat_interp_label]["strat_interp_lith_cond"]["cermak1982_source_lith_conds"] = {'strat_interp_lith_cond_name':strat_interp_label+'_c82_source_lith_conds', 'strat_interp_lith_cond_file':strat_interp_lith_outfile + '_cermak1982_source_lith_conds', 'strat_interp_lith_cond_calcs_file':strat_interp_lith_outfile + '_cermak1982_source_lith_conds_calcformat', 'run_calcs':True, 'plot_lith_fill_dict':cermak1982_unit_plot_dict, 'plot_lith_fill_dict_keyword':'source_lith_cermak1982', 'k_distribution':'uniform'}
		
		
		# # TODO Update so useful for all interpretations. Or move to plotting.
		# z0 = np.array(strat_interp_lith_df['z0_m'])
		# z1 = np.array(strat_interp_lith_df['z1_m'])
		# strat_interp_dict[strat_interp_label]["z_int_plot"] = np.concatenate(np.column_stack((z0, z1)))
		
		
		
		
		### Save stratigraphic interpretations with conductivities to csv
		strat_interp_lith_df.to_csv(strat_interp_lith_outfile + '_std_format_conds.csv', index=False, na_rep=np.nan, columns=('source','z0_m','z0_quoted_error_m','z0_assigned_error_m','z1_m','z1_quoted_error_m','z1_assigned_error_m','source_lith','source_group','source_formation','source_member','source_age','source_notes','strat_log_lookup_key','bgs_group_name','bgs_formation_name','bgs_formation_code','rollin1987_unit','rollin1987_unit_number_k_samples','rollin1987_unit_mean_k','rollin1987_unit_stdev_k','rollin1987_unit_min_k','rollin1987_unit_max_k','source_lith_cermak1982','source_lith_cermak1982_number_k_samples','source_lith_cermak1982_mean_k','source_lith_cermak1982_stdev_k','source_lith_cermak1982_min_k','source_lith_cermak1982_max_k','unit_lith_cermak1982','unit_lith_cermak1982_number_k_samples','unit_lith_cermak1982_mean_k','unit_lith_cermak1982_stdev_k','unit_lith_cermak1982_min_k','unit_lith_cermak1982_max_k'))
		strat_interp_lith_rollin1987_unit_df.to_csv(strat_interp_lith_outfile + '_rollin1987_unit_conds.csv', index=False, na_rep=np.nan)
		strat_interp_lith_unit_lith_cermak1982_df.to_csv(strat_interp_lith_outfile + '_cermak1982_unit_lith_conds.csv', index=False, na_rep=np.nan)
		strat_interp_lith_source_lith_cermak1982_df.to_csv(strat_interp_lith_outfile + '_cermak1982_source_lith_conds.csv', index=False, na_rep=np.nan)
		
		strat_interp_lith_rollin1987_unit_calcformat_df.to_csv(strat_interp_lith_outfile + '_rollin1987_unit_conds_calcformat.csv', index=False, na_rep=np.nan)
		strat_interp_lith_unit_lith_cermak1982_calcformat_df.to_csv(strat_interp_lith_outfile + '_cermak1982_unit_lith_conds_calcformat.csv', index=False, na_rep=np.nan)
		strat_interp_lith_source_lith_cermak1982_calcformat_df.to_csv(strat_interp_lith_outfile + '_cermak1982_source_lith_conds_calcformat.csv', index=False, na_rep=np.nan)
	
	return(borehole_path, raw_data_path, figures_path, borehole_year, temperatures_df, in_situ_conductivities_df, strat_interp_dict, skip_borehole, in_situ_conductivity_flag, borehole_name, number_strat_interps, strat_interp_names)


# ----------------------------------------------------------------

### FUNCTIONS FOR READING DICTIONARIES AND DATAFRAMES ###

def read_strat_interp_lith_cond_dict(strat_interp_dict, strat_interp_name, strat_interp_lith_cond_option):
	strat_interp_lith_cond_dict = strat_interp_dict[strat_interp_name]['strat_interp_lith_cond'][strat_interp_lith_cond_option]
	strat_interp_lith_cond_name = strat_interp_lith_cond_dict['strat_interp_lith_cond_name']
	strat_interp_lith_cond_file = strat_interp_lith_cond_dict['strat_interp_lith_cond_file']
	strat_interp_lith_cond_calcs_file = strat_interp_lith_cond_dict['strat_interp_lith_cond_calcs_file']
	run_calcs_option = strat_interp_lith_cond_dict['run_calcs']
	plot_lith_fill_dict = strat_interp_lith_cond_dict['plot_lith_fill_dict']
	plot_lith_fill_dict_keyword = strat_interp_lith_cond_dict['plot_lith_fill_dict_keyword']
	k_distribution = strat_interp_lith_cond_dict['k_distribution']
	return(strat_interp_lith_cond_dict, strat_interp_lith_cond_name, strat_interp_lith_cond_file, strat_interp_lith_cond_calcs_file, run_calcs_option, plot_lith_fill_dict, plot_lith_fill_dict_keyword, k_distribution)

### Read temperature dataframe into individual numpy arrays for ease of use ###
def read_temperature_dataframe_to_numpy_arrays(temperatures_df, errors_option):
	zT_m = np.array(temperatures_df['z_m'].copy(deep=True))
	T = np.array(temperatures_df['T'].copy(deep=True))
	if errors_option == 'quoted':
		zT_error_m = np.array(temperatures_df['z_quoted_error_m'].copy(deep=True))
		T_error = np.array(temperatures_df['T_quoted_error'].copy(deep=True))
	elif errors_option == 'assigned':
		zT_error_m = np.array(temperatures_df['z_assigned_error_m'].copy(deep=True))
		T_error = np.array(temperatures_df['T_assigned_error'].copy(deep=True))
	return(zT_m, T, zT_error_m, T_error)

### If in situ conductivities exist, load as individual numpy arrays for ease of use ###
def read_in_situ_conductivity_dataframe_to_numpy_arrays(in_situ_conductivity_flag, in_situ_conductivities_df, errors_option):
	if in_situ_conductivity_flag == 'y':
		zk_m = np.array(in_situ_conductivities_df['z_m'].copy(deep=True))
		k = np.array(in_situ_conductivities_df['k'].copy(deep=True))
		if errors_option == 'quoted':
			zk_error_m = np.array(in_situ_conductivities_df['z_quoted_error_m'].copy(deep=True))
			k_error = np.array(in_situ_conductivities_df['k_quoted_error'].copy(deep=True))
		elif errors_option == 'assigned':
			zk_error_m = np.array(in_situ_conductivities_df['z_assigned_error_m'].copy(deep=True))
			k_error = np.array(in_situ_conductivities_df['k_assigned_error'].copy(deep=True))
	else:
		zk_m, k, zk_error_m, k_error = np.nan, np.nan, np.nan, np.nan
	return(zk_m, k, zk_error_m, k_error)

### Read conductivity data into individual numpy arrays for ease of use ###
def read_conductivity_dataframe_to_numpy_arrays(k_distribution, strat_interp_lith_cond_calcs_df, errors_option):
	# Drop dataframe rows that have undefined conductivity
	if k_distribution == 'normal' or k_distribution == 'in_situ_normal':
		strat_interp_lith_cond_calcs_df = strat_interp_lith_cond_calcs_df[strat_interp_lith_cond_calcs_df['mean_k'].notna()]
		print('dropping conductivity na')
	elif k_distribution == 'uniform':
		strat_interp_lith_cond_calcs_df = strat_interp_lith_cond_calcs_df[strat_interp_lith_cond_calcs_df['min_k'].notna()]
		strat_interp_lith_cond_calcs_df = strat_interp_lith_cond_calcs_df[strat_interp_lith_cond_calcs_df['max_k'].notna()]
		print('dropping conductivity na')
	# Read dataframes into individual numpy arrays for ease of use
	zk_m = np.array(strat_interp_lith_cond_calcs_df['zk_m'].copy(deep=True))
	z0_m = np.array(strat_interp_lith_cond_calcs_df['z0_m'].copy(deep=True))
	z1_m = np.array(strat_interp_lith_cond_calcs_df['z1_m'].copy(deep=True))
	mean_k = np.array(strat_interp_lith_cond_calcs_df['mean_k'].copy(deep=True))
	if errors_option == 'quoted':
		zk_error_m = np.array(strat_interp_lith_cond_calcs_df['zk_quoted_error_m'].copy(deep=True))
		z0_error_m = np.array(strat_interp_lith_cond_calcs_df['z0_quoted_error_m'].copy(deep=True))
		z1_error_m = np.array(strat_interp_lith_cond_calcs_df['z1_quoted_error_m'].copy(deep=True))
		mean_k_error = np.array(strat_interp_lith_cond_calcs_df['k_quoted_error'].copy(deep=True))
	elif errors_option == 'assigned':
		zk_error_m = np.array(strat_interp_lith_cond_calcs_df['zk_assigned_error_m'].copy(deep=True))
		z0_error_m = np.array(strat_interp_lith_cond_calcs_df['z0_assigned_error_m'].copy(deep=True))
		z1_error_m = np.array(strat_interp_lith_cond_calcs_df['z1_assigned_error_m'].copy(deep=True))
		mean_k_error = np.array(strat_interp_lith_cond_calcs_df['k_assigned_error'].copy(deep=True))
	min_k = np.array(strat_interp_lith_cond_calcs_df['min_k'].copy(deep=True))
	max_k = np.array(strat_interp_lith_cond_calcs_df['max_k'].copy(deep=True))
	# Create arrays for plotting of conductivity layers
	z_plot = np.concatenate(np.column_stack((z0_m, z1_m)))
	min_k_plot = np.concatenate(np.column_stack((min_k, min_k))).astype('float64')
	max_k_plot = np.concatenate(np.column_stack((max_k, max_k))).astype('float64')
	mean_k_plot = np.concatenate(np.column_stack((mean_k, mean_k))).astype('float64')
	stdev_k_plot = np.concatenate(np.column_stack((mean_k_error, mean_k_error))).astype('float64')
	return(zk_m, zk_error_m, z0_m, z1_m, mean_k, z0_error_m, z1_error_m, mean_k_error, min_k, max_k, z_plot, mean_k_plot, stdev_k_plot, min_k_plot, max_k_plot)

# ----------------------------------------------------------------

### FUNCTIONS FOR SETTING UP STRUCTURE OF DIRECTORIES FOR CALCULATED DATA ###

### Set up top-level directories for calculations and save temperature dataframe within this structure ###
def set_up_calcs_folder_top(calc_data_path, borehole, temperatures_df):
	cond_calcs_path = str(calc_data_path) + "/cond_profs"
	general_python_functions.set_up_directory(cond_calcs_path)
	# Save temperature dataframe to file within calcs folder
	local_temps_calcs_path = str(calc_data_path) + "/input_temps"
	general_python_functions.set_up_directory(local_temps_calcs_path)
	local_temps_calcs_file = str(local_temps_calcs_path) + "/" + str(borehole) + '_temperatures'
	temperatures_df.to_csv(local_temps_calcs_file + '.csv', index=False)
	return(cond_calcs_path, local_temps_calcs_path, local_temps_calcs_file)

def set_up_calcs_folder_cond_top(cond_calcs_path, strat_interp_name, strat_interp_lith_cond_name, borehole, strat_interp_lith_cond_calcs_file):
	# Set up filenames
	strat_interp_cond_calcs_path = str(cond_calcs_path) + "/" + str(strat_interp_name) + "_conds"
	general_python_functions.set_up_directory(strat_interp_cond_calcs_path)
	strat_interp_lith_cond_calcs_path = strat_interp_cond_calcs_path + "/" + str(strat_interp_lith_cond_name)
	general_python_functions.set_up_directory(strat_interp_lith_cond_calcs_path)
	# Copy file with liths and conds into calcs folder and load this file
	local_strat_interp_lith_cond_calcs_file = strat_interp_lith_cond_calcs_path + "/" + str(borehole) + '_' + str(strat_interp_lith_cond_name)
	copyfile(strat_interp_lith_cond_calcs_file + '.csv', local_strat_interp_lith_cond_calcs_file + '.csv')
	strat_interp_lith_cond_calcs_df = pd.read_csv(local_strat_interp_lith_cond_calcs_file + '.csv')
	return(strat_interp_cond_calcs_path, strat_interp_lith_cond_calcs_path, local_strat_interp_lith_cond_calcs_file, strat_interp_lith_cond_calcs_file, strat_interp_lith_cond_calcs_df)

# ----------------------------------------------------------------

### FUNCTIONS FOR PREPROCESSING TEMPERATURE ###

def cut_top_depth(z_m_uncut, cut_top_m, suffix_root, *args):
	if cut_top_m == None:
		cut_list = [z_m_uncut, suffix_root]
		for arg in args: cut_list.append(arg)
	else:
		cut_indices = np.where(z_m_uncut > cut_top_m)
		cut_list = [z_m_uncut[cut_indices], suffix_root + 'ct' + str(cut_top_m) + 'm']
		for arg in args: cut_list.append(arg[cut_indices])
	return(tuple(cut_list))
	

















