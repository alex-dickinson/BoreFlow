import numpy as np
import pandas as pd
# import dataclasses

from BoreFlow import general_python_functions


def call_function():
	print('hello')

# Functions to be used with different data types
def cut_top_depth(z_m_uncut, cut_top_m, suffix_root, *args):
	print('cutting')
	if cut_top_m == None:
		cut_list = [z_m_uncut, suffix_root]
		for arg in args: cut_list.append(arg)
	else:
		print(type(z_m_uncut))
		print(type(cut_top_m))
		cut_indices = np.where(z_m_uncut > cut_top_m)[0]
		print(cut_indices)
		cut_list = [z_m_uncut[cut_indices], str(suffix_root) + 'ct' + str(cut_top_m) + 'm']
		print('here')
		for arg in args: cut_list.append(arg[cut_indices])
		print('here3')
	return(tuple(cut_list))



# -------------------

class BoreHole:
	def __init__(self, cfg):
		self.config = cfg['borehole']
		self.data = general_python_functions.EmptyClass()
		self.data.borehole = general_python_functions.EmptyClass()
	
	def load_temperature_measurements(self):
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
		self.data.borehole.temperatures_df = temperatures_df
		self.data.borehole.T_suffix = temperatures_df
		
		### Read temperature dataframe into individual numpy arrays for ease of use ###
		errors_option = self.config['errors_option']
		self.data.borehole.zT_m = np.array(temperatures_df['z_m'].copy(deep=True))
		self.data.borehole.T = np.array(temperatures_df['T'].copy(deep=True))
		if errors_option == 'quoted':
			self.data.borehole.zT_error_m = np.array(temperatures_df['z_quoted_error_m'].copy(deep=True))
			self.data.borehole.T_error = np.array(temperatures_df['T_quoted_error'].copy(deep=True))
		elif errors_option == 'assigned':
			self.data.borehole.zT_error_m = np.array(temperatures_df['z_assigned_error_m'].copy(deep=True))
			self.data.borehole.T_error = np.array(temperatures_df['T_assigned_error'].copy(deep=True))
	
	def cut_top_temperature_measurements(self):
		self.data.borehole.Tcut_top_m = self.config['temperature_preprocessing']['cut_top_m']
		
		self.data.borehole.zT_m_cut, self.data.borehole.T_cut_suffix, self.data.borehole.T_cut, self.data.borehole.zT_error_m_cut, self.data.borehole.T_error_cut = cut_top_depth(self.data.borehole.zT_m, self.data.borehole.Tcut_top_m, self.data.borehole.T_suffix, self.data.borehole.T, self.data.borehole.zT_error_m, self.data.borehole.T_error)
		
		print('finished')
		
		
		
		#
		# if self.config['cut_top_m'] == None:
		# 	self.data.borehole.zT_m_cut, self.data.borehole.T_cut_suffix, self.data.borehole.T_cut, self.data.borehole.zT_error_m_cut, self.data.borehole.T_error_cut = self.data.borehole.zT_m, self.data.borehole.suffix_root, self.data.borehole.T, self.data.borehole.zT_error_m, self.data.borehole.T_error
		# else:
		# 	cut_indices = np.where(self.data.borehole.zT_m > self.config['cut_top_m'])
		#
		# 	self.data.borehole.zT_m_cut, self.data.borehole.T_cut, self.data.borehole.zT_error_m_cut, self.data.borehole.T_error_cut = self.data.borehole.zT_m[cut_indices], self.data.borehole.T[cut_indices], self.data.borehole.zT_error_m[cut_indices], self.data.borehole.T_error[cut_indices]
		# 	suffix_root + 'ct' + str(cut_top_m) + 'm']
		# 	for arg in args: cut_list.append(arg[cut_indices])
		# return(tuple(cut_list))
		#
		#
		# zT_m_cut, T_cut_suffix, T_cut, zT_error_m_cut, T_error_cut
		
		
		
		