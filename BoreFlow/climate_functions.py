import numpy as np
import pandas as pd

from BoreFlow import general_python_functions


### FUNCTIONS FOR LOADING SURFACE TEMPERATURE HISTORIES ###




class PalaeoClimate:
	def __init__(self, cfg):
		self.config = cfg['climate']['palaeoclimate']
		self.data = general_python_functions.EmptyClass()

	def load_palaeoclimate(self):
		# TODO Include uncertainties in t0 and t1
		palaeoclimate_df = pd.read_excel(self.config['pc_excel'], usecols=["t1(ka)", "sigma_t1(ka)", "t2(ka)", "sigma_t2(ka)", "deltaT(degreesC)", "sigma_deltaT(degreesC)"])
		
		### Load palaeoclimate ages
		# Flip arrays so that oldest values (i.e. largest t0) are given first
		self.data.palaeoclimate_t0_seconds = np.array(np.flip(general_python_functions.y2s(palaeoclimate_df["t1(ka)"]*1e3))) # Start of period of temperature palaeoclimate_deltaTs in seconds
		self.data.palaeoclimate_t1_seconds = np.array(np.flip(general_python_functions.y2s(palaeoclimate_df["t2(ka)"]*1e3))) # End of period of temperature palaeoclimate_deltaTs in seconds
		self.data.palaeoclimate_t1_seconds[-1] = 3e9 # Set to non-zero to prevent division by zero
		
		if self.config['pc_sigma_ty_type'] == 'from_file':
			if palaeoclimate_df["sigma_t1(ka)"].any() == 'unstated' or palaeoclimate_df["sigma_t1(ka)"].any() == 'unstated':
				raise Exception("uncertainties in palaeoclimate ages are unspecified in file")
			else:
				self.data.palaeoclimate_sigma_t0_seconds = np.flip(general_python_functions.y2s(palaeoclimate_df["sigma_t1(ka)"]*1e3)) 
				self.data.palaeoclimate_sigma_t1_seconds = np.flip(general_python_functions.y2s(palaeoclimate_df["sigma_t2(ka)"]*1e3))
		elif self.config['pc_sigma_ty_type'] == 'constant_assigned':
			self.data.palaeoclimate_sigma_t0_seconds = general_python_functions.y2s(np.full(np.size(self.data.palaeoclimate_t0_seconds), float(self.config['pc_sigma_ty_cst'])))
			self.data.palaeoclimate_sigma_t1_seconds = self.data.palaeoclimate_sigma_t0_seconds
		else:
			raise Exception("option for uncertainties in palaeoclimate ages is unspecified")
		
		print('here')	
		
		
		### Load palaeoclimate temperature changes
		self.data.palaeoclimate_deltaTs = np.array(np.flip(palaeoclimate_df["deltaT(degreesC)"])) # Palaeoclimatic temperature
		print(np.size(self.data.palaeoclimate_deltaTs))
		
		if self.config['pc_sigma_deltaTs_type'] == 'from_file':
			if palaeoclimate_df["sigma_deltaT(degreesC)"].any() == 'unstated':
				raise Exception("uncertainties in palaeoclimate temperature changes are unspecified in file")
			else:
				self.data.palaeoclimate_sigma_deltaTs = palaeoclimate_df["sigma_deltaT(degreesC)"]
		elif self.config['pc_sigma_deltaTs_type'] == 'constant_assigned':
			self.data.palaeoclimate_sigma_deltaTs = np.full(np.size(self.data.palaeoclimate_deltaTs), float(self.config['pc_sigma_deltaTs_cst']))
		else:
			raise Exception("option for uncertainties in palaeoclimate temperature changes is unspecified")
		
		self.data.palaeoclimate_suffix = "_pc"



class RecentClimate:
	def __init__(self, cfg):
		self.config = cfg['climate']['recent_climate']
		self.data = general_python_functions.EmptyClass()
		
	def load_recent_climate(self):
		# Load time series of recent temperature changes at surface - NASA record of global temperature change within latitude bands. British mainland approx 50 to 60 N therefore use latitude band 44N - 64N. In ninth column
		# TODO - add uncertainty
		self.data.recent_climtemp_year, self.data.recent_climtemp_deltaTs = np.loadtxt(self.config['rc_input_csv'], delimiter=',', skiprows=1, unpack=True, usecols=(0,8))
		self.data.recent_climtemp_suffix = "_rc"
	
	def smooth_recent_climate(self):
		# Smooth temperature history
		if self.config['rc_smoother'] == "boxcar":
			self.data.recent_climtemp_year_smoothing_cutoff, self.data.recent_climtemp_deltaTs_smoothed = general_python_functions.smooth_data_boxcar(self.data.recent_climtemp_year, self.data.recent_climtemp_deltaTs, self.config['rc_smoothing_length'])
			# dtemp = {'recent_climtemp_year_smoothing_cutoff':recent_climtemp_year_smoothing_cutoff, 'recent_climtemp_deltaTs_smoothed':recent_climtemp_deltaTs_smoothed}
			# recent_climtemp_smoothed_df = pd.DataFrame(data=dtemp)
			# ftemp = open(recent_climtemp_smoothed_outfile + '.csv', 'w')
			# ftemp.write('### Original data: ' + str(recent_climtemp_hist_csv) + "\n")
			# ftemp.write('### Smoothed with boxcar filter of length ' + str(recent_climtemp_smoothing_length) + "\n")
			# recent_climtemp_smoothed_df.to_csv(ftemp, index=False)
			# ftemp.close()
			self.data.recent_climtemp_suffix = self.data.recent_climtemp_suffix + '_bcs' + str(self.config['rc_smoothing_length'])
		else:
			print("Smoothing for recent climate not specified - exiting")
			exit()
	
	def cut_recent_climate_to_borehole_year(self, borehole_year):
		def cut(year, sigma_year, T, sigma_T,):
			# TODO Add uncertainties
			t0_seconds = general_python_functions.y2s(borehole_year - year) # Convert years to seconds before borehole measurements
			# print(np.size(t0_seconds))
			t1_seconds = t0_seconds[1::]
			t1_seconds = np.hstack([t1_seconds, t1_seconds[-1] - general_python_functions.y2s(1)])
			sigma_t0_seconds = general_python_functions.y2s(sigma_year)
			sigma_t1_seconds = general_python_functions.y2s(sigma_year)
			# Keep only values from before borehole temperatures were measured
			year_index = np.min(np.where(year>(borehole_year-1)))
			year_cut = year[0:year_index]
			sigma_year_cut = sigma_year[0:year_index]
			t0_seconds_cut = general_python_functions.y2s(borehole_year - year_cut)
			# print(np.size(t0_seconds_cut))
			t1_seconds_cut = t0_seconds_cut[0:-1] + np.diff(t0_seconds_cut)
			# Set year 0 to 1e6 seconds to avoid division by zero
			t1_seconds_cut = np.hstack([t1_seconds_cut, [1e6]])
			sigma_t0_seconds_cut = general_python_functions.y2s(sigma_year[0:year_index])
			sigma_t1_seconds_cut = general_python_functions.y2s(sigma_year[1:year_index+1])
			T_cut = T[0:year_index]
			sigma_T_cut = sigma_T[0:year_index]
			return(t0_seconds, sigma_t0_seconds, t1_seconds, sigma_t1_seconds, year_cut, sigma_year_cut, t0_seconds_cut, sigma_t0_seconds_cut, t1_seconds_cut, sigma_t1_seconds_cut, T_cut, sigma_T_cut)
		
		self.data.recent_climtemp_t0_seconds, self.data.recent_climtemp_sigma_t0_seconds, self.data.recent_climtemp_t1_seconds, self.data.recent_climtemp_sigma_t1_seconds, self.data.recent_climtemp_year_cut, self.data.recent_climtemp_sigma_year_cut, self.data.recent_climtemp_t0_seconds_cut, self.data.recent_climtemp_sigma_t0_seconds_cut, self.data.recent_climtemp_t1_seconds_cut, self.data.recent_climtemp_sigma_t1_seconds_cut, self.data.recent_climtemp_deltaTs_cut, self.data.recent_climtemp_sigma_deltaTs_cut = cut(self.data.recent_climtemp_year, self.data.recent_climtemp_sigma_year, self.data.recent_climtemp_deltaTs, self.data.recent_climtemp_sigma_deltaTs, self.data.borehole_year)
		
		
		
		