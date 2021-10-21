import numpy as np
import pandas as pd

from BoreFlow import general_python_functions


def cut_recent_climate_history(year, sigma_year, T, sigma_T, borehole_year):
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


def process_recent_climate_history(recent_climtemp_path, recent_climtemp_filename, borehole, recent_climtemp_year, recent_climtemp_sigma_year, recent_climtemp_deltaTs, recent_climtemp_sigma_deltaTs, borehole_year, recent_climtemp_hist_csv, recent_climtemp_smoothed_filename, recent_climtemp_year_smoothing_cutoff, recent_climtemp_sigma_year_smoothing_cutoff, recent_climtemp_deltaTs_smoothed, recent_climtemp_sigma_deltaTs_smoothed, recent_climtemp_smoothed_outfile):
	### PROCESS CLIMATE HISTORIES ###
	### Cut recent temperature history to year before borehole acquired
	recent_climtemp_borehole_cut_outfile = str(recent_climtemp_path) + '/' + str(recent_climtemp_filename) + '_cut_' + str(borehole)
	recent_climtemp_t0_seconds, recent_climtemp_sigma_t0_seconds, recent_climtemp_t1_seconds, recent_climtemp_sigma_t1_seconds, recent_climtemp_year_cut, recent_climtemp_sigma_year_cut, recent_climtemp_t0_seconds_cut, recent_climtemp_sigma_t0_seconds_cut, recent_climtemp_t1_seconds_cut, recent_climtemp_sigma_t1_seconds_cut, recent_climtemp_deltaTs_cut, recent_climtemp_sigma_deltaTs_cut = cut_recent_climate_history(recent_climtemp_year, recent_climtemp_sigma_year, recent_climtemp_deltaTs, recent_climtemp_sigma_deltaTs, borehole_year)
	dtemp = {'recent_climtemp_year_cut':recent_climtemp_year_cut, 'recent_climtemp_sigma_year_cut':recent_climtemp_sigma_year_cut, 'recent_climtemp_t0_seconds_cut':recent_climtemp_t0_seconds_cut, 'recent_climtemp_sigma_t0_seconds_cut':recent_climtemp_sigma_t0_seconds_cut, 'recent_climtemp_t1_seconds_cut':recent_climtemp_t1_seconds_cut, 'recent_climtemp_sigma_t1_seconds_cut':recent_climtemp_sigma_t1_seconds_cut, 'recent_climtemp_deltaTs_cut':recent_climtemp_deltaTs_cut, 'recent_climtemp_sigma_deltaTs_cut':recent_climtemp_sigma_deltaTs_cut}
	recent_climtemp_borehole_cut_df = pd.DataFrame(data=dtemp)
	ftemp = open(recent_climtemp_borehole_cut_outfile + '.csv', 'w')
	ftemp.write('### Input data: ' + str(recent_climtemp_hist_csv) + "\n")
	ftemp.write('### Cut off at year ' + str(borehole_year) + "\n")
	recent_climtemp_borehole_cut_df.to_csv(ftemp, index=False)
	ftemp.close()
	### Cut smoothed recent temperature history to year before borehole acquired
	recent_climtemp_smoothed_borehole_cut_outfile = str(recent_climtemp_path) + '/' + str(recent_climtemp_smoothed_filename) + '_cut_' + str(borehole)
	recent_climtemp_t0_seconds_smoothing_cutoff, recent_climtemp_sigma_t0_seconds_smoothing_cutoff, recent_climtemp_t1_seconds_smoothing_cutoff, recent_climtemp_sigma_t1_seconds_smoothing_cutoff, recent_climtemp_year_smoothing_cutoff_cut, recent_climtemp_sigma_year_smoothing_cutoff_cut, recent_climtemp_t0_seconds_smoothing_cutoff_cut, recent_climtemp_sigma_t0_seconds_smoothing_cutoff_cut, recent_climtemp_t1_seconds_smoothing_cutoff_cut, recent_climtemp_sigma_t1_seconds_smoothing_cutoff_cut, recent_climtemp_deltaTs_smoothed_cut, recent_climtemp_sigma_deltaTs_smoothed_cut = cut_recent_climate_history(recent_climtemp_year_smoothing_cutoff, recent_climtemp_sigma_year_smoothing_cutoff, recent_climtemp_deltaTs_smoothed, recent_climtemp_sigma_deltaTs_smoothed, borehole_year)
	dtemp = {'recent_climtemp_year_smoothing_cutoff_cut':recent_climtemp_year_smoothing_cutoff_cut, 'recent_climtemp_sigma_year_smoothing_cutoff_cut':recent_climtemp_sigma_year_smoothing_cutoff_cut, 'recent_climtemp_t0_seconds_smoothing_cutoff_cut':recent_climtemp_t0_seconds_smoothing_cutoff_cut, 'recent_climtemp_sigma_t0_seconds_smoothing_cutoff_cut':recent_climtemp_sigma_t0_seconds_smoothing_cutoff_cut, 'recent_climtemp_t1_seconds_smoothing_cutoff_cut':recent_climtemp_t1_seconds_smoothing_cutoff_cut, 'recent_climtemp_sigma_t1_seconds_smoothing_cutoff_cut':recent_climtemp_sigma_t1_seconds_smoothing_cutoff_cut, 'recent_climtemp_deltaTs_smoothed_cut':recent_climtemp_deltaTs_smoothed_cut, 'recent_climtemp_sigma_deltaTs_smoothed_cut':recent_climtemp_sigma_deltaTs_smoothed_cut}
	recent_climtemp_smoothed_borehole_cut_df = pd.DataFrame(data=dtemp)
	ftemp = open(recent_climtemp_smoothed_borehole_cut_outfile + '.csv', 'w')
	ftemp.write('### Input data: ' + str(recent_climtemp_smoothed_outfile) + ".csv\n")
	ftemp.write('### Cut off at year ' + str(borehole_year) + "\n")
	recent_climtemp_smoothed_borehole_cut_df.to_csv(ftemp, index=False)
	ftemp.close()
	return(recent_climtemp_t0_seconds, recent_climtemp_sigma_t0_seconds, recent_climtemp_t1_seconds, recent_climtemp_sigma_t1_seconds, recent_climtemp_year_cut, recent_climtemp_sigma_year_cut, recent_climtemp_t0_seconds_cut, recent_climtemp_sigma_t0_seconds_cut, recent_climtemp_t1_seconds_cut, recent_climtemp_sigma_t1_seconds_cut, recent_climtemp_deltaTs_cut, recent_climtemp_sigma_deltaTs_cut, recent_climtemp_t0_seconds_smoothing_cutoff, recent_climtemp_sigma_t0_seconds_smoothing_cutoff, recent_climtemp_t1_seconds_smoothing_cutoff, recent_climtemp_sigma_t1_seconds_smoothing_cutoff, recent_climtemp_year_smoothing_cutoff_cut, recent_climtemp_sigma_year_smoothing_cutoff_cut, recent_climtemp_t0_seconds_smoothing_cutoff_cut, recent_climtemp_sigma_t0_seconds_smoothing_cutoff_cut, recent_climtemp_t1_seconds_smoothing_cutoff_cut, recent_climtemp_sigma_t1_seconds_smoothing_cutoff_cut, recent_climtemp_deltaTs_smoothed_cut, recent_climtemp_sigma_deltaTs_smoothed_cut)



















