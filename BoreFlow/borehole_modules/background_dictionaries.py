import numpy as np

def import_borehole_metadata_dict():
	borehole_metadata_dict = {
		'becklees':{
			'borehole_name_plot':'Becklees'
		}
	}
	return(borehole_metadata_dict)

def import_comp_conds_dict():
	comp_conds_dict = {
		'cermak1982conds':{
			'granite':{
				'number_samples':356,
				'mean_cond':3.05,
				'cond_low':1.25,
				'cond_high':4.45
			},
			'limestone':{
				'number_samples':487,
				'mean_cond':2.29,
				'cond_low':0.62,
				'cond_high':4.4
			},
			'sandstone':{
				'number_samples':1262,
				'mean_cond':2.47,
				'cond_low':0.9,
				'cond_high':6.5
			},
			'shale':{
				'number_samples':377,
				'mean_cond':2.07,
				'cond_low':0.55,
				'cond_high':4.25
			}	
		},
		'jessop1990conds':{
			'granite':{
				'number_samples':153,
				'mean_cond':3.43,
				'cond_low':2.3,
				'cond_high':3.6
			},
			'limestone':{
				'number_samples':445,
				'mean_cond':3.44,
				'cond_low':1.3,
				'cond_high':6.26
			},
			'sandstone':{
				'number_samples':11,
				'mean_cond':3.72,
				'cond_low':1.88,
				'cond_high':4.98
			}
		},
		'rollin1987conds':{
			# 'keuper_marl':{
			# 	'number_samples':41,
			# 	'mean_cond':2.28,
			# 	'sterr_cond':0.33
			# },
			'sherwood_sandstone':{
				'number_samples':64,
				'mean_cond':3.41,
				'sterr_cond':0.09
			},
			'magnesian_limestone':{
				'number_samples':12,
				'mean_cond':3.32,
				'sterr_cond':0.17
			},
			# 'westphalian_sandstone':{
			# 	'number_samples':37,
			# 	'mean_cond':3.31,
			# 	'stdev_cond':0.62
			# },
			# 'westphalian_siltstone':{
			# 	'number_samples':12,
			# 	'mean_cond':2.22,
			# 	'stdev_cond':0.29
			# },
			# 'westphalian_mudstone':{
			# 	'number_samples':25,
			# 	'mean_cond':1.49,
			# 	'stdev_cond':0.41
			# },
			# 'westphalian_coal':{
			# 	'number_samples':8,
			# 	'mean_cond':0.31,
			# 	'stdev_cond':0.08
			# },
			'mercia_mudstone':{
				'number_samples':225,
				'mean_cond':1.88,
				'sterr_cond':0.03
			}
		},
		'richardson1978conds':{
			'coal_measures_sandstone':{
				'number_samples':37,
				'mean_cond':3.31,
				'stdev_cond':0.62
			},
			'coal_measures_siltstone':{
				'number_samples':12,
				'mean_cond':2.22,
				'stdev_cond':0.29
			},
			'coal_measures_mudstone':{
				'number_samples':25,
				'mean_cond':1.49,
				'stdev_cond':0.41
			},
			'coal_measures_coal':{
				'number_samples':8,
				'mean_cond':0.31,
				'stdev_cond':0.08
			}
		}
	}
	return(comp_conds_dict)

def import_lith_plot_format_dict():
	# TODO Update with correct lithological colors
	lith_plot_format_dict = {
		'sandstone':'red',
		'shale':'purple',
		'coalmeasures':'blue',
		'mudrock':'green',
		'unstated':'gray'	
	}
	return(lith_plot_format_dict)
	
	
def import_rollin1987_lith_type_plot_format_dict():
	# TODO Update with correct lithological colors
	lith_plot_format_dict = {
		'mudstone':'---',
		'sandy_mudstone':'..---',
		'siltstone':'+',
		'silty_clay':'',
		'sandstone':'..',
		'silty_mudstone':'',
		'marl':'',
		'anhydrite':'x',
		'halite':'',
		'coal':'',
		'limestone':''
	}
	return(lith_plot_format_dict)

def import_cermak1982_lith_type_plot_format_dict():
	# TODO Update with correct lithological colors
	lith_plot_format_dict = {
		'mudrock':'---',
		'sandstone':'..',
		'evaporite':'x',
		'coal':'',
		'limestone':'+',
		'granite':'o'
	}
	return(lith_plot_format_dict)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	