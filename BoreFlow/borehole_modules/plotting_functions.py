import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
from matplotlib import transforms
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredAuxTransformBox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from PIL import Image

import pprint

from borehole_modules import general_python_functions




	# temp_ago = np.column_stack([deltaTs, deltaTs]).ravel()
	# time_ago = np.column_stack([t0_seconds, t1_seconds]).ravel()
	# time_ago = general_python_functions.s2y(time_ago)
	# fig = plt.figure(figsize=(5,3.5))
	# ax1 = fig.add_subplot(111, xlim=(0,150))
	# ax1.set_ylabel(r'$\Delta T_s$ / $^{\circ}$C')
	# ax1.set_xlabel('Time before present / ka')
	# ax1.plot(time_ago/1e3, temp_ago, 'k-', label=r'$\Delta T_c$')
	# fig.savefig(figure_name, bbox_inches='tight', dpi=300)
	# plt.close(fig)

def plot_palaeoclimate(palaeoclimate_deltaTs, palaeoclimate_sigma_deltaTs, palaeoclimate_t0_seconds, palaeoclimate_t1_seconds, palaeoclimate_sigma_t0_seconds, palaeoclimate_sigma_t1_seconds, palaeoclimate_deltaTs_input, palaeoclimate_t0_seconds_input, palaeoclimate_t1_seconds_input, pc_plot_input_option, figure_name):

	temp_ago = palaeoclimate_deltaTs
	temp_ago_ravel = np.column_stack([temp_ago, temp_ago]).ravel()
	temp_ago_low = palaeoclimate_deltaTs - palaeoclimate_sigma_deltaTs
	temp_ago_low_ravel = np.column_stack([temp_ago_low, temp_ago_low]).ravel()
	temp_ago_high = palaeoclimate_deltaTs + palaeoclimate_sigma_deltaTs
	temp_ago_high_ravel = np.column_stack([temp_ago_high, temp_ago_high]).ravel()
	time0_ago = general_python_functions.s2y(palaeoclimate_t0_seconds)
	time1_ago = general_python_functions.s2y(palaeoclimate_t1_seconds)
	time1_ago_low = general_python_functions.s2y(palaeoclimate_t1_seconds - palaeoclimate_sigma_t1_seconds)
	time1_ago_high = general_python_functions.s2y(palaeoclimate_t1_seconds + palaeoclimate_sigma_t1_seconds)
	time_ago_ravel = general_python_functions.s2y(np.column_stack([palaeoclimate_t0_seconds, palaeoclimate_t1_seconds]).ravel())
	time_ago_low_ravel = general_python_functions.s2y(np.column_stack([palaeoclimate_t0_seconds - palaeoclimate_sigma_t0_seconds, palaeoclimate_t1_seconds - palaeoclimate_sigma_t1_seconds]).ravel())
	time_ago_high_ravel = general_python_functions.s2y(np.column_stack([palaeoclimate_t0_seconds + palaeoclimate_sigma_t0_seconds, palaeoclimate_t1_seconds + palaeoclimate_sigma_t1_seconds]).ravel())
	temp_ago_mc = np.column_stack([palaeoclimate_deltaTs_input, palaeoclimate_deltaTs_input]).ravel()
	time_ago_mc = np.column_stack([palaeoclimate_t0_seconds_input, palaeoclimate_t1_seconds_input]).ravel()
	time_ago_mc = general_python_functions.s2y(time_ago_mc)

	fig = plt.figure(figsize=(10,3.5))
	ax1 = fig.add_subplot(111, xlim=(0,150))
	ax1.xaxis.set_label_position('top') 
	### TODO Make sure no white spaces in infill
	ax1.fill_between(time0_ago/1e3, temp_ago_low, temp_ago_high, color='lightgray', step='post')
	ax1.fill_between(time1_ago/1e3, temp_ago_low, temp_ago_high, color='lightgray', step='pre')
	ax1.fill_betweenx(temp_ago, time1_ago_low/1e3, time1_ago_high/1e3, where=np.full(np.size(temp_ago), True), interpolate=True, color='lightgray', step='post')
	ax1.fill_betweenx(temp_ago_low, time1_ago_low/1e3, time1_ago_high/1e3, where=np.full(np.size(temp_ago), True), interpolate=True, color='lightgray', step='post')
	ax1.fill_betweenx(temp_ago_high, time1_ago_low/1e3, time1_ago_high/1e3, where=np.full(np.size(temp_ago), True), interpolate=True, color='lightgray', step='post')
	# ax1.fill_betweenx(temp_ago, time1_ago/1e3, time1_ago_high/1e3, color='blue', step='post', alpha=0.5)
	# ax1.fill_between(time_ago_ravel/1e3, temp_ago_low_ravel, temp_ago_high_ravel, color='gray', alpha=0.2)
# 											ax1.fill_between(time_ago_high/1e3, temp_ago_low, temp_ago_high, color='gray', alpha=1)
# 											ax1.fill_between(time_ago_low/1e3, temp_ago_low, temp_ago_high, color='gray', alpha=1)
	# ax1.fill_betweenx(temp_ago, time_ago_low/1e3, time_ago_high/1e3, color='gray', alpha=1)
	# ax1.fill_betweenx(temp_ago_high, time_ago_low/1e3, time_ago_high/1e3, color='gray', alpha=1)
	ax1.plot(time_ago_ravel/1e3, temp_ago_ravel, 'k-', label=r'$\Delta T_c$')
	if pc_plot_input_option == 'yes':
		ax1.plot(time_ago_mc/1e3, temp_ago_mc, 'r-', label=r'$\Delta T_c$', alpha=1)
	else:
		ax1.plot(time_ago_mc/1e3, temp_ago_mc, 'r-', label=r'$\Delta T_c$', alpha=0)
	ax1.set_ylabel(r'$\Delta T_s$ / $^{\circ}$C')
	ax1.set_xlabel('Time before present / ka')
	
	fig.savefig(figure_name, bbox_inches='tight', dpi=300)
	plt.close(fig)
	
	return()


def plot_recent_climate_history_new(recent_climtemp_deltaTs, recent_climtemp_t0_seconds, recent_climtemp_t1_seconds, recent_climtemp_deltaTs_cut, recent_climtemp_t0_seconds_cut, recent_climtemp_t1_seconds_cut, recent_climtemp_deltaTs_smoothed_cut, recent_climtemp_t0_seconds_smoothing_cutoff_cut, recent_climtemp_t1_seconds_smoothing_cutoff_cut, recent_climtemp_deltaTs_input, recent_climtemp_t0_seconds_input, recent_climtemp_t1_seconds_input, recent_climtemp_year, rc_plot_input_option, figure_name):
	
	recent_climtemp_deltaTs_ago = np.column_stack([recent_climtemp_deltaTs, recent_climtemp_deltaTs]).ravel()
	recent_climtemp_year_ago = general_python_functions.s2y(np.column_stack([recent_climtemp_t0_seconds, recent_climtemp_t1_seconds]).ravel())
	recent_climtemp_deltaTs_ago_cut = np.column_stack([recent_climtemp_deltaTs_cut, recent_climtemp_deltaTs_cut]).ravel()
	recent_climtemp_year_ago_cut = general_python_functions.s2y(np.column_stack([recent_climtemp_t0_seconds_cut, recent_climtemp_t1_seconds_cut]).ravel())
	recent_climtemp_deltaTs_ago_smoothed_cut = np.column_stack([recent_climtemp_deltaTs_smoothed_cut, recent_climtemp_deltaTs_smoothed_cut]).ravel()
	recent_climtemp_year_ago_smoothing_cutoff_cut = general_python_functions.s2y(np.column_stack([recent_climtemp_t0_seconds_smoothing_cutoff_cut, recent_climtemp_t1_seconds_smoothing_cutoff_cut]).ravel())
	recent_climtemp_deltaTs_ago_input = np.column_stack([recent_climtemp_deltaTs_input, recent_climtemp_deltaTs_input]).ravel()
	recent_climtemp_year_ago_input = general_python_functions.s2y(np.column_stack([recent_climtemp_t0_seconds_input, recent_climtemp_t1_seconds_input]).ravel())
	
	fig = plt.figure(figsize=(10,3.5))
	ax1 = fig.add_subplot(111)
	ax1.set_ylabel(r'$\Delta T_s$ / $^{\circ}$C')
	ax1.set_xlabel('Time before borehole drilled / years')
	if rc_plot_input_option == 'yes':
		ax1.plot(recent_climtemp_year_ago, recent_climtemp_deltaTs_ago, color='black', alpha=0.25)
		# ax1.plot(recent_climtemp_year_ago_cut, recent_climtemp_deltaTs_ago_cut, color='black')
		ax1.plot(recent_climtemp_year_ago_smoothing_cutoff_cut, recent_climtemp_deltaTs_ago_smoothed_cut, color='black', alpha=0.5)
		ax1.plot(recent_climtemp_year_ago_input, recent_climtemp_deltaTs_ago_input, color='red')
	else:
		ax1.plot(recent_climtemp_year_ago, recent_climtemp_deltaTs_ago, color='black')
		ax1.plot(recent_climtemp_year_ago_smoothing_cutoff_cut, recent_climtemp_deltaTs_ago_smoothed_cut, color='black', alpha=0)
		ax1.plot(recent_climtemp_year_ago_input, recent_climtemp_deltaTs_ago_input, color='red', alpha=0)
	ax1year = ax1.twiny()
	ax1year.set_xlabel('Year')
	ax1year.invert_xaxis()
	ax1year.plot(recent_climtemp_year, recent_climtemp_deltaTs, alpha=0)
	
	fig.savefig(figure_name, bbox_inches='tight', dpi=300)
	plt.close(fig)
	
	# print('here')
	# print(figure_name)
	
	return()
	

def plot_recent_climate_history(year, deltaTs, year_smoothing_cutoff, deltaTs_smoothed, year_cut, deltaTs_cut, year_smoothing_cutoff_cut, deltaTs_smoothed_cut, figures_path):
	figure_name = figures_path + '_' + 'recent_climate_history.jpg'
	year_ago = year_cut[-1] - np.flip(year)
	year_smoothing_cutoff_ago = year_smoothing_cutoff_cut[-1] - np.flip(year_smoothing_cutoff)
	year_cut_ago = year_cut[-1] - np.flip(year_cut)
	year_smoothing_cutoff_cut_ago = year_smoothing_cutoff_cut[-1] - np.flip(year_smoothing_cutoff_cut)
	deltaTs_ago = np.flip(deltaTs)
	deltaTs_smoothed_ago = np.flip(deltaTs_smoothed)
	deltaTs_cut_ago = np.flip(deltaTs_cut)
	deltaTs_smoothed_cut_ago = np.flip(deltaTs_smoothed_cut)

	# deltaTs_ago = deltaTs[-1] - np.flip(deltaTs)
	# year_ago = year[-1] - np.flip(year)
	fig = plt.figure(figsize=(5,3.5))
	ax1 = fig.add_subplot(111)
	ax1.set_ylabel(r'$\Delta T_s$ / $^{\circ}$C')
	ax1.set_xlabel('Years before borehole drilled')
	plt.plot(year_ago, deltaTs_ago, color='black', alpha=0.25)
	plt.plot(year_smoothing_cutoff_ago, deltaTs_smoothed_ago, color='red', alpha=0.25)
	plt.plot(year_cut_ago, deltaTs_cut_ago, color='black')
	plt.plot(year_smoothing_cutoff_cut_ago, deltaTs_smoothed_cut_ago, color='red')
	ax2 = ax1.twiny()
	ax2.set_xlabel('Year')
	ax2.invert_xaxis()
	ax2.plot(year, deltaTs, alpha=0)
	
	fig.savefig(figure_name, bbox_inches='tight', dpi=300)
	plt.close(fig)
	
	return()



def set_up_plot_title(borehole_analysis_dict, int_option, heat_flow_estimation_method, plot_label_dict, figures_path):
	
	if int_option == 'orig_source_strat':
		if heat_flow_estimation_method == 'interval_method':
			plot_title = str(borehole_analysis_dict['metadata']['borehole_name_plot']) + ': Estimation of heat flow using interval method for measured conductivites averaged within stratigraphic divisions of ' + str(borehole_analysis_dict['metadata']['orig_source_ref'])
			figure_name = figures_path + '_' + str(plot_label_dict['orig_source_ref']) + '_ave_k_int_meth_q_est'
		elif heat_flow_estimation_method == 'bullard_method':
			plot_title = str(borehole_analysis_dict['metadata']['borehole_name_plot']) + ': Estimation of heat flow using Bullard method for measured conductivites averaged within stratigraphic divisions of ' + str(borehole_analysis_dict['metadata']['orig_source_ref'])
			figure_name = figures_path + '_' + str(plot_label_dict['orig_source_ref']) + '_ave_k_bull_meth_q_est'
		else:
			plot_title = 'undefined'
	elif plot_label_dict['layer_option'] == 'cst_depth_intervals':
		if heat_flow_estimation_method == 'interval_method': 
			plot_title = str(borehole_analysis_dict['metadata']['borehole_name_plot']) + ': Estimation of heat flow using interval method for measured conductivites averaged within ' + str(plot_label_dict['cst_z_int']) + ' m depth intervals'
			figure_name = figures_path + '_cst_z_int' + str(plot_label_dict['cst_z_int']) + 'm_ave_k_int_meth_q_est'
		elif heat_flow_estimation_method == 'bullard_method':
			plot_title = str(borehole_analysis_dict['metadata']['borehole_name_plot']) + ': Estimation of heat flow using Bullard method for measured conductivites averaged within ' + str(plot_label_dict['cst_z_int']) + ' m depth intervals'
			figure_name = figures_path + '_cst_z_int' + str(plot_label_dict['cst_z_int']) + 'm_ave_k_bull_meth_q_est'
		else:
			plot_title = 'undefined'
	else:
		plot_title = 'undefined'
	
	return(figure_name, plot_title)


def set_up_plot_formatting_constant_height(axis_setup_dict, axis_position_dict):
	
	figure_width = axis_setup_dict['figure_width']
	figure_height = axis_setup_dict['figure_height']
	xpos = axis_setup_dict['init_xpos']
	ypos = axis_setup_dict['init_ypos']
	figure_label_inset = axis_setup_dict['figure_label_inset']
	
	# Unpack specification of axis positioning
	fractional_pos_dict = {}
	for axes in axis_position_dict.keys():
		fractional_pos_dict['axis_' + str(axes)] = {}
		
		fractional_pos_dict['axis_' + str(axes)]['frac_xpos'] = xpos / figure_width
		fractional_pos_dict['axis_' + str(axes)]['frac_ypos'] = ypos / figure_height
		
		fractional_pos_dict['axis_' + str(axes)]['panel_frac_height'] = axis_position_dict[axes]['height'] / figure_height
		fractional_pos_dict['axis_' + str(axes)]['panel_frac_width'] = axis_position_dict[axes]['width'] / figure_width
		
		fractional_pos_dict['axis_' + str(axes)]['figure_label_frac_inset_width'] = figure_label_inset / axis_position_dict[axes]['width']
		fractional_pos_dict['axis_' + str(axes)]['figure_label_frac_inset_height'] = figure_label_inset / axis_position_dict[axes]['height']
		
		xpos = xpos + axis_position_dict[axes]['hspace_after']
		ypos = ypos + axis_position_dict[axes]['vspace_after']
		
		
		
# 		ypos = ypos / figure_height
#
# 		row_frac_ypos
#
#
# 		fractional_pos_dict['row' + str(row_index) + 'col' + str(column_index)] = {}
#
#
# 		fractional_pos_dict['row' + str(row_index) + 'col' + str(column_index)]['frac_xpos'] = start_width
# 		fractional_pos_dict['row' + str(row_index) + 'col' + str(column_index)]['panel_frac_height'] = row_fractional_height
# 		fractional_pos_dict['row' + str(row_index) + 'col' + str(column_index)]['figure_label_frac_inset_height'] = figure_label_inset / row_height
#
# 		if column == 'temp':
# 			fractional_pos_dict['row' + str(row_index) + 'col' + str(column_index)]['panel_frac_width'] = temp_fractional_width
# 			fractional_pos_dict['row' + str(row_index) + 'col' + str(column_index)]['figure_label_frac_inset_width'] = figure_label_inset / temp_width
# 			start_width = start_width + temp_fractional_width
#
#
# 	# Unpack dictionary of column dimensions
# 	strat_width = axis_setup_dict['strat']['width']
# 	cond_width = axis_setup_dict['cond']['width']
# 	temp_width = axis_setup_dict['temp']['width']
# 	pc_width = axis_setup_dict['pc']['width']
# 	rc_width = axis_setup_dict['rc']['width']
# 	res_width = axis_setup_dict['res']['width']
# 	bullard_width = axis_setup_dict['bullard']['width']
# 	figure_label_inset = axis_setup_dict['figure_label_inset']
# 	row_spacings = axis_setup_dict['row_spacings']
# 	column_spacing = axis_setup_dict['column_spacing']
#
# 	# Set up vertical dimensions
# 	figure_height = 0
# 	number_rows = len(axis_position_dict.keys())
# 	for row_key in axis_position_dict.keys():
# 		figure_height = figure_height + axis_position_dict[row_key]['height'] + row_spacings[row_key]
# 	# figure_height = figure_height + (number_rows-1) * row_spacing
# 	# figure_height = number_rows * row_height + (number_rows-1) * row_spacing
# 	# figure_height= figure_height + 0.5 * row_height
# 	# row_fractional_height = row_height / figure_height
# # 	row_fractional_spacing = row_spacing / figure_height
#
# 	# Unpack specification of axis positioning
# 	fractional_pos_dict = {}
# 	row_index = 0
# 	row_frac_ypos = 1
#
# 	for row_key in axis_position_dict.keys():
# 		number_temp_panels = axis_position_dict[row_key]['panels'].count('temp')
# 		number_strat_panels = axis_position_dict[row_key]['panels'].count('strat')
# 		number_cond_panels = axis_position_dict[row_key]['panels'].count('cond')
# 		number_pc_panels = axis_position_dict[row_key]['panels'].count('pc')
# 		number_rc_panels = axis_position_dict[row_key]['panels'].count('rc')
# 		number_res_panels = axis_position_dict[row_key]['panels'].count('res')
# 		number_bullard_panels = axis_position_dict[row_key]['panels'].count('bullard')
# 		number_column_spaces = axis_position_dict[row_key]['panels'].count('column_space')
#
# 		row_height = axis_position_dict[row_key]['height']
#
# 		row_fractional_height = row_height / figure_height
# 		row_fractional_spacing = row_spacing / figure_height
# 		row_frac_ypos = row_frac_ypos - row_fractional_height - row_fractional_spacing
#
# 		# Set figure width based on plots in top row
# 		if row_index == 0:
# 			figure_width = number_temp_panels * temp_width + number_strat_panels * strat_width + number_cond_panels * cond_width + number_pc_panels * pc_width + number_rc_panels * rc_width + number_res_panels * res_width + number_bullard_panels * bullard_width + number_column_spaces * column_spacing
# 		# figure_width = figure_width + 2 * row_spacing
#
#
#
# 		temp_fractional_width = temp_width / figure_width
# 		strat_fractional_width = strat_width / figure_width
# 		cond_fractional_width = cond_width / figure_width
# 		pc_fractional_width = pc_width / figure_width
# 		rc_fractional_width = rc_width / figure_width
# 		res_fractional_width = res_width / figure_width
# 		bullard_fractional_width = bullard_width / figure_width
# 		column_fractional_spacing = column_spacing / figure_width
#
# 		start_width = row_fractional_spacing
# 		column_index = 0
#
# 		for column in axis_position_dict[row_key]['panels']:
#
# 			fractional_pos_dict['row' + str(row_index) + 'col' + str(column_index)] = {}
# 			fractional_pos_dict['row' + str(row_index) + 'col' + str(column_index)]['frac_ypos'] = row_frac_ypos
#
# 			fractional_pos_dict['row' + str(row_index) + 'col' + str(column_index)]['frac_xpos'] = start_width
# 			fractional_pos_dict['row' + str(row_index) + 'col' + str(column_index)]['panel_frac_height'] = row_fractional_height
# 			fractional_pos_dict['row' + str(row_index) + 'col' + str(column_index)]['figure_label_frac_inset_height'] = figure_label_inset / row_height
#
# 			if column == 'temp':
# 				fractional_pos_dict['row' + str(row_index) + 'col' + str(column_index)]['panel_frac_width'] = temp_fractional_width
# 				fractional_pos_dict['row' + str(row_index) + 'col' + str(column_index)]['figure_label_frac_inset_width'] = figure_label_inset / temp_width
# 				start_width = start_width + temp_fractional_width
# 				column_index += 1
# 			elif column == 'strat':
# 				fractional_pos_dict['row' + str(row_index) + 'col' + str(column_index)]['panel_frac_width'] = strat_fractional_width
# 				fractional_pos_dict['row' + str(row_index) + 'col' + str(column_index)]['figure_label_frac_inset_width'] = figure_label_inset / strat_width
# 				start_width = start_width + strat_fractional_width
# 				column_index += 1
# 			elif column == 'cond':
# 				fractional_pos_dict['row' + str(row_index) + 'col' + str(column_index)]['panel_frac_width'] = cond_fractional_width
# 				fractional_pos_dict['row' + str(row_index) + 'col' + str(column_index)]['figure_label_frac_inset_width'] = figure_label_inset / cond_width
# 				start_width = start_width + cond_fractional_width
# 				column_index += 1
# 			elif column == 'pc':
# 				fractional_pos_dict['row' + str(row_index) + 'col' + str(column_index)]['panel_frac_width'] = pc_fractional_width
# 				fractional_pos_dict['row' + str(row_index) + 'col' + str(column_index)]['figure_label_frac_inset_width'] = figure_label_inset / pc_width
# 				start_width = start_width + pc_fractional_width
# 				column_index += 1
# 			elif column == 'rc':
# 				fractional_pos_dict['row' + str(row_index) + 'col' + str(column_index)]['panel_frac_width'] = rc_fractional_width
# 				fractional_pos_dict['row' + str(row_index) + 'col' + str(column_index)]['figure_label_frac_inset_width'] = figure_label_inset / rc_width
# 				start_width = start_width + rc_fractional_width
# 				column_index += 1
# 			elif column == 'res':
# 				fractional_pos_dict['row' + str(row_index) + 'col' + str(column_index)]['panel_frac_width'] = res_fractional_width
# 				fractional_pos_dict['row' + str(row_index) + 'col' + str(column_index)]['figure_label_frac_inset_width'] = figure_label_inset / res_width
# 				start_width = start_width + res_fractional_width
# 				column_index += 1
# 			elif column == 'bullard':
# 				fractional_pos_dict['row' + str(row_index) + 'col' + str(column_index)]['panel_frac_width'] = bullard_fractional_width
# 				fractional_pos_dict['row' + str(row_index) + 'col' + str(column_index)]['figure_label_frac_inset_width'] = figure_label_inset / bullard_width
# 				start_width = start_width + bullard_fractional_width
# 				column_index += 1
# 			elif column == 'column_space':
# 				start_width = start_width + column_fractional_spacing
# 		row_index += 1
	
	# Set up formatting options for axes
	axes_linewidth = 1.5
	axes_tick_labels_fontsize = 12
	axes_labels_fontsize = 12
	figure_label_fontsize = 12
	label_fontsize = 10
	annotation_fontdict={'fontweight': 'bold','size':label_fontsize}
	figure_label_fontdict={'fontweight':'bold', 'size':figure_label_fontsize}
	pad_fraction = 0.2*figure_width
	borderpad_fraction = 0.6*figure_width
	figure_label_pad = pad_fraction / figure_label_fontsize
	figure_label_borderpad = borderpad_fraction / figure_label_fontsize
	label_pad = pad_fraction / label_fontsize
	label_borderpad = borderpad_fraction / label_fontsize
	# these are matplotlib.patch.Patch properties
	inset_props = dict(boxstyle='square', facecolor='none', edgecolor='black', linewidth=axes_linewidth)
	white_bbox_props = dict(boxstyle='square', facecolor='white', edgecolor='black', linewidth=axes_linewidth/2)
	figure_label_white_bbox_props = dict(boxstyle='square', facecolor='white', edgecolor='black', linewidth=axes_linewidth)
	
	plt.rcParams['axes.linewidth'] = axes_linewidth 
	plt.rcParams['xtick.major.size'] = 7.5
	plt.rcParams['xtick.major.width'] = axes_linewidth
	plt.rcParams['xtick.minor.size'] = 3.75
	plt.rcParams['xtick.minor.width'] = axes_linewidth
	plt.rcParams['ytick.major.size'] = 7.5
	plt.rcParams['ytick.major.width'] = axes_linewidth
	plt.rcParams['ytick.minor.size'] = 3.75
	plt.rcParams['ytick.minor.width'] = axes_linewidth
	plt.rcParams['font.family'] = 'Helvetica'
	plt.rcParams['font.weight'] = 'normal' 
	plt.rcParams['font.size'] = axes_tick_labels_fontsize 
	plt.rcParams['axes.labelsize'] = axes_labels_fontsize
	plt.rcParams['axes.labelpad'] = axes_labels_fontsize
	
	# Set up formatting options for data
	interior_linewidth = 1.5
	
	plt.rcParams['lines.linewidth'] = interior_linewidth
	plt.rcParams['lines.markersize'] = 4
	
	# Return all local variables as dictionary
	plot_format_dict = dict()
	plot_format_dict.update(locals())
	plot_format_dict.pop('plot_format_dict', None)
	
	return(plot_format_dict)


# ### Plot temp against depth with fitted line
# def plot_simple_heat_flow(z_T_f, sigma_z_T_f, T_f, sigma_T_f, z_k_f, sigma_z_k_f, k_f, sigma_k_f, dTdz_f, sigma_dTdz_f, dTdz_km_f, sigma_dTdz_km_f, T0_f, sigma_T0_f, cov_dTdz_T0_f, r_dTdz_T0_f, dof_f, dTdz_T0_chi_sq_f, dTdz_T0_reduced_chi_sq_f, mean_k_round_f, std_mean_k_round_f, weighted_mean_k_round_f, std_weighted_mean_k_round_f, Q_f, sigma_Q_f, image):
#
# 	# Set up figure limits
# 	min_depth_m=0
# 	max_depth_m=600
# 	min_k=1
# 	max_k=5.75
#
#
#
#
# 	# fig = plt.figure(figsize=(figure_width, figure_height), dpi=100)
#
# 	# ax1 = fig.add_axes([ax1_fractional_xpos, ax1_fractional_ypos, ax1_fractional_width, ax1_fractional_height], xlabel=r'$k$ / W m$^{-1}$ K$^{-1}$', ylabel=r'$z$ / m')
# 	# ax1.set_ylim(top=min_depth_m, bottom=max_depth_m)
# 	# ax1.set_xlim(left=min_k, right=max_k)
# 	# ax1.xaxis.set_label_position('top')
# 	# ax1.yaxis.set_ticks_position('both')
# 	# ax1.xaxis.set_ticks_position('both')
# 	# ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
# 	# ax1.yaxis.set_minor_locator(MultipleLocator(50))
# 	ax1.plot(np.array([mean_k_round_f, mean_k_round_f]), np.array([0, np.max(z_k_f)]), color='red', linewidth=axes_linewidth)
# 	ax1.fill_betweenx(np.array([0, np.max(z_k_f)]), mean_k_round_f-std_mean_k_round_f, mean_k_round_f+std_mean_k_round_f, facecolor='red', alpha=0.25)
# 	ax1.errorbar(k_f, z_k_f, xerr=sigma_k_f, yerr=sigma_z_k_f, fmt='.k', markeredgecolor='k', zorder=5)
# 	# ax1.text(0.95, 0.95, r'$\mathbf{\bar{k}=%s \pm %s}$ W m$^{-1}$ K$^{-1}$' % (mean_k_round_f, std_mean_k_round_f), transform=ax1.transAxes, fontdict=annotation_fontdict, verticalalignment='top', horizontalalignment='right', bbox=props)
# 	at = AnchoredText("a", prop=dict(size=figure_label_fontsize, weight='bold'), frameon=True, loc='lower left', pad=figure_label_pad, borderpad=figure_label_borderpad, bbox_transform=ax1.transAxes)
# 	at.patch.set_boxstyle("square, pad=0")
# 	at.patch.set_linewidth(axes_linewidth)
# 	ax1.add_artist(at)
# 	at = AnchoredText(r'$\mathbf{\bar{k}=%s \pm %s}$ W m$\mathbf{^{-1}}$ K$\mathbf{^{-1}}$' % (mean_k_round_f, std_mean_k_round_f), prop=dict(size=label_fontsize, weight='bold'), frameon=True, loc='upper right', pad=label_pad, borderpad=label_borderpad, bbox_transform=ax1.transAxes)
# 	at.patch.set_boxstyle("square, pad=0")
# 	at.patch.set_linewidth(axes_linewidth)
# 	ax1.add_artist(at)
#
# 	# ax2 = fig.add_axes([ax2_fractional_xpos, ax2_fractional_ypos, ax2_fractional_width, ax2_fractional_height], xlabel=r'$T$ / $^{\circ}$C')
# # 	ax2.invert_yaxis()
# # 	ax2.set_ylim(top=min_depth_m, bottom=max_depth_m)
# # 	ax2.xaxis.set_label_position('top')
# # 	ax2.yaxis.set_ticks_position('both')
# # 	ax2.xaxis.set_ticks_position('both')
# # 	ax2.xaxis.set_minor_locator(MultipleLocator(1))
# # 	ax2.yaxis.set_minor_locator(MultipleLocator(50))
# # 	ax2.yaxis.set_major_formatter(FormatStrFormatter(''))
# 	ax2.errorbar(T_f, z_T_f, xerr=sigma_T_f, yerr=sigma_z_T_f, fmt='.k', color='k', markeredgewidth=0.1)
# 	ax2.fill_betweenx(z_T_f, (dTdz_f+sigma_dTdz_f) * z_T_f + (T0_f+sigma_T0_f), (dTdz_f-sigma_dTdz_f) * z_T_f + (T0_f-sigma_T0_f), facecolor='black', alpha=0.25)
# 	ax2.plot(dTdz_f * z_T_f + T0_f, z_T_f, color='red', linewidth=axes_linewidth)
# 	# place a text box in upper left in axes coords
# 	at = AnchoredText("b", prop=dict(size=figure_label_fontsize, weight='bold'), frameon=True, loc='lower left', pad=figure_label_pad, borderpad=figure_label_borderpad, bbox_transform=ax2.transAxes)
# 	at.patch.set_boxstyle("square, pad=0")
# 	at.patch.set_linewidth(axes_linewidth)
# 	ax2.add_artist(at)
# 	at = AnchoredText(r'$\mathbf{\frac{dT}{dz}=%s \pm %s \ ^{\circ}}$C km$\mathbf{^{-1}}$' % (dTdz_km_f, sigma_dTdz_km_f), prop=dict(size=label_fontsize, weight='bold'), frameon=True, loc='upper right', pad=label_pad, borderpad=label_borderpad, bbox_transform=ax2.transAxes)
# 	at.patch.set_boxstyle("square, pad=0")
# 	at.patch.set_linewidth(axes_linewidth)
# 	ax2.add_artist(at)
# 	ax2.text(0.9725, 0.91, r'$\mathbf{\chi^2_\nu=%i}$' % (dTdz_T0_reduced_chi_sq_f), transform=ax2.transAxes, fontdict=annotation_fontdict, verticalalignment='top', horizontalalignment='right', bbox=props)
# 	ax2.text(0.9725, 0.8375, r'$\mathbf{q=%s \pm %s}$ mW m$\mathbf{^{-2}}$' % (Q_f, sigma_Q_f), transform=ax2.transAxes, fontdict=annotation_fontdict, verticalalignment='top', horizontalalignment='right', bbox=props)
#
# 	fig.savefig(image, bbox_inches='tight', transparent=True)
# 	plt.close(fig)
# 	return()


def plot_stratigraphy(plot_lith_fill_dict, plot_lith_fill_dict_keyword, strat_interp_lith_k_calcs_df, z_int_plot, plot_format_dict, strat_label_option, axis):
	number_layers = np.size(strat_interp_lith_k_calcs_df['z0_m'].values)
	layer_index = 0
	for layer_index in range(number_layers):
		axis.axhspan(
			ymin = strat_interp_lith_k_calcs_df['z0_m'][layer_index],
			ymax = strat_interp_lith_k_calcs_df['z1_m'][layer_index],
			color = plot_lith_fill_dict[strat_interp_lith_k_calcs_df['geological_description'][layer_index]]['mplt_color'],
			hatch = plot_lith_fill_dict[strat_interp_lith_k_calcs_df['geological_description'][layer_index]]['mplt_hatch_key'],
			alpha = 0.5,
			label = plot_lith_fill_dict[strat_interp_lith_k_calcs_df['geological_description'][layer_index]]['mplt_label']
		)
		axis.axhline(
			y = strat_interp_lith_k_calcs_df['z0_m'][layer_index],
			color = 'black'
		)
		axis.axhline(
			y = strat_interp_lith_k_calcs_df['z1_m'][layer_index],
			color = 'black'
		)
		if strat_label_option == 'yes':
			# Plot layer label if it is non-nan
			layer_label_option = plot_lith_fill_dict[strat_interp_lith_k_calcs_df['geological_description'][layer_index]]['mplt_label'] != plot_lith_fill_dict[strat_interp_lith_k_calcs_df['geological_description'][layer_index]]['mplt_label']
			if layer_label_option == False:
				axis.text(
					0.5,
					(strat_interp_lith_k_calcs_df['z0_m'][layer_index] + strat_interp_lith_k_calcs_df['z1_m'][layer_index]) / 2,
					plot_lith_fill_dict[strat_interp_lith_k_calcs_df['geological_description'][layer_index]]['mplt_label'],
					verticalalignment='center',
					horizontalalignment='center',
					bbox=plot_format_dict['white_bbox_props']
				)
	return()

def plot_temperatures(tempdict, axis, plot_format_dict_local, plot_format_dict):
	if tempdict['line_type'] == 'errorbar':
		axis.errorbar(tempdict['T_plot'], tempdict['zT_m_plot'], xerr=tempdict['T_error_plot'], yerr=tempdict['zT_error_m_plot'], fmt=tempdict['fmt'], markeredgecolor=tempdict['markeredgecolor'], alpha=tempdict['alpha'])
	# if spl_temp_option_f == 'no':
	# 	axis.errorbar(T, zT_m, xerr=T_error, yerr=zT_error_m, fmt='.k', markeredgecolor='k')
	# elif spl_temp_option_f == 'yes':
	# 	axis.errorbar(T, zT_m, xerr=T_error, yerr=zT_error_m, fmt='.', color='black', markeredgecolor='black', alpha=0.25)
		# axis.errorbar(T_k_f, zk_m_f, yerr=zk_error_m_f, color='red', fmt='.')
	if tempdict['line_type'] == 'mc_results':
		# TODO Resample
		heatmap, xedges, yedges = np.histogram2d(tempdict['T_plot'][~np.isnan(tempdict['T_plot'])].ravel(), tempdict['zT_m_plot'][~np.isnan(tempdict['zT_m_plot'])].ravel(), bins=[200,100], density=True)
		heatmap = heatmap.T
		X, Y = np.meshgrid(xedges, yedges)
		axis.pcolormesh(X, Y, heatmap, cmap=tempdict['colors'])
	# Plot figure label
	if tempdict['figure_label'] != None:
		axis.text(plot_format_dict_local['figure_label_frac_inset_width'], plot_format_dict_local['figure_label_frac_inset_height'], tempdict['figure_label'], verticalalignment='bottom', horizontalalignment='left', bbox=plot_format_dict['figure_label_white_bbox_props'], transform=axis.transAxes, fontdict=plot_format_dict['figure_label_fontdict'])
	return()

def plot_conductivity(k_distribution, tempdict, axis, plot_format_dict_local, plot_format_dict):
	if tempdict['line_type'] == 'strat_interp_lith_k':
		z_int_plot = np.concatenate(np.column_stack((tempdict['z0_m'], tempdict['z1_m'])))
		if k_distribution == 'uniform':
			min_k_int_plot = np.concatenate(np.column_stack((tempdict['min_k'], tempdict['min_k']))).astype('float64')
			max_k_int_plot = np.concatenate(np.column_stack((tempdict['max_k'], tempdict['max_k']))).astype('float64')
			axis.fill_betweenx(z_int_plot, min_k_int_plot, max_k_int_plot, where=max_k_int_plot >= min_k_int_plot, facecolor=tempdict['color'], alpha=0.25*tempdict['alpha'])
			axis.plot(min_k_int_plot, z_int_plot, color=tempdict['color'], linestyle='--', alpha=tempdict['alpha'], zorder=tempdict['zorder'])
			axis.plot(max_k_int_plot, z_int_plot, color=tempdict['color'], linestyle='--', alpha=tempdict['alpha'], zorder=tempdict['zorder'])
		if k_distribution == 'normal' or k_distribution == 'in_situ_normal':
			mean_k_int_plot = np.concatenate(np.column_stack((tempdict['mean_k'], tempdict['mean_k']))).astype('float64')
			stdev_k_int_plot = np.concatenate(np.column_stack((tempdict['k_assigned_error'], tempdict['k_assigned_error']))).astype('float64')
			axis.fill_betweenx(z_int_plot, mean_k_int_plot - stdev_k_int_plot, mean_k_int_plot + stdev_k_int_plot, where=mean_k_int_plot+stdev_k_int_plot >= mean_k_int_plot-stdev_k_int_plot, facecolor=tempdict['color'], alpha=0.25*tempdict['alpha'])
			axis.plot(mean_k_int_plot, z_int_plot, color=tempdict['color'], alpha=tempdict['alpha'], zorder=tempdict['zorder'])
	elif tempdict['line_type'] == 'mc_input':
		axis.plot(tempdict['layer_k_input_plot'], tempdict['layer_zk_m_input_plot'], color=tempdict['color'], alpha=tempdict['alpha'], zorder=tempdict['zorder'])
	elif tempdict['line_type'] == 'mean_errorbar':
		axis.errorbar(tempdict['layer_k_plot'], tempdict['layer_zk_m_plot'], xerr=tempdict['layer_k_error_plot'], yerr=tempdict['layer_zk_error_m_plot'], fmt=tempdict['fmt'], alpha=tempdict['alpha'], zorder=tempdict['zorder'])
	elif tempdict['line_type'] == 'mc_results':
		# TODO Sample every 1 m to make smooth distribution of conductivity profile
		heatmap, xedges, yedges = np.histogram2d(tempdict['k_plot'][~np.isnan(tempdict['k_plot'])].ravel(), tempdict['zk_m_plot'][~np.isnan(tempdict['zk_m_plot'])].ravel(), bins=[200,100], density=True)
		heatmap = heatmap.T
		X, Y = np.meshgrid(xedges, yedges)
		axis.pcolormesh(X, Y, heatmap, cmap=tempdict['colors'])
	if tempdict['figure_label'] != None:
		axis.text(1-plot_format_dict_local['figure_label_frac_inset_width'], plot_format_dict_local['figure_label_frac_inset_height'], tempdict['figure_label'], verticalalignment='bottom', horizontalalignment='right', bbox=plot_format_dict['figure_label_white_bbox_props'], transform=axis.transAxes, fontdict=plot_format_dict['figure_label_fontdict'])
	return()

def plot_palaeoclimate_subplot(axis, plot_format_dict_local, tempdict, plot_format_dict):
	if tempdict['pc_plot_input_option'] != 'mc_results':
		temp_ago = tempdict['pc_deltaTs']
		temp_ago_ravel = np.column_stack([temp_ago, temp_ago]).ravel()
		temp_ago_low = tempdict['pc_deltaTs'] - tempdict['pc_sigma_deltaTs']
		temp_ago_low_ravel = np.column_stack([temp_ago_low, temp_ago_low]).ravel()
		temp_ago_high = tempdict['pc_deltaTs'] + tempdict['pc_sigma_deltaTs']
		temp_ago_high_ravel = np.column_stack([temp_ago_high, temp_ago_high]).ravel()
		time0_ago = general_python_functions.s2y(tempdict['pc_t0'])
		time1_ago = general_python_functions.s2y(tempdict['pc_t1'])
		time1_ago_low = general_python_functions.s2y(tempdict['pc_t1'] - tempdict['pc_sigma_t1'])
		time1_ago_high = general_python_functions.s2y(tempdict['pc_t1'] + tempdict['pc_sigma_t1'])
		time_ago_ravel = general_python_functions.s2y(np.column_stack([tempdict['pc_t0'], tempdict['pc_t1']]).ravel())
		time_ago_low_ravel = general_python_functions.s2y(np.column_stack([tempdict['pc_t0'] - tempdict['pc_sigma_t0'], tempdict['pc_t1'] - tempdict['pc_sigma_t1']]).ravel())
		time_ago_high_ravel = general_python_functions.s2y(np.column_stack([tempdict['pc_t0'] + tempdict['pc_sigma_t0'], tempdict['pc_t1'] + tempdict['pc_sigma_t1']]).ravel())
		temp_ago_mc = np.column_stack([tempdict['pc_deltaTs_input'], tempdict['pc_deltaTs_input']]).ravel()
		time_ago_mc = np.column_stack([tempdict['pc_t0_input'], tempdict['pc_t1_input']]).ravel()
		time_ago_mc = general_python_functions.s2y(time_ago_mc)

		# fig = plt.figure(figsize=(10,3.5))
		# axis = fig.add_subplot(111, xlim=(0,150))
		# axis.xaxis.set_label_position('top')
		### TODO Make sure no white spaces in infill
		axis.fill_between(time0_ago/1e3, temp_ago_low, temp_ago_high, color='lightgray', step='post')
		axis.fill_between(time1_ago/1e3, temp_ago_low, temp_ago_high, color='lightgray', step='pre')
		axis.fill_betweenx(temp_ago, time1_ago_low/1e3, time1_ago_high/1e3, where=np.full(np.size(temp_ago), True), interpolate=True, color='lightgray', step='post')
		axis.fill_betweenx(temp_ago_low, time1_ago_low/1e3, time1_ago_high/1e3, where=np.full(np.size(temp_ago), True), interpolate=True, color='lightgray', step='post')
		axis.fill_betweenx(temp_ago_high, time1_ago_low/1e3, time1_ago_high/1e3, where=np.full(np.size(temp_ago), True), interpolate=True, color='lightgray', step='post')
		# axis.fill_betweenx(temp_ago, time1_ago/1e3, time1_ago_high/1e3, color='blue', step='post', alpha=0.5)
		# axis.fill_between(time_ago_ravel/1e3, temp_ago_low_ravel, temp_ago_high_ravel, color='gray', alpha=0.2)
	# 											axis.fill_between(time_ago_high/1e3, temp_ago_low, temp_ago_high, color='gray', alpha=1)
	# 											axis.fill_between(time_ago_low/1e3, temp_ago_low, temp_ago_high, color='gray', alpha=1)
		# axis.fill_betweenx(temp_ago, time_ago_low/1e3, time_ago_high/1e3, color='gray', alpha=1)
		# axis.fill_betweenx(temp_ago_high, time_ago_low/1e3, time_ago_high/1e3, color='gray', alpha=1)
		axis.plot(time_ago_ravel/1e3, temp_ago_ravel, 'k-', label=r'$\Delta T_c$')
		if tempdict['pc_plot_input_option'] == 'yes':
			axis.plot(time_ago_mc/1e3, temp_ago_mc, tempdict['fmt'], label=r'$\Delta T_c$', alpha=1)
		else:
			axis.plot(time_ago_mc/1e3, temp_ago_mc, tempdict['fmt'], label=r'$\Delta T_c$', alpha=0)
	else:
		print(tempdict['pc_t'])
		print(type(tempdict['pc_t']))
		print(tempdict['pc_deltaTs'])
		print(type(tempdict['pc_deltaTs']))
		heatmap, xedges, yedges = np.histogram2d(tempdict['pc_t'][~np.isnan(tempdict['pc_deltaTs'])].ravel(), tempdict['pc_deltaTs'][~np.isnan(tempdict['pc_deltaTs'])].ravel(), bins=[200,100], density=True)
		heatmap = heatmap.T
		X, Y = np.meshgrid(xedges, yedges)
		axis.pcolormesh(X, Y, heatmap, cmap=tempdict['colors'])
	if tempdict['figure_label'] != None:
		axis.text(1 - plot_format_dict_local['figure_label_frac_inset_width'], plot_format_dict_local['figure_label_frac_inset_height'], tempdict['figure_label'], verticalalignment='bottom', horizontalalignment='right', bbox=plot_format_dict['figure_label_white_bbox_props'], transform=axis.transAxes, fontdict=plot_format_dict['figure_label_fontdict'])
	return()

def plot_recent_climate_subplot(axis, plot_format_dict_local, tempdict, plot_format_dict):

# rc_deltaTs, rc_t0, rc_t1, rc_deltaTs_cut, rc_t0_cut, rc_t1_cut, rc_deltaTs_smoothed_cut, rc_t0_smoothing_cutoff_cut, rc_t1_smoothing_cutoff_cut, rc_deltaTs_input, rc_t0_input, rc_t1_input, rc_ty, rc_plot_input_option
#
# recent_climtemp_deltaTs, recent_climtemp_t0_seconds, recent_climtemp_t1_seconds, recent_climtemp_deltaTs_cut, recent_climtemp_t0_seconds_cut, recent_climtemp_t1_seconds_cut, recent_climtemp_deltaTs_smoothed_cut, recent_climtemp_t0_seconds_smoothing_cutoff_cut, recent_climtemp_t1_seconds_smoothing_cutoff_cut, recent_climtemp_deltaTs_input, recent_climtemp_t0_seconds_input, recent_climtemp_t1_seconds_input, recent_climtemp_year, rc_plot_input_option, figure_name):
	
	recent_climtemp_deltaTs_ago = np.column_stack([tempdict['rc_deltaTs'], tempdict['rc_deltaTs']]).ravel()
	recent_climtemp_year_ago = general_python_functions.s2y(np.column_stack([tempdict['rc_t0'], tempdict['rc_t1']]).ravel())
	recent_climtemp_deltaTs_ago_cut = np.column_stack([tempdict['rc_deltaTs_cut'], tempdict['rc_deltaTs_cut']]).ravel()
	recent_climtemp_year_ago_cut = general_python_functions.s2y(np.column_stack([tempdict['rc_t0_cut'], tempdict['rc_t1_cut']]).ravel())
	recent_climtemp_deltaTs_ago_smoothed_cut = np.column_stack([tempdict['rc_deltaTs_smoothed_cut'], tempdict['rc_deltaTs_smoothed_cut']]).ravel()
	recent_climtemp_year_ago_smoothing_cutoff_cut = general_python_functions.s2y(np.column_stack([tempdict['rc_t0_smoothing_cutoff_cut'], tempdict['rc_t1_smoothing_cutoff_cut']]).ravel())
	recent_climtemp_deltaTs_ago_input = np.column_stack([tempdict['rc_deltaTs_input'], tempdict['rc_deltaTs_input']]).ravel()
	recent_climtemp_year_ago_input = general_python_functions.s2y(np.column_stack([tempdict['rc_t0_input'], tempdict['rc_t1_input']]).ravel())
	
	axis.plot(recent_climtemp_year_ago, recent_climtemp_deltaTs_ago, color='black', alpha=0.25)
	if tempdict['rc_plot_input_option'] == 'smoothed':
		axis.plot(recent_climtemp_year_ago_smoothing_cutoff_cut, recent_climtemp_deltaTs_ago_smoothed_cut, color='black', alpha=1)
	elif tempdict['rc_plot_input_option'] == 'mc':
		axis.plot(recent_climtemp_year_ago_smoothing_cutoff_cut, recent_climtemp_deltaTs_ago_smoothed_cut, color='black', alpha=1)
		axis.plot(recent_climtemp_year_ago_input, recent_climtemp_deltaTs_ago_input, color=tempdict['color'])
	else:
		axis.plot(recent_climtemp_year_ago_cut, recent_climtemp_deltaTs_ago_cut, color='black')
	
	# axisyear = axis.twiny()
	# axisyear.set_xlabel('Year')
	# axisyear.invert_xaxis()
	# axisyear.plot(tempdict['rc_ty'], tempdict['rc_deltaTs'], alpha=0)
	if tempdict['figure_label'] != None:
		axis.text(plot_format_dict_local['figure_label_frac_inset_width'], plot_format_dict_local['figure_label_frac_inset_height'], tempdict['figure_label'], verticalalignment='bottom', horizontalalignment='left', bbox=plot_format_dict['figure_label_white_bbox_props'], transform=axis.transAxes, fontdict=plot_format_dict['figure_label_fontdict'])
	return()

def plot_resistivity(tempdict, axis, plot_format_dict_local, plot_format_dict):
	if tempdict['line_type'] == 'errorbar':
		axis.errorbar(tempdict['R_plot'], tempdict['zR_m_plot'], xerr=tempdict['R_error_plot'], yerr=tempdict['zR_error_m_plot'], fmt=tempdict['fmt'], markeredgecolor=tempdict['markeredgecolor'], alpha=tempdict['alpha'])
	# if spl_temp_option_f == 'no':
	# 	axis.errorbar(T, zT_m, xerr=T_error, yerr=zT_error_m, fmt='.k', markeredgecolor='k')
	# elif spl_temp_option_f == 'yes':
	# 	axis.errorbar(T, zT_m, xerr=T_error, yerr=zT_error_m, fmt='.', color='black', markeredgecolor='black', alpha=0.25)
		# axis.errorbar(T_k_f, zk_m_f, yerr=zk_error_m_f, color='red', fmt='.')
	elif tempdict['line_type'] == 'mc_results':
		# TODO Sample every 1 m to make smooth distribution of conductivity profile
		heatmap, xedges, yedges = np.histogram2d(tempdict['R_plot'][~np.isnan(tempdict['R_plot'])].ravel(), tempdict['zR_m_plot'][~np.isnan(tempdict['zR_m_plot'])].ravel(), bins=[200,100], density=True)
		heatmap = heatmap.T
		X, Y = np.meshgrid(xedges, yedges)
		axis.pcolormesh(X, Y, heatmap, cmap=tempdict['colors'])
	if tempdict['figure_label'] != None:
		axis.text(plot_format_dict_local['figure_label_frac_inset_width'], plot_format_dict_local['figure_label_frac_inset_height'], tempdict['figure_label'], verticalalignment='bottom', horizontalalignment='left', bbox=plot_format_dict['figure_label_white_bbox_props'], transform=axis.transAxes, fontdict=plot_format_dict['figure_label_fontdict'])
	return()

def plot_bullard(tempdict, axis, plot_format_dict_local, plot_format_dict):
	legend_option = 'no'
	if tempdict['line_type'] == 'emptyframe':
		axis.plot(tempdict['empty_x'], tempdict['empty_y'], alpha=0)
	if tempdict['line_type'] == 'TvR_errorbar':
		axis.errorbar(tempdict['R_plot'], tempdict['T_plot'], xerr=tempdict['R_error_plot'], yerr=tempdict['T_error_plot'], fmt=tempdict['fmt'], markeredgecolor=tempdict['markeredgecolor'], alpha=tempdict['alpha'])
	if tempdict['line_type'] == 'RvT_errorbar':
		axis.errorbar(tempdict['T_plot'], tempdict['R_plot'], xerr=tempdict['T_error_plot'], yerr=tempdict['R_error_plot'], fmt=tempdict['fmt'], markeredgecolor=tempdict['markeredgecolor'], alpha=tempdict['alpha'])
	if tempdict['line_type'] == 'fitted_line_x':
		axis.plot(tempdict['x_plot'], tempdict['gradient']*tempdict['x_plot'] + tempdict['intercept'], color=tempdict['color'], linestyle=tempdict['linestyle'], alpha=tempdict['alpha'], label=tempdict['legend_label'])
		legend_option = 'yes'
		# axis.text(
		# 	0.5,
		# 	0.5,
		# 	'Testing',
		# 	verticalalignment='center',
		# 	horizontalalignment='center',
		# 	bbox=plot_format_dict['white_bbox_props'],
		# 	transform=axis.transAxes
		# )
	if tempdict['line_type'] == 'fitted_line_y':
		axis.plot(tempdict['gradient']*tempdict['y_plot'] + tempdict['intercept'], tempdict['y_plot'], color=tempdict['color'], linestyle=tempdict['linestyle'], alpha=tempdict['alpha'], label=tempdict['legend_label'])
		legend_option = 'yes'
		# axis.text(
		# 	0.5,
		# 	0.5,
		# 	'Testing',
		# 	verticalalignment='center',
		# 	horizontalalignment='center',
		# 	bbox=plot_format_dict['white_bbox_props'],
		# 	transform=axis.transAxes
		# )
	elif tempdict['line_type'] == 'mc_results':
		# TODO Sample every 1 m to make smooth distribution of conductivity profile
		heatmap, xedges, yedges = np.histogram2d(tempdict['R_plot'][~np.isnan(tempdict['R_plot'])].ravel(), tempdict['T_plot'][~np.isnan(tempdict['T_plot'])].ravel(), bins=[200,100], density=True)
		heatmap = heatmap.T
		X, Y = np.meshgrid(xedges, yedges)
		axis.pcolormesh(X, Y, heatmap, cmap=tempdict['colors'])
	if legend_option == 'yes':
		axis.legend(loc='lower right', fancybox=False, framealpha=1, edgecolor='black', fontsize=8, borderaxespad=0.75)
		# axis.legend().get_frame().set_linewidth(plot_format_dict['axes_linewidth'])
	if tempdict['figure_label'] != None:
		axis.text(plot_format_dict_local['figure_label_frac_inset_width'], plot_format_dict_local['figure_label_frac_inset_height'], tempdict['figure_label'], verticalalignment='bottom', horizontalalignment='left', bbox=plot_format_dict['figure_label_white_bbox_props'], transform=axis.transAxes, fontdict=plot_format_dict['figure_label_fontdict'])
	# if spl_temp_option_f == 'no':
	# 	axis.errorbar(T, zT_m, xerr=T_error, yerr=zT_error_m, fmt='.k', markeredgecolor='k')
	# elif spl_temp_option_f == 'yes':
	# 	axis.errorbar(T, zT_m, xerr=T_error, yerr=zT_error_m, fmt='.', color='black', markeredgecolor='black', alpha=0.25)
		# axis.errorbar(T_k_f, zk_m_f, yerr=zk_error_m_f, color='red', fmt='.')
	return()

def plot_heat_flow_histograms(tempdict, axis, plot_format_dict_local, plot_format_dict):
	
	# qhist_plotting_dict = {'number_hists':1, 'hist0':{'hist_type':'unweighted_histogram', 'set':whole_bullard_q_downward_TvR_mc_all, 'sigma_set':whole_bullard_sigma_q_downward_TvR_mc_all, 'color':'red', 'alpha':1, 'number_bins':50, 'figure_label':'f'}}
	
	if 'legend_label' in tempdict:
		legend_option = 'yes'
	else:
		legend_option = 'no'
		tempdict['legend_label'] = 'na'
	
	if tempdict['hist_type'] == 'unweighted_histogram':
		probdens, edges = np.histogram(tempdict['set'] * 1e3, bins=tempdict['number_bins'], density=True)
		axis.bar(edges[:-1], probdens, width=np.diff(edges), align="edge", color=tempdict['color'], alpha=tempdict['alpha'], label=tempdict['legend_label'])
	
	if legend_option == 'yes':
		axis.legend(loc='upper right', fancybox=False, framealpha=1, edgecolor='black', fontsize=8, borderaxespad=0.75)
	
	if tempdict['figure_label'] != None:
		axis.text(plot_format_dict_local['figure_label_frac_inset_width'], plot_format_dict_local['figure_label_frac_inset_height'], tempdict['figure_label'], verticalalignment='bottom', horizontalalignment='left', bbox=plot_format_dict['figure_label_white_bbox_props'], transform=axis.transAxes, fontdict=plot_format_dict['figure_label_fontdict'])


def plot_temperature_stratigraphy_conductivity(max_depth_m_plot, k_distribution, plot_lith_fill_dict, plot_lith_fill_dict_keyword, strat_interp_lith_k_calcs_df, T_plotting_dict, k_plotting_dict, res_plotting_dict, bullard_TvR_plotting_dict, bullard_RvT_plotting_dict, pc_plotting_dict, rc_plotting_dict, qhist_plotting_dict, T, figure_name):
	
	# Set up figure limits
	min_depth_m=0
	max_depth_m=max_depth_m_plot
	
	min_T=np.min(T) - 0.05 * np.ptp(T)
	max_T=np.max(T) + 0.05 * np.ptp(T)
	
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
	

	
	
	
	z_int_plot = np.concatenate(np.column_stack((strat_interp_lith_k_calcs_df['z0_m'], strat_interp_lith_k_calcs_df['z1_m'])))
	
	# Set up plotting preferences
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
	plot_format_dict = set_up_plot_formatting_constant_height(axis_setup_dict, axis_position_dict)
	
	# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
	plt.rcParams['xtick.bottom'] = False
	plt.rcParams['xtick.labelbottom'] = False
	plt.rcParams['xtick.top'] = True
	plt.rcParams['xtick.labeltop'] = True
	
	fig = plt.figure(figsize=(plot_format_dict['figure_width'], plot_format_dict['figure_height']), dpi=100)
	
	# Plot temperature
	plot_format_dict_local = plot_format_dict['fractional_pos_dict']['axis_temp']
	ax00 = fig.add_axes([plot_format_dict_local['frac_xpos'], plot_format_dict_local['frac_ypos'], plot_format_dict_local['panel_frac_width'], plot_format_dict_local['panel_frac_height']], xlabel=r'$T$ / $^{\circ}$C', ylabel=r'$z$ / m')
	ax00.set_ylim(top=0, bottom=max_depth_m_plot)
	ax00.set_xlim(left=min_T, right=max_T)
	ax00.yaxis.set_ticks_position('both')
	ax00.xaxis.set_ticks_position('both')
	ax00.xaxis.set_minor_locator(MultipleLocator(2.5))
	ax00.xaxis.set_major_locator(MultipleLocator(5))
	ax00.yaxis.set_minor_locator(MultipleLocator(25))
	ax00.yaxis.set_major_locator(MultipleLocator(50))
	# ax00.yaxis.set_major_formatter(FormatStrFormatter(''))
	ax00.xaxis.set_label_position('top') 
	spl_temp_option_f = 'no'
	# plot_temperatures(max_depth_m_plot, zT_m, T, zT_error_m, T_error, ax00)
	if T_plotting_dict['number_lines'] > 0:
		for line_number in range(T_plotting_dict['number_lines']):
			tempdict = T_plotting_dict['line' + str(line_number)]
			plot_temperatures(tempdict, ax00, plot_format_dict_local, plot_format_dict)
	
	# Plot stratigraphy
	if k_distribution != 'in_situ_normal':
		### TODO temp
		strat_label_option = 'no'
		plot_format_dict_local = plot_format_dict['fractional_pos_dict']['axis_strat']
		ax01 = fig.add_axes([plot_format_dict_local['frac_xpos'], plot_format_dict_local['frac_ypos'], plot_format_dict_local['panel_frac_width'], plot_format_dict_local['panel_frac_height']], ylabel=None)
		ax01.set_ylim(top=0, bottom=max_depth_m_plot)
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
	plot_format_dict_local = plot_format_dict['fractional_pos_dict']['axis_cond']
	ax02 = fig.add_axes([plot_format_dict_local['frac_xpos'], plot_format_dict_local['frac_ypos'], plot_format_dict_local['panel_frac_width'], plot_format_dict_local['panel_frac_height']], xlabel=r'$k$ / W m$^{-1}$ K$^{-1}$', ylabel=None)
	ax02.set_ylim(top=0, bottom=max_depth_m_plot)
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
	if res_plotting_dict != None:
		plot_format_dict_local = plot_format_dict['fractional_pos_dict']['axis_res']
		ax03 = fig.add_axes([plot_format_dict_local['frac_xpos'], plot_format_dict_local['frac_ypos'], plot_format_dict_local['panel_frac_width'], plot_format_dict_local['panel_frac_height']], xlabel=r'$R$ / W$^{-1}$ K m$^2$', ylabel=r'$z$ / m')
		ax03.set_ylim(top=0, bottom=max_depth_m_plot)
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
	if bullard_TvR_plotting_dict != None:
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
	if bullard_RvT_plotting_dict != None:
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
	if qhist_plotting_dict != None:
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
	
	fig.savefig(figure_name + ".jpg", dpi=300, bbox_inches='tight', transparent=True)
	plt.close(fig)
	
	return()

















#### UNUSED below this line



def plot_lithological_conductivity_temperature(max_depth_m_plot, plot_lith_fill_dict, plot_lith_fill_dict_keyword, lith_df, k_distribution, temps_df, zk_m_f, k_f, zk_error_m_f, k_error_f, T_k_f, in_situ_option_f, spl_temp_option_f, strat_cond_option_f, heat_flow_option_f, heat_flow_file_f, figure_name):
	number_layers = np.size(lith_df['z0_m'].values)
	
	# print(heat_flow_option_f)
	
	# Set up figure limits
	min_depth_m=0
	max_depth_m=max_depth_m_plot
	min_k=0
	max_k=8
	min_Q = 25
	max_Q = 90
	
	min_T=np.min(temps_df['T']) - 0.05 * np.ptp(temps_df['T'])
	max_T=np.max(temps_df['T']) + 0.05 * np.ptp(temps_df['T'])
	
	number_rows = 1
	number_columns = 3
	plot_strat_option = 'yes'
	if plot_strat_option == 'yes':
		number_columns = number_columns + 1
	plot_format_dict = set_up_plot_formatting(plot_strat_option, number_rows, number_columns)
	
	# Set up plotting preferences
	# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
	plt.rcParams['xtick.bottom'] = False
	plt.rcParams['xtick.labelbottom'] = False
	plt.rcParams['xtick.top'] = True
	plt.rcParams['xtick.labeltop'] = True
	
	fig = plt.figure(figsize=(plot_format_dict['figure_width'], plot_format_dict['figure_height']), dpi=100)
	
	ax00 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column0'], plot_format_dict['row_fractional_ypos_dict']['row0'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], ylabel=r'$z$ / m')
	ax00.set_ylim(top=0, bottom=max_depth_m_plot)
	ax00.set_xlim(left=0, right=1)
	ax00.yaxis.set_ticks_position('both')
	ax00.xaxis.set_ticks_position('none')
	ax00.xaxis.set_minor_locator(MultipleLocator(5))
	ax00.yaxis.set_minor_locator(MultipleLocator(50))
	# ax00.yaxis.set_major_formatter(FormatStrFormatter(''))
	ax00.xaxis.set_major_formatter(FormatStrFormatter(''))
	
	ax10 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column1'], plot_format_dict['row_fractional_ypos_dict']['row0'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$k$ / W m$^{-1}$ K$^{-1}$', ylabel=None)
	ax10.set_ylim(top=0, bottom=max_depth_m_plot)
	ax10.set_xlim(left=min_k, right=max_k)
	ax10.yaxis.set_ticks_position('both')
	ax10.xaxis.set_ticks_position('both')
	ax10.xaxis.set_minor_locator(MultipleLocator(5))
	ax10.yaxis.set_major_locator(MultipleLocator(50))
	ax10.yaxis.set_major_formatter(FormatStrFormatter(''))
	ax10.xaxis.set_label_position('top') 
	# ax10.xaxis.set_major_formatter(FormatStrFormatter(''))
	
	ax20 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column2'], plot_format_dict['row_fractional_ypos_dict']['row0'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$T$ / $^{\circ}$C', ylabel=None)
	ax20.set_ylim(top=0, bottom=max_depth_m_plot)
	ax20.set_xlim(left=min_T, right=max_T)
	ax20.yaxis.set_ticks_position('both')
	ax20.xaxis.set_ticks_position('both')
	ax20.xaxis.set_minor_locator(MultipleLocator(5))
	ax20.yaxis.set_major_locator(MultipleLocator(50))
	ax20.yaxis.set_major_formatter(FormatStrFormatter(''))
	ax20.xaxis.set_label_position('top') 
	# ax20.xaxis.set_major_formatter(FormatStrFormatter(''))
	
	ax30 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column3'], plot_format_dict['row_fractional_ypos_dict']['row0'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$Q$ / mW m$^{-2}$', ylabel=None)
	ax30.set_ylim(top=0, bottom=max_depth_m_plot)
	ax30.set_xlim(left=min_Q, right=max_Q)
	ax30.yaxis.set_ticks_position('both')
	ax30.xaxis.set_ticks_position('both')
	ax30.xaxis.set_minor_locator(MultipleLocator(5))
	ax30.yaxis.set_major_locator(MultipleLocator(50))
	ax30.yaxis.set_major_formatter(FormatStrFormatter(''))
	ax30.xaxis.set_label_position('top') 
	# ax20.xaxis.set_major_formatter(FormatStrFormatter(''))
	
	
	# TODO Set up one function that plots all data and is passed axis position, figure labels, plot options
	
	z_int_plot = np.concatenate(np.column_stack((lith_df['z0_m'], lith_df['z1_m'])))
	
	if strat_cond_option_f == 'yes':
		if k_distribution != 'in_situ_normal':
			for layer_index in range(number_layers):
				ax00.axhspan(
					ymin = lith_df['z0_m'][layer_index],
					ymax = lith_df['z1_m'][layer_index],
					color = plot_lith_fill_dict[lith_df['geological_description'][layer_index]]['mplt_color'],
					hatch = plot_lith_fill_dict[lith_df['geological_description'][layer_index]]['mplt_hatch_key'],
					alpha = 0.5,
					label = plot_lith_fill_dict[lith_df['geological_description'][layer_index]]['mplt_label']
				)
				ax00.axhline(
					linewidth=2,
					y = lith_df['z0_m'][layer_index],
					color = 'black'
				)
				ax00.axhline(
					linewidth=2,
					y = lith_df['z1_m'][layer_index],
					color = 'black'
				)
				# Plot layer label if it is non-nan
				layer_label_option = plot_lith_fill_dict[lith_df['geological_description'][layer_index]]['mplt_label'] != plot_lith_fill_dict[lith_df['geological_description'][layer_index]]['mplt_label']
				if layer_label_option == False:
					ax00.text(
						0.5,
						(lith_df['z0_m'][layer_index] + lith_df['z1_m'][layer_index]) / 2,
						plot_lith_fill_dict[lith_df['geological_description'][layer_index]]['mplt_label'],
						verticalalignment='center',
						horizontalalignment='center',
						bbox=plot_format_dict['white_bbox_props']
					)
		if k_distribution == 'uniform':
			min_k_int_plot = np.concatenate(np.column_stack((lith_df['min_k'], lith_df['min_k']))).astype('float64')
			max_k_int_plot = np.concatenate(np.column_stack((lith_df['max_k'], lith_df['max_k']))).astype('float64')
			ax10.fill_betweenx(z_int_plot, min_k_int_plot, max_k_int_plot, where=max_k_int_plot >= min_k_int_plot, facecolor='red', alpha=0.25)
			ax10.plot(min_k_int_plot, z_int_plot, color='red', linestyle='--', alpha=1, zorder=5)
			ax10.plot(max_k_int_plot, z_int_plot, color='red', linestyle='--', alpha=1, zorder=5)
		if k_distribution == 'normal' or k_distribution == 'in_situ_normal':
			mean_k_int_plot = np.concatenate(np.column_stack((lith_df['mean_k'], lith_df['mean_k']))).astype('float64')
			stdev_k_int_plot = np.concatenate(np.column_stack((lith_df['k_assigned_error'], lith_df['k_assigned_error']))).astype('float64')
			ax10.fill_betweenx(z_int_plot, mean_k_int_plot - stdev_k_int_plot, mean_k_int_plot + stdev_k_int_plot, where=mean_k_int_plot+stdev_k_int_plot >= mean_k_int_plot-stdev_k_int_plot, facecolor='red', alpha=0.25)
			ax10.plot(mean_k_int_plot, z_int_plot, color='red', alpha=1, zorder=5)
	
	if in_situ_option_f == 'overlay':
		### Plot in situ conductivities at measured locations
		ax10.errorbar(k_f, zk_m_f, xerr=k_error_f, yerr=zk_error_m_f, fmt='.', color='black', alpha=0.4, zorder=6)
	elif in_situ_option_f == 'yes':
		### Plot in situ conductivities at measured locations
		ax10.errorbar(k_f, zk_m_f, xerr=k_error_f, yerr=zk_error_m_f, fmt='.', color='black', zorder=6)
		
		
	### Plot temperatures
	if spl_temp_option_f == 'no':
		ax20.errorbar(temps_df['T'], temps_df['z_m'], xerr=temps_df['T_assigned_error'], yerr=temps_df['z_assigned_error_m'], fmt='.k', markeredgecolor='k')
	elif spl_temp_option_f == 'yes':
		ax20.errorbar(temps_df['T'], temps_df['z_m'], xerr=temps_df['T_assigned_error'], yerr=temps_df['z_assigned_error_m'], fmt='.', color='black', markeredgecolor='black', alpha=0.25)
		ax20.errorbar(T_k_f, zk_m_f, yerr=zk_error_m_f, color='red', fmt='.')
	# if spl_temp_datapoints_option_f == 'yes':
	# 	ax20.errorbar(T_k_f, zk_m_f, yerr=zk_error_m_f, color='red', alpha=0.5)
	
	
	### Plot estimated heat flow
	if heat_flow_option_f == 'bullard_whole':
		
		# Load commented lines with description and values of heat flow as dataframe
		heat_flow_df = pd.read_csv(heat_flow_file_f + '.csv', nrows=1, delimiter=', ', engine='python')
		
		# TODO Pass this from outside
		Q_plot = np.append(heat_flow_df['Q_round'].values, heat_flow_df['Q_round'].values)
		z_Q_plot = np.append(heat_flow_df['z_top'], heat_flow_df['z_base'])
		sigma_Q_plot = np.append(heat_flow_df['sigma_Q_round'], heat_flow_df['sigma_Q_round'])
		
		ax30.plot(Q_plot, z_Q_plot, color='red')
		ax30.fill_betweenx(z_Q_plot, Q_plot - sigma_Q_plot, Q_plot + sigma_Q_plot, facecolor='red', alpha=0.25)


		#  borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='red')
		# ax30.plot(borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['T_versus_R']['Q_round_plot']-borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['T_versus_R']['sigma_Q_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='red', linestyle='dashed')
		# ax30.plot(borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['T_versus_R']['Q_round_plot']+borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['T_versus_R']['sigma_Q_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='red', linestyle='dashed')
		#
		# print(heat_flow_df)
		#
		# print(heat_flow_df['Q_round'], heat_flow_df['sigma_Q_round'])
		
		
		
		# R_T_df = pd.read_csv(heat_flow_file_f + '.csv', comment="#")
		
		# pprint.pprint(R_T_df)
		# exit()
		
	
	# print(figure_name)
	
	fig.savefig(figure_name + ".jpg", bbox_inches='tight', transparent=True)
	plt.close(fig)
	
	return()


def plot_interval_method(borehole_analysis_dict, int_option, lith_plot_format_dict, max_depth_m_plot, plot_label_dict, figures_path):
		
	heat_flow_estimation_method = 'interval_method'
	figure_name, plot_title = set_up_plot_title(borehole_analysis_dict, int_option, heat_flow_estimation_method, plot_label_dict, figures_path)
	
	number_layers = borehole_analysis_dict[int_option]['layers_overview']['number_layers']
	plot_strat_option = borehole_analysis_dict[int_option]['layers_overview']['plot_strat_option']
	
	# for layer_index in range(number_layers):
	# 	x = 'layer'+str(layer_index)
	# 	print(x)
	# 	pprint.pprint(print(borehole_analysis_dict[int_option]['individual_layers'][x].keys())) #['interval_method'].keys()))
	# 	# exit()
	# #
	# exit()
	
	# TODO Still need to set this up
	number_rows = 4
	number_columns = 3
	if plot_strat_option == 'yes':
		number_columns = number_columns + 1
	plot_format_dict = set_up_plot_formatting(borehole_analysis_dict[int_option]['layers_overview']['plot_strat_option'], number_rows, number_columns)
	
	# Set up plotting preferences
	plt.rcParams['xtick.bottom'] = False
	plt.rcParams['xtick.labelbottom'] = False
	plt.rcParams['xtick.top'] = True
	plt.rcParams['xtick.labeltop'] = True
	
	# Set up figure limits
	min_depth_m=0
	max_depth_m=600
	min_k=1
	max_k=5.75
	
	# Set up axes
	fig = plt.figure(figsize=(plot_format_dict['figure_width'], plot_format_dict['figure_height']), dpi=100)
	fig.suptitle(plot_title)
	
	ax30 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column0'], plot_format_dict['row_fractional_ypos_dict']['row3'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$k$ / W m$^{-1}$ K$^{-1}$', ylabel=r'$z$ / m')
	ax30.set_ylim(top=min_depth_m, bottom=max_depth_m)
	# ax30.set_xlim(left=min_k, right=max_k)
	ax30.xaxis.set_label_position('top')
	ax30.yaxis.set_ticks_position('both')
	ax30.xaxis.set_ticks_position('both')
	ax30.xaxis.set_minor_locator(MultipleLocator(0.5))
	ax30.yaxis.set_minor_locator(MultipleLocator(50))
	
	ax31 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column1'], plot_format_dict['row_fractional_ypos_dict']['row3'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$T$ / $^{\circ}$C')
	ax31.set_ylim(top=min_depth_m, bottom=max_depth_m)
	ax31.xaxis.set_label_position('top')
	ax31.yaxis.set_ticks_position('both')
	ax31.xaxis.set_ticks_position('both')
	ax31.xaxis.set_minor_locator(MultipleLocator(1))
	ax31.yaxis.set_minor_locator(MultipleLocator(50))
	ax31.yaxis.set_major_formatter(FormatStrFormatter(''))
	
	ax32 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column2'], plot_format_dict['row_fractional_ypos_dict']['row3'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$Q$ / mW m$^{-2}$')
	ax32.set_ylim(top=min_depth_m, bottom=max_depth_m)
	ax32.xaxis.set_label_position('top')
	ax32.yaxis.set_ticks_position('both')
	ax32.xaxis.set_ticks_position('both')
	ax32.xaxis.set_minor_locator(MultipleLocator(1))
	ax32.yaxis.set_minor_locator(MultipleLocator(50))
	ax32.yaxis.set_major_formatter(FormatStrFormatter(''))
	
	if plot_strat_option == 'yes':
		ax33 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column3'], plot_format_dict['row_fractional_ypos_dict']['row3'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']])
		ax33.set_ylim(top=0, bottom=max_depth_m_plot)
		ax33.set_xlim(left=0, right=1)
		ax33.yaxis.set_ticks_position('both')
		ax33.xaxis.set_ticks_position('none')
		ax33.xaxis.set_minor_locator(MultipleLocator(5))
		ax33.yaxis.set_minor_locator(MultipleLocator(50))
		ax33.yaxis.set_major_formatter(FormatStrFormatter(''))
		ax33.xaxis.set_major_formatter(FormatStrFormatter(''))
	
		
	ax20 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column0'], plot_format_dict['row_fractional_ypos_dict']['row2'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$k$ / W m$^{-1}$ K$^{-1}$', ylabel=r'$z$ / m')
	ax20.set_ylim(top=min_depth_m, bottom=max_depth_m)
	# ax20.set_xlim(left=min_k, right=max_k)
	ax20.xaxis.set_label_position('top')
	ax20.yaxis.set_ticks_position('both')
	ax20.xaxis.set_ticks_position('both')
	ax20.xaxis.set_minor_locator(MultipleLocator(0.5))
	ax20.yaxis.set_minor_locator(MultipleLocator(50))
	
	ax21 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column1'], plot_format_dict['row_fractional_ypos_dict']['row2'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$T$ / $^{\circ}$C')
	ax21.set_ylim(top=min_depth_m, bottom=max_depth_m)
	ax21.xaxis.set_label_position('top')
	ax21.yaxis.set_ticks_position('both')
	ax21.xaxis.set_ticks_position('both')
	ax21.xaxis.set_minor_locator(MultipleLocator(1))
	ax21.yaxis.set_minor_locator(MultipleLocator(50))
	ax21.yaxis.set_major_formatter(FormatStrFormatter(''))
	
	ax22 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column2'], plot_format_dict['row_fractional_ypos_dict']['row2'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$Q$ / mW m$^{-2}$')
	ax22.set_ylim(top=min_depth_m, bottom=max_depth_m)
	ax22.xaxis.set_label_position('top')
	ax22.yaxis.set_ticks_position('both')
	ax22.xaxis.set_ticks_position('both')
	ax22.xaxis.set_minor_locator(MultipleLocator(1))
	ax22.yaxis.set_minor_locator(MultipleLocator(50))
	ax22.yaxis.set_major_formatter(FormatStrFormatter(''))
	
	if plot_strat_option == 'yes':
		ax23 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column3'], plot_format_dict['row_fractional_ypos_dict']['row2'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']])
		ax23.set_ylim(top=0, bottom=max_depth_m_plot)
		ax23.set_xlim(left=0, right=1)
		ax23.yaxis.set_ticks_position('both')
		ax23.xaxis.set_ticks_position('none')
		ax23.xaxis.set_minor_locator(MultipleLocator(5))
		ax23.yaxis.set_minor_locator(MultipleLocator(50))
		ax23.yaxis.set_major_formatter(FormatStrFormatter(''))
		ax23.xaxis.set_major_formatter(FormatStrFormatter(''))
	
	ax10 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column0'], plot_format_dict['row_fractional_ypos_dict']['row1'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$k$ / W m$^{-1}$ K$^{-1}$', ylabel=r'$z$ / m')
	ax10.set_ylim(top=min_depth_m, bottom=max_depth_m)
	# ax10.set_xlim(left=min_k, right=max_k)
	ax10.xaxis.set_label_position('top')
	ax10.yaxis.set_ticks_position('both')
	ax10.xaxis.set_ticks_position('both')
	ax10.xaxis.set_minor_locator(MultipleLocator(0.5))
	ax10.yaxis.set_minor_locator(MultipleLocator(50))
	
	ax11 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column1'], plot_format_dict['row_fractional_ypos_dict']['row1'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$T$ / $^{\circ}$C')
	ax11.set_ylim(top=min_depth_m, bottom=max_depth_m)
	ax11.xaxis.set_label_position('top')
	ax11.yaxis.set_ticks_position('both')
	ax11.xaxis.set_ticks_position('both')
	ax11.xaxis.set_minor_locator(MultipleLocator(1))
	ax11.yaxis.set_minor_locator(MultipleLocator(50))
	ax11.yaxis.set_major_formatter(FormatStrFormatter(''))
	
	ax12 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column2'], plot_format_dict['row_fractional_ypos_dict']['row1'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$Q$ / mW m$^{-2}$')
	ax12.set_ylim(top=min_depth_m, bottom=max_depth_m)
	ax12.xaxis.set_label_position('top')
	ax12.yaxis.set_ticks_position('both')
	ax12.xaxis.set_ticks_position('both')
	ax12.xaxis.set_minor_locator(MultipleLocator(1))
	ax12.yaxis.set_minor_locator(MultipleLocator(50))
	ax12.yaxis.set_major_formatter(FormatStrFormatter(''))
	
	if plot_strat_option == 'yes':
		ax13 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column3'], plot_format_dict['row_fractional_ypos_dict']['row1'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']])
		ax13.set_ylim(top=0, bottom=max_depth_m_plot)
		ax13.set_xlim(left=0, right=1)
		ax13.yaxis.set_ticks_position('both')
		ax13.xaxis.set_ticks_position('none')
		ax13.xaxis.set_minor_locator(MultipleLocator(5))
		ax13.yaxis.set_minor_locator(MultipleLocator(50))
		ax13.yaxis.set_major_formatter(FormatStrFormatter(''))
		ax13.xaxis.set_major_formatter(FormatStrFormatter(''))
		
	ax00 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column0'], plot_format_dict['row_fractional_ypos_dict']['row0'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$k$ / W m$^{-1}$ K$^{-1}$', ylabel=r'$z$ / m')
	ax00.set_ylim(top=min_depth_m, bottom=max_depth_m)
	# ax00.set_xlim(left=min_k, right=max_k)
	ax00.xaxis.set_label_position('top')
	ax00.yaxis.set_ticks_position('both')
	ax00.xaxis.set_ticks_position('both')
	ax00.xaxis.set_minor_locator(MultipleLocator(0.5))
	ax00.yaxis.set_minor_locator(MultipleLocator(50))
	
	ax01 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column1'], plot_format_dict['row_fractional_ypos_dict']['row0'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$T$ / $^{\circ}$C')
	ax01.set_ylim(top=min_depth_m, bottom=max_depth_m)
	ax01.xaxis.set_label_position('top')
	ax01.yaxis.set_ticks_position('both')
	ax01.xaxis.set_ticks_position('both')
	ax01.xaxis.set_minor_locator(MultipleLocator(1))
	ax01.yaxis.set_minor_locator(MultipleLocator(50))
	ax01.yaxis.set_major_formatter(FormatStrFormatter(''))
	
	ax02 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column2'], plot_format_dict['row_fractional_ypos_dict']['row0'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$Q$ / mW m$^{-2}$')
	ax02.set_ylim(top=min_depth_m, bottom=max_depth_m)
	ax02.xaxis.set_label_position('top')
	ax02.yaxis.set_ticks_position('both')
	ax02.xaxis.set_ticks_position('both')
	ax02.xaxis.set_minor_locator(MultipleLocator(1))
	ax02.yaxis.set_minor_locator(MultipleLocator(50))
	ax02.yaxis.set_major_formatter(FormatStrFormatter(''))
	
	if plot_strat_option == 'yes':
		ax03 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column3'], plot_format_dict['row_fractional_ypos_dict']['row0'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']])
		ax03.set_ylim(top=0, bottom=max_depth_m_plot)
		ax03.set_xlim(left=0, right=1)
		ax03.yaxis.set_ticks_position('both')
		ax03.xaxis.set_ticks_position('none')
		ax03.xaxis.set_minor_locator(MultipleLocator(5))
		ax03.yaxis.set_minor_locator(MultipleLocator(50))
		ax03.yaxis.set_major_formatter(FormatStrFormatter(''))
		ax03.xaxis.set_major_formatter(FormatStrFormatter(''))
	
	# Plot data
	### Plot all estimates overlying each other ###
	ax30.plot(borehole_analysis_dict[int_option]['layers_overview']['mean_k_int_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='red')
	ax30.plot(borehole_analysis_dict[int_option]['layers_overview']['mean_k_int_plot']-borehole_analysis_dict[int_option]['layers_overview']['std_mean_k_int_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='red', linestyle='dashed')
	ax30.plot(borehole_analysis_dict[int_option]['layers_overview']['mean_k_int_plot']+borehole_analysis_dict[int_option]['layers_overview']['std_mean_k_int_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='red', linestyle='dashed')
	ax30.plot(borehole_analysis_dict[int_option]['layers_overview']['gmean_k_int_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='blue', linestyle='solid')
	ax30.plot(borehole_analysis_dict[int_option]['layers_overview']['gmean_k_int_plot']-borehole_analysis_dict[int_option]['layers_overview']['gmean_minus_k_int_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='blue', linestyle='dashed')
	ax30.plot(borehole_analysis_dict[int_option]['layers_overview']['gmean_k_int_plot']+borehole_analysis_dict[int_option]['layers_overview']['gmean_plus_k_int_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='blue', linestyle='dashed')
	# TODO Sort out correct uncertainties on harmonic mean
	ax30.plot(borehole_analysis_dict[int_option]['layers_overview']['hmean_k_int_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='green', linestyle='solid')
	ax30.plot(borehole_analysis_dict[int_option]['layers_overview']['hmean_k_int_plot']-borehole_analysis_dict[int_option]['layers_overview']['std_mean_k_int_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='green', linestyle='dashed')
	ax30.plot(borehole_analysis_dict[int_option]['layers_overview']['hmean_k_int_plot']+borehole_analysis_dict[int_option]['layers_overview']['std_mean_k_int_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='green', linestyle='dashed')
	ax30.errorbar(borehole_analysis_dict['raw_borehole_measurements']['k'], borehole_analysis_dict['raw_borehole_measurements']['z_k'], xerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_k'], yerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_z_k'], fmt='.k', markeredgecolor='k', zorder=5)
	ax31.errorbar(borehole_analysis_dict['raw_borehole_measurements']['T'], borehole_analysis_dict['raw_borehole_measurements']['z_T'], xerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_T'], yerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_z_T'], fmt='.k', color='k', markeredgewidth=0.1)
	# ax31.errorbar(z_T_int[:,2], z_T_int[:,0], xerr=z_T_int[:,3], yerr=z_T_int[:,1], fmt='.k', color='k', markeredgewidth=0.1)
	# ax31.fill_betweenx(z_T_int[:,0], (dTdz+sigma_dTdz) * z_T_int[:,0] + (T0+sigma_T0), (dTdz-sigma_dTdz) * z_T_int[:,0] + (T0-sigma_T0), facecolor='black', alpha=0.25)
	for layer_index in range(number_layers):
		x = 'layer'+str(layer_index)	
		ax31.plot(
			borehole_analysis_dict[int_option]['individual_layers'][x]['interval_method']['dTdz'] * np.append(borehole_analysis_dict[int_option]['individual_layers'][x]['z0_int'], borehole_analysis_dict[int_option]['individual_layers'][x]['z1_int']) + borehole_analysis_dict[int_option]['individual_layers'][x]['interval_method']['T0'],
			np.append(borehole_analysis_dict[int_option]['individual_layers'][x]['z0_int'], borehole_analysis_dict[int_option]['individual_layers'][x]['z1_int']),
			color='red', zorder=5
		)
	ax32.plot(borehole_analysis_dict[int_option]['layers_overview']['interval_method']['Q_int_mean_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='red')
	ax32.plot(borehole_analysis_dict[int_option]['layers_overview']['interval_method']['Q_int_mean_round_plot']-borehole_analysis_dict[int_option]['layers_overview']['interval_method']['sigma_Q_int_mean_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='red', linestyle='dashed')
	ax32.plot(borehole_analysis_dict[int_option]['layers_overview']['interval_method']['Q_int_mean_round_plot']+borehole_analysis_dict[int_option]['layers_overview']['interval_method']['sigma_Q_int_mean_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='red', linestyle='dashed')
	ax32.plot(borehole_analysis_dict[int_option]['layers_overview']['interval_method']['Q_int_gmean_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='blue')
	# TODO Sort out correct errors on heat flow estimated from geometric mean
	ax32.plot(borehole_analysis_dict[int_option]['layers_overview']['interval_method']['Q_int_gmean_round_plot']-borehole_analysis_dict[int_option]['layers_overview']['interval_method']['sigma_Q_int_gmean_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='blue', linestyle='dashed')
	ax32.plot(borehole_analysis_dict[int_option]['layers_overview']['interval_method']['Q_int_gmean_round_plot']+borehole_analysis_dict[int_option]['layers_overview']['interval_method']['sigma_Q_int_gmean_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='blue', linestyle='dashed')
	ax32.plot(borehole_analysis_dict[int_option]['layers_overview']['interval_method']['Q_int_hmean_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='green')
	# TODO Sort out correct errors on heat flow estimated from harmonic mean
	ax32.plot(borehole_analysis_dict[int_option]['layers_overview']['interval_method']['Q_int_hmean_round_plot']-borehole_analysis_dict[int_option]['layers_overview']['interval_method']['sigma_Q_int_hmean_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='green', linestyle='dashed')
	ax32.plot(borehole_analysis_dict[int_option]['layers_overview']['interval_method']['Q_int_hmean_round_plot']+borehole_analysis_dict[int_option]['layers_overview']['interval_method']['sigma_Q_int_hmean_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='green', linestyle='dashed')
	if plot_strat_option == 'yes':
		for layer_index in range(number_layers):
			
			ax33.axhspan(
				ymin = borehole_analysis_dict[int_option]['layers_overview']['z0_int'][layer_index],
				ymax = borehole_analysis_dict[int_option]['layers_overview']['z1_int'][layer_index],
				color = lith_plot_format_dict[borehole_analysis_dict[int_option]['layers_overview']['int_lith_type'][layer_index]],
				alpha = 0.5
			)
			ax33.axhline(
			linewidth=2,
			y = borehole_analysis_dict[int_option]['layers_overview']['z1_int'][layer_index],
			color = 'black'
			)
			ax33.text(
				0.5,
				(borehole_analysis_dict[int_option]['layers_overview']['z0_int'][layer_index] + borehole_analysis_dict[int_option]['layers_overview']['z1_int'][layer_index]) / 2,
				borehole_analysis_dict[int_option]['layers_overview']['int_lith_name'][layer_index],
				verticalalignment='center',
				horizontalalignment='center'
			)
	
	### Plot estimate using arithmetic mean
	ax20.errorbar(borehole_analysis_dict['raw_borehole_measurements']['k'], borehole_analysis_dict['raw_borehole_measurements']['z_k'], xerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_k'], yerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_z_k'], fmt='.k', markeredgecolor='k', zorder=5)
	ax20.plot(borehole_analysis_dict[int_option]['layers_overview']['mean_k_int_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='red')
	ax20.fill_betweenx(
		borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['mean_k_int_plot']-borehole_analysis_dict[int_option]['layers_overview']['std_mean_k_int_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['mean_k_int_plot']+borehole_analysis_dict[int_option]['layers_overview']['std_mean_k_int_plot'],
		facecolor='red', alpha=0.25
	)
	ax21.errorbar(borehole_analysis_dict['raw_borehole_measurements']['T'], borehole_analysis_dict['raw_borehole_measurements']['z_T'], xerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_T'], yerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_z_T'], fmt='.k', color='k', markeredgewidth=0.1)
	for layer_index in range(number_layers):
		x = 'layer'+str(layer_index)	
		ax21.plot(
			borehole_analysis_dict[int_option]['individual_layers'][x]['interval_method']['dTdz'] * np.append(borehole_analysis_dict[int_option]['individual_layers'][x]['z0_int'], borehole_analysis_dict[int_option]['individual_layers'][x]['z1_int']) + borehole_analysis_dict[int_option]['individual_layers'][x]['interval_method']['T0'],
			np.append(borehole_analysis_dict[int_option]['individual_layers'][x]['z0_int'], borehole_analysis_dict[int_option]['individual_layers'][x]['z1_int']),
			color='red', zorder=5
		)
	ax22.plot(borehole_analysis_dict[int_option]['layers_overview']['interval_method']['Q_int_mean_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='red')
	ax22.fill_betweenx(
		borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['interval_method']['Q_int_mean_round_plot']-borehole_analysis_dict[int_option]['layers_overview']['interval_method']['sigma_Q_int_mean_round_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['interval_method']['Q_int_mean_round_plot']+borehole_analysis_dict[int_option]['layers_overview']['interval_method']['sigma_Q_int_mean_round_plot'],
		facecolor='red', alpha=0.25
	)
	if plot_strat_option == 'yes':
		for layer_index in range(number_layers):
			
			ax23.axhspan(
				ymin = borehole_analysis_dict[int_option]['layers_overview']['z0_int'][layer_index],
				ymax = borehole_analysis_dict[int_option]['layers_overview']['z1_int'][layer_index],
				color = lith_plot_format_dict[borehole_analysis_dict[int_option]['layers_overview']['int_lith_type'][layer_index]],
				alpha = 0.5
			)
			ax23.axhline(
			linewidth=2,
			y = borehole_analysis_dict[int_option]['layers_overview']['z1_int'][layer_index],
			color = 'black'
			)
			ax23.text(
				0.5,
				(borehole_analysis_dict[int_option]['layers_overview']['z0_int'][layer_index] + borehole_analysis_dict[int_option]['layers_overview']['z1_int'][layer_index]) / 2,
				borehole_analysis_dict[int_option]['layers_overview']['int_lith_name'][layer_index],
				verticalalignment='center',
				horizontalalignment='center'
			)
	
	### Plot estimate using geometric mean
	ax10.errorbar(borehole_analysis_dict['raw_borehole_measurements']['k'], borehole_analysis_dict['raw_borehole_measurements']['z_k'], xerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_k'], yerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_z_k'], fmt='.k', markeredgecolor='k', zorder=5)
	ax10.plot(borehole_analysis_dict[int_option]['layers_overview']['gmean_k_int_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='blue')
	ax10.fill_betweenx(
		borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['gmean_k_int_plot']-borehole_analysis_dict[int_option]['layers_overview']['gmean_minus_k_int_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['gmean_k_int_plot']+borehole_analysis_dict[int_option]['layers_overview']['gmean_plus_k_int_plot'],
		facecolor='blue', alpha=0.25
	)
	ax11.errorbar(borehole_analysis_dict['raw_borehole_measurements']['T'], borehole_analysis_dict['raw_borehole_measurements']['z_T'], xerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_T'], yerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_z_T'], fmt='.k', color='k', markeredgewidth=0.1)
	for layer_index in range(number_layers):
		x = 'layer'+str(layer_index)	
		ax11.plot(
			borehole_analysis_dict[int_option]['individual_layers'][x]['interval_method']['dTdz'] * np.append(borehole_analysis_dict[int_option]['individual_layers'][x]['z0_int'], borehole_analysis_dict[int_option]['individual_layers'][x]['z1_int']) + borehole_analysis_dict[int_option]['individual_layers'][x]['interval_method']['T0'],
			np.append(borehole_analysis_dict[int_option]['individual_layers'][x]['z0_int'], borehole_analysis_dict[int_option]['individual_layers'][x]['z1_int']),
			color='blue', zorder=5
		)
	ax12.plot(borehole_analysis_dict[int_option]['layers_overview']['interval_method']['Q_int_gmean_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='blue')
	# TODO Sort out correct errors on heat flow estimated from geometric mean
	ax12.fill_betweenx(
		borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['interval_method']['Q_int_gmean_round_plot']-borehole_analysis_dict[int_option]['layers_overview']['interval_method']['sigma_Q_int_gmean_round_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['interval_method']['Q_int_gmean_round_plot']+borehole_analysis_dict[int_option]['layers_overview']['interval_method']['sigma_Q_int_gmean_round_plot'],
		facecolor='blue', alpha=0.25
	)
	if plot_strat_option == 'yes':
		for layer_index in range(number_layers):
			ax13.axhspan(
				ymin = borehole_analysis_dict[int_option]['layers_overview']['z0_int'][layer_index],
				ymax = borehole_analysis_dict[int_option]['layers_overview']['z1_int'][layer_index],
				color = lith_plot_format_dict[borehole_analysis_dict[int_option]['layers_overview']['int_lith_type'][layer_index]],
				alpha = 0.5
			)
			ax13.axhline(
			linewidth=2,
			y = borehole_analysis_dict[int_option]['layers_overview']['z1_int'][layer_index],
			color = 'black'
			)
			ax13.text(
				0.5,
				(borehole_analysis_dict[int_option]['layers_overview']['z0_int'][layer_index] + borehole_analysis_dict[int_option]['layers_overview']['z1_int'][layer_index]) / 2,
				borehole_analysis_dict[int_option]['layers_overview']['int_lith_name'][layer_index],
				verticalalignment='center',
				horizontalalignment='center'
			)
	
	### Plot estimate using harmonic mean
	# TODO Sort out correct uncertainties on harmonic mean
	ax00.errorbar(borehole_analysis_dict['raw_borehole_measurements']['k'], borehole_analysis_dict['raw_borehole_measurements']['z_k'], xerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_k'], yerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_z_k'], fmt='.k', markeredgecolor='k', zorder=5)
	ax00.plot(borehole_analysis_dict[int_option]['layers_overview']['hmean_k_int_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='green')
	ax00.fill_betweenx(
		borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['hmean_k_int_plot']-borehole_analysis_dict[int_option]['layers_overview']['std_mean_k_int_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['hmean_k_int_plot']+borehole_analysis_dict[int_option]['layers_overview']['std_mean_k_int_plot'],
		facecolor='green', alpha=0.25
	)
	ax01.errorbar(borehole_analysis_dict['raw_borehole_measurements']['T'], borehole_analysis_dict['raw_borehole_measurements']['z_T'], xerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_T'], yerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_z_T'], fmt='.k', color='k', markeredgewidth=0.1)
	for layer_index in range(number_layers):
		x = 'layer'+str(layer_index)	
		ax01.plot(
			borehole_analysis_dict[int_option]['individual_layers'][x]['interval_method']['dTdz'] * np.append(borehole_analysis_dict[int_option]['individual_layers'][x]['z0_int'], borehole_analysis_dict[int_option]['individual_layers'][x]['z1_int']) + borehole_analysis_dict[int_option]['individual_layers'][x]['interval_method']['T0'],
			np.append(borehole_analysis_dict[int_option]['individual_layers'][x]['z0_int'], borehole_analysis_dict[int_option]['individual_layers'][x]['z1_int']),
			color='green', zorder=5
		)
	ax02.plot(borehole_analysis_dict[int_option]['layers_overview']['interval_method']['Q_int_hmean_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='green')
	# TODO Sort out correct errors on heat flow estimated from harmonic mean
	ax02.fill_betweenx(
		borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['interval_method']['Q_int_hmean_round_plot']-borehole_analysis_dict[int_option]['layers_overview']['interval_method']['sigma_Q_int_hmean_round_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['interval_method']['Q_int_hmean_round_plot']+borehole_analysis_dict[int_option]['layers_overview']['interval_method']['sigma_Q_int_hmean_round_plot'],
		facecolor='green', alpha=0.25
	)
	if plot_strat_option == 'yes':
		for layer_index in range(number_layers):
			
			ax03.axhspan(
				ymin = borehole_analysis_dict[int_option]['layers_overview']['z0_int'][layer_index],
				ymax = borehole_analysis_dict[int_option]['layers_overview']['z1_int'][layer_index],
				color = lith_plot_format_dict[borehole_analysis_dict[int_option]['layers_overview']['int_lith_type'][layer_index]],
				alpha = 0.5
			)
			ax03.axhline(
			linewidth=2,
			y = borehole_analysis_dict[int_option]['layers_overview']['z1_int'][layer_index],
			color = 'black'
			)
			ax03.text(
				0.5,
				(borehole_analysis_dict[int_option]['layers_overview']['z0_int'][layer_index] + borehole_analysis_dict[int_option]['layers_overview']['z1_int'][layer_index]) / 2,
				borehole_analysis_dict[int_option]['layers_overview']['int_lith_name'][layer_index],
				verticalalignment='center',
				horizontalalignment='center'
			)
	
	# plt.show()
	fig.savefig(figure_name, bbox_inches='tight', transparent=True)
	plt.close(fig)
		
	return()


###------------------------------------------------------------------------------------------------------------

def plot_layers_bullard_method(borehole_analysis_dict, int_option, lith_plot_format_dict, max_depth_m_plot, plot_label_dict, figures_path):
	
	heat_flow_estimation_method = 'bullard_method'
	figure_name, plot_title = set_up_plot_title(borehole_analysis_dict, int_option, heat_flow_estimation_method, plot_label_dict, figures_path)
	
	
	# TODO Still need to set this up
	number_rows = 3
	number_columns = 19
	plot_format_dict = set_up_plot_formatting(borehole_analysis_dict[int_option]['layers_overview']['plot_strat_option'], number_rows, number_columns)
	
	number_layers = borehole_analysis_dict[int_option]['layers_overview']['number_layers']
	plot_strat_option = borehole_analysis_dict[int_option]['layers_overview']['plot_strat_option']
		
	# Set up plotting preferences
	plt.rcParams['xtick.bottom'] = False
	plt.rcParams['xtick.labelbottom'] = False
	plt.rcParams['xtick.top'] = True
	plt.rcParams['xtick.labeltop'] = True
	
	# Set up figure limits
	min_depth_m=0
	max_depth_m=600
	min_k=1
	max_k=5.75
	
	# Set up axes
	fig = plt.figure(figsize=(plot_format_dict['figure_width'], plot_format_dict['figure_height']), dpi=100)
	fig.suptitle(plot_title)
	
	
	# Plot all estimates overlain
	if plot_strat_option == 'yes':
		ax20 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column0'], plot_format_dict['row_fractional_ypos_dict']['row2'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], ylabel=r'$z$ / m')
		ax20.set_ylim(top=0, bottom=max_depth_m_plot)
		ax20.set_xlim(left=0, right=1)
		ax20.yaxis.set_ticks_position('both')
		ax20.xaxis.set_ticks_position('none')
		ax20.xaxis.set_minor_locator(MultipleLocator(5))
		ax20.yaxis.set_minor_locator(MultipleLocator(50))
		ax20.xaxis.set_major_formatter(FormatStrFormatter(''))
		ax21 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column1'], plot_format_dict['row_fractional_ypos_dict']['row2'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$k$ / W m$^{-1}$ K$^{-1}$')
		ax21.yaxis.set_major_formatter(FormatStrFormatter(''))
	else:	
		ax21 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column1'], plot_format_dict['row_fractional_ypos_dict']['row2'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$k$ / W m$^{-1}$ K$^{-1}$', ylabel=r'$z$ / m')
	
	ax21.set_ylim(top=min_depth_m, bottom=max_depth_m)
	# ax21.set_xlim(left=min_k, right=max_k)
	ax21.xaxis.set_label_position('top')
	ax21.yaxis.set_ticks_position('both')
	ax21.xaxis.set_ticks_position('both')
	ax21.xaxis.set_minor_locator(MultipleLocator(0.5))
	ax21.yaxis.set_minor_locator(MultipleLocator(50))
	
	ax22 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column2'], plot_format_dict['row_fractional_ypos_dict']['row2'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$R$ / K m$^2$ W$^{-1}$')
	ax22.set_ylim(top=min_depth_m, bottom=max_depth_m)
	ax22.xaxis.set_label_position('top')
	ax22.yaxis.set_ticks_position('both')
	ax22.xaxis.set_ticks_position('both')
	ax22.xaxis.set_minor_locator(MultipleLocator(25))
	ax22.yaxis.set_minor_locator(MultipleLocator(50))
	ax22.yaxis.set_major_formatter(FormatStrFormatter(''))
	
	ax23 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column3'], plot_format_dict['row_fractional_ypos_dict']['row2'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$T$ / $^{\circ}$C')
	ax23.set_ylim(top=min_depth_m, bottom=max_depth_m)
	ax23.xaxis.set_label_position('top')
	ax23.yaxis.set_ticks_position('both')
	ax23.xaxis.set_ticks_position('both')
	ax23.xaxis.set_minor_locator(MultipleLocator(1))
	ax23.yaxis.set_minor_locator(MultipleLocator(50))
	ax23.yaxis.set_major_formatter(FormatStrFormatter(''))
	

	
	ax11to12 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column1'], plot_format_dict['row_fractional_ypos_dict']['row1'], 2*plot_format_dict['column_fractional_width'] + plot_format_dict['column_fractional_spacing'], plot_format_dict['row_fractional_height']], xlabel=r'$R$ / K m$^2$ W$^{-1}$', ylabel='$T$ / $^{\circ}$C')
	# ax11to12.set_ylim(top=0, bottom=max_depth_m_plot)
	# ax11to12.set_xlim(left=0, right=1)
	ax11to12.xaxis.set_label_position('top')
	ax11to12.yaxis.set_ticks_position('both')
	ax11to12.xaxis.set_ticks_position('both')
	# ax11to12.xaxis.set_minor_locator(MultipleLocator(5))
	# ax11to12.yaxis.set_minor_locator(MultipleLocator(50))
	# ax11to12.yaxis.set_major_formatter(FormatStrFormatter(''))
	# ax11to12.xaxis.set_major_formatter(FormatStrFormatter(''))
	
	ax13 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column3'], plot_format_dict['row_fractional_ypos_dict']['row1'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$Q$ / mW m$^{-2}$', ylabel=r'$z$ / m')
	ax13.set_xlim(left=35, right=51)
	ax13.set_ylim(top=min_depth_m, bottom=max_depth_m)
	ax13.xaxis.set_label_position('top')
	ax13.yaxis.set_label_position('right')
	ax13.yaxis.set_ticks_position('right')
	ax13.xaxis.set_ticks_position('both')
	ax13.xaxis.set_minor_locator(MultipleLocator(1))
	ax13.yaxis.set_minor_locator(MultipleLocator(50))
	# ax13.yaxis.set_major_formatter(FormatStrFormatter(50))
	
	ax01to02 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column1'], plot_format_dict['row_fractional_ypos_dict']['row0'], 2*plot_format_dict['column_fractional_width'] + plot_format_dict['column_fractional_spacing'], plot_format_dict['row_fractional_height']], xlabel='$T$ / $^{\circ}$C', ylabel=r'$R$ / K m$^2$ W$^{-1}$')
	# ax01to02.set_ylim(top=0, bottom=max_depth_m_plot)
	# ax01to02.set_xlim(left=0, right=1)
	ax01to02.xaxis.set_label_position('top')
	ax01to02.yaxis.set_ticks_position('both')
	ax01to02.xaxis.set_ticks_position('both')
	# ax01to02.xaxis.set_minor_locator(MultipleLocator(5))
	# ax01to02.yaxis.set_minor_locator(MultipleLocator(50))
	# ax01to02.yaxis.set_major_formatter(FormatStrFormatter(''))
	# ax01to02.xaxis.set_major_formatter(FormatStrFormatter(''))
	
	ax03 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column3'], plot_format_dict['row_fractional_ypos_dict']['row0'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$Q$ / mW m$^{-2}$', ylabel=r'$z$ / m')
	ax03.set_xlim(left=35, right=51)
	ax03.set_ylim(top=min_depth_m, bottom=max_depth_m)
	ax03.xaxis.set_label_position('top')
	ax03.yaxis.set_label_position('right')
	ax03.yaxis.set_ticks_position('right')
	ax03.xaxis.set_ticks_position('both')
	ax03.xaxis.set_minor_locator(MultipleLocator(1))
	ax03.yaxis.set_minor_locator(MultipleLocator(50))
	# ax03.yaxis.set_major_formatter(FormatStrFormatter(50))
	
	# Plot data
	ax21.plot(borehole_analysis_dict[int_option]['layers_overview']['mean_k_int_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='red')
	ax21.plot(borehole_analysis_dict[int_option]['layers_overview']['mean_k_int_plot']-borehole_analysis_dict[int_option]['layers_overview']['std_mean_k_int_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='red', linestyle='dashed')
	ax21.plot(borehole_analysis_dict[int_option]['layers_overview']['mean_k_int_plot']+borehole_analysis_dict[int_option]['layers_overview']['std_mean_k_int_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='red', linestyle='dashed')
	# ax21.fill_betweenx(
	# 	borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'],
	# 	borehole_analysis_dict[int_option]['layers_overview']['mean_k_int_plot']-borehole_analysis_dict[int_option]['layers_overview']['std_mean_k_int_plot'],
	# 	borehole_analysis_dict[int_option]['layers_overview']['mean_k_int_plot']+borehole_analysis_dict[int_option]['layers_overview']['std_mean_k_int_plot'],
	# 	facecolor='red', alpha=0.25
	# )
	ax21.plot(borehole_analysis_dict[int_option]['layers_overview']['gmean_k_int_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='blue', linestyle='solid')
	ax21.plot(borehole_analysis_dict[int_option]['layers_overview']['gmean_k_int_plot']-borehole_analysis_dict[int_option]['layers_overview']['gmean_minus_k_int_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='blue', linestyle='dashed')
	ax21.plot(borehole_analysis_dict[int_option]['layers_overview']['gmean_k_int_plot']+borehole_analysis_dict[int_option]['layers_overview']['gmean_plus_k_int_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='blue', linestyle='dashed')
	# ax21.fill_betweenx(
	# 	borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'],
	# 	borehole_analysis_dict[int_option]['layers_overview']['gmean_k_int_plot']-borehole_analysis_dict[int_option]['layers_overview']['gmean_minus_k_int_plot'],
	# 	borehole_analysis_dict[int_option]['layers_overview']['gmean_k_int_plot']+borehole_analysis_dict[int_option]['layers_overview']['gmean_plus_k_int_plot'],
	# 	facecolor='blue', alpha=0.25
	# )
	# TODO Sort out correct uncertainties on harmonic mean
	ax21.plot(borehole_analysis_dict[int_option]['layers_overview']['hmean_k_int_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='green', linestyle='solid')
	ax21.plot(borehole_analysis_dict[int_option]['layers_overview']['hmean_k_int_plot']-borehole_analysis_dict[int_option]['layers_overview']['std_mean_k_int_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='green', linestyle='dashed')
	ax21.plot(borehole_analysis_dict[int_option]['layers_overview']['hmean_k_int_plot']+borehole_analysis_dict[int_option]['layers_overview']['std_mean_k_int_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='green', linestyle='dashed')
	# ax21.fill_betweenx(
	# 	borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'],
	# 	borehole_analysis_dict[int_option]['layers_overview']['gmean_k_int_plot']-borehole_analysis_dict[int_option]['layers_overview']['gmean_minus_k_int_plot'],
	# 	borehole_analysis_dict[int_option]['layers_overview']['gmean_k_int_plot']+borehole_analysis_dict[int_option]['layers_overview']['gmean_plus_k_int_plot'],
	# 	facecolor='blue', alpha=0.25
	# )
	ax21.errorbar(borehole_analysis_dict['raw_borehole_measurements']['k'], borehole_analysis_dict['raw_borehole_measurements']['z_k'], xerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_k'], yerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_z_k'], fmt='.k', markeredgecolor='k', zorder=5)

	# ax22.errorbar(z_T_int[:,2], z_T_int[:,0], xerr=z_T_int[:,3], yerr=z_T_int[:,1], fmt='.k', color='k', markeredgewidth=0.1)
	# ax22.fill_betweenx(z_T_int[:,0], (dTdz+sigma_dTdz) * z_T_int[:,0] + (T0+sigma_T0), (dTdz-sigma_dTdz) * z_T_int[:,0] + (T0-sigma_T0), facecolor='black', alpha=0.25)
	# for layer_index in range(number_layers):
	# 	x = 'layer'+str(layer_index)
	# 	print(borehole_analysis_dict[int_option]['individual_layers'][x]['interval_method']['dTdz'])
	# 	ax22.plot(
	# 		borehole_analysis_dict[int_option]['individual_layers'][x]['interval_method']['dTdz'] * np.append(borehole_analysis_dict[int_option]['individual_layers'][x]['z0_int'], borehole_analysis_dict[int_option]['individual_layers'][x]['z1_int']) + borehole_analysis_dict[int_option]['individual_layers'][x]['interval_method']['T0'],
	# 		np.append(borehole_analysis_dict[int_option]['individual_layers'][x]['z0_int'], borehole_analysis_dict[int_option]['individual_layers'][x]['z1_int']),
	# 		color='red', zorder=5
	# 	)
		
	ax22.plot(borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_mean_k_int_z_T'], borehole_analysis_dict['raw_borehole_measurements']['z_T'], color='red')
	ax22.plot(borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_mean_k_int_z_T'] + borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['sigma_R_mean_k_int_z_T'], borehole_analysis_dict['raw_borehole_measurements']['z_T'], color='red', linestyle = 'dashed')
	ax22.plot(borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_mean_k_int_z_T'] - borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['sigma_R_mean_k_int_z_T'], borehole_analysis_dict['raw_borehole_measurements']['z_T'], color='red', linestyle = 'dashed')
	
	ax22.plot(borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_gmean_k_int_z_T'], borehole_analysis_dict['raw_borehole_measurements']['z_T'], color='blue')
	ax22.plot(borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_gmean_k_int_z_T'] + borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['sigma_R_gmean_k_int_z_T'], borehole_analysis_dict['raw_borehole_measurements']['z_T'], color='blue', linestyle = 'dashed')
	ax22.plot(borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_gmean_k_int_z_T'] - borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['sigma_R_gmean_k_int_z_T'], borehole_analysis_dict['raw_borehole_measurements']['z_T'], color='blue', linestyle = 'dashed')
	
	ax22.plot(borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_hmean_k_int_z_T'], borehole_analysis_dict['raw_borehole_measurements']['z_T'], color='green')
	ax22.plot(borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_hmean_k_int_z_T'] + borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['sigma_R_hmean_k_int_z_T'], borehole_analysis_dict['raw_borehole_measurements']['z_T'], color='green', linestyle = 'dashed')
	ax22.plot(borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_hmean_k_int_z_T'] - borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['sigma_R_hmean_k_int_z_T'], borehole_analysis_dict['raw_borehole_measurements']['z_T'], color='green', linestyle = 'dashed')
	
	ax23.errorbar(borehole_analysis_dict['raw_borehole_measurements']['T'], borehole_analysis_dict['raw_borehole_measurements']['z_T'], xerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_T'], yerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_z_T'], fmt='.k', color='k', markeredgewidth=0.1)
	
	# pprint.pprint(borehole_analysis_dict[int_option]['individual_layers'][x]['bullard_method'])
	# pprint.pprint(borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance'].keys())
	
	if plot_strat_option == 'yes':
		for layer_index in range(number_layers):
			
			ax20.axhspan(
				ymin = borehole_analysis_dict[int_option]['layers_overview']['z0_int'][layer_index],
				ymax = borehole_analysis_dict[int_option]['layers_overview']['z1_int'][layer_index],
				color = lith_plot_format_dict[borehole_analysis_dict[int_option]['layers_overview']['int_lith_type'][layer_index]],
				alpha = 0.5
			)
			ax20.axhline(
			linewidth=2,
			y = borehole_analysis_dict[int_option]['layers_overview']['z1_int'][layer_index],
			color = 'black'
			)
			ax20.text(
				0.5,
				(borehole_analysis_dict[int_option]['layers_overview']['z0_int'][layer_index] + borehole_analysis_dict[int_option]['layers_overview']['z1_int'][layer_index]) / 2,
				borehole_analysis_dict[int_option]['layers_overview']['int_lith_name'][layer_index],
				verticalalignment='center',
				horizontalalignment='center'
			)
	
	ax11to12.errorbar(borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_mean_k_int_z_T'],
	borehole_analysis_dict['raw_borehole_measurements']['T'],
	xerr=borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['sigma_R_mean_k_int_z_T'],
	yerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_T'], fmt='.', color='red', markeredgewidth=0.1)
	
	# TODO Sort out error on geometric mean
	ax11to12.errorbar(borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_gmean_k_int_z_T'],
	borehole_analysis_dict['raw_borehole_measurements']['T'],
	xerr=borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['sigma_R_mean_k_int_z_T'],
	yerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_T'], fmt='.', color='blue', markeredgewidth=0.1)
	
	# TODO Sort out error on harmonic mean
	ax11to12.errorbar(borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_hmean_k_int_z_T'],
	borehole_analysis_dict['raw_borehole_measurements']['T'],
	xerr=borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['sigma_R_mean_k_int_z_T'],
	yerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_T'], fmt='.', color='green', markeredgewidth=0.1)
	
	
	
	ax13.plot(borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['T_versus_R']['Q_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='red')
	ax13.plot(borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['T_versus_R']['Q_round_plot']-borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['T_versus_R']['sigma_Q_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='red', linestyle='dashed')
	ax13.plot(borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['T_versus_R']['Q_round_plot']+borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['T_versus_R']['sigma_Q_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='red', linestyle='dashed')
	# ax13.fill_betweenx(
	# 	borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'],
	# 	borehole_analysis_dict[int_option]['layers_overview']['interval_method']['Q_int_mean_round_plot']-borehole_analysis_dict[int_option]['layers_overview']['interval_method']['sigma_Q_int_mean_round_plot'],
	# 	borehole_analysis_dict[int_option]['layers_overview']['interval_method']['Q_int_mean_round_plot']+borehole_analysis_dict[int_option]['layers_overview']['interval_method']['sigma_Q_int_mean_round_plot'],
	# 	facecolor='red', alpha=0.25
	# )
	
	# TODO Sort out errors on heat flow estimated from geometric mean
	ax13.plot(borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['gmean_k']['T_versus_R']['Q_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='blue')
	ax13.plot(borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['gmean_k']['T_versus_R']['Q_round_plot']-borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['gmean_k']['T_versus_R']['sigma_Q_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='blue', linestyle='dashed')
	ax13.plot(borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['gmean_k']['T_versus_R']['Q_round_plot']+borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['gmean_k']['T_versus_R']['sigma_Q_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='blue', linestyle='dashed')
	
	# TODO Sort out errors on heat flow estimated from harmonic mean
	ax13.plot(borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['hmean_k']['T_versus_R']['Q_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='green')
	ax13.plot(borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['hmean_k']['T_versus_R']['Q_round_plot']-borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['hmean_k']['T_versus_R']['sigma_Q_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='green', linestyle='dashed')
	ax13.plot(borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['hmean_k']['T_versus_R']['Q_round_plot']+borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['hmean_k']['T_versus_R']['sigma_Q_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='green', linestyle='dashed')
	
	
	
	
	
	ax01to02.errorbar(borehole_analysis_dict['raw_borehole_measurements']['T'], borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_mean_k_int_z_T'], xerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_T'], yerr=borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['sigma_R_mean_k_int_z_T'], fmt='.', color='red', markeredgewidth=0.1)
	
	# TODO Sort out error on geometric mean
	ax01to02.errorbar(borehole_analysis_dict['raw_borehole_measurements']['T'], borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_gmean_k_int_z_T'], xerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_T'], yerr=borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['sigma_R_gmean_k_int_z_T'], fmt='.', color='blue', markeredgewidth=0.1)
	
	# TODO Sort out error on harmonic mean
	ax01to02.errorbar(borehole_analysis_dict['raw_borehole_measurements']['T'], borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_hmean_k_int_z_T'], xerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_T'], yerr=borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['sigma_R_hmean_k_int_z_T'], fmt='.', color='green', markeredgewidth=0.1)
	
	
	
	ax03.plot(borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['R_versus_T']['Q_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='red')
	ax03.plot(borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['R_versus_T']['Q_round_plot']-borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['R_versus_T']['sigma_Q_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='red', linestyle='dashed')
	ax03.plot(borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['R_versus_T']['Q_round_plot']+borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['R_versus_T']['sigma_Q_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='red', linestyle='dashed')
	# ax03.fill_betweenx(
	# 	borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'],
	# 	borehole_analysis_dict[int_option]['layers_overview']['interval_method']['Q_int_mean_round_plot']-borehole_analysis_dict[int_option]['layers_overview']['interval_method']['sigma_Q_int_mean_round_plot'],
	# 	borehole_analysis_dict[int_option]['layers_overview']['interval_method']['Q_int_mean_round_plot']+borehole_analysis_dict[int_option]['layers_overview']['interval_method']['sigma_Q_int_mean_round_plot'],
	# 	facecolor='red', alpha=0.25
	# )
	
	# TODO Sort out errors on heat flow estimated from geometric mean
	ax03.plot(borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['gmean_k']['R_versus_T']['Q_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='blue')
	ax03.plot(borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['gmean_k']['R_versus_T']['Q_round_plot']-borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['gmean_k']['R_versus_T']['sigma_Q_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='blue', linestyle='dashed')
	ax03.plot(borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['gmean_k']['R_versus_T']['Q_round_plot']+borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['gmean_k']['R_versus_T']['sigma_Q_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='blue', linestyle='dashed')
	
	# TODO Sort out errors on heat flow estimated from harmonic mean
	ax03.plot(borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['hmean_k']['R_versus_T']['Q_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='green')
	ax03.plot(borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['hmean_k']['R_versus_T']['Q_round_plot']-borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['hmean_k']['R_versus_T']['sigma_Q_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='green', linestyle='dashed')
	ax03.plot(borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['hmean_k']['R_versus_T']['Q_round_plot']+borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['hmean_k']['R_versus_T']['sigma_Q_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='green', linestyle='dashed')
	
	
	
	
	
	
	# Plot estimates of Q from arithmetic mean
	if plot_strat_option == 'yes':
		ax25 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column5'], plot_format_dict['row_fractional_ypos_dict']['row2'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], ylabel=r'$z$ / m')
		ax25.set_ylim(top=0, bottom=max_depth_m_plot)
		ax25.set_xlim(left=0, right=1)
		ax25.yaxis.set_ticks_position('both')
		ax25.xaxis.set_ticks_position('none')
		ax25.xaxis.set_minor_locator(MultipleLocator(5))
		ax25.yaxis.set_minor_locator(MultipleLocator(50))
		ax25.xaxis.set_major_formatter(FormatStrFormatter(''))
		ax26 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column6'], plot_format_dict['row_fractional_ypos_dict']['row2'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$k$ / W m$^{-1}$ K$^{-1}$')
		ax26.yaxis.set_major_formatter(FormatStrFormatter(''))
	else:	
		ax26 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column6'], plot_format_dict['row_fractional_ypos_dict']['row2'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$k$ / W m$^{-1}$ K$^{-1}$', ylabel=r'$z$ / m')
	
	ax26.set_ylim(top=min_depth_m, bottom=max_depth_m)
	# ax26.set_xlim(left=min_k, right=max_k)
	ax26.xaxis.set_label_position('top')
	ax26.yaxis.set_ticks_position('both')
	ax26.xaxis.set_ticks_position('both')
	ax26.xaxis.set_minor_locator(MultipleLocator(0.5))
	ax26.yaxis.set_minor_locator(MultipleLocator(50))
	
	ax27 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column7'], plot_format_dict['row_fractional_ypos_dict']['row2'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$R$ / K m$^2$ W$^{-1}$')
	ax27.set_ylim(top=min_depth_m, bottom=max_depth_m)
	ax27.xaxis.set_label_position('top')
	ax27.yaxis.set_ticks_position('both')
	ax27.xaxis.set_ticks_position('both')
	ax27.xaxis.set_minor_locator(MultipleLocator(25))
	ax27.yaxis.set_minor_locator(MultipleLocator(50))
	ax27.yaxis.set_major_formatter(FormatStrFormatter(''))
	
	ax28 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column8'], plot_format_dict['row_fractional_ypos_dict']['row2'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$T$ / $^{\circ}$C')
	ax28.set_ylim(top=min_depth_m, bottom=max_depth_m)
	ax28.xaxis.set_label_position('top')
	ax28.yaxis.set_ticks_position('both')
	ax28.xaxis.set_ticks_position('both')
	ax28.xaxis.set_minor_locator(MultipleLocator(1))
	ax28.yaxis.set_minor_locator(MultipleLocator(50))
	ax28.yaxis.set_major_formatter(FormatStrFormatter(''))
	

	
	ax16to17 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column6'], plot_format_dict['row_fractional_ypos_dict']['row1'], 2*plot_format_dict['column_fractional_width'] + plot_format_dict['column_fractional_spacing'], plot_format_dict['row_fractional_height']], xlabel=r'$R$ / K m$^2$ W$^{-1}$', ylabel='$T$ / $^{\circ}$C')
	# ax16to17.set_ylim(top=0, bottom=max_depth_m_plot)
	# ax16to17.set_xlim(left=0, right=1)
	ax16to17.xaxis.set_label_position('top')
	ax16to17.yaxis.set_ticks_position('both')
	ax16to17.xaxis.set_ticks_position('both')
	ax16to17.invert_yaxis()
	# ax16to17.xaxis.set_minor_locator(MultipleLocator(5))
	# ax16to17.yaxis.set_minor_locator(MultipleLocator(50))
	# ax16to17.yaxis.set_major_formatter(FormatStrFormatter(''))
	# ax16to17.xaxis.set_major_formatter(FormatStrFormatter(''))
	
	ax18 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column8'], plot_format_dict['row_fractional_ypos_dict']['row1'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$Q$ / mW m$^{-2}$', ylabel=r'$z$ / m')
	ax18.set_xlim(left=35, right=51)
	ax18.set_ylim(top=min_depth_m, bottom=max_depth_m)
	ax18.xaxis.set_label_position('top')
	ax18.yaxis.set_label_position('right')
	ax18.yaxis.set_ticks_position('right')
	ax18.xaxis.set_ticks_position('both')
	ax18.xaxis.set_minor_locator(MultipleLocator(1))
	ax18.yaxis.set_minor_locator(MultipleLocator(50))
	# ax18.yaxis.set_major_formatter(FormatStrFormatter(50))
	
	ax06to07 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column6'], plot_format_dict['row_fractional_ypos_dict']['row0'], 2*plot_format_dict['column_fractional_width'] + plot_format_dict['column_fractional_spacing'], plot_format_dict['row_fractional_height']], xlabel='$T$ / $^{\circ}$C', ylabel=r'$R$ / K m$^2$ W$^{-1}$')
	# ax06to07.set_ylim(top=0, bottom=max_depth_m_plot)
	# ax06to07.set_xlim(left=0, right=1)
	ax06to07.xaxis.set_label_position('top')
	ax06to07.yaxis.set_ticks_position('both')
	ax06to07.xaxis.set_ticks_position('both')
	ax06to07.invert_yaxis()
	# ax06to07.xaxis.set_minor_locator(MultipleLocator(5))
	# ax06to07.yaxis.set_minor_locator(MultipleLocator(50))
	# ax06to07.yaxis.set_major_formatter(FormatStrFormatter(''))
	# ax06to07.xaxis.set_major_formatter(FormatStrFormatter(''))
	
	ax08 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column8'], plot_format_dict['row_fractional_ypos_dict']['row0'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$Q$ / mW m$^{-2}$', ylabel=r'$z$ / m')
	ax08.set_xlim(left=35, right=51)
	ax08.set_ylim(top=min_depth_m, bottom=max_depth_m)
	ax08.xaxis.set_label_position('top')
	ax08.yaxis.set_label_position('right')
	ax08.yaxis.set_ticks_position('right')
	ax08.xaxis.set_ticks_position('both')
	ax08.xaxis.set_minor_locator(MultipleLocator(1))
	ax08.yaxis.set_minor_locator(MultipleLocator(50))
	# ax08.yaxis.set_major_formatter(FormatStrFormatter(50))
	
	# Plot data
	if plot_strat_option == 'yes':
		for layer_index in range(number_layers):
			
			ax25.axhspan(
				ymin = borehole_analysis_dict[int_option]['layers_overview']['z0_int'][layer_index],
				ymax = borehole_analysis_dict[int_option]['layers_overview']['z1_int'][layer_index],
				color = lith_plot_format_dict[borehole_analysis_dict[int_option]['layers_overview']['int_lith_type'][layer_index]],
				alpha = 0.5
			)
			ax25.axhline(
			linewidth=2,
			y = borehole_analysis_dict[int_option]['layers_overview']['z1_int'][layer_index],
			color = 'black'
			)
			ax25.text(
				0.5,
				(borehole_analysis_dict[int_option]['layers_overview']['z0_int'][layer_index] + borehole_analysis_dict[int_option]['layers_overview']['z1_int'][layer_index]) / 2,
				borehole_analysis_dict[int_option]['layers_overview']['int_lith_name'][layer_index],
				verticalalignment='center',
				horizontalalignment='center'
			)
	
	ax26.plot(borehole_analysis_dict[int_option]['layers_overview']['mean_k_int_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='red')
	ax26.fill_betweenx(
		borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['mean_k_int_plot']-borehole_analysis_dict[int_option]['layers_overview']['std_mean_k_int_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['mean_k_int_plot']+borehole_analysis_dict[int_option]['layers_overview']['std_mean_k_int_plot'],
		facecolor='red', alpha=0.25
	)
	ax26.errorbar(borehole_analysis_dict['raw_borehole_measurements']['k'], borehole_analysis_dict['raw_borehole_measurements']['z_k'], xerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_k'], yerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_z_k'], fmt='.k', markeredgecolor='k', zorder=5)
	
	ax27.fill_betweenx(
		borehole_analysis_dict['raw_borehole_measurements']['z_T'],
		borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_mean_k_int_z_T']-borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['sigma_R_mean_k_int_z_T'],
		borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_mean_k_int_z_T']+borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['sigma_R_mean_k_int_z_T'],
		facecolor='red', alpha=0.25
	)
	ax27.plot(borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_mean_k_int_z_T'], borehole_analysis_dict['raw_borehole_measurements']['z_T'], color='red')
	
	ax28.errorbar(borehole_analysis_dict['raw_borehole_measurements']['T'], borehole_analysis_dict['raw_borehole_measurements']['z_T'], xerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_T'], yerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_z_T'], fmt='.k', color='k', markeredgewidth=0.1)
	
	ax16to17.errorbar(borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_mean_k_int_z_T'],
	borehole_analysis_dict['raw_borehole_measurements']['T'],
	xerr=borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['sigma_R_mean_k_int_z_T'],
	yerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_T'], fmt='.', color='black', markeredgewidth=0.1)
	for layer_index in range(number_layers):
		x = 'layer'+str(layer_index)
		ax16to17.plot(
			borehole_analysis_dict[int_option]['individual_layers'][x]['R_mean_k_int_z_T'],
			borehole_analysis_dict[int_option]['individual_layers'][x]['bullard_method']['mean_k']['T_versus_R']['q'] * borehole_analysis_dict[int_option]['individual_layers'][x]['R_mean_k_int_z_T'] + borehole_analysis_dict[int_option]['individual_layers'][x]['bullard_method']['mean_k']['T_versus_R']['c'],
			color='red', zorder=5
		)
	
	ax18.plot(borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['T_versus_R']['Q_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='red')
	ax18.fill_betweenx(
		borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['T_versus_R']['Q_round_plot']-borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['T_versus_R']['sigma_Q_round_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['T_versus_R']['Q_round_plot']+borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['T_versus_R']['sigma_Q_round_plot'],
		facecolor='red', alpha=0.25
	)
	
	ax06to07.errorbar(borehole_analysis_dict['raw_borehole_measurements']['T'], borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_mean_k_int_z_T'], xerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_T'], yerr=borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['sigma_R_mean_k_int_z_T'], fmt='.', color='black', markeredgewidth=0.1)
	for layer_index in range(number_layers):
		x = 'layer'+str(layer_index)
		ax06to07.plot(
			borehole_analysis_dict[int_option]['individual_layers'][x]['T'],
			borehole_analysis_dict[int_option]['individual_layers'][x]['bullard_method']['mean_k']['R_versus_T']['invq'] * borehole_analysis_dict[int_option]['individual_layers'][x]['T'] + borehole_analysis_dict[int_option]['individual_layers'][x]['bullard_method']['mean_k']['R_versus_T']['invc'],
			color='red', zorder=5
		)
	
	ax08.plot(borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['R_versus_T']['Q_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='red')
	ax08.fill_betweenx(
		borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['R_versus_T']['Q_round_plot']-borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['R_versus_T']['sigma_Q_round_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['R_versus_T']['Q_round_plot']+borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['R_versus_T']['sigma_Q_round_plot'],
		facecolor='red', alpha=0.25
	)
	



	# Plot estimates of Q from geometric mean
	if plot_strat_option == 'yes':
		ax2_10 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column10'], plot_format_dict['row_fractional_ypos_dict']['row2'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], ylabel=r'$z$ / m')
		ax2_10.set_ylim(top=0, bottom=max_depth_m_plot)
		ax2_10.set_xlim(left=0, right=1)
		ax2_10.yaxis.set_ticks_position('both')
		ax2_10.xaxis.set_ticks_position('none')
		ax2_10.xaxis.set_minor_locator(MultipleLocator(5))
		ax2_10.yaxis.set_minor_locator(MultipleLocator(50))
		ax2_10.xaxis.set_major_formatter(FormatStrFormatter(''))
		ax2_11 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column11'], plot_format_dict['row_fractional_ypos_dict']['row2'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$k$ / W m$^{-1}$ K$^{-1}$')
		ax2_11.yaxis.set_major_formatter(FormatStrFormatter(''))
	else:	
		ax2_11 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column11'], plot_format_dict['row_fractional_ypos_dict']['row2'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$k$ / W m$^{-1}$ K$^{-1}$', ylabel=r'$z$ / m')
	
	ax2_11.set_ylim(top=min_depth_m, bottom=max_depth_m)
	# ax2_11.set_xlim(left=min_k, right=max_k)
	ax2_11.xaxis.set_label_position('top')
	ax2_11.yaxis.set_ticks_position('both')
	ax2_11.xaxis.set_ticks_position('both')
	ax2_11.xaxis.set_minor_locator(MultipleLocator(0.5))
	ax2_11.yaxis.set_minor_locator(MultipleLocator(50))
	
	ax2_12 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column12'], plot_format_dict['row_fractional_ypos_dict']['row2'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$R$ / K m$^2$ W$^{-1}$')
	ax2_12.set_ylim(top=min_depth_m, bottom=max_depth_m)
	ax2_12.xaxis.set_label_position('top')
	ax2_12.yaxis.set_ticks_position('both')
	ax2_12.xaxis.set_ticks_position('both')
	ax2_12.xaxis.set_minor_locator(MultipleLocator(25))
	ax2_12.yaxis.set_minor_locator(MultipleLocator(50))
	ax2_12.yaxis.set_major_formatter(FormatStrFormatter(''))
	
	ax2_13 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column13'], plot_format_dict['row_fractional_ypos_dict']['row2'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$T$ / $^{\circ}$C')
	ax2_13.set_ylim(top=min_depth_m, bottom=max_depth_m)
	ax2_13.xaxis.set_label_position('top')
	ax2_13.yaxis.set_ticks_position('both')
	ax2_13.xaxis.set_ticks_position('both')
	ax2_13.xaxis.set_minor_locator(MultipleLocator(1))
	ax2_13.yaxis.set_minor_locator(MultipleLocator(50))
	ax2_13.yaxis.set_major_formatter(FormatStrFormatter(''))
	

	
	ax1_11to1_12 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column11'], plot_format_dict['row_fractional_ypos_dict']['row1'], 2*plot_format_dict['column_fractional_width'] + plot_format_dict['column_fractional_spacing'], plot_format_dict['row_fractional_height']], xlabel=r'$R$ / K m$^2$ W$^{-1}$', ylabel='$T$ / $^{\circ}$C')
	# ax1_11to1_12.set_ylim(top=0, bottom=max_depth_m_plot)
	# ax1_11to1_12.set_xlim(left=0, right=1)
	ax1_11to1_12.xaxis.set_label_position('top')
	ax1_11to1_12.yaxis.set_ticks_position('both')
	ax1_11to1_12.xaxis.set_ticks_position('both')
	ax1_11to1_12.invert_yaxis()
	# ax1_11to1_12.xaxis.set_minor_locator(MultipleLocator(5))
	# ax1_11to1_12.yaxis.set_minor_locator(MultipleLocator(50))
	# ax1_11to1_12.yaxis.set_major_formatter(FormatStrFormatter(''))
	# ax1_11to1_12.xaxis.set_major_formatter(FormatStrFormatter(''))
	
	ax23 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column13'], plot_format_dict['row_fractional_ypos_dict']['row1'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$Q$ / mW m$^{-2}$', ylabel=r'$z$ / m')
	ax23.set_xlim(left=35, right=51)
	ax23.set_ylim(top=min_depth_m, bottom=max_depth_m)
	ax23.xaxis.set_label_position('top')
	ax23.yaxis.set_label_position('right')
	ax23.yaxis.set_ticks_position('right')
	ax23.xaxis.set_ticks_position('both')
	ax23.xaxis.set_minor_locator(MultipleLocator(1))
	ax23.yaxis.set_minor_locator(MultipleLocator(50))
	# ax23.yaxis.set_major_formatter(FormatStrFormatter(50))
	
	ax0_11to0_12 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column11'], plot_format_dict['row_fractional_ypos_dict']['row0'], 2*plot_format_dict['column_fractional_width'] + plot_format_dict['column_fractional_spacing'], plot_format_dict['row_fractional_height']], xlabel='$T$ / $^{\circ}$C', ylabel=r'$R$ / K m$^2$ W$^{-1}$')
	# ax0_11to0_12.set_ylim(top=0, bottom=max_depth_m_plot)
	# ax0_11to0_12.set_xlim(left=0, right=1)
	ax0_11to0_12.xaxis.set_label_position('top')
	ax0_11to0_12.yaxis.set_ticks_position('both')
	ax0_11to0_12.xaxis.set_ticks_position('both')
	ax0_11to0_12.invert_yaxis()
	# ax0_11to0_12.xaxis.set_minor_locator(MultipleLocator(5))
	# ax0_11to0_12.yaxis.set_minor_locator(MultipleLocator(50))
	# ax0_11to0_12.yaxis.set_major_formatter(FormatStrFormatter(''))
	# ax0_11to0_12.xaxis.set_major_formatter(FormatStrFormatter(''))
	
	ax0_13 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column13'], plot_format_dict['row_fractional_ypos_dict']['row0'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$Q$ / mW m$^{-2}$', ylabel=r'$z$ / m')
	ax0_13.set_xlim(left=35, right=51)
	ax0_13.set_ylim(top=min_depth_m, bottom=max_depth_m)
	ax0_13.xaxis.set_label_position('top')
	ax0_13.yaxis.set_label_position('right')
	ax0_13.yaxis.set_ticks_position('right')
	ax0_13.xaxis.set_ticks_position('both')
	ax0_13.xaxis.set_minor_locator(MultipleLocator(1))
	ax0_13.yaxis.set_minor_locator(MultipleLocator(50))
	# ax0_13.yaxis.set_major_formatter(FormatStrFormatter(50))
	
	# Plot data
	if plot_strat_option == 'yes':
		for layer_index in range(number_layers):
			
			ax2_10.axhspan(
				ymin = borehole_analysis_dict[int_option]['layers_overview']['z0_int'][layer_index],
				ymax = borehole_analysis_dict[int_option]['layers_overview']['z1_int'][layer_index],
				color = lith_plot_format_dict[borehole_analysis_dict[int_option]['layers_overview']['int_lith_type'][layer_index]],
				alpha = 0.5
			)
			ax2_10.axhline(
			linewidth=2,
			y = borehole_analysis_dict[int_option]['layers_overview']['z1_int'][layer_index],
			color = 'black'
			)
			ax2_10.text(
				0.5,
				(borehole_analysis_dict[int_option]['layers_overview']['z0_int'][layer_index] + borehole_analysis_dict[int_option]['layers_overview']['z1_int'][layer_index]) / 2,
				borehole_analysis_dict[int_option]['layers_overview']['int_lith_name'][layer_index],
				verticalalignment='center',
				horizontalalignment='center'
			)
	
	ax2_11.plot(borehole_analysis_dict[int_option]['layers_overview']['mean_k_int_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='green')
	ax2_11.fill_betweenx(
		borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['mean_k_int_plot']-borehole_analysis_dict[int_option]['layers_overview']['std_mean_k_int_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['mean_k_int_plot']+borehole_analysis_dict[int_option]['layers_overview']['std_mean_k_int_plot'],
		facecolor='green', alpha=0.25
	)
	ax2_11.errorbar(borehole_analysis_dict['raw_borehole_measurements']['k'], borehole_analysis_dict['raw_borehole_measurements']['z_k'], xerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_k'], yerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_z_k'], fmt='.k', markeredgecolor='k', zorder=5)
	
	ax2_12.fill_betweenx(
		borehole_analysis_dict['raw_borehole_measurements']['z_T'],
		borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_mean_k_int_z_T']-borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['sigma_R_mean_k_int_z_T'],
		borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_mean_k_int_z_T']+borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['sigma_R_mean_k_int_z_T'],
		facecolor='green', alpha=0.25
	)
	ax2_12.plot(borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_mean_k_int_z_T'], borehole_analysis_dict['raw_borehole_measurements']['z_T'], color='green')
	
	ax2_13.errorbar(borehole_analysis_dict['raw_borehole_measurements']['T'], borehole_analysis_dict['raw_borehole_measurements']['z_T'], xerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_T'], yerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_z_T'], fmt='.k', color='k', markeredgewidth=0.1)
	
	ax1_11to1_12.errorbar(borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_mean_k_int_z_T'],
	borehole_analysis_dict['raw_borehole_measurements']['T'],
	xerr=borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['sigma_R_mean_k_int_z_T'],
	yerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_T'], fmt='.', color='black', markeredgewidth=0.1)
	for layer_index in range(number_layers):
		x = 'layer'+str(layer_index)
		ax1_11to1_12.plot(
			borehole_analysis_dict[int_option]['individual_layers'][x]['R_mean_k_int_z_T'],
			borehole_analysis_dict[int_option]['individual_layers'][x]['bullard_method']['mean_k']['T_versus_R']['q'] * borehole_analysis_dict[int_option]['individual_layers'][x]['R_mean_k_int_z_T'] + borehole_analysis_dict[int_option]['individual_layers'][x]['bullard_method']['mean_k']['T_versus_R']['c'],
			color='green', zorder=5
		)
	
	ax23.plot(borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['T_versus_R']['Q_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='green')
	ax23.fill_betweenx(
		borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['T_versus_R']['Q_round_plot']-borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['T_versus_R']['sigma_Q_round_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['T_versus_R']['Q_round_plot']+borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['T_versus_R']['sigma_Q_round_plot'],
		facecolor='green', alpha=0.25
	)
	
	ax0_11to0_12.errorbar(borehole_analysis_dict['raw_borehole_measurements']['T'], borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_mean_k_int_z_T'], xerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_T'], yerr=borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['sigma_R_mean_k_int_z_T'], fmt='.', color='black', markeredgewidth=0.1)
	for layer_index in range(number_layers):
		x = 'layer'+str(layer_index)
		ax0_11to0_12.plot(
			borehole_analysis_dict[int_option]['individual_layers'][x]['T'],
			borehole_analysis_dict[int_option]['individual_layers'][x]['bullard_method']['mean_k']['R_versus_T']['invq'] * borehole_analysis_dict[int_option]['individual_layers'][x]['T'] + borehole_analysis_dict[int_option]['individual_layers'][x]['bullard_method']['mean_k']['R_versus_T']['invc'],
			color='green', zorder=5
		)
	
	ax0_13.plot(borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['R_versus_T']['Q_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='green')
	ax0_13.fill_betweenx(
		borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['R_versus_T']['Q_round_plot']-borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['R_versus_T']['sigma_Q_round_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['R_versus_T']['Q_round_plot']+borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['R_versus_T']['sigma_Q_round_plot'],
		facecolor='green', alpha=0.25
	)






	# Plot estimates of Q from harmonic mean
	if plot_strat_option == 'yes':
		ax2_15 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column15'], plot_format_dict['row_fractional_ypos_dict']['row2'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], ylabel=r'$z$ / m')
		ax2_15.set_ylim(top=0, bottom=max_depth_m_plot)
		ax2_15.set_xlim(left=0, right=1)
		ax2_15.yaxis.set_ticks_position('both')
		ax2_15.xaxis.set_ticks_position('none')
		ax2_15.xaxis.set_minor_locator(MultipleLocator(5))
		ax2_15.yaxis.set_minor_locator(MultipleLocator(50))
		ax2_15.xaxis.set_major_formatter(FormatStrFormatter(''))
		ax2_16 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column16'], plot_format_dict['row_fractional_ypos_dict']['row2'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$k$ / W m$^{-1}$ K$^{-1}$')
		ax2_16.yaxis.set_major_formatter(FormatStrFormatter(''))
	else:	
		ax2_16 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column16'], plot_format_dict['row_fractional_ypos_dict']['row2'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$k$ / W m$^{-1}$ K$^{-1}$', ylabel=r'$z$ / m')
	
	ax2_16.set_ylim(top=min_depth_m, bottom=max_depth_m)
	# ax2_16.set_xlim(left=min_k, right=max_k)
	ax2_16.xaxis.set_label_position('top')
	ax2_16.yaxis.set_ticks_position('both')
	ax2_16.xaxis.set_ticks_position('both')
	ax2_16.xaxis.set_minor_locator(MultipleLocator(0.5))
	ax2_16.yaxis.set_minor_locator(MultipleLocator(50))
	
	ax2_17 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column17'], plot_format_dict['row_fractional_ypos_dict']['row2'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$R$ / K m$^2$ W$^{-1}$')
	ax2_17.set_ylim(top=min_depth_m, bottom=max_depth_m)
	ax2_17.xaxis.set_label_position('top')
	ax2_17.yaxis.set_ticks_position('both')
	ax2_17.xaxis.set_ticks_position('both')
	ax2_17.xaxis.set_minor_locator(MultipleLocator(25))
	ax2_17.yaxis.set_minor_locator(MultipleLocator(50))
	ax2_17.yaxis.set_major_formatter(FormatStrFormatter(''))
	
	ax2_18 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column18'], plot_format_dict['row_fractional_ypos_dict']['row2'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$T$ / $^{\circ}$C')
	ax2_18.set_ylim(top=min_depth_m, bottom=max_depth_m)
	ax2_18.xaxis.set_label_position('top')
	ax2_18.yaxis.set_ticks_position('both')
	ax2_18.xaxis.set_ticks_position('both')
	ax2_18.xaxis.set_minor_locator(MultipleLocator(1))
	ax2_18.yaxis.set_minor_locator(MultipleLocator(50))
	ax2_18.yaxis.set_major_formatter(FormatStrFormatter(''))
	

	
	ax1_16to1_17 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column16'], plot_format_dict['row_fractional_ypos_dict']['row1'], 2*plot_format_dict['column_fractional_width'] + plot_format_dict['column_fractional_spacing'], plot_format_dict['row_fractional_height']], xlabel=r'$R$ / K m$^2$ W$^{-1}$', ylabel='$T$ / $^{\circ}$C')
	# ax1_16to1_17.set_ylim(top=0, bottom=max_depth_m_plot)
	# ax1_16to1_17.set_xlim(left=0, right=1)
	ax1_16to1_17.xaxis.set_label_position('top')
	ax1_16to1_17.yaxis.set_ticks_position('both')
	ax1_16to1_17.xaxis.set_ticks_position('both')
	ax1_16to1_17.invert_yaxis()
	# ax1_16to1_17.xaxis.set_minor_locator(MultipleLocator(5))
	# ax1_16to1_17.yaxis.set_minor_locator(MultipleLocator(50))
	# ax1_16to1_17.yaxis.set_major_formatter(FormatStrFormatter(''))
	# ax1_16to1_17.xaxis.set_major_formatter(FormatStrFormatter(''))
	
	ax1_18 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column18'], plot_format_dict['row_fractional_ypos_dict']['row1'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$Q$ / mW m$^{-2}$', ylabel=r'$z$ / m')
	ax1_18.set_xlim(left=35, right=51)
	ax1_18.set_ylim(top=min_depth_m, bottom=max_depth_m)
	ax1_18.xaxis.set_label_position('top')
	ax1_18.yaxis.set_label_position('right')
	ax1_18.yaxis.set_ticks_position('right')
	ax1_18.xaxis.set_ticks_position('both')
	ax1_18.xaxis.set_minor_locator(MultipleLocator(1))
	ax1_18.yaxis.set_minor_locator(MultipleLocator(50))
	# ax1_18.yaxis.set_major_formatter(FormatStrFormatter(50))
	
	ax0_16to0_17 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column16'], plot_format_dict['row_fractional_ypos_dict']['row0'], 2*plot_format_dict['column_fractional_width'] + plot_format_dict['column_fractional_spacing'], plot_format_dict['row_fractional_height']], xlabel='$T$ / $^{\circ}$C', ylabel=r'$R$ / K m$^2$ W$^{-1}$')
	# ax0_16to0_17.set_ylim(top=0, bottom=max_depth_m_plot)
	# ax0_16to0_17.set_xlim(left=0, right=1)
	ax0_16to0_17.xaxis.set_label_position('top')
	ax0_16to0_17.yaxis.set_ticks_position('both')
	ax0_16to0_17.xaxis.set_ticks_position('both')
	ax0_16to0_17.invert_yaxis()
	# ax0_16to0_17.xaxis.set_minor_locator(MultipleLocator(5))
	# ax0_16to0_17.yaxis.set_minor_locator(MultipleLocator(50))
	# ax0_16to0_17.yaxis.set_major_formatter(FormatStrFormatter(''))
	# ax0_16to0_17.xaxis.set_major_formatter(FormatStrFormatter(''))
	
	ax0_18 = fig.add_axes([plot_format_dict['column_fractional_xpos_dict']['column18'], plot_format_dict['row_fractional_ypos_dict']['row0'], plot_format_dict['column_fractional_width'], plot_format_dict['row_fractional_height']], xlabel=r'$Q$ / mW m$^{-2}$', ylabel=r'$z$ / m')
	ax0_18.set_xlim(left=35, right=51)
	ax0_18.set_ylim(top=min_depth_m, bottom=max_depth_m)
	ax0_18.xaxis.set_label_position('top')
	ax0_18.yaxis.set_label_position('right')
	ax0_18.yaxis.set_ticks_position('right')
	ax0_18.xaxis.set_ticks_position('both')
	ax0_18.xaxis.set_minor_locator(MultipleLocator(1))
	ax0_18.yaxis.set_minor_locator(MultipleLocator(50))
	# ax0_18.yaxis.set_major_formatter(FormatStrFormatter(50))
	
	# Plot data
	if plot_strat_option == 'yes':
		for layer_index in range(number_layers):
			
			ax2_15.axhspan(
				ymin = borehole_analysis_dict[int_option]['layers_overview']['z0_int'][layer_index],
				ymax = borehole_analysis_dict[int_option]['layers_overview']['z1_int'][layer_index],
				color = lith_plot_format_dict[borehole_analysis_dict[int_option]['layers_overview']['int_lith_type'][layer_index]],
				alpha = 0.5
			)
			ax2_15.axhline(
			linewidth=2,
			y = borehole_analysis_dict[int_option]['layers_overview']['z1_int'][layer_index],
			color = 'black'
			)
			ax2_15.text(
				0.5,
				(borehole_analysis_dict[int_option]['layers_overview']['z0_int'][layer_index] + borehole_analysis_dict[int_option]['layers_overview']['z1_int'][layer_index]) / 2,
				borehole_analysis_dict[int_option]['layers_overview']['int_lith_name'][layer_index],
				verticalalignment='center',
				horizontalalignment='center'
			)
	
	ax2_16.plot(borehole_analysis_dict[int_option]['layers_overview']['mean_k_int_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='blue')
	ax2_16.fill_betweenx(
		borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['mean_k_int_plot']-borehole_analysis_dict[int_option]['layers_overview']['std_mean_k_int_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['mean_k_int_plot']+borehole_analysis_dict[int_option]['layers_overview']['std_mean_k_int_plot'],
		facecolor='blue', alpha=0.25
	)
	ax2_16.errorbar(borehole_analysis_dict['raw_borehole_measurements']['k'], borehole_analysis_dict['raw_borehole_measurements']['z_k'], xerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_k'], yerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_z_k'], fmt='.k', markeredgecolor='k', zorder=5)
	
	ax2_17.fill_betweenx(
		borehole_analysis_dict['raw_borehole_measurements']['z_T'],
		borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_mean_k_int_z_T']-borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['sigma_R_mean_k_int_z_T'],
		borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_mean_k_int_z_T']+borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['sigma_R_mean_k_int_z_T'],
		facecolor='blue', alpha=0.25
	)
	ax2_17.plot(borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_mean_k_int_z_T'], borehole_analysis_dict['raw_borehole_measurements']['z_T'], color='blue')
	
	ax2_18.errorbar(borehole_analysis_dict['raw_borehole_measurements']['T'], borehole_analysis_dict['raw_borehole_measurements']['z_T'], xerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_T'], yerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_z_T'], fmt='.k', color='k', markeredgewidth=0.1)
	
	ax1_16to1_17.errorbar(borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_mean_k_int_z_T'],
	borehole_analysis_dict['raw_borehole_measurements']['T'],
	xerr=borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['sigma_R_mean_k_int_z_T'],
	yerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_T'], fmt='.', color='black', markeredgewidth=0.1)
	for layer_index in range(number_layers):
		x = 'layer'+str(layer_index)
		ax1_16to1_17.plot(
			borehole_analysis_dict[int_option]['individual_layers'][x]['R_mean_k_int_z_T'],
			borehole_analysis_dict[int_option]['individual_layers'][x]['bullard_method']['mean_k']['T_versus_R']['q'] * borehole_analysis_dict[int_option]['individual_layers'][x]['R_mean_k_int_z_T'] + borehole_analysis_dict[int_option]['individual_layers'][x]['bullard_method']['mean_k']['T_versus_R']['c'],
			color='blue', zorder=5
		)
	
	ax1_18.plot(borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['T_versus_R']['Q_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='blue')
	ax1_18.fill_betweenx(
		borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['T_versus_R']['Q_round_plot']-borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['T_versus_R']['sigma_Q_round_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['T_versus_R']['Q_round_plot']+borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['T_versus_R']['sigma_Q_round_plot'],
		facecolor='blue', alpha=0.25
	)
	
	ax0_16to0_17.errorbar(borehole_analysis_dict['raw_borehole_measurements']['T'], borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['R_mean_k_int_z_T'], xerr=borehole_analysis_dict['raw_borehole_measurements']['sigma_T'], yerr=borehole_analysis_dict['whole_borehole_calculations'][int_option]['thermal_resistance']['sigma_R_mean_k_int_z_T'], fmt='.', color='black', markeredgewidth=0.1)
	for layer_index in range(number_layers):
		x = 'layer'+str(layer_index)
		ax0_16to0_17.plot(
			borehole_analysis_dict[int_option]['individual_layers'][x]['T'],
			borehole_analysis_dict[int_option]['individual_layers'][x]['bullard_method']['mean_k']['R_versus_T']['invq'] * borehole_analysis_dict[int_option]['individual_layers'][x]['T'] + borehole_analysis_dict[int_option]['individual_layers'][x]['bullard_method']['mean_k']['R_versus_T']['invc'],
			color='blue', zorder=5
		)
	
	ax0_18.plot(borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['R_versus_T']['Q_round_plot'], borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'], color='blue')
	ax0_18.fill_betweenx(
		borehole_analysis_dict[int_option]['layers_overview']['z_int_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['R_versus_T']['Q_round_plot']-borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['R_versus_T']['sigma_Q_round_plot'],
		borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['R_versus_T']['Q_round_plot']+borehole_analysis_dict[int_option]['layers_overview']['bullard_method']['mean_k']['R_versus_T']['sigma_Q_round_plot'],
		facecolor='blue', alpha=0.25
	)
	

	
	
	fig.savefig(figure_name, bbox_inches='tight', transparent=True)
	plt.close(fig)
	
	
	


	return()
