def plot_stratigraphic_conductivity(borehole_analysis_dict, int_option, lith_plot_format_dict, max_depth_m_plot, plot_label_dict, figure_name, plot_title, figures_path):
		
	heat_flow_estimation_method = 'interval_method'
	figure_name, plot_title = set_up_plot_title(borehole_analysis_dict, int_option, heat_flow_estimation_method, plot_label_dict, figures_path)
	
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