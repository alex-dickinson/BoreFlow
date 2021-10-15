# Numerical packages
import numpy as np
import pandas as pd


### Set up paths for borehole data
boreholes_dir = "/Users/alex/Documents/newcastle_postdoc/nelep_heat_flow_project/nelep_heat_flow_data/boreholes/standardised_borehole_records"
all_boreholes_spreadsheet_dir = str(boreholes_dir) + "/ne_england_heat_flow_boreholes_manual_entry"
# Northern England
all_boreholes_overview_file = str(all_boreholes_spreadsheet_dir) + "/ne_england_heat_flow_boreholes_manual_entry.xlsx"
# NELEP region
all_boreholes_overview_file = str(all_boreholes_spreadsheet_dir) + "/ne_england_heat_flow_boreholes_manual_entry.xlsx"



all_borehole_overview_df = pd.read_excel(all_boreholes_overview_file, comment="%!%#", mangle_dupe_cols=True, usecols=["name","file_name","os_grid_code","ukng_easting","ukng_northing","temperature?","type_temp","number_uncorrected_temp_measurements","max_depth_temp(m)","conductivity?","max_depth_conductivities(m)","number_conductivity_measurements","max_depth_conductivities(m)","year_drilling_completed","year_temp_measurements","number_strat_interps","strat_interp1","strat_interp1_ext","strat_interp2","strat_interp2_ext","strat_interp3","strat_interp3_ext","strat_interp4","strat_interp4_ext","strat_interp5","strat_interp5_ext"])

print(len(all_borehole_overview_df['name']))
print(all_borehole_overview_df['name'].nunique())
