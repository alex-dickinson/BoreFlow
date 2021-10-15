#!/bin/bash

# ### Arrange into csv with lon, lat, whether borehole has T, whether borehole has k. One line for each borehole, even if several measurements taken. Do not differentiate between different kinds of temperature measurement. Lon and lat are in UKNG eastings and northings
# sort_for_plotting_lon_lat_T_k() {
# 	# Remove rows with % in first column - shows that not to be used. Remove rows belonging to same borehole
# 	cat $spreadsheet_csv.csv | awk 'BEGIN{FS=","}{if ($1!="%" && NR>1) print $0}' | sort -u -t, -k2,2 | awk 'BEGIN{FS=","}{OFS=","}{print $6,$7,$2,$12,$19}' > easting_northing_T_k_temp.txt
# 	# Convert UKNG to lonlat
# 	python ukng_to_lonlat.py easting_northing_T_k_temp.txt lon_lat_T_k_temp.txt
#
#
#
# 	# cat $spreadsheet_csv | sort -u -t, -k1,1 > test.csv
# }
#
#
# # Spreadsheet contains columns Commented,Name,latitude,longitude,os_grid_code,ukng_easting,ukng_northing,source_of_my_temperature_values,source_of_my_conductivity_values,other_sources,max_depth_borehole(m),temperature?,burley1984type,rollin1987_temp_type,type_temp,number_uncorrected_temp_measurements,number_corrected_temperature_measurements,max_depth_temp(m),conductivity?,number_conductivity_measurements,max_depth_conductivities(m),Year of drilling,Year_of_measurements,Name,L19_name,SOBI_name,SOBI_code,B84_name,BGS Ref (ID),BGS scan link,UKOGL_name,UKOGL_code,UKOGL_well_tops,OGA_code,TGS_name,CGG_name,Have well logs?,B84_k_HF_nT_listed_source,B84_nk,B84_nT_listed,B84_HF(mW/m2),B84_Tsource,B84_HF_to_plot(mW/m2),Notes,Name,,,,,,,,,,,,,,,,,,,,,,
#
# spreadsheet_csv=ne_england_heat_flow_boreholes_manual_entry
#
#
# sort_for_plotting_lon_lat_T_k



