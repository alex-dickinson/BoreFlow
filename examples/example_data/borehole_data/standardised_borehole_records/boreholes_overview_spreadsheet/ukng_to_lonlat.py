import os
from sys import argv

import numpy as np
import pandas as pd
import geopandas as gpd
import fiona
from shapely.geometry import Point, Polygon


script, infile, outfile = argv

### Read ascii file with UKNG easting in column 1 and UKNG northing in column 2, name in column 3, temp? in column 4, cond? in column 5
df_ukng = pd.read_csv(infile, names=['easting','northing','name','temp?','cond?'])

# Add geometry and convert to spatial dataframe in source CRS (UKNG in this case. Code is EPSG:27700)
df_ukng['geometry'] = list(zip(df_ukng['easting'], df_ukng['northing']))
df_ukng['geometry'] = df_ukng['geometry'].apply(Point)
gdf_ukng = gpd.GeoDataFrame(df_ukng, geometry='geometry', crs='EPSG:27700')

# Reproject data from UKNG coordinates to WGS84 (code is EPSG:4326)
gdf_wgs84 = gdf_ukng.to_crs('EPSG:4326')

### Set latitude and longitude as new columns
gdf_wgs84['longitude'] = gdf_wgs84['geometry'].x
gdf_wgs84['latitude'] = gdf_wgs84['geometry'].y

### Convert geodataframe to dataframe and drop geometry column
df_wgs84 = pd.DataFrame(gdf_wgs84)
df_wgs84.drop(columns=['geometry'], axis=1, inplace=True)

### Save as csv file
df_wgs84.to_csv(str(outfile), index=False, header=False, sep='\t')