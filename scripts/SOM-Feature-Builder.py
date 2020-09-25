"""Preprocess RGA Phase 2 table and GIS data to build SOM features."""

import os
import sys

import numpy
import pandas
from osgeo import gdal 
# GIS processing module developed by Natural Capital Project
# https://github.com/natcap/pygeoprocessing
import pygeoprocessing
import TaskGraph
# Modules from D. Denu repository at EPSCoR
import utils
import geospatialutils as geoutils

## Set up data directory paths
# Directory where ALL files will be saved, if it does not exist it is created
# IMPORTANT: Existing files will be overwritten
workspace_dir = os.path.join(
    "C:", os.sep, "Users", "ddenu", "Workspace", "UVM", "feature_build_winooski")
                                
if not os.path.isdir(workspace_dir):
    os.mkdir(workspace_dir)

# Directory path where most data can be found, a convenience path if most 
# data is stored in one place
#data_dir_path = os.path.join(
#    "C:", os.sep, "Users", "ddenu", "Desktop", "Preprocess Data Workspace")
#data_dir_path = os.path.join(
#    "C:", os.sep, "Users", "ddenu", "Workspace", "UVM", "DougsFINALCode", 
#    "DougsFINALCode", "Preprocess Data Workspace", "Preprocess Data Workspace")
data_dir_path = os.path.join(
    "C:", os.sep, "Users", "ddenu", "Workspace", "UVM", "WinooskiCode3", 
    "WinooskiCode3", "Preprocess Data Workspace", "Preprocess Data Workspace")
 
# Convenience directory path if tabular RGA data is located in a single folder
tabular_data_path = os.path.join(data_dir_path, 'Tabular', 'RGA')


# Path to data excel file with reach segments and variables
# Download from https://anrweb.vt.gov/DEC/SGA/projects/exports/combined.aspx
#rga_data_path = os.path.join(
#    tabular_data_path, 'Missisquoi', 'Missisquoi_P1P2_All_Miss_Projects.xls')
rga_data_path = os.path.join(
    tabular_data_path, 'Winooski', 'Winooski_P1P2_All_Wino_Projects.xls')
# Define Digital Elevation Map (DEM) Raster Path
dem_path = os.path.join(
    data_dir_path, 'Geospatial', 'ElevationDEM_VTHYDRODEM', 'vthydrodem', 
    'hdr.adf')

# Where the project workbooks are located for the watershed of interest
#workbook_parent_dir = os.path.join(
#    tabular_data_path, 'Missisquoi', 'Workbooks')
workbook_parent_dir = os.path.join(
    tabular_data_path, 'Winooski', 'Workbooks')
# Path to a shapefile of the watershed of interest.
# IMPORTANT: The projection or coordinate system of this shapefile is what 
# will be used for all outputs. Should be projected in meters, ideally using 
# Vermont Stateplane coordinate system
#watershed_path = os.path.join(
#    data_dir_path, 'Geospatial', 'Missisquoi', 'Missisquoi_Watershed.shp')
watershed_path = os.path.join(
    data_dir_path, 'Geospatial', 'Winooski', 'Winooski_Watershed.shp')

# Path for Assessed Phase 2 shapefile
segment_line_path = os.path.join(
    data_dir_path, 'Geospatial', 'SGA_Phase_2_Assessed_Reaches', 
    'SGA_Phase_2_Assessed_Reaches.shp')
# Path for Vermont hydrography shapefile
hydrography_stream_path = os.path.join(
    data_dir_path, 'Geospatial', 
    'VT_Hydrography_Dataset_cartographic_extract_lines', 
    'VT_Hydrography_Dataset__cartographic_extract_lines.shp')

## Collect workbook pebble count data for Phase 2 segments 
# IDs added are 'd16', 'd35', 'd50', 'd84', 'SGAT_ID'
# Path to save pebble counts
pebble_count_path = os.path.join(workspace_dir, 'pebble_count.csv')
print('Getting Pebble Counts')
# Get pebble counts
utils.get_pebble_counts(
    workbook_parent_dir, pebble_count_path, 'Segment Pebble Counts',
    rga_data_path)

## Get elevation difference in order to get slopes for Phase 2 segments
# Get dem raster info, inluding nodata value
dem_info = pygeoprocessing.get_raster_info(dem_path)
dem_nodata = dem_info['nodata'][0]
# Output path for DEM converted to meter elevation values
dem_meters_path = os.path.join(workspace_dir, 'dem_meters.tif')
def cm_to_meters(in_array):
    return numpy.where(in_array!=dem_nodata, in_array * 0.01, dem_nodata)

# Convert DEM pixel values from centimeters units (cm) to (meters)
# Metadata Link: http://maps.vcgi.vermont.gov/gisdata/metadata/ElevationDEM_VTHYDRODEM.htm
# If elevation height values are already in meters, uncomment line below and 
# comment the 3 lines below that!
# dem_meters_path = dem_path
pygeoprocessing.raster_calculator(
    [(dem_path, 1)], cm_to_meters, dem_meters_path, gdal.GDT_Float32, 
    dem_nodata)

# Get shapefile info such as the projection
watershed_info = pygeoprocessing.get_vector_info(watershed_path)
# Use this projection going forward for all GIS data
project_projection = watershed_info['projection_wkt']

# Output path for projected phase 2 assessed shapefile
segment_line_projected_path = os.path.join(
    workspace_dir, 'Phase2_Reaches_Projcted.shp')
# Output path for projected hydrography shapefile
hydrography_stream_projected_path = os.path.join(
        workspace_dir, 'hydrography_streams_projected.shp')
# Reproject the phase 2 and hydrography shapefiles to the watershed 
# coordinate projection                                 
pygeoprocessing.reproject_vector(
    segment_line_path, project_projection, segment_line_projected_path)
pygeoprocessing.reproject_vector(
    hydrography_stream_path, project_projection, 
    hydrography_stream_projected_path)

# Output path for phase 2 reaches with elevation difference
reach_slopes_path = os.path.join(
    workspace_dir, 'Phase_2_Assessed_Reaches_Slopes.shp')
# Output csv path for the elevation differences
elev_diff_csv_path = os.path.join(workspace_dir, 'elevation_diff.csv')
# These attributes are saved to the CSV file: 'SGAT_ID', 'ElevDiff'
print('Calculating Elevation Differences')
geoutils.calculate_line_slope_from_dem(
    segment_line_projected_path, dem_meters_path, reach_slopes_path, 
    'SGAT_ID', elev_diff_csv_path,  aoi_path=watershed_path)

## Calculate slope from elevation differences and segment length
# Read in main RGA data from excel file into a pandas dataframe
rga_df = pandas.read_excel(rga_data_path)
# Output path to save slope values in csv
slopes_path = os.path.join(workspace_dir, 'slopes.csv')
# Read in the elevation differences csv
elev_diff_df = pandas.read_csv(elev_diff_csv_path, index_col=False)
# Only get the columns we need
elev_sub_diff_df = elev_diff_df.loc[:, ('SGAT_ID', 'ElevDiff')]
# Get the ID and segment_length columns from RGA main table
segment_length_df = rga_df.loc[:,('SGAT_PID_P2', 'segment_length')]
# Convert segment length to meters from feet
segment_length_df['segment_length_m'] = segment_length_df['segment_length'] * 0.3048
# Merge the two dataframes into one using the proper ID fields to join on
slope_df = segment_length_df.merge(
    elev_sub_diff_df, how='left', left_on='SGAT_PID_P2', right_on='SGAT_ID')
# Calculate slope as rise over run, elevation difference over segment length
slope_df['slope'] = slope_df['ElevDiff'] / slope_df['segment_length_m']
# Save the dataframe and slope values to a csv
slope_df.to_csv(slopes_path)

## Calculate Drainage Area (DA)
# Output path for drainage area csv
darea_csv_path = os.path.join(workspace_dir, 'drainage_area.csv')
# Output path for downstream pour point shapefile
pour_points_path = os.path.join(workspace_dir, 'downstream_points.shp')
# Create pour points at the downstream end of the reaches
geoutils.create_downstream_point_vector(
    segment_line_projected_path, dem_meters_path, pour_points_path, 'SGAT_ID')
# A 'DA' attribute is added to CSV output
print('Calculate Drainage Area')
# Calculate drainage area
geoutils.calculate_drainage_area(
    workspace_dir, dem_meters_path, watershed_path, 
    hydrography_stream_projected_path, darea_csv_path, pour_points_path)

## Calculate VC and VC Ratio 
# Output path for csv with upstream IDs
upstream_csv_path = os.path.join(workspace_dir, 'upstream_ids.csv')
print('Calculating Upstream IDs')
utils.calculate_upstream_id(rga_data_path, 'SGAT_PID_P2', upstream_csv_path)

# Output path for VC and VC Ratio csv
vc_csv_path = os.path.join(workspace_dir, 'vc_ratio.csv')
# The following attributes are added to the csv: 'VC', 'VC_RATIO'
print('Calculating VC, VC Ratio')
utils.calculate_vc_vcratio(
    upstream_csv_path, 'SGAT_PID_P2', 'upstream_ID', vc_csv_path)

## Calculate Streampower and SSP Balance

# Output path for the SSP csv and SSP Balance csv
ssp_path = os.path.join(workspace_dir, 'ssp.csv')
ssp_balance_path = os.path.join(workspace_dir, 'ssp_bal.csv')
# Calculate Stream Power and Stream Power Balance
print('Calculating SSP and SSP Balance')
utils.calculate_streampower(
    ssp_path, darea_csv_path, slopes_path, rga_data_path)
utils.calculate_streampower_balance(
    ssp_balance_path, ssp_path, upstream_csv_path)

## Compute IR, ER, WtoD, pARMOR, nBars, nFCs
# The ID column to merge and join everything on
segment_ID_column = 'SGAT_PID_P2'

# List of variables to include as inputs
print('Calculating Features From RGA Table')
# The features or columns we want to end up with
new_features = ["SegmentID", "ER", "WtoD", "IR", "nBars", "nFCs", "pArmor"]
# Create a new dataframe with these columns
segments_df = pandas.DataFrame(columns=new_features)
# Fill in the ID column with the IDs from the RGA table
segments_df['SegmentID'] = rga_df[segment_ID_column]
# These features are straight forward ports from the RGA table
segments_df['IR'] = rga_df['Incision_ratio']
segments_df['ER'] = rga_df['Entrenchment_ratio']
segments_df['WtoD'] = rga_df['Width_to_depth_ratio']

# I THINK it's OK to do pArmor using feet as units here
# Use Phase 2 left / right bank revetment lengths first, if available
# These percent armor calculations come from Kristen Underwood
segments_df['pArmor'] = (
    (rga_df['bank_revetment_length_left'] + rga_df['bank_revetment_length_right'])
    / (2 * rga_df['segment_length']))
# If no Phase 2 revetment, then use Phase 1 armoring left / right values
phase1_armor_na = (
    (rga_df['BankArmoringLengthLeft'] + rga_df['BankArmoringLengthRight']) / 
    (2 * rga_df['segment_length']))
segments_df['pArmor'] = segments_df['pArmor'].where(segments_df['pArmor'].notna(), other=phase1_armor_na)

# Need segment length in km for depositional bars. So feet to meters to km
segment_len_km = rga_df['segment_length'] * 0.001 * 0.3048

# Number of depositional bars ( # per kilometer )
segments_df['nBars'] = ((rga_df['BarMidNumber'] + 
                         rga_df['BarPointNumber'] + 
                         rga_df['BarSideNumber'] + 
                         rga_df['BarDiagonalNumber'] + 
                         rga_df['BarDeltaNumber']) / segment_len_km)
                         
segments_df['nFCs'] = rga_df['FloodChutesNumber'] / segment_len_km 
# Remove any entry in the dataframe table where there is not a valid ID
segments_df.dropna(subset=['SegmentID'], inplace=True)
# Save this dataframe as a csv table to the below path
basic_feats_path = os.path.join(workspace_dir, 'basic_features.csv')
segments_df.to_csv(basic_feats_path)

## Merge everything together into one final feature table
print('Merge All Features')
# Merge in Pebble count DF (dataframe)
pebble_df = pandas.read_csv(pebble_count_path)
pebble_sub_df = pebble_df[['SGAT_ID', 'd16', 'd35', 'd50', 'd84']]
print('Merge Pebble Counts')
rga_complete_df = rga_df.merge(
    pebble_sub_df, how='left', left_on='SGAT_PID_P2', right_on='SGAT_ID')

# Merge in Slope DF
slope_sub_df = slope_df.loc[:,('SGAT_PID_P2','slope')]
# Drop any rows where slope data has no valid ID
slope_sub_df.dropna(subset=['SGAT_PID_P2'], inplace=True)
print('Merge Slope')
rga_complete_df = rga_complete_df.merge(
    slope_sub_df, how='left', on='SGAT_PID_P2')

# Merge in Streampower DF
ssp_df = pandas.read_csv(ssp_path)
ssp_bal_df = pandas.read_csv(ssp_balance_path)
# Only need the ID and ssp field to merge, so subset those
ssp_sub_df = ssp_df[['SGAT_PID_P2', 'ssp']]
ssp_bal_sub_df = ssp_bal_df[['SGAT_PID_P2', 'ssp_bal']]
print('Merge SSP')
rga_complete_df = rga_complete_df.merge(
    ssp_sub_df, how='left', on='SGAT_PID_P2')
rga_complete_df = rga_complete_df.merge(
    ssp_bal_sub_df, how='left', on='SGAT_PID_P2')

# Merge in Drainage Area (DA) DF
da_df = pandas.read_csv(darea_csv_path)
# Only need the ID and DA attribute for merging
da_sub_df = da_df.loc[:, ('LineID', 'DA')]
# Drop any rows of data with missing ID
da_sub_df.dropna(subset=['LineID'], inplace=True)
print('Merge DA')
rga_complete_df = rga_complete_df.merge(
    da_sub_df, how='left', left_on='SGAT_PID_P2', right_on='LineID')

# Merge in VC VC Ratio DF
vc_df = pandas.read_csv(vc_csv_path)
# Subset only the necessary features
vc_sub_df = vc_df.loc[:, ('SGAT_PID_P2', 'VC', 'VC_RATIO')]
# Drop any rows missing ID values
vc_sub_df.dropna(subset=['SGAT_PID_P2'], inplace=True)
print('Merge VC')
rga_complete_df = rga_complete_df.merge(
    vc_sub_df, how='left', on='SGAT_PID_P2')

# Merge in the ported features from the main RGA table
print('Merge RGA table features')
segments_df.dropna(subset=['SegmentID'], inplace=True)
rga_complete_df = rga_complete_df.merge(
    segments_df, how='left', left_on='SGAT_PID_P2', right_on='SegmentID')
# Output path for the complete feature table
rga_complete_df.to_csv(os.path.join(workspace_dir, 'rga_complete.csv'))

# A list (really tuple) of the desired features for the SOM
som_features_list = (
    'SGAT_PID_P2', 'DA', 'd84_d16', 'd50', 'slope', 'ssp', 'ssp_bal', 'VC', 
    'VC_RATIO', 'pArmor', 'IR', 'ER', 'WtoD', 'nBars', 'nFCs')
# Quickly compute the d84 - d16 pebble count values
rga_complete_df['d84_d16'] = rga_complete_df['d84'] - rga_complete_df['d16']
rga_som_df = rga_complete_df.loc[:, som_features_list]
rga_som_df.dropna(subset=['SGAT_PID_P2'], inplace=True)
# Output path for the feature table that would plug in to the SOM model
som_features_path = os.path.join(workspace_dir, 'som_features_selected.csv')
rga_som_df.to_csv(som_features_path)

