import os
import sys

import pygeoprocessing as pygeo
import pygeoprocessing.routing.routing as routing
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import shapely.wkb
import shapely.ops
import shapely.prepared

import pandas as pd
import numpy as np

import geospatialutils as geoutils

#data_dir_path = os.path.join('..', 'Data', 'Mississquoi', 'Mississquoi_gis')
data_dir_path = os.path.join("Y:", "EPSCoR", "My-Project", "Data")
#"Y:\EPSCoR\My-Project\Data\workspace\clipped_hydrodem.tif"

workspace_dir = os.path.join(data_dir_path, "workspace")
dem_clipped_extent_path = os.path.join(workspace_dir, 'clipped_extent_dem.tif')
downstream_points_path = os.path.join(workspace_dir, 'downstream_points.shp')
phase2_stream_path = os.path.join(data_dir_path, 'Geospatial', 'Test-Tmp', 'phase_2_segments_clipped_miss.shp')

#geoutils.create_downstream_point_vector(phase2_stream_path, dem_clipped_extent_path, downstream_points_path, 'SGAT_ID')

sampled_output_path = os.path.join(workspace_dir, 'flow_accum_sampled_down.shp')


# Try running Routing Flow Accumulation Alg
flow_accum_path = os.path.join(workspace_dir, 'flow_accum.tif')

geoutils.sample_values_to_points(flow_accum_path, downstream_points_path, sampled_output_path)

flow_info = pygeo.geoprocessing.get_raster_info(flow_accum_path)
            
#sampled_output_path = os.path.join(workspace_dir, 'flow_accum_sampled.shp')

# Need to get sample points to AREA using flow acumm cell size, save as csv
csv_out_path = os.path.join(workspace_dir, 'drainage_area.csv')
key_field = 'SGAT_ID'
geoutils.shape_to_csv(sampled_output_path, csv_out_path)

da_df = pd.read_csv(csv_out_path)
print(da_df.head())

pixel_area = abs(flow_info['pixel_size'][0] * flow_info['pixel_size'][1])
da_df['rasterVal'] = da_df['rasterVal'] * pixel_area
print(flow_info['pixel_size'])
print(da_df.head())