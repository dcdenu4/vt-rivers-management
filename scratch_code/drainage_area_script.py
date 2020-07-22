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

# Home Desktop Path
#data_dir_path = os.path.join('..', '..', 'Desktop', 'Mississquoi', 'clipped_segments_test')
# Work Desktop Path
#data_dir_path = os.path.join('..', '..', '..', 'Desktop', 'Mississquoi', 'clipped_segments_test')

#data_dir_path = os.path.join('..', 'Data', 'Mississquoi', 'Mississquoi_gis')
data_dir_path = os.path.join("Y:", "EPSCoR", "My-Project", "Data")
#"Y:\EPSCoR\My-Project\Data\workspace\clipped_hydrodem.tif"

workspace_dir = os.path.join(data_dir_path, "workspace")

# Define DEM Raster Path
dem_path = os.path.join(data_dir_path, 'Geospatial', 'ElevationDEM_VTHYDRODEM', 'vthydrodem', 'hdr.adf')
dem_info = pygeo.geoprocessing.get_raster_info(dem_path)
       
missisquoi_ws_path = os.path.join(data_dir_path, 'Geospatial', 'Missisquoi', 'Missisquoi_Watershed.shp')
state_hydrography_dir = 'VT_Hydrography_Dataset__cartographic_extract_lines'
state_hydrography_name = 'VT_Hydrography_Dataset__cartographic_extract_lines.shp'
stream_path = os.path.join(data_dir_path, 'Geospatial', state_hydrography_dir, state_hydrography_name)
clipped_streams_path = os.path.join(workspace_dir, 'clipped_state_streams.shp')
print('Clip state stream paths')
geoutils.clip_vectors(stream_path, [0], missisquoi_ws_path, clipped_streams_path)

# TODO: Instead of passing in a clipped DEM, do the clipping in house
dem_clipped_extent_path = os.path.join(workspace_dir, 'clipped_extent_dem.tif')
print('clip dem extents')
pygeo.geoprocessing.align_and_resize_raster_stack(
        [dem_path], [dem_clipped_extent_path], ['near'],
        dem_info['pixel_size'], 'intersection', 
        base_vector_path_list=[missisquoi_ws_path],
        raster_align_index=0)

watershed_mask_path = os.path.join(workspace_dir, 'watershed_mask.tif')
print('create new raster')
pygeo.geoprocessing.new_raster_from_base(
    dem_clipped_extent_path, watershed_mask_path, gdal.GDT_Int32, [-3246], 
    fill_value_list=[-3246])
print('burn watershed layer into mask')
pygeo.geoprocessing.rasterize(missisquoi_ws_path, watershed_mask_path, [1], ['ALL_TOUCHED=TRUE'])        
        
mask_info = pygeo.geoprocessing.get_raster_info(watershed_mask_path)
        
def raster_mask_op(dem_pix, mask_pix):
    valid_mask = ( (dem_pix != dem_info['nodata'][0]) & (mask_pix != mask_info['nodata'][0]) )
    result = np.empty(dem_pix.shape)
    result[:] = dem_info['nodata'][0]
    result[valid_mask] = dem_pix[valid_mask]
    return result

dem_clipped_path = os.path.join(workspace_dir, 'clipped_dem.tif')
print('finally clip the dem to mask')
pygeo.geoprocessing.raster_calculator(
    [(dem_clipped_extent_path, 1), (watershed_mask_path, 1)], raster_mask_op, 
    dem_clipped_path, gdal.GDT_Int32, dem_info['nodata'][0])
# END TODO

# Try burning stream layer into DEM by lowering DEM / Stream Layer 
# overlap by 5 feet.

burned_streams_path = os.path.join(workspace_dir, 'burned_streams_state.tif')
print('create new raster')
pygeo.geoprocessing.new_raster_from_base(
    dem_clipped_path, burned_streams_path, gdal.GDT_Int32, [dem_info['nodata'][0]], 
    fill_value_list=[0])
print('burn first stream layer')
#pygeo.geoprocessing.rasterize(clipped_streams_path, burned_streams_path, [-50], ['ALL_TOUCHED=TRUE'])

# Attempt to burn Phase 2 stream layer specifically to try and force flow 
# accumulation through phase 2 break points
phase2_stream_path = os.path.join(data_dir_path, 'Geospatial', 'Test-Tmp', 'phase_2_segments_clipped_miss.shp')
#burned_streams_path = os.path.join(workspace_dir, 'burned_streams_phase2.tif')
print('burn second stream layer')
#pygeo.geoprocessing.rasterize(phase2_stream_path, burned_streams_path, [-1000], ['ALL_TOUCHED=TRUE, MERGE_ALG=ADD'])
pygeo.geoprocessing.rasterize(phase2_stream_path, burned_streams_path, [-1000], ['ALL_TOUCHED=TRUE'])

burned_raster_info = pygeo.geoprocessing.get_raster_info(burned_streams_path)
dem_raster_info = pygeo.geoprocessing.get_raster_info(dem_clipped_path)

def raster_add_op(dem_pix, burn_pix):
    valid_mask = ( (dem_pix != dem_raster_info['nodata'][0]) & (burn_pix != burned_raster_info['nodata'][0]) )
    result = np.empty(dem_pix.shape)
    result[:] = dem_raster_info['nodata'][0]
    result[valid_mask] = dem_pix[valid_mask] + burn_pix[valid_mask]
    return result

dem_burned_streams_path = os.path.join(workspace_dir, 'hydrodem_burned.tif')
print('adjust dem to sink stream network')
pygeo.geoprocessing.raster_calculator(
    [(dem_clipped_path, 1), (burned_streams_path, 1)], raster_add_op, 
    dem_burned_streams_path, gdal.GDT_Int32, dem_raster_info['nodata'][0])

# Try running DEM Pit Fill
dem_filled_path = os.path.join(workspace_dir, 'hydrodem_filled.tif')
print('fill pits')
routing.fill_pits((dem_burned_streams_path, 1), dem_filled_path)

# Try running Routing Flow Alg
flow_direction_path = os.path.join(workspace_dir, 'flow_direction.tif')
print('flow direction')
routing.flow_dir_d8((dem_filled_path, 1), flow_direction_path)

# Try running Routing Flow Accumulation Alg
flow_accum_path = os.path.join(workspace_dir, 'flow_accum.tif')
print('flow accumulation')
routing.flow_accumulation_d8((flow_direction_path, 1), flow_accum_path)
            
# TODO: Instead of using phase 2 break points, use points created from 
# phase 2 segments when getting slope. Or just do that same process, 
# Since those have the proper SGAT_ID. Just need to determine if 
# entrance or exit point is wanted for DA.
downstream_points_path = os.path.join(workspace_dir, 'downstream_points.shp')
geoutils.create_downstream_point_vector(phase2_stream_path, dem_clipped_extent_path, downstream_points_path, 'SGAT_ID')


phase2_breaks_path = os.path.join(data_dir_path, 'Geospatial', 'Test-Tmp', 'phase2_breaks_clipped_miss.shp')
sampled_output_path = os.path.join(workspace_dir, 'flow_accum_sampled.shp')
print('sample point values')
geoutils.sample_values_to_points(flow_accum_path, phase2_breaks_path, sampled_output_path)

# Need to get sample points to AREA using flow acumm cell size, save as csv
geoutils.shape_to_csv(vector_path, csv_out_path, key_field)