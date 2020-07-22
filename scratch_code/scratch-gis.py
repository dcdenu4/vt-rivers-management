# Scratch code for thinking about getting the end points of 
# a polyline and then taking those points, getting the 
# DEM value underneath and computing slope based on the 
# elevation difference and segment length.

# https://gis.stackexchange.com/questions/86040/how-would-one-get-the-end-points-of-a-polyline


from osgeo import ogr
"""
ds=ogr.Open(somepolylines)
lyr=ds.GetLayer()
for i in range(lyr.GetFeatureCount()):
    feat=lyr.GetFeature(i)
    geom=feat.GetGeometryRef()
    firstpoint=geom.GetPoint(0)
    lastpoint=geom.GetPoint(geom.GetPointCount()-1)
    print firstpoint[0],firstpoint[1],lastpoint[0],lastpoint[1] #X,Y,X,Y
    
""" 
# Possibly can get length right from geometry of line instead of 
# needing to request from table / pandas df lookup

# https://pcjericks.github.io/py-gdalogr-cookbook/geometry.html

# https://shapely.readthedocs.io/en/stable/manual.html#linestrings

# https://gis.stackexchange.com/questions/4022/looking-for-a-pythonic-way-to-calculate-the-length-of-a-wkt-linestring


import utils
import geospatialutils as gutils
import optimizer
import numpy as np
import somutils
import pandas as pd
import os
import sys

import pygeoprocessing as pygeo


data_dir_path = os.path.join("Y:", "EPSCoR", "My-Project", "Data")
tabular_data_path = os.path.join(data_dir_path, 'Tabular', 'RGA')
workspace_dir = os.path.join(data_dir_path, "wkspace_test3")
pareto_dir = os.path.join(data_dir_path, "Prelim-Results")

#### Determine percentage of LULC type in buffer of reach
# Buffer stream reaches
buffer_stream_path = os.path.join(workspace_dir, 'buffered_streams.shp')

# Get LULC percentage under stream buffer
vector_info = pygeo.geoprocessing.get_vector_info(buffer_stream_path)
lulc_warped = os.path.join(workspace_dir, 'projected_lulc.tif')

#lulc_coverage_dict = gutils.zonal_statistics_nominal(lulc_warped, buffer_stream_path, 'SGAT_ID')

lulc_dict_path = os.path.join(workspace_dir, 'lulc_coverages.npy')
#np.save(lulc_dict_path, lulc_coverage_dict)

lulc_coverage_dict = np.load(lulc_dict_path).item()
#print lulc_coverage_dict

rga_staged_path = os.path.join(workspace_dir, 'staged_rga_prioritize.csv')


# RUN OPTIMIZATION
prioritized_params = 'Y:/EPSCoR/My-Project/Data/test_sample_data/optimizer-params.csv'
objective_weights = 'Y:/EPSCoR/My-Project/Data/test_sample_data/objective_weights.csv'
optimizer.run_ga(prioritized_params, objective_weights, rga_staged_path, lulc_coverage_dict, pareto_dir)

sys.exit()




cluster_df_path = os.path.join(workspace_dir, 'cluster_results.csv')
class_df_path = os.path.join(data_dir_path, 'Kristen', 'SedRegDataV2g-Copy.csv')

class_df = pd.read_csv(class_df_path)
norm = somutils.normalize(class_df, subset=['DA', 'S', 'L'])
print(norm.describe())

somutils.map_cluster_to_class(class_df_path, cluster_df_path)

streams_intersect_csv = os.path.join(workspace_dir, 'streams_intersect.csv')
#gutils.shape_to_csv(streams_intersect_towns_path, streams_intersect_csv, key_field=None, fields=None)

# Get upstream segments
rga_path = os.path.join(tabular_data_path, 'Missisquoi', 'Missisquoi_P1P2_All_Miss_Projects.xls')
rga_upstream_path = os.path.join(workspace_dir, 'rga_upstream.csv')
utils.calculate_upstream_id(rga_path, 'SGAT_PID_P2', rga_upstream_path)
dist_upstream_csv = os.path.join(workspace_dir, 'dist_upstream.csv')
utils.upstream_towns(rga_upstream_path, streams_intersect_csv, 'SGAT_PID_P2', 'SGAT_ID', dist_upstream_csv)
#### Determine percentage of LULC type in buffer of reach
# Buffer stream reaches
clipped_streams_path = os.path.join(workspace_dir, 'classified_streams.shp')
buffer_stream_path = os.path.join(workspace_dir, 'buffered_streams.shp')
buffer_distance = 100
gutils.buffer_vector(clipped_streams_path, buffer_stream_path, buffer_distance)

vector_info = pygeo.geoprocessing.get_vector_info(buffer_stream_path)

lulc = 'Y:/EPSCoR/My-Project/Data/Geospatial/NLCD2001_LC_Vermont/NLCD2011_LC_Vermont.tif'
lulc_warped = os.path.join(workspace_dir, 'projected_lulc.tif')

gutils.zonal_statistics_nominal(lulc_warped, buffer_stream_path, 'SGAT_ID')

sys.exit()

#pygeo.geoprocessing.warp_raster(
#        lulc, [30,30], lulc_warped,
#        'near', target_sr_wkt=vector_info['projection'])

lulc_results = pygeo.geoprocessing.zonal_statistics(
    (lulc_warped,1), buffer_stream_path,
    'FID', aggregate_layer_name=None,
    ignore_nodata=True, all_touched=False, polygons_might_overlap=True,
    working_dir=workspace_dir)
    
#print(lulc_results)