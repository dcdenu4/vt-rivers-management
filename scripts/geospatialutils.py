"""Utility module for GIS based operations."""

import os
import sys
import tempfile
from decimal import Decimal

import pandas
import numpy

import pygeoprocessing
import pygeoprocessing.routing.routing as routing

from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import shapely.ops
import shapely.prepared
import shapely.wkt


class NodataException(Exception):
    pass

# Get flow_accum value from under phase 2 break points
# Calculate Drainage area by number of pixels * area of pixels
# Units in meters, pixel size ~ 30 x 30 ( DO NOT leave this hardcoded in release )
_gdal_type_to_ogr_lookup = {
    gdal.GDT_Byte: ogr.OFTInteger,
    gdal.GDT_Int16: ogr.OFTInteger,
    gdal.GDT_Int32: ogr.OFTInteger,
    gdal.GDT_UInt16: ogr.OFTInteger,
    gdal.GDT_UInt32: ogr.OFTInteger,
    gdal.GDT_Float32: ogr.OFTReal,
    gdal.GDT_Float64: ogr.OFTReal
}

def sample_values_to_points(raster_path, vector_path, output_path):
    """Sample raster value under points.
    
    Parameters:
    raster_path (string) - file path to a raster from which to get cell values
        from
    vector_path (string) - 
    output_path (string) - 
    
    """
    
    raster_info = pygeoprocessing.get_raster_info(raster_path)
    
    raster_ds = gdal.OpenEx(raster_path, gdal.OF_RASTER)
    raster_band = raster_ds.GetRasterBand(1)
    raster_gt = raster_info['geotransform']
    
    source_vector = gdal.OpenEx(vector_path)
    new_attribute_field = 'rasterVal'
    new_field_datatype = _gdal_type_to_ogr_lookup[raster_info['datatype']]

    # create a new shapefile from the orginal_datasource
    output_driver = ogr.GetDriverByName('ESRI Shapefile')
    output_datasource = output_driver.CreateDataSource(output_path)

    for layer_index in range(source_vector.GetLayerCount()):
        layer = source_vector.GetLayer(layer_index)
        layer_dfn = layer.GetLayerDefn()
        
        output_srs = layer.GetSpatialRef()
        output_layer = output_datasource.CreateLayer(
            layer_dfn.GetName(), output_srs, layer_dfn.GetGeomType())

        # Get the number of fields in original_layer
        original_field_count = layer_dfn.GetFieldCount()

        # For every field, create a duplicate field in the new layer
        for fld_index in range(original_field_count):
            original_field = layer_dfn.GetFieldDefn(fld_index)
            target_field = ogr.FieldDefn(
                original_field.GetName(), original_field.GetType())
            output_layer.CreateField(target_field)
    
        # Add new field coming from raster value
        output_field_defn = ogr.FieldDefn(new_attribute_field, new_field_datatype)
        output_layer.CreateField(output_field_defn)
        
        out_dfn = output_layer.GetLayerDefn()
            
        for point_feature in layer:

            geometry = point_feature.GetGeometryRef()
            
            # Copy original_datasource's feature and set as new shapes feature
            output_feature = ogr.Feature(output_layer.GetLayerDefn())
            output_feature.SetGeometry(geometry)

            # For all the fields in the feature set the field values from the
            # source field
            new_field_index = output_feature.GetFieldIndex(new_attribute_field)
            
            for fld_index in range(output_feature.GetFieldCount()):
                if fld_index != new_field_index:
                    output_feature.SetField(
                        fld_index, point_feature.GetField(fld_index))
            
            # How to get x,y from point geometry
            mx, my = geometry.GetX(), geometry.GetY()
            
            px = int((mx - raster_gt[0]) / raster_gt[1]) # x pixel
            py = int((my - raster_gt[3]) / raster_gt[5]) # y pixel
            
            raster_val = raster_band.ReadAsArray(px, py, 1, 1)
            raster_val = raster_val.flatten()
            
            output_feature.SetField(new_field_index, raster_val[0])
  
            output_layer.CreateFeature(output_feature)
            output_feature = None
            point_feature = None          

def vector_intersection(
        input_vector_path, input_vector_layers, method_vector_path, 
        result_vector_path):
    """ """
    input_vector = gdal.OpenEx(input_vector_path)
    method_vector = gdal.OpenEx(method_vector_path)

    # create a new shapefile from the orginal_datasource
    output_driver = ogr.GetDriverByName('ESRI Shapefile')
    output_datasource = output_driver.CreateDataSource(result_vector_path)

    method_layer = method_vector.GetLayer(0)
    
    for layer_index in input_vector_layers:
        input_layer = input_vector.GetLayer(layer_index)
        input_layer_dfn = input_layer.GetLayerDefn()       
        
        output_srs = input_layer.GetSpatialRef()
        output_layer = output_datasource.CreateLayer(
            input_layer_dfn.GetName(), output_srs, input_layer_dfn.GetGeomType())
            
        input_layer.Intersection(method_layer, output_layer)

def clip_vectors(
        input_vector_path, input_vector_layers, method_vector_path,
        result_vector_path):
    """ """
    input_vector = gdal.OpenEx(input_vector_path)
    method_vector = gdal.OpenEx(method_vector_path)

    # create a new shapefile from the orginal_datasource
    output_driver = ogr.GetDriverByName('ESRI Shapefile')
    output_datasource = output_driver.CreateDataSource(result_vector_path)

    method_layer = method_vector.GetLayer(0)
    
    for layer_index in input_vector_layers:
        input_layer = input_vector.GetLayer(layer_index)
        input_layer_dfn = input_layer.GetLayerDefn()       
        
        output_srs = input_layer.GetSpatialRef()
        output_layer = output_datasource.CreateLayer(
            input_layer_dfn.GetName(), output_srs, input_layer_dfn.GetGeomType())
            
        input_layer.Clip(method_layer, output_layer)     
        
def buffer_vector(input_vector_path, buffer_vector_path, buffer_distance):
    """ """
    input_vector = gdal.OpenEx(input_vector_path)
    
    # create a new shapefile from the orginal_datasource
    output_driver = ogr.GetDriverByName('ESRI Shapefile')
    output_datasource = output_driver.CreateDataSource(buffer_vector_path)

    for layer_index in range(input_vector.GetLayerCount()):
        layer = input_vector.GetLayer(layer_index)
        layer_dfn = layer.GetLayerDefn()
        
        output_srs = layer.GetSpatialRef()
        output_layer = output_datasource.CreateLayer(
            layer_dfn.GetName(), output_srs, ogr.wkbPolygon)

        # Get the number of fields in original_layer
        original_field_count = layer_dfn.GetFieldCount()
        
        for fld_index in range(original_field_count):
            original_field = layer_dfn.GetFieldDefn(fld_index)
            target_field = ogr.FieldDefn(
                original_field.GetName(), original_field.GetType())
            output_layer.CreateField(target_field)
        
        for input_feat in layer:
            geometry = input_feat.GetGeometryRef()
            geom_wkt = geometry.ExportToWkt()
            shapely_geom = shapely.wkt.loads(geom_wkt)
            shapely_buffered = shapely_geom.buffer(buffer_distance, resolution=16, cap_style=2, join_style=1)
            ogr_buffered_geom = ogr.CreateGeometryFromWkt(shapely_buffered.wkt)
            
            output_feature = ogr.Feature(output_layer.GetLayerDefn())
            output_layer.CreateFeature(output_feature)
            
            #buffer_geom = geometry.Buffer(buffer_distance)
            
            #output_feature.SetGeometry(buffer_geom)
            output_feature.SetGeometry(ogr_buffered_geom)
            
            for fld_index in range(output_feature.GetFieldCount()):
                output_feature.SetField(
                    fld_index, input_feat.GetField(fld_index))
                                            
            output_layer.SetFeature(output_feature)

def zonal_statistics_nominal(
        nominal_raster_path, zone_vector_path, target_field):
    """ """
    print("zonal_stats_nominal target_field: ", target_field)
    unique_lulc = numpy.array([])
    nominal_raster_info = pygeoprocessing.get_raster_info(nominal_raster_path)
    nominal_raster_nodata = nominal_raster_info['nodata']
    
    for _ , nominal_block in pygeoprocessing.iterblocks(nominal_raster_path):
        unique_block = numpy.unique(nominal_block)
        set_diff = numpy.setdiff1d(unique_block, unique_lulc, assume_unique=True)
        unique_lulc = numpy.concatenate((unique_lulc, set_diff))
    
    unique_lulc = numpy.setdiff1d(unique_lulc, numpy.array([nominal_raster_nodata]))
    unique_lulc_int = unique_lulc.astype(int)
    
    aggregate_results = {}
    
    for lulc_val in unique_lulc_int:
        
        print("Aggregating Nominal Value: %s" % lulc_val)
        
        with tempfile.NamedTemporaryFile(
                prefix='lulc_raster', delete=False) as tmp_raster_file:
            tmp_raster_path = tmp_raster_file.name
        
        def mask_op(lulc_array):
            return numpy.where(
                lulc_array == nominal_raster_nodata, nominal_raster_nodata, 
                lulc_array == lulc_val)
            
        pygeoprocessing.raster_calculator(
            [(nominal_raster_path, 1)], mask_op, tmp_raster_path,
            gdal.GDT_Int32, int(nominal_raster_nodata[0]))
        
        lulc_results = pygeoprocessing.zonal_statistics(
            (tmp_raster_path, 1), zone_vector_path, aggregate_layer_name=None,
            ignore_nodata=True, all_touched=False, polygons_might_overlap=True)
            
        for key, val in lulc_results.iteritems():
            if key not in aggregate_results:
                aggregate_results[key] = {}
            
            aggregate_results[key][lulc_val] = {
                'percent': float(Decimal('%.3f' % (100 * (val['sum'] / val['count'])))), 'count':val['sum']}
        
    return aggregate_results
    
def shape_to_csv(vector_path, csv_out_path, key_field=None, fields=None):
    """ """
    input_vector = gdal.OpenEx(vector_path)
    
    data = []
    
    for layer_index in range(input_vector.GetLayerCount()):
        layer = input_vector.GetLayer(layer_index)
        layer_dfn = layer.GetLayerDefn()
        
        # Get the number of fields in original_layer
        original_field_count = layer_dfn.GetFieldCount()

        for feat in layer:
            tmp_dict = {}
            # For every field, create a duplicate field in the new layer
            for fld_index in range(original_field_count):
                original_field = layer_dfn.GetFieldDefn(fld_index)
                tmp_dict[original_field.GetName()] = feat.GetField(fld_index)
        
            data.append(tmp_dict)
            
            
    vector_df = pandas.DataFrame(data)
    if key_field:
        vector_df.set_index(key_field)
        
    vector_df.to_csv(csv_out_path)
        
def calculate_line_slope_from_dem(
        vector_line_path, dem_path, output_path, input_vector_id, csv_out_path,
        aoi_path):
    """ """
    #TODO: is it possible to get length of line segment and ACTUALLY calculate
    # slope here? Need to make sure vector_line_path is projected in 
    #ft / meters
    # see: geom.Length() -> test that against observed segment length in table
    
    dem_ds = gdal.OpenEx(dem_path, gdal.OF_RASTER)
    #dem_info = pygeo.geoprocessing.get_raster_info(dem_path)
    dem_info = pygeoprocessing.get_raster_info(dem_path)
    dem_nodata = dem_info['nodata'][0]
    dem_band = dem_ds.GetRasterBand(1)
    dem_gt = dem_ds.GetGeoTransform()
    
    # Clip streams layer to AOI if desired
    if aoi_path != None:
        vector_clip_path = os.path.join(
            os.path.dirname(output_path), 'clipped_lines.shp')
        clip_vectors(vector_line_path, [0], aoi_path, vector_clip_path)
        vector_line_path = vector_clip_path
    
    # Need to get end points of each line segment and save somehow
    source_vector = gdal.OpenEx(vector_line_path)
    vector_attribute_field = input_vector_id

    # create a new shapefile from the orginal_datasource
    output_driver = ogr.GetDriverByName('ESRI Shapefile')
    output_datasource = output_driver.CreateDataSource(output_path)

    # PANDAS lists
    vectorIDS = []
    line_slopes = []

    numpy.seterr(all='raise')
    
    for layer_index in range(source_vector.GetLayerCount()):
        layer = source_vector.GetLayer(layer_index)
        
        output_srs = layer.GetSpatialRef()
        output_layer = output_datasource.CreateLayer(
            layer.GetName(), output_srs, ogr.wkbPoint)
        
        new_types = [ogr.OFTInteger, ogr.OFTInteger, ogr.OFTString]
        new_fields = ['ID', 'ElevDiff', 'LineID']
        
        for field, type in zip(new_fields, new_types):
            output_field_defn = ogr.FieldDefn(field, type)
            output_layer.CreateField(output_field_defn)
            
        for line_feature in layer:
            point_list = []
        
            segment_ID = line_feature.GetField(vector_attribute_field)
            
            vectorIDS.append(segment_ID)
            
            # Add in the numpy notation which is row, col
            # Here the point geometry is in the form x, y (col, row)
            geometry = line_feature.GetGeometryRef()
            first_point = geometry.GetPoint(0)
            last_point = geometry.GetPoint(geometry.GetPointCount() - 1)
            
            # How to get x,y from point geometry
            #mx, my = first_point.GetX(), first_point.GetY()
            # Currently we have the point x,y from GetPoint of line
            mx, my = first_point[0], first_point[1]
            
            px = int((mx - dem_gt[0]) / dem_gt[1]) # x pixel
            py = int((my - dem_gt[3]) / dem_gt[5]) # y pixel
            
            dem_val_first = dem_band.ReadAsArray(px, py, 1, 1)
            dem_val_first = dem_val_first.flatten()
                
            #print("DEM Value for first point: %s" % dem_val_first[0])

            mx, my = last_point[0], last_point[1]
            
            px = int((mx - dem_gt[0]) / dem_gt[1]) # x pixel
            py = int((my - dem_gt[3]) / dem_gt[5]) # y pixel
            
            dem_val_last = dem_band.ReadAsArray(px, py, 1, 1)
            dem_val_last = dem_val_last.flatten()
            if dem_val_first == dem_nodata or dem_val_last == dem_nodata:
                raise NodataException("End of a stream segment did not "
                    "intersect with a valid DEM value. Please make sure "
                    "the DEM has no Nodata holes and that the DEM covers "
                    "the stream segments.")
                    
            elev_diff = int(abs(dem_val_first[0] - dem_val_last[0]))
                       
            line_slopes.append(elev_diff)
            
            #print("first point: ", first_point)
            #print("last point: ", last_point)
            
            point_list.append(first_point)
            point_list.append(last_point)
            
            for idx, point in enumerate(point_list):
                output_feature = ogr.Feature(output_layer.GetLayerDefn())
                output_layer.CreateFeature(output_feature)
                
                for field, value in zip(new_fields, [idx, elev_diff, segment_ID]):
                    index = output_feature.GetFieldIndex(field)
                    output_feature.SetField(index, value)
                
                geom = ogr.Geometry(ogr.wkbPoint)
                geom.AddPoint_2D(point[0], point[1])
                
                output_feature.SetGeometryDirectly(geom)
                output_layer.SetFeature(output_feature)

    df_data = {'SGAT_ID' : vectorIDS, 'ElevDiff': line_slopes}
    df_slopes = pandas.DataFrame(data=df_data)
    df_slopes.to_csv(csv_out_path)
    
    #return vectorIDS, line_slopes
    
def add_feature_to_shape(
        input_vector_path, input_csv, vector_key, csv_key, join_feature,
        output_path):
    """ """
    
    input_df = pandas.read_csv(input_csv)
    
    input_vector = gdal.OpenEx(input_vector_path)

    # create a new shapefile from the orginal_datasource
    output_driver = ogr.GetDriverByName('ESRI Shapefile')
    output_datasource = output_driver.CreateDataSource(output_path)
    
    for layer_index in range(input_vector.GetLayerCount()):
        layer = input_vector.GetLayer(layer_index)
        layer_dfn = layer.GetLayerDefn()
        
        output_layer = output_datasource.CreateLayer(
            layer_dfn.GetName(), layer.GetSpatialRef(), layer_dfn.GetGeomType())
            
        new_types = [ogr.OFTInteger]
        new_fields = [join_feature]
        
        for fld_index in range(layer_dfn.GetFieldCount()):
            output_field_defn = layer_dfn.GetFieldDefn(fld_index)
            output_layer.CreateField(output_field_defn)

        for field, type in zip(new_fields, new_types):
            output_field_defn = ogr.FieldDefn(field, type)
            output_layer.CreateField(output_field_defn)
            
        for input_feat in layer:        
            vector_key_val = input_feat.GetField(vector_key)
            feat_series = input_df.loc[input_df[csv_key]==vector_key_val][join_feature]
            #print(vector_key_val)
            if not feat_series.empty:         
            #if True:         
                
                feat_val = int(feat_series.item())
                #print(feat_val)
                geometry = input_feat.GetGeometryRef()    

                output_feature = ogr.Feature(output_layer.GetLayerDefn())
                output_layer.CreateFeature(output_feature)
                
                output_feature.SetGeometry(geometry)
                        
                join_index = output_feature.GetFieldIndex(join_feature)
                output_feature.SetField(join_index, feat_val)
                #output_feature.SetField(join_index, 4)
                
                
                for fld_index in range(output_feature.GetFieldCount()):
                    if fld_index != join_index:
                        out_fld_dfn = output_feature.GetFieldDefnRef(fld_index)
                        out_fld_name = out_fld_dfn.GetNameRef()
                        input_feat_index = input_feat.GetFieldIndex(out_fld_name)
                        
                        output_feature.SetField(
                            fld_index, input_feat.GetField(input_feat_index))
                            
                output_layer.SetFeature(output_feature)

    
def create_downstream_point_vector(
        vector_line_path, dem_path, output_path, input_vector_id):
    """ """
    
    dem_ds = gdal.OpenEx(dem_path, gdal.OF_RASTER)
    dem_band = dem_ds.GetRasterBand(1)
    dem_gt = dem_ds.GetGeoTransform()
    
    # Need to get end points of each line segment and save somehow
    source_vector = gdal.OpenEx(vector_line_path)
    vector_attribute_field = input_vector_id

    # create a new shapefile from the orginal_datasource
    output_driver = ogr.GetDriverByName('ESRI Shapefile')
    output_datasource = output_driver.CreateDataSource(output_path)

    for layer_index in range(source_vector.GetLayerCount()):
        layer = source_vector.GetLayer(layer_index)
        
        output_srs = layer.GetSpatialRef()
        output_layer = output_datasource.CreateLayer(
            layer.GetName(), output_srs, ogr.wkbPoint)
        
        new_types = [ogr.OFTInteger, ogr.OFTString]
        new_fields = ['ID', 'LineID']
        
        for field, type in zip(new_fields, new_types):
            output_field_defn = ogr.FieldDefn(field, type)
            output_layer.CreateField(output_field_defn)
            
        for line_feature in layer:
            point_list = []
        
            segment_ID = line_feature.GetField(vector_attribute_field)
            
            # Add in the numpy notation which is row, col
            # Here the point geometry is in the form x, y (col, row)
            geometry = line_feature.GetGeometryRef()
            first_point = geometry.GetPoint(0)
            last_point = geometry.GetPoint(geometry.GetPointCount() - 1)
            
            # How to get x,y from point geometry
            #mx, my = first_point.GetX(), first_point.GetY()
            # Currently we have the point x,y from GetPoint of line
            mx, my = first_point[0], first_point[1]
            
            px = int((mx - dem_gt[0]) / dem_gt[1]) # x pixel
            py = int((my - dem_gt[3]) / dem_gt[5]) # y pixel
            
            dem_val_first = dem_band.ReadAsArray(px, py, 1, 1)
            dem_val_first = dem_val_first.flatten()
            #print("DEM Value for first point: %s" % dem_val_first[0])
            
            downstream_point = first_point
            
            mx, my = last_point[0], last_point[1]
            
            px = int((mx - dem_gt[0]) / dem_gt[1]) # x pixel
            py = int((my - dem_gt[3]) / dem_gt[5]) # y pixel
            
            dem_val_last = dem_band.ReadAsArray(px, py, 1, 1)
            dem_val_last = dem_val_last.flatten()
            
            elev_diff = int(abs(dem_val_first[0] - dem_val_last[0]))
            
            if dem_val_last[0] < dem_val_first[0]:
                downstream_point = last_point

            output_feature = ogr.Feature(output_layer.GetLayerDefn())
            output_layer.CreateFeature(output_feature)
                
            for field, value in zip(new_fields, [1, segment_ID]):
                index = output_feature.GetFieldIndex(field)
                output_feature.SetField(index, value)
                
                geom = ogr.Geometry(ogr.wkbPoint)
                geom.AddPoint_2D(downstream_point[0], downstream_point[1])
                
                output_feature.SetGeometryDirectly(geom)
                output_layer.SetFeature(output_feature)


def calculate_drainage_area(
        workspace_dir, dem_path, watershed_path, stream_path, darea_csv_path,
        pour_pts_path):
    """ """
    #dem_info = pygeo.geoprocessing.get_raster_info(dem_path)
    dem_info = pygeoprocessing.get_raster_info(dem_path)
    dem_nodata = dem_info['nodata'][0]
    
    clipped_streams_path = os.path.join(
        workspace_dir, 'clipped_state_streams.shp')
    print('Clip state stream paths')
    clip_vectors(stream_path, [0], watershed_path, clipped_streams_path)

    # TODO: Instead of passing in a clipped DEM, do the clipping in house
    dem_clipped_extent_path = os.path.join(
        workspace_dir, 'clipped_extent_dem.tif')
    print('clip dem extents')
    pygeoprocessing.align_and_resize_raster_stack(
            [dem_path], [dem_clipped_extent_path], ['near'],
            dem_info['pixel_size'], 'intersection', 
            base_vector_path_list=[watershed_path],
            raster_align_index=0)
    
    burned_streams_path = os.path.join(workspace_dir, 'burned_streams_state.tif')
    print('create new raster')
    pygeoprocessing.new_raster_from_base(
        dem_clipped_extent_path, burned_streams_path, gdal.GDT_Int32,
        [dem_nodata], fill_value_list=[0])
    print('burn stream layer')

    pygeoprocessing.rasterize(
        clipped_streams_path, burned_streams_path, [1], ['ALL_TOUCHED=TRUE'])

    burned_raster_info = pygeoprocessing.get_raster_info(burned_streams_path)
    burned_raster_nodata = burned_raster_info['nodata'][0]
    
    dem_raster_info = pygeoprocessing.get_raster_info(dem_clipped_extent_path)
    dem_raster_nodata = dem_raster_info['nodata'][0]
    
    tmp_stats_path = os.path.join(workspace_dir, "tmp_stats_raster.tif")
    def identity_op(x):
        return x
    pygeoprocessing.raster_calculator(
        [(dem_clipped_extent_path, 1)], identity_op, tmp_stats_path, 
        gdal.GDT_Int32, dem_raster_nodata, calc_raster_stats=True)
        
    raster_ds = gdal.OpenEx(tmp_stats_path, gdal.OF_RASTER)
    raster_band = raster_ds.GetRasterBand(1)
    raster_stats = raster_band.GetStatistics(True,True)
    raster_band = None
    raster_ds = None

    def raise_dem_op(dem_pix, burn_pix):
        # mask for valid pixels that are NOT stream and NOT nodata
        valid_mask = ((dem_pix != dem_raster_nodata) & (burn_pix != burned_raster_nodata) & (burn_pix==0))
        # mask for valid pixels that are stream and NOT nodata
        valid_stream_mask = ((dem_pix != dem_raster_nodata) & (burn_pix != burned_raster_nodata) & (burn_pix==1))
        result = numpy.empty(dem_pix.shape)
        result[:] = dem_raster_nodata
        result[valid_mask] = dem_pix[valid_mask] + raster_stats[1]
        result[valid_stream_mask] = dem_pix[valid_stream_mask]
        return result

    dem_burned_streams_path = os.path.join(workspace_dir, 'hydrodem_burned.tif')
    print('adjust dem to sink stream network')
    pygeoprocessing.raster_calculator(
        [(dem_clipped_extent_path, 1), (burned_streams_path, 1)], raise_dem_op, 
        dem_burned_streams_path, gdal.GDT_Int32, dem_raster_nodata)

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
    clipped_pour_points_path = os.path.join(workspace_dir, 'clipped_pour_pts.shp')
    clip_vectors(pour_pts_path, [0], watershed_path, clipped_pour_points_path)
    #clipped_pour_points_path = pour_pts_path
    # TODO write a snap points function
    #snap_pour_points()
    
    sampled_output_path = os.path.join(workspace_dir, 'flow_accum_sampled.shp')
    print('sample point values')
    sample_values_to_points(flow_accum_path, clipped_pour_points_path, sampled_output_path)

    # Need to get sample points to AREA using flow acumm cell size, save as csv
    flow_accum_csv_path = os.path.join(workspace_dir, 'flow_accum_table.csv')
    shape_to_csv(sampled_output_path, flow_accum_csv_path)
    
    flow_info = pygeoprocessing.get_raster_info(flow_accum_path)
    
    da_df = pandas.read_csv(flow_accum_csv_path)
    #print(da_df.head())

    pixel_area = abs(flow_info['pixel_size'][0] * flow_info['pixel_size'][1])
    # Area will be in squared meters
    drainage_area = da_df['rasterVal'] * pixel_area
    da_df['DA'] = drainage_area
    
    da_df.to_csv(darea_csv_path)
    
