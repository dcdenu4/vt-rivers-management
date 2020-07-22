# river prioritization
import os
import sys

import pandas as pd
import numpy as np
import pygeoprocessing as pygeo

import somutils
import utils
import geospatialutils as gutils
import optimizer

def execute(args):
    """ 
    
    args : 
    
    '_output_dir_path_'
    '_streams_path_'
    '_function_weights_'
    '_som_input_path_'
    '_som_model_'
    '_som_params_path_'
    '_town_path_'
    '_rga_path_'
    '_LULC_'
    '_prioritized_params_'
    
    """
    workspace_dir = args['_output_dir_path_']
    if not os.path.isdir(workspace_dir):
        os.mkdir(workspace_dir)
    
    #som_weights = np.load(args['_som_model_'])
    som_weights, som_details = somutils.load_som_model(args['_som_model_'])
    # Number of clusters to cluster SOM
    if 'cluster' in som_details:
        nclusters = som_details['cluster']
    else:
        nclusters = 7
    # Cluster SOM nodes
    clustering = somutils.cluster_som(som_weights, nclusters) 

    # SOM cluster to class map
    cluster_to_class_df = pd.read_csv(args['_som_params_path_'])
    print(cluster_to_class_df)
    cluster_list = cluster_to_class_df['cluster'].tolist()
    class_list = cluster_to_class_df['class'].tolist()
    cluster_to_class = {key:val for key, val in zip(cluster_list, class_list)}
    
    # Predict cluster (sediment regime type) for new data
    som_to_model_path = args['_som_input_path_']
    
    som_input_df = pd.read_csv(som_to_model_path)
    som_input_df.columns = som_input_df.columns.str.lower()
    som_input_df.set_index('sgat_pid_p2', inplace=True)
    print(som_input_df.head())
            
    som_input_features = ["Slope", "VC", "VC_ratio", "ER", "WtoD", "IR", "d50", 
                         "d84_d16", "nBars", "nFCs", "pArmor", "SSP", "SSP_bal"]
    
    som_input_features_lower = [x.lower() for x in som_input_features]
    
    # The number of features for the SOM                     
    num_som_input_feats = len(som_input_features_lower)  

    som_input_df_dropna = som_input_df.dropna(subset=som_input_features_lower)

    # Get only the data from the features of interest
    som_input_feats_dropna_df = som_input_df_dropna.loc[:, som_input_features_lower]
    # NORMALIZE DATA by min, max normalization approach
    som_input_df_norm = somutils.normalize(som_input_feats_dropna_df)
    # Display statistics on our normalized data
    print(som_input_df_norm.describe())
    # Get the data as a matrix dropping the dataframe wrapper
    som_input_data = som_input_df_norm.as_matrix()
    som_input_index_list = som_input_df_norm.index.values
    
    #som_input_feats_df = som_input_df.loc[:, som_input_features_lower]
    #print(selected_data_feats_df.head(n=50))
    # Handle NODATA / Missing data by removing (for now)
    #som_input_feats_df.dropna(how='any', inplace=True)
    
    som_input_object_distances = somutils.fill_bmu_distances(som_input_data, som_weights)
    
    # The number of rows for the grid and number of columns. This dictates 
    # how many nodes the SOM will consist of. Currently not calculated 
    # using PCA or other analyses methods.
    # THESE SHOULD COME FROM PARAM FILE
    nrows = som_details['shape'][0]
    ncols = som_details['shape'][1]
    grid_type = som_details['grid_type']
    # Create the SOM grid (which initializes the SOM network)
    som_grid = somutils.create_grid(nrows, ncols, grid_type=grid_type)
    
    som_input_model_results_path = os.path.join(workspace_dir, 'som_input_cluster_results.csv')
    somutils.save_cluster_results(som_input_feats_dropna_df, som_input_model_results_path, clustering.labels_, (nrows, ncols), som_input_object_distances)
   
    som_input_figure_path = os.path.join(workspace_dir, 'predicted_input_regimes.jpg')
    somutils.basic_som_figure(som_input_data, som_weights, som_grid, clustering.labels_,
                                grid_type, som_input_figure_path, dframe=som_input_feats_dropna_df)
    
    # Save Predicted results with new field to CSV output
    classified_df = pd.read_csv(som_input_model_results_path)
    cluster_result_list = classified_df['cluster'].tolist()
    classified_df['class'] = pd.Series([cluster_to_class[key] for key in cluster_result_list])
    classified_df.to_csv(som_input_model_results_path)
    
    # Save predicted results to streams shapefile layer
    streams_path = args['_streams_path_']
    watershed_path = args['_watershed_path_']
    clipped_streams_path = os.path.join(workspace_dir, 'clipped_copied_streams.shp')
    gutils.clip_vectors(streams_path, [0], watershed_path, clipped_streams_path)
    classified_streams_path = os.path.join(workspace_dir, 'classified_streams.shp')
    gutils.add_feature_to_shape(clipped_streams_path, som_input_model_results_path, 'SGAT_ID', 'sgat_pid_p2', 'class', classified_streams_path)
    
    # Get parameters set to call and run optimization function
    
    ### Determine if river is upstream from "town"
    # Get intersection of "towns" and rivers
    town_vector_path = args['_town_path_']
    streams_intersect_towns_path = os.path.join(workspace_dir, 'streams_towns.shp')
    gutils.vector_intersection(classified_streams_path, [0], town_vector_path, streams_intersect_towns_path)
    
    # Now save the intersected streams as a csv file
    streams_intersect_csv = os.path.join(workspace_dir, 'streams_intersect.csv')
    gutils.shape_to_csv(streams_intersect_towns_path, streams_intersect_csv, key_field=None, fields=None)
    
    # Get upstream segments
    rga_upstream_path = os.path.join(workspace_dir, 'rga_upstream.csv')
    utils.calculate_upstream_id(args['_rga_path_'], 'SGAT_PID_P2', rga_upstream_path)
    dist_upstream_csv = os.path.join(workspace_dir, 'dist_upstream.csv')
    utils.upstream_towns(rga_upstream_path, streams_intersect_csv, 'SGAT_PID_P2', 'SGAT_ID', dist_upstream_csv)
    
    #### Determine percentage of LULC type in buffer of reach
    # Buffer stream reaches
    buffer_stream_path = os.path.join(workspace_dir, 'buffered_streams.shp')
    # 200 Foot Buffer to meters, since projection is in meters
    buffer_distance = int(100 / 3.281)

    gutils.buffer_vector(clipped_streams_path, buffer_stream_path, buffer_distance)
    # Get LULC percentage under stream buffer
    vector_info = pygeo.geoprocessing.get_vector_info(buffer_stream_path)
    lulc_warped = os.path.join(workspace_dir, 'projected_lulc.tif')
    pygeo.geoprocessing.warp_raster(
        args['_LULC_'], [30,30], lulc_warped,
        'near', target_sr_wkt=vector_info['projection'])
        
    lulc_coverage_dict = gutils.zonal_statistics_nominal(lulc_warped, buffer_stream_path, 'SGAT_ID')
    
    ### Stage inputs for prioritization
    # Combine rga data
    upstream_df = pd.read_csv(dist_upstream_csv)
    upstream_sub_df = upstream_df[['SGAT_PID_P2', 'segment_length', 'upstream_ID', 'upstream_town', 'upstream_dist']]
    classified_sub_df = classified_df[['sgat_pid_p2', 'ssp', 'cluster', 'class']]
    rga_staged_df = classified_sub_df.merge(upstream_sub_df, how='left', left_on='sgat_pid_p2', right_on='SGAT_PID_P2')
    rga_staged_path = os.path.join(workspace_dir, 'staged_rga_prioritize.csv')
    rga_staged_df.to_csv(rga_staged_path)
    
    # RUN OPTIMIZATION
    optimizer.run_ga(args['_prioritized_params_'], args['_objective_weights_'], rga_staged_path, lulc_coverage_dict, workspace_dir)