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

data_dir_path = os.path.join("Y:", "EPSCoR", "My-Project", "Data")

workspace_dir = os.path.join(data_dir_path, "workspace")

workbook_dir = os.path.join(data_dir_path, 'Tabular', 'RGA', 'Missisquoi', 'Workbooks')
#workbook_path = os.path.join(data_dir_path, 'Tabular', 'RGA', 'Missisquoi', 'Workbooks', 'Mudd Creek', 'M4S3.01.xlt')

# Get workbook values for D50 from Y31 cell
segment_data = []

for root, dirs, files in os.walk(workbook_dir):

    project_name = os.path.basename(root)
    #print(os.path.basename(root))
    for file in files:
        #print(os.path.basename(root))
        workbook_path = os.path.join(root, file)
        
        print(workbook_path)
        workbook_df = pd.read_excel(workbook_path, sheetname=2, header=None)

        #print(workbook_df)
        #print(workbook_df.sheetnames)

        pebble_sizes = ['d16', 'd35', 'd50', 'd85']
        sub_ids = ['A', 'B', 'C', 'D']

        d16 = workbook_df.iloc[30:34][22]
        d16 = d16.dropna()
        d35 = workbook_df.iloc[30:34][23]
        d35 = d35.dropna()
        d50 = workbook_df.iloc[30:34][24]
        d50 = d50.dropna()
        d85 = workbook_df.iloc[30:34][25]
        d85 = d85.dropna()
        #for idx, peb in enumerate([d16, d35, d50, d85]):
        #    peb = peb.dropna()
        #print(d16.head())
        # TODO probably should use the file name, as it has a higher chance of
        # matching actual id name
        #sgat_id = workbook_df.iloc[3][22]
        sgat_id, _ext = os.path.splitext(file)
        segment_name = workbook_df.iloc[2][22]
        
        data_shape = d16.shape


        print(sgat_id)

        for sub_idx in range(data_shape[0]):
            #print(sub_idx)
            combined_id = '%s%s' % (sgat_id, sub_ids[sub_idx])
            #print(combined_id)
            #print(d16.iloc[sub_idx])
            segment_data.append({'SGAT_ID':combined_id, 
                                 'Project_Name':project_name,
                                 'Stream_Name':segment_name, 
                                 'd16':d16.iloc[sub_idx], 
                                 'd35':d35.iloc[sub_idx], 
                                 'd50':d50.iloc[sub_idx],
                                 'd85':d85.iloc[sub_idx]})

        #print(segment_data)                         
        #print(workbook_df.iloc[30:34][22:25])
        
pebble_count_path = os.path.join(workspace_dir, 'pebble_count.csv')
pebble_count_df = pd.DataFrame(segment_data)
pebble_count_df.to_csv(pebble_count_path)