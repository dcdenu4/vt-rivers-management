""" Utilities """

import os

import numpy
import pandas


def get_pebble_counts(workbook_dir, out_path, pebble_sheet_name, rga_path):
    """TODO: Fill in docstring. """
    # Get workbook values for D16, D35, D50, D85 
    # from cells: Y30, Y31, Y32, Y33
    segment_data = []
    
    # Use the Master RGA file to match IDs
    rga_df = pandas.read_excel(rga_path)

    for root, dirs, files in os.walk(workbook_dir):

        project_name = os.path.basename(root)
        rga_project_df = rga_df.loc[rga_df['project_name']==project_name]
   
        for file in files:
            workbook_path = os.path.join(root, file)
            
            # TODO probably should use the file name, as it has a higher chance of
            # matching actual id name
            sgat_id, _ext = os.path.splitext(file)
            rga_sub_df = rga_project_df.loc[rga_project_df['reachpoint_id']==sgat_id]
            segment_id_list = rga_sub_df['segment_id'].tolist()
            sgat_p2_id_list = rga_sub_df['SGAT_PID_P2'].tolist()
            
            subid_len = len(segment_id_list)
            
            #print(workbook_path)
            workbook_df = pandas.read_excel(
                workbook_path, sheet_name=pebble_sheet_name, header=None)

            pebble_sizes = ['d16', 'd35', 'd50', 'd84']
            sub_ids = ['A', 'B', 'C', 'D']

            row_end = 30 + subid_len
            
            d16 = workbook_df.iloc[30:row_end][22]
            d35 = workbook_df.iloc[30:row_end][23]
            d50 = workbook_df.iloc[30:row_end][24]
            d85 = workbook_df.iloc[30:row_end][25]

            segment_name = workbook_df.iloc[2][22]
            
            data_shape = d16.shape
            
            for sub_idx in range(len(segment_id_list)):
                if sub_idx >= data_shape[0]:
                    segment_data.append({'SGAT_ID':sgat_p2_id_list[sub_idx], 
                                 'Project_Name':project_name,
                                 'Stream_Name':segment_name, 
                                 'd16':numpy.nan, 
                                 'd35':numpy.nan, 
                                 'd50':numpy.nan,
                                 'd84':numpy.nan})

                else: 
                    segment_data.append({'SGAT_ID':sgat_p2_id_list[sub_idx], 
                                     'Project_Name':project_name,
                                     'Stream_Name':segment_name, 
                                     'd16':d16.iloc[sub_idx], 
                                     'd35':d35.iloc[sub_idx], 
                                     'd50':d50.iloc[sub_idx],
                                     'd84':d85.iloc[sub_idx]})

    pebble_count_df = pandas.DataFrame(segment_data)
    pebble_count_df.to_csv(out_path)
    
def upstream_towns(upstream_csv, town_csv, key_left, key_right, result_csv):
    """TODO: Fill in docstring. """
    upstream_df = pandas.read_csv(upstream_csv)
    town_df = pandas.read_csv(town_csv)
    
    town_id_list = town_df[key_right].tolist()
    
    upstream_df["upstream_town"] = numpy.nan
    upstream_df["upstream_dist"] = numpy.nan
    
    # For each intersected town work back upstream and mark all 
    # those segments with being upstream of town and distance
    for town_id in town_id_list:
        cur_id = town_id
        upstream_df.loc[upstream_df[key_left]==cur_id, 'upstream_town'] = True
        upstream_df.loc[upstream_df[key_left]==cur_id, 'upstream_dist'] = 0
        up_id = upstream_df[upstream_df[key_left]==town_id]['upstream_ID']
        cur_dist = 0
        
        while(up_id.item()!='None'):
            
            up_id = up_id.item()
            
            upstream_df.loc[upstream_df[key_left]==up_id, 'upstream_town'] = True
            
            if upstream_df[upstream_df[key_left]==up_id]['segment_length'].isna().item():
                up_dist = 2500
            else:
                up_dist = upstream_df[upstream_df[key_left]==up_id]['segment_length'].item()
            cur_dist = cur_dist + up_dist
            #print(cur_dist)
            if upstream_df[upstream_df[key_left]==up_id]['upstream_dist'].notna().item():
                if cur_dist < upstream_df[upstream_df[key_left]==up_id]['upstream_dist'].item():
                    upstream_df.loc[upstream_df[key_left]==up_id, 'upstream_dist'] = cur_dist
            else:
                upstream_df.loc[upstream_df[key_left]==up_id, 'upstream_dist'] = cur_dist
                
            cur_id = up_id
            up_id = upstream_df[upstream_df[key_left]==cur_id]['upstream_ID']

    upstream_df.loc[upstream_df['upstream_town']!= True, 'upstream_town'] = False
    #print(upstream_df.head(n=20))
    upstream_df.to_csv(result_csv)
    
def calculate_upstream_id(input_csv, id_field, out_csv):
    """TODO: Fill in docstring. """
    input_df = pandas.read_excel(input_csv)
    
    segment_ID_column = id_field

    # Get subset of data where id column is NOT missing
    input_df_id_subset = input_df.dropna(subset=[segment_ID_column])
    
    # Get the ids as a list
    id_list = input_df_id_subset[segment_ID_column].tolist()
    
    # Sort the String IDs. This is a huge assumption that sorting the IDs
    # will mean the following ID from some subject ID SHOULD be it's 
    # upstream neighbour based on conventions of Lettering (A, B, ...) 
    # and Numbering (01, 02, ...)
    id_sorted = sorted(id_list)
    
    upstream_list = []
    
    subid_idx = {'A':0, 'B':1, 'C':2, 'D':3,'E':4, 'F':5, 'G':6, 'H':7, 
                 'I':8, 'J':9}
    subid_list = ['A','B','C','D','E','F','G','None']
    
    # Iterate through all of the IDs
    for index, id in enumerate(id_sorted):
        # Since we are looking ahead in the list, need to check that we 
        # haven't reached the last element in the list
        if index == len(id_sorted) - 1:
            # If last element than there this no upstream reach specified so 
            # set as None
            upstream_list.append('None')
            #return upstream_list, id_sorted
            break
        # The Phase 2 ID indicator will be the last element in the ID String
        # This will either be a letter, 'A', or a dash, '-'. 
        subid = id[-1]
        
        if subid == '-':
            # If current subject ID ends with '-', then it could have an upstream 
            # ID ending incremented by 1 with a '-' or an 'A'. 
            segnum = int(id[-2])
            upstream_id = id[0:-2] + str(segnum + 1) + '-'
            upstream_id2 = id[0:-2] + str(segnum + 1) + 'A'
            if id_sorted[index+1] == upstream_id:
                upstream_list.append(upstream_id)
            elif id_sorted[index+1] == upstream_id2:
                upstream_list.append(upstream_id2)
            else:
                upstream_list.append('None')
        else:
            # If current subject ID ends with a letter, then it could have an
            # upstream ID ending incremented by 1 with a '-' or an 'A' OR 
            # it could have the same numeric ending with the next letter
            segnum = int(id[-2])
            next_letter = subid_list[subid_idx[subid]+1]
            upstream_id = id[0:-1] + next_letter
            upstream_id2 = id[0:-2] + str(segnum + 1) + '-'
            upstream_id3 = id[0:-2] + str(segnum + 1) + 'A'

            if id_sorted[index+1] == upstream_id:
                upstream_list.append(upstream_id)
            elif id_sorted[index+1] == upstream_id2:
                upstream_list.append(upstream_id2)
            elif id_sorted[index+1] == upstream_id3:
                upstream_list.append(upstream_id3)
            else:
                upstream_list.append('None')
            
    sorted_ids_series = pandas.Series(id_sorted)
    upstream_series = pandas.Series(upstream_list)
    df_data = {segment_ID_column: sorted_ids_series, 'upstream_ID': upstream_series}
    upstream_df = pandas.DataFrame(data=df_data)
    input_upstream_merge_df = input_df.merge(upstream_df, how='left', on=segment_ID_column)
    
    input_upstream_merge_df.to_csv(out_csv)
    
def calculate_vc_vcratio(rga_csv, id_field, upstream_id_field, out_csv):
    """TODO: Fill in docstring. """
    input_df = pandas.read_csv(rga_csv)

    subset_df = input_df[[id_field, upstream_id_field, 'P2valley_width', 'bankfull_width']]
    
    # Get subset of data where columns are NOT missing data
    subset_df = subset_df.dropna(subset=['P2valley_width', 'bankfull_width'])
    
    subset_df['VC'] = subset_df['P2valley_width'] / subset_df['bankfull_width']

    vc_list = subset_df['VC'].tolist()
    #print(len(vc_list))
    id_sorted = subset_df[id_field].tolist()
    upstream_list = subset_df[upstream_id_field].tolist()
    #print(upstream_list)
    vc_ratio_list = []
    #print(subset_df.head())

    for index, cur_id in enumerate(id_sorted):
        up_id = upstream_list[index]
        if up_id in id_sorted:
            cur_vc = vc_list[index]
            if up_id == numpy.nan:
                continue
            elif up_id == 'None':
                up_vc = 0.5
            else:
                up_vc = subset_df[subset_df[id_field]==up_id]['VC'].item()
        else:
            up_vc = 0.5
            
        vc_ratio_list.append({id_field:cur_id, 'VC_RATIO': up_vc / cur_vc})

    vc_ratio_df = pandas.DataFrame(data=vc_ratio_list)
    #print(subset_df.head())

    subset_df = subset_df.merge(vc_ratio_df, how='left', on=id_field)
    vc_df = subset_df[[id_field, 'VC', 'VC_RATIO']]
    #print(vc_df.head())
    input_vc_merge_df = input_df.merge(vc_df, how='left', on=id_field)
    
    input_vc_merge_df.to_csv(out_csv)

    
def calculate_streampower(ssp_path, darea_path, slopes_path, data_path):
    """TODO: Fill in docstring. """
    # Open DA, slope, and RGA files to get slope, DA, and width
    darea_df = pandas.read_csv(darea_path, usecols=['LineID','DA'])
    slope_df = pandas.read_csv(slopes_path, usecols=['SGAT_PID_P2','slope'])
    #rga_df = pandas.read_excel(data_path, usecols=['SGAT_PID_P2','P2valley_width','bankfull_width'])
    rga_df = pandas.read_excel(data_path)
    rga_df = rga_df.loc[:, ['SGAT_PID_P2','P2valley_width','bankfull_width']]
    # 1 foot is 0.3048 meters
    rga_df['P2valley_width_m'] = rga_df['P2valley_width'] * 0.3048
    rga_df['bankfull_width_m'] = rga_df['bankfull_width'] * 0.3048
    #print(rga_df.head())
    # Merge these together
    combined_df = rga_df.merge(darea_df, how='left', left_on='SGAT_PID_P2', right_on='LineID')
    combined_df = combined_df.merge(slope_df, how='left', on='SGAT_PID_P2')
    
    combined_df.dropna(how='any', inplace=True)
    
    # Unit weight of water lb/ft^3
    #unit_weight_water = 2.4 
    # Unit weight of water N/m^3
    unit_weight_water = 9087 
    
    # Discharge (cfs | cubic ft per second).
    # 'DA' in meters squared from flow accumulation / DEM (looks like it should be in sq. miles)
    # m^2 / 2.59e+6 = sq. miles
    #combined_df['disch_Q'] = 17.69 * (combined_df['DA'] / 2.59e6)**1.07
    
    # Discharge (cms | cubic meters per second)
    # 'DA' in squared meters from flow accumulation / DEM (looks like it should be in sq. km)
    # 1 m^2 = 1e-6 km (.000006)
    combined_df['disch_Q'] = 0.3376 * (combined_df['DA'] * 1e-6)**0.9487
    
    # Regional Channel Width (rcw) (ft) 
    #combined_df['rcw'] = 10.18 * (combined_df['DA'] / 2.59e6)
    
    # Regional Channel Width (rcw) (m) 
    combined_df['rcw'] = 2.6176 * (combined_df['DA'] * 1e-6)**0.4415
    
    # ((unit_weight_water) * Q * Slope ) / regional channel width
    combined_df['ssp'] = ((unit_weight_water) * combined_df['disch_Q'] * combined_df['slope'] ) / combined_df['rcw']
    
    combined_df.to_csv(ssp_path)
    
        
def calculate_streampower_balance(ssp_balance_path, ssp_path, upstream_ids_path):
    """TODO: Fill in docstring. """
    # Open ssp and upstream ids datasets
    ssp_df = pandas.read_csv(ssp_path, usecols=['SGAT_PID_P2','ssp'])
    upstream_df = pandas.read_csv(upstream_ids_path, usecols=['SGAT_PID_P2','upstream_ID'])
    # Merge these together
    combined_df = ssp_df.merge(upstream_df, how='left', on='SGAT_PID_P2')
    
    combined_df.dropna(how='any', inplace=True)
    
    id_sorted = combined_df['SGAT_PID_P2'].tolist()
    upstream_list = combined_df['upstream_ID'].tolist()
    ssp_list = combined_df['ssp'].tolist()
    
    ssp_bal_list = []
    
    for index, cur_id in enumerate(id_sorted):
        up_id = upstream_list[index]
        if up_id in id_sorted:
            cur_ssp = ssp_list[index]
            if up_id == numpy.nan:
                continue
            elif up_id == 'None':
                # Dumby value at the moment in case there is no upstream
                # segment. There are more clever ways to do this
                ssp_bal = 0.3
            else:
                up_ssp = combined_df[combined_df['SGAT_PID_P2']==up_id]['ssp'].item()
                try:
                    ssp_bal = up_ssp / cur_ssp
                except ZeroDivisionError:
                    print('SSP is 0 and causing a divide by zero error: setting to ssp_bal of 0')
                    ssp_bal = 0               
        else:
            # Dumby value at the moment in case there is no upstream
            # segment. There are more clever ways to do this    
            ssp_bal = 0.3
    
        ssp_bal_list.append({'SGAT_PID_P2':cur_id, 'ssp_bal': ssp_bal})

    ssp_bal_df = pandas.DataFrame(data=ssp_bal_list)
    
    ssp_bal_df.to_csv(ssp_balance_path)
