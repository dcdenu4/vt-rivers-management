import os
import pandas as pd
import numpy as np


def calculate_upstream_id(input_csv, id_field, out_csv):
    """ """
    input_df = pd.read_excel(input_csv)
    
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
            
    sorted_ids_series = pd.Series(id_sorted)
    upstream_series = pd.Series(upstream_list)
    df_data = {segment_ID_column: sorted_ids_series, 'upstream_ID': upstream_series}
    upstream_df = pd.DataFrame(data=df_data)
    input_upstream_merge_df = input_df.merge(upstream_df, how='left', on=segment_ID_column)
    
    input_upstream_merge_df.to_csv(out_csv)
    
def calculate_vc_vcratio(rga_csv, id_field, upstream_id_field, out_csv):
    """ """
    input_df = pd.read_csv(rga_csv)

    subset_df = input_df[[id_field, upstream_id_field, 'P2valley_width', 'bankfull_width']]
    
    # Get subset of data where columns are NOT missing data
    subset_df = subset_df.dropna(subset=['P2valley_width', 'bankfull_width'])
    
    subset_df['VC'] = subset_df['P2valley_width'] / subset_df['bankfull_width']

    vc_list = subset_df['VC'].tolist()
    print(len(vc_list))
    id_sorted = subset_df[id_field].tolist()
    upstream_list = subset_df[upstream_id_field].tolist()
    print(upstream_list)
    vc_ratio_list = []
    print(subset_df.head())

    for index, cur_id in enumerate(id_sorted):
        up_id = upstream_list[index]
        if up_id in id_sorted:
            cur_vc = vc_list[index]
            if up_id == np.nan:
                continue
            elif up_id == 'None':
                up_vc = 0.5
            else:
                up_vc = subset_df[subset_df[id_field]==up_id]['VC'].item()
        else:
            up_vc = 0.5
            
        vc_ratio_list.append({id_field:cur_id, 'VC_RATIO': up_vc / cur_vc})

    vc_ratio_df = pd.DataFrame(data=vc_ratio_list)
    print(subset_df.head())

    subset_df = subset_df.merge(vc_ratio_df, how='left', on=id_field)
    vc_df = subset_df[[id_field, 'VC', 'VC_RATIO']]
    print(vc_df.head())
    input_vc_merge_df = input_df.merge(vc_df, how='left', on=id_field)
    
    input_vc_merge_df.to_csv(out_csv)

            
data_dir_path = os.path.join("Y:", "EPSCoR", "My-Project", "Data")
workspace_dir = os.path.join(data_dir_path, "workspace")

# Path to data file with reach segments and variables
data_path = os.path.join(data_dir_path, 'Tabular', 'RGA', 'Missisquoi', 'Missisquoi_P1P2_All_Miss_Projects.xls')
upstream_csv_path = os.path.join(workspace_dir, 'upstream_ids.csv')

calculate_upstream_id(data_path, 'SGAT_PID_P2', upstream_csv_path)

vc_csv_path = os.path.join(workspace_dir, 'vc_ratio.csv')
calculate_vc_vcratio(upstream_csv_path, 'SGAT_PID_P2', 'upstream_ID', vc_csv_path)