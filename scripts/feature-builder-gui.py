import sys
import os
import shutil

import PySimpleGUI as sg

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasAgg
import matplotlib.backends.tkagg as tkagg
import tkinter as Tk

from som_feature_builder import execute

#printsq = sg.Print
#sg.ChangeLookAndFeel('GreenTan')

#root = Tkinter.Tk()
#defaultbg = root.cget('bg')
#print(defaultbg)
"workspace_dir" : "C://Users//ddenu//Workspace//UVM//feature_build_winooski",
"tabular_data_path" : "C://Users//denu//Desktop//Preprocess Data Workspace//Tabular//RGA",
"rga_data_path" : "C://Users//denu//Desktop//Preprocess Data Workspace//Tabular//RGA//Winooski//Winooski_P1P2_All_Wino_Projects.xls",
"dem_path" : "C://Users//denu//Desktop//Preprocess Data Workspace//Geospatial//ElevationDEM_VTHYDRODEM//vthydrodem//hdr.adf",
"workbook_parent_dir" : "C://Users//denu//Desktop//Preprocess Data Workspace//Tabular//RGA//Winooski//Workbooks",
"watershed_path" : "C://Users//denu//Desktop//Preprocess Data Workspace//Geospatial//Winooski//Winooski_Watershed.shp",
"segment_line_path" : "C://Users//denu//desktop//Preprocess Data Workspace//Geospatial//SGA_Phase_2_Assessed_Reaches//SGA_Phase_2_Assessed_Reaches.shp",
"hydrography_stream_path" : "C://Users//denu//Desktop//Preprocess Data Workspace//Geospatial//VT_Hydrography_Dataset_cartographic_extract_lines//VT_Hydrography_Dataset__cartographic_extract_lines.shp",


layout = [[sg.Text('Feature Builder Tool', size=(30, 1), 
    font=("Helvetica", 14))],      
[sg.Text('_'  * 100, size=(70, 1))],
[sg.Text('Output Workspace:', size=(25, 1), auto_size_text=True, justification='right'), 
    sg.InputText(do_not_clear=True, key="_output_workspace_"), sg.FolderBrowse()],
[sg.Text('_'  * 100, size=(70, 1))],    
[sg.Text('DEM:', size=(25, 1), auto_size_text=True, justification='right'), 
    sg.InputText(do_not_clear=True, key="_dem_path_"), sg.FileBrowse()],
[sg.Text('Phase 2 Reaches:', size=(25, 1), auto_size_text=True, justification='right'), 
    sg.InputText(do_not_clear=True, key='_phase_2_reach_path_'), sg.FileBrowse()],
[sg.Text('Watershed:', size=(25, 1), auto_size_text=True, justification='right'), 
    sg.InputText(do_not_clear=True, key='_watershed_path_'), sg.FileBrowse()],
[sg.Text('Stream Hydrography:', size=(25, 1), auto_size_text=True, justification='right'), 
    sg.InputText(do_not_clear=True, key='_hydrography_path_'), sg.FileBrowse()],
[sg.Text('RGA Table:', size=(25, 1), auto_size_text=True, justification='right'), 
    sg.InputText(do_not_clear=True, key='_rga_table_path_'), sg.FileBrowse()],

[sg.Text('', text_color=None, background_color=None, size=(70, 1), key='_success_message_')],
[sg.Submit(), sg.Cancel(key='_cancel_')]]

window = sg.Window('Feature Builder Tool', auto_size_text=True, default_element_size=(40, 1)).Layout(layout)

# Event Loop      
while True:      
    event, values = window.Read()      
    print(event)
    if event is None:      
        break
    if event == '_cancel_':
        break
    if event == 'Submit':
        try:
            # TODO: Should try to validate file inputs first           
            results = som_selector.execute(values)
            som_fig = results['som_figure']
            som_model_path = results['model_weights_path']
            show_figure(som_fig, som_model_path)
            #print(values)
            pass
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise
        window.FindElement('_success_message_').Update(text_color="#24b73d")
        window.FindElement('_success_message_').Update('The Model Completed Successfully')
        window.FindElement('_cancel_').Update('Close')

print("Done.")

