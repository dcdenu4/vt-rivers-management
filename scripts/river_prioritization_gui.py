import sys
import os

if sys.version_info[0] >= 3:
    import PySimpleGUI as sg
else:
    import PySimpleGUI27 as sg
    
#printsq = sg.Print
#sg.ChangeLookAndFeel('GreenTan')

import river_prioritization

#root = Tkinter.Tk()
#defaultbg = root.cget('bg')
#print(defaultbg)

  

validated = False

layout = [[sg.Text('River Reach Prioritization', size=(30, 1), font=("Helvetica", 14))],      
[sg.Text('_'  * 100, size=(70, 1))],
[sg.Text('Workspace:', size=(25, 1), auto_size_text=True, justification='right'), 
    sg.InputText(default_text='Y:/EPSCoR/My-Project/Data/guiWkspace', do_not_clear=True, key='_output_dir_path_'),
    sg.FolderBrowse()],
[sg.Text('_'  * 100, size=(70, 1))],
[sg.Text('SOM')], 
[sg.Text('SOM Parameters:', size=(25, 1), auto_size_text=True, justification='right'), 
    sg.InputText(default_text='Y:/EPSCoR/My-Project/Data/test_sample_data/som_classified_clusters.csv', do_not_clear=True, key='_som_params_path_'),
    sg.FileBrowse()],
[sg.Text('SOM Trained Model:', size=(25, 1), auto_size_text=True, justification='right'), 
    sg.InputText(default_text='Y:/EPSCoR/My-Project/Data/wkspace_test_som/som_model.npy', do_not_clear=True, disabled=False, key='_som_model_'),
    sg.FileBrowse()],
    
[sg.Text('SOM Input Features:', size=(25, 1), auto_size_text=True, justification='right'), 
    sg.InputText(default_text='Y:/EPSCoR/My-Project/Data/feature_build_1/som_features_selected.csv', do_not_clear=True, key='_som_input_path_'),
    sg.FileBrowse()],
[sg.Text('_'  * 100, size=(70, 1))],

[sg.Text('Prioritization')],
[sg.Checkbox('Run Prioritization:', default=True, change_submits = True, key='_prioritize_')],
[sg.Text('Phase 2 Stream Reaches:', size=(25, 1), auto_size_text=True, justification='right'), 
    sg.InputText(default_text='Y:/EPSCoR/My-Project/Data/Geospatial/SGA_Phase_2_Assessed_Reaches/SGA_Phase_2_Assessed_Reaches.shp', do_not_clear=True, key='_streams_path_'),
    sg.FileBrowse()],
[sg.Text('Watershed Shapefile:', size=(25, 1), auto_size_text=True, justification='right'), 
    sg.InputText(default_text='Y:/EPSCoR/My-Project/Data/Geospatial/Missisquoi/Missisquoi_Watershed.shp', do_not_clear=True, key='_watershed_path_'),
    sg.FileBrowse()],
[sg.Text('LULC:', size=(25, 1), auto_size_text=True, justification='right'), 
    sg.InputText(default_text='Y:/EPSCoR/My-Project/Data/Geospatial/NLCD2011_LC_Vermont/NLCD2011_LC_Vermont.tif', do_not_clear=True, key='_LULC_'),
    sg.FileBrowse()],
[sg.Text('Town Layer Shapefile:', size=(25, 1), auto_size_text=True, justification='right'), 
    sg.InputText(default_text='Y:/EPSCoR/My-Project/Data/trialrun/pop_density_as_town.shp', do_not_clear=True, key='_town_path_'),
    sg.FileBrowse()],
[sg.Text('RGA Table:', size=(25, 1), auto_size_text=True, justification='right'), 
    sg.InputText(default_text='Y:/EPSCoR/My-Project/Data/Tabular/RGA/Missisquoi/Missisquoi_P1P2_All_Miss_Projects.xls', do_not_clear=True, key='_rga_path_'),
    sg.FileBrowse()],
[sg.Text('Prioritization Parameters:', size=(25, 1), auto_size_text=True, justification='right'), 
    sg.InputText(default_text='Y:/EPSCoR/My-Project/Data/test_sample_data/optimizer-params.csv', do_not_clear=True, key='_prioritized_params_'),
    sg.FileBrowse()],
[sg.Text('Objective Function Weights:', size=(25, 1), auto_size_text=True, justification='right'), 
    sg.InputText(default_text='Y:/EPSCoR/My-Project/Data/test_sample_data/objective_weights.csv', do_not_clear=True, key='_objective_weights_'),
    sg.FileBrowse()],
[sg.Text('', text_color=None, background_color=None, size=(70, 1), key='_success_message_')],
[sg.Submit(), sg.Cancel(key='_cancel_')]]

som_training_elements = ['_som_training_input_path_', '_som_training_params_path_', '_som_model_']
prioritize_elements = ['_streams_path_', '_function_weights_']

window = sg.Window('River Reach Prioritization Tool', auto_size_text=True, default_element_size=(40, 1)).Layout(layout)

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
            river_prioritization.execute(values)
            #print(values)
            pass
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise
        window.FindElement('_success_message_').Update(text_color="#24b73d")
        window.FindElement('_success_message_').Update('The Model Completed Successfully')
        window.FindElement('_cancel_').Update('OK')
        
    #if event=='_prioritize_':
    #    prioritize = values['_prioritize_']
    #    update_values = [not prioritize, not prioritize]
    #    for key, val in zip(prioritize_elements, update_values):
    #        window.FindElement(key).Update(disabled=val)

print("Done.")


