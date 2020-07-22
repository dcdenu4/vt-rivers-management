import somutils
import numpy as np
import os
import math

from matplotlib import pyplot as plt
from matplotlib import patches as patches
from matplotlib import colors
from matplotlib import cm

def basic_som_figure(data, som_weights, map_coords, grid, out_path):
    """ """
    x_loc = map_coords[:,:,0]
    y_loc = map_coords[:,:,1]
    #cm_subsection = np.linspace(0, 1, 193)
    
    som_weights_flat = som_weights.flatten()
    
    if grid == 'hex':
        # Compute radius
        hex_radius = 2 / math.sqrt(3) * 0.5
        fig, axs = plt.subplots(ncols=1)
        plt.set_cmap('hsv')
        axs.set_aspect('equal')
        patch_collection = []
        indexer = 0
        for xc, yc, in zip(x_loc.flatten(), y_loc.flatten()):
            #print xc, yc
            #hex = patches.RegularPolygon((xc,yc), numVertices=6, radius=hex_radius, alpha=0.5, edgecolor='k', facecolor=som_weights_flat[0])
            hex = patches.RegularPolygon((xc,yc), numVertices=6, radius=hex_radius, alpha=0.5, edgecolor='k')#, facecolor=som_weights_flat[indexer:indexer + 3])
            axs.add_patch(hex)
            indexer+=3
            
        #pl.text(m[1], m[0], color_names[i], ha='center', va='center',
        #   bbox=dict(facecolor='white', alpha=0.5, lw=0))
            
        xmin = np.min(x_loc)
        xmax = np.max(x_loc)
        ymin = np.min(y_loc)
        ymax = np.max(y_loc)
        axs.set_xlim(xmin - 1.0, xmax + 1.0)
        axs.set_ylim(ymin - 1.0, ymax + 1.0)
        
        num_segments = data.shape[0]
        num_feats = data.shape[1]
                
        for i in range(num_segments):
            t = data[i, :].reshape(np.array([num_feats, 1]))
            id = color_names[i]
            
            # find its Best Matching Unit (BMU)
            bmu, bmu_idx = somutils.find_bmu(t, som_weights, num_feats)
            x_cord = map_coords[bmu_idx[0], bmu_idx[1], 0]
            y_cord = map_coords[bmu_idx[0], bmu_idx[1], 1]
            #print x_cord, y_cord
            #axs.text(x_cord, y_cord, s=id, ha='center', va='center',
            #            bbox=dict(facecolor='white', alpha=0.5, lw=0))
        
    #plt.show(block=False)
    plt.savefig(out_path)
    plt.show()

salmon = [244, 95, 66]
yellow = [252, 252, 65]
green = [59, 155, 60]
blue = [10, 47, 127]
purple = [181, 86, 196]
red = [214, 19, 35]
orange = [237, 130, 9]
aqua = [31, 191, 193]
pink = [249, 32, 195]

color_rgb = np.array([salmon, yellow, green, blue, purple, red, orange, aqua, pink])
print color_rgb
color_rgb_norm = color_rgb / 255.0

color_rgb_norm = np.array(
         [[0., 0., 0.],
          [0., 0., 1.],
          [0., 0., 0.5],
          [0.125, 0.529, 1.0],
          [0.33, 0.4, 0.67],
          [0.6, 0.5, 1.0],
          [0., 1., 0.],
          [1., 0., 0.],
          [0., 1., 1.],
          [1., 0., 1.],
          [1., 1., 0.],
          [1., 1., 1.],
          [.33, .33, .33],
          [.5, .5, .5],
          [.66, .66, .66]])
          
color_names = \
        ['black', 'blue', 'darkblue', 'skyblue',
         'greyblue', 'lilac', 'green', 'red',
         'cyan', 'violet', 'yellow', 'white',
         'darkgrey', 'mediumgrey', 'lightgrey']

# NORMALIZE DATA by min, max normalization approach
#selected_feats_df_norm = somutils.normalize(selected_data_feats_df)

# Display statistics on our normalized data
#print(selected_feats_df_norm.describe())

# Initial learning rate for SOM. Will decay to 0.01 linearly
init_learning_rate = 0.05

# The number of rows for the grid and number of columns. This dictates 
# how many nodes the SOM will consist of. Currently not calculated 
# using PCA or other analyses methods.
nrows = 20
ncols = 30
# Create the SOM grid (which initializes the SOM network)
som_grid = somutils.create_grid(nrows, ncols, grid_type='hex')

# Initial neighbourhood radius is defaulted to 2/3 of the longest distance
# Should be set up similar to R package Kohonen
# https://cran.r-project.org/web/packages/kohonen/kohonen.pdf
# Radius will decay to 0.0 linearly
init_radius = somutils.default_radius(som_grid)

# Get the data as a matrix dropping the dataframe wrapper
#data = selected_feats_df_norm.as_matrix()
data = color_rgb_norm

# Number of iterations to run SOM
niter = 350

# Run SOM
som_weights, object_distances = somutils.run_som(data, som_grid, 'hex', niter, init_radius, init_learning_rate)
# Save SOM model. This is done by saving the weights (numpy ndarray)
#som_model_weights_path = os.path.join(workspace_dir, 'som_model.npy')
#np.save(som_model_weights_path, som_weights)

# It's possible that some data samples were not selected for training, thus do
# do not have a latest bmu
object_distances = somutils.fill_bmu_distances(data, som_weights, object_distances)

# Number of clusters to cluster SOM
nclusters = 6
# Cluster SOM nodes
clustering = somutils.cluster_som(som_weights, nclusters)

# Let's save the clusters corresponding to the samples now
#results_path = os.path.join(workspace_dir, 'cluster_results.csv')
#somutils.save_cluster_results(selected_data_feats_df, results_path, clustering.labels_, (nrows, ncols), object_distances)
# Display the SOM, coloring the nodes into different clusters from 
# 'clustering' above
# Optional: pass in original dataframe to plot 
# the IDs onto their respective nodes
som_figure_path = os.path.join('./','color_figure_blank.jpg')
basic_som_figure(data, som_weights, som_grid, 'hex', som_figure_path)

som_figure_path = os.path.join('./','color_figure_cluster.jpg')                            
#somutils.basic_som_figure(data, som_weights, som_grid, clustering.labels_,
#                                'hex', som_figure_path)
                            
