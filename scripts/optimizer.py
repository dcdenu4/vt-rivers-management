import random
import os
import sys
import itertools
import pickle

from deap import creator, base, tools, algorithms
import pandas as pd
import numpy as np

import somutils

from matplotlib import pyplot as plt
from matplotlib import patches as patches
from matplotlib import colors

# These should be updated from a param file
REACH_TYPES = {1: .3, 2: .8, 3: .9, 4: 1.0, 5: .6, 6: .2}
TMP_COSTS = {1: 500, 2: 200, 3: 150, 4: 400, 5:0, 6: 0, 7: 0}

OBJECTIVE_WEIGHTS = None
NORMALIZED_SEGS = None

def connected_segments():
    """Determine if any of the segments are adjacent """

    return None
    
def upstream_of_town():
    """Determine if any of the segments are upstream of a town """
    
    return None
    
def streampower_reduction_value():
    """Calculate the stream power reduction for each segment if it were restored """
    
    return None
    
def ecosystem_service_benefits():
    """Placeholder for any ES attempts """
    
    return None
    
def evaluate(individual, segments, lulc_coverage):
    """ 
    
        individual (list) : binary list indicating which segments are 'on' 
            or being chosen for conservation / restoration projects
            
        segments (pd df) : pandas dataframe with columns:
            sgat_pid_p2, class, cost, upstream_id, upstream_town, 
            upstream_dist, ssp
            
        lulc_coverage (dict) : dictionary with keys same as sgat_pid_p2 column 
            in `segments`. Each key points to a dictionary with lulc types 
            as keys and percent coverage as values
        
        
        Cost = 0-1 normalized , Cost Easement
        
        Benefit = Class regime normalized from ranking +
                  Distance Town, % length * Inverse dist from town (then normalize ?) +
                  % Ag LULC + 
                  Normalized SSP, only considering 34 - 300 values
        
        
    """
    
    MAX_PROJECTS = 30
    MAX_BUDGET = 1200000
    
    # All these should be normalized, I wonder if they can be normalized and 
    # Stored globally outside of here
    class_list = segments['class'].tolist()
    ids_list = segments['sgat_pid_p2'].tolist()
    #cost_list = segments['cost'].values
    upstream_ids_list = segments['upstream_ID'].tolist()
    upstream_town_list = segments['upstream_town'].tolist()
    upstream_town_dist_list = segments['upstream_dist'].tolist()
    ssp_list = segments['ssp_norm'].tolist()
    segment_length_list = segments['segment_length'].tolist()
    
    
    # Score based on selected segments and their SOM predicted types
    reach_type_score = np.zeros(len(individual))
    #reach_type_score = np.array([REACH_TYPES[class_list[i]] for i,j in enumerate(individual) if j == 1])
    for i,j in enumerate(individual):
        if j == 1:
            #print(i)
            #print(class_list[i])
            #print(reach_type_score[i])
            reach_type_score[i] = REACH_TYPES[class_list[i]]
            
    # Score based on Distance from town and segment length
    town_dist_score = np.zeros(len(individual))
    for i,j in enumerate(individual):
        if j == 1:
            if upstream_town_list[i]:
                if upstream_town_dist_list[i] == 0:
                    inverse_dist = 1.0
                    percent_length = 1.0
                else:
                    inverse_dist = 1.0 / upstream_town_dist_list[i]
                    percent_length = segment_length_list[i] / upstream_town_dist_list[i]
            else:
                inverse_dist = 0
                percent_length = 0
                
            town_dist_score[i] = percent_length * inverse_dist
                
                
    # Score based on LULC percent cover
    ag_lulc_code1 = 81
    ag_lulc_code2 = 82
    lulc_score = np.zeros(len(individual))
    for i,j in enumerate(individual):
        if j == 1:
            reach_id = ids_list[i]
            ag_value = 0
            if 81 in lulc_coverage[reach_id]:
                ag_value += lulc_coverage[reach_id][81]['percent'] / 100.0
            if 82 in lulc_coverage[reach_id]:
                ag_value += lulc_coverage[reach_id][82]['percent'] / 100.0
              
            lulc_score[i] = ag_value
    
    # Score based on ssp assuming already normalized
    ssp_selected =  np.zeros(len(individual))
    #ssp_selected = np.array([ssp_list[i] for i,j in enumerate(individual) if j == 1])
    for i,j in enumerate(individual):
        if j == 1:
            ssp_selected[i] = ssp_list[i]
    
    
    # Score based on adjacent segments
    #adjacency_count = 0
    #for i, j in enumerate(individual):
    #    if j == 1:
    #        cur_id = ids_list[i]
    #        for x, y in enumerate(individual):
    #            if y == 1 and x != i:
    #                if upstream_ids_list[i] == ids_list[x]:
    #                    adjacency_count += 1
    #                    break
                        
    #adjacency_score = adjacency_count * -1.0
    
    
    # Score based on cost
    #reach_cost_score = sum([segment_length_list[i] * segments['costPerLength'] for i,j in enumerate(individual) if j == 1])
    cost_per_acre = 3000
    cost_per_hectre = cost_per_acre / 0.404686
    
    # ((segment_length_list[i] * 200.0) / 107639.104) - converting square feet to hectare
    
    minimize_project_cost = sum([((segment_length_list[i] * 200.0) / 107639.104) * cost_per_hectre for i,j in enumerate(individual) if j == 1])
    
    #reach_cost_score = sum([segment_length_list[i] * 50 for i,j in enumerate(individual) if j == 1])
    
    #print(reach_type_score + town_dist_score + ssp_selected + lulc_score)
    
    reach_benefits_matrix = np.array([
        reach_type_score, town_dist_score, ssp_selected, lulc_score])
        
    reach_benefits_averaged = reach_benefits_matrix.mean(axis=0)
        
    maximize_benefits = sum(reach_benefits_averaged)    
    
    maximize_benefits = maximize_benefits / sum(individual)

    #if sum(individual) > MAX_PROJECTS:
    #    return minimize_project_cost * 2.0, 1
    
    return minimize_project_cost, maximize_benefits
    
    
def run_ga(param_file_path, objective_weights_path, rga_csv, lulc_coverage_dict, save_dir):
    """ """
    
    # Read in ga parameters
    params_df = pd.read_csv(param_file_path, header=None)
    # Since header names are row vertical, transpose to traditional
    params_df = params_df.T
    # Get the first row which now has header names
    new_header = params_df.iloc[0,:]
    # Get the dataframe without first row, so they can be set as proper header
    params_df = params_df.loc[1:]
    params_df.columns = new_header
    
    # Set population size, the number of possible solutions to carry
    population_size = params_df['Population Size']
    # Generations is the number of iterations to run the GA through
    # looking for better solutions
    number_generations = params_df['Generations']
    print('Pop Size: %d | Generations : %d' % (population_size, number_generations))
    
    OBJECTIVE_WEIGHTS = pd.read_csv(objective_weights_path)
    
    
    # Read in predicted reach types for segments
    segments_df = pd.read_csv(rga_csv)
    NORMALIZED_SEGS = somutils.normalize(segments_df, subset=['upstream_dist', 'ssp', 'segment_length'])
    segments_df['ssp_norm'] = NORMALIZED_SEGS['ssp']
    
    class_array = segments_df['class'].values
    # The individual for the GA will be a binary list of all the possible
    # segments that could be restored. That is, each index of the list 
    # represents a river segment, and a 0 indicates that segment is not 
    # part of the solution and a 1 indicates it is in the solution.
    # Where a solution is the combination of river segments that give the 
    # best bang for buck
    individual_size = len(segments_df['class'])
    print('Individual Size: %d' % (individual_size))

    # Set up the GA by defining fitness metric and what weights to track, 
    # so that a tuple with a single value represents a single objective 
    # problem, a tuple with two values represents a multi objective problem.
    creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))
    #creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
    # Create individual as a list with above fitness metrics
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    # A random binary generator
    toolbox.register("attr_bool", random.randint, 0, 1)
    # Individual definition
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     toolbox.attr_bool, n=individual_size)
    # Population definition
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # Evaluation function
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    #toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("select", tools.selNSGA2)
    # Create population of individuals
    population = toolbox.population(n=population_size)
    
    # Set up statistics to track
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    logbook = tools.Logbook()
    
    hof = tools.ParetoFront()
    
    print(population[0:5])

    # Start iterating over generations and searching for best individuals
    for gen in range(number_generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        segments_generator = itertools.repeat(segments_df, len(offspring))
        lulc_generator = itertools.repeat(lulc_coverage_dict, len(offspring))
        #print("Num Offspring: %d" % len(offspring))
        fits = toolbox.map(toolbox.evaluate, offspring, segments_generator, lulc_generator)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
            
        hof.update(offspring)
        
        population = toolbox.select(offspring, k=len(population))
        record = stats.compile(population)
        #print(record)
        #print(population[0:5])
        
        #logbook.record(gen=gen, evals=30, **record)
        logbook.record(gen=gen, **record)
    
    gen, avg, std = logbook.select("gen", "avg", "std")
    logbook.header = "gen", "avg", "min", "max", "std"
    
    print("Logbook:")
    np.set_printoptions(suppress=True)
    print(logbook)
    np.set_printoptions(suppress=False)
    
    logbook_file = open('logbook_test', 'wb')
    pickle.dump(logbook, logbook_file)
    
    top40 = tools.selBest(population, k=len(population))
    print(top40[0:3])
    top40_score = [list(toolbox.evaluate(i, segments_df, lulc_coverage_dict)) for i in top40]
    top40_num = [sum(i) for i in top40]
    hof_score = [list(toolbox.evaluate(i, segments_df, lulc_coverage_dict)) for i in hof]
    hof_num = [sum(i) for i in hof]
    
    top40_hof = top40_score + hof_score
    top40_hof_num = top40_num + hof_num
    
    top40_score = np.array(top40_score)
    hof_score = np.array(hof_score)
    
    print(top40_hof)
    #top40_hof = np.array(top40_hof)
    jet_cmap = plt.cm.get_cmap('viridis')
    plt.figure(figsize=(10,8))
    #plt.scatter(top40_hof[:,0] / 1000000.0, top40_hof[:,1], c=top40_hof_num, cmap=jet_cmap, label=top40_hof_num)
    
    top_plt = plt.scatter(top40_score[:,0] / 1000000.0, top40_score[:,1], c=top40_num, cmap=jet_cmap, label=top40_hof_num)
    hof_plt = plt.scatter(hof_score[:,0] / 1000000.0, hof_score[:,1], c=hof_num, cmap=jet_cmap, marker="x", s=52, label=top40_hof_num)
    
    plt.xlabel('Cost (Millions)', fontsize=16)
    plt.ylabel('Benefit Index', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    clb = plt.colorbar()
    clb.ax.tick_params(labelsize=14)
    clb.set_label('Number of Reaches', fontsize=16, labelpad=20)
    plt.title("Final Population and Hall of Fame Results", fontsize=16)
    plt.legend((top_plt, hof_plt), ('Population', 'HOF'), scatterpoints=1, loc='upper right', fontsize=10)
 
    pareto_results_path = os.path.join(save_dir, 'pareto_results_normalized.png')
    plt.savefig(pareto_results_path, bbox_inches='tight')
    plt.show()
    
    avg = np.array(avg)
    std = np.array(std)

    plt.figure(figsize=(10,8))
    #plt.errorbar(gen, avg[:,0] / 1000000.0, std[:,0] / 1000000.0, errorevery=20, barsabove=True, ecolor="black", c="red")
    plt.plot(gen, avg[:,0] / 1000000.0, c="black")
    plt.xlabel('Generation', fontsize=16)
    plt.ylabel('Average Cost (Millions)', fontsize=16)
    plt.title("Average Cost", fontsize=16)
    pareto_results_path = os.path.join(save_dir, 'avg_cost_results.png')
    plt.savefig(pareto_results_path, bbox_inches='tight')
    #plt.show()
    
    plt.figure(figsize=(10,8))
    #plt.errorbar(gen, avg[:,1], std[:,1], errorevery=20, barsabove=True, ecolor="black", c="green")
    plt.errorbar(gen, avg[:,1], c="blue")
    plt.xlabel('Generation', fontsize=16)
    plt.ylabel('Average Benefit Index', fontsize=16)
    plt.title("Average Benefit Index", fontsize=16)
    pareto_results_path = os.path.join(save_dir, 'avg_benefits_results.png')
    plt.savefig(pareto_results_path, bbox_inches='tight')
    #plt.show()
    
    
if __name__ == "__main__":
    """Running as main program, NOT as import."""
  
    segment_class_path = os.path.join('.', 'segment-fake-data-optimizer.csv')
    optimizer_param_path = os.path.join('.', 'optimizer-params.csv')
        
    run_ga(optimizer_param_path, segment_class_path)
    
    print("Finished")