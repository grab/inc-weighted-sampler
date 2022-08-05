# Copyright 2022 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.

# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

import argparse
import os
import sys
import subprocess
import pobdd_parser
import pobdd_inference
import pobdd_utils
import obdd_parser
import obdd_smoother_utilities
import numpy as np
import time
import pickle
import smoother

def preprocess_cnf(cnf_file):
    '''
    Function to parse cnf file, create a temporary unweighted version of the cnf file and also extract the variable weights from it.
    Unweighter version cnf file is created at same location as the original cnf.
    Returns:
    - dict of {literal : weights}
    - file path of unweighted cnf
    - set of variables to sample against (we can trim the samples as a post processing step)
    '''
    # extract the file name
    cnf_path, cnf_name = os.path.split(cnf_file)
    cnf_path = cnf_path + '/'
    # generate name for unweighted cnf file
    unweighted_cnf_name = cnf_name + '.unweighted'

    # if the cnf exist from previous run, remove and regenerate
    try:
        os.remove(cnf_path + unweighted_cnf_name)
    except OSError:
        pass

    unweighted_file = open(cnf_path + unweighted_cnf_name, 'a+')

    weight_dict = {}

    sampling_set = set()

    with open(cnf_file, 'r') as fp:
        line = fp.readline()
        while line:
            line_list = line.split()
            if line_list[0].lower() == 'w':
                if line_list[2].lower() == 'inf':
                    weight_dict[int(line_list[1])] = float('inf')
                else:
                    weight_dict[int(line_list[1])] = float(line_list[2])
            elif len(line_list) > 2 and line_list[0] == 'c' and line_list[1] == 'ind':
                sampling_set.update([int(x) for x in line_list[2: -1]])
            else:
                # write the line to unweighted file
                unweighted_file.write(line)
            line = fp.readline()
    unweighted_file.close()

    # normalize the weights
    var_set = {x for x in weight_dict.keys() if x > 0}
    for var in var_set:
        if -var not in weight_dict:
            weight_dict[-var] = 1.0 - weight_dict[var]
        else:
            lo_weight = weight_dict[-var]
            hi_weight = weight_dict[var]
            if lo_weight == float('inf') and hi_weight == float('inf'):
                weight_dict[-var] = 0.5
                weight_dict[var] = 0.5
            elif lo_weight == float('inf'):
                weight_dict[-var] = 1.0
                weight_dict[var] = 0.0
            elif hi_weight == float('inf'):
                weight_dict[-var] = 0.0
                weight_dict[var] = 1.0
            else:
                weight_dict[-var] = lo_weight / (lo_weight + hi_weight)
                weight_dict[var] = 1.0 - weight_dict[-var]

    return weight_dict, cnf_path + unweighted_cnf_name, sampling_set

def compile_cnf_file(kc_compiler_path, unweighted_cnf_file, timeout=None, robdd=False):
    '''
    This function calls the kc compiler to compile the unweighted_cnf_file into an OBDD[AND]. This function is specifically designed to work with KCBOX Panini compiler.
    The output obdd file will be at the same location of the unweighted_cnf_file
    '''
    obdd_file = unweighted_cnf_file + '.obdd'
    # if exist from previous run, remove
    try:
        os.remove(obdd_file)
    except OSError:
        pass
    # determine if compiling to OBDD[AND] or regular OBDD
    if not robdd:
        command_string = kc_compiler_path + ' Panini --lang 1 --out ' + obdd_file + ' ' + unweighted_cnf_file
    else:
        command_string = kc_compiler_path + ' Panini --lang 0 --out ' + obdd_file + ' ' + unweighted_cnf_file
    if  timeout != None:
        command_string = 'timeout ' + str(timeout) + 's ' + command_string
    subprocess.call(command_string, shell=True)
    # check if compiled file exists
    compile_success = False
    if os.path.isfile(unweighted_cnf_file):
        compile_success = True
    return compile_success, obdd_file

def call_smoother(obdd_file):
    smooth_obdd_file = obdd_file + '.smooth'
    # if exist from previous run, remove
    try:
        os.remove(smooth_obdd_file)
    except OSError:
        pass

    variable_order = obdd_parser.get_variable_order(obdd_file)
    num_nodes = obdd_parser.get_num_nodes(obdd_file)
    # root is the last element in the list
    node_list = obdd_parser.parse_nodes(obdd_file)
    root_node = node_list[-1]
    true_node = None

    for node in node_list:
        if node.node_type == 'T':
            true_node = node
            break

    if true_node == None:
        print('No node indicating True in obdd file')
        sys.exit('No node indicating True in obdd file')
    
    node_list = smoother.smooth(node_list, root_node, true_node)
    obdd_smoother_utilities.write_obdd_to_file(node_list, root_node, variable_order, smooth_obdd_file)
    # check if smoothed file exists
    smooth_success = False
    if os.path.isfile(smooth_obdd_file):
        smooth_success = True
    return smooth_success, smooth_obdd_file

def parse_weight_file(weight_file):
    '''either specify all positive literal weights between 0 and 1 or
    specify weights for both positive and negative literals.'''
    data = open(weight_file).read()
    lines = data.strip().split("\n")
    weights = {int(x.split(',')[0]) : float(x.split(',')[1]) for x in lines}
    for lit, value in weights.items():
        if lit > 0:
            if -lit in weights:
                # normalize
                total = value + weights[-lit]
                weights[lit] = value / total
                weights[-lit] = 1.0 - (value / total)
    return weights

def apply_weights(node_list, weight_dict):
    '''
    Applies weights to decision nodes in node_list, the weights are from weight_dict, which is already normalized previously
    Function assumes that the weight to positive literal is always present.
    Does not return anything, modifies nodes in node_list in place
    '''
    for node in node_list:
        if node.node_type == 'D':
            var = node.variable
            if -var in weight_dict:
                node.branch_parameters = [weight_dict[-var], weight_dict[var]]
                node.normalize_weights()
            else:
                node.branch_parameters = [1.0 - weight_dict[var], weight_dict[var]]
    return

def parse_conditions(condition_file):
    '''
    Function to parse the condition file: contains literals to condition on.
    Assume condition is correct, that is positive and negative literal cannot appear in the file at the same time.
    File contains the literals separated by space within quotes on which you want to condition, should only have a single line
    Returns a set of literals that we can treat as an assignment.
    '''
    if condition_file == '':
        return set()
    with open(condition_file, 'r') as f:
        line = f.readline()
        condition_literals = line.split()
    return set(condition_literals)

def trim_samples(samples, sampling_set):
    lit_list = list(sampling_set)
    lit_list = lit_list + [-x for x in lit_list]
    original_shape = samples.shape
    mask = np.isin(samples, lit_list)
    samples = samples[mask].reshape([original_shape[0],-1])
    return samples

def sample_assignments(initial_partial_assignment_set, node_list, root_node, true_node, layer_lists, num_samples=1, seed=None, high_precision=False):
    '''
    A more general sampling function to determine which computation to perform while sampling - log space computation or arbitrary precision math computations
    '''
    if high_precision:
        return pobdd_inference.sample_assignment_pzddc_hp(initial_partial_assignment_set, 
        node_list, 
        root_node, 
        true_node, 
        layer_lists, 
        num_samples=num_samples, 
        num_processes=1, 
        seed=seed)
    else:
        return pobdd_inference.sample_assignment_pzddc(initial_partial_assignment_set, 
        node_list, 
        root_node, 
        true_node, 
        layer_lists, 
        num_samples=num_samples, 
        num_processes=1, 
        seed=seed)

if __name__ == '__main__':
    script_description = None

    parser = argparse.ArgumentParser(description=script_description)
    
    parser.add_argument('--cnf_file', type=str, help='/location/to/file.cnf', dest='cnf_file', required=True)

    parser.add_argument('--kc_compiler_path', type=str, help='/location/to/kcbox compiler executable', dest='kc_compiler_path', required=True)

    parser.add_argument('--num_samples', type=int, help='Number of samples to draw', dest='num_samples', required=True)

    parser.add_argument('--timeout', type=int, default=0, help='timeout in seconds for each component, suggest to put total timeout and run script with timeout python in bash, so total time taken abides by the timeout.', dest='timeout')

    parser.add_argument('--seed', type=int, default=42, help='seed for sampling', dest='seed')

    parser.add_argument('--condition_file', type=str, default='', help='specify the file containing the literals on which you want to condition, literals should be seperated by space and the file only contains 1 line', dest='condition_file')

    args = parser.parse_args()
    cnf_file = args.cnf_file

    # cleaning up cnf file
    weight_dict, unweighted_cnf_file, sampling_set = preprocess_cnf(cnf_file)

    # compiling
    compile_start = time.time()
    compile_success, obdd_file = compile_cnf_file(args.kc_compiler_path, unweighted_cnf_file, args.timeout)
    compile_end = time.time()
    
    if compile_success:
        print('Completed compilation')
    else:
        print('Something went wrong with KCBOX OBDD[AND] compilation')
        sys.exit()

    # smoothing 
    smoothing_start = time.time()
    smooth_success, smooth_obdd_file = call_smoother(obdd_file)
    smoothing_end = time.time()

    if smooth_success:
        print('Completed smoothing')
    else:
        print('Something went wrong with smoothing process')
        sys.exit()
    
    preprocessing_start = time.time()
    variable_ordering_list, num_nodes, root_node_id, node_list = pobdd_parser.parse_obdd(smooth_obdd_file)
    apply_weights(node_list, weight_dict)
    root_node = node_list[int(root_node_id)]

    # parse literals to be conditioned on if any
    conditional_assignment_set = parse_conditions(args.condition_file)

    # preprocessing for sampling
    forward_edge_zdd_adj_dict = pobdd_parser.parse_pobdd_zero_surpassed_forwardedges(node_list)
    toposort_result_sets = pobdd_parser.perform_toposort(forward_edge_zdd_adj_dict)
    layer_node_list = pobdd_inference.prepare_layer_node_lists(toposort_result_sets, node_list)
    true_node_id, false_node_id = pobdd_utils.get_true_false_node_id(node_list)
    true_node = node_list[true_node_id]
    preprocessing_end = time.time()

    sampling_start = time.time()
    samples = sample_assignments(conditional_assignment_set=conditional_assignment_set, 
                        node_list=node_list, 
                        root_node=root_node, 
                        true_node=true_node, 
                        layer_node_list=layer_node_list, 
                        num_samples=args.num_samples, 
                        seed=args.seed, 
                        high_precision=False)
    sampling_end = time.time()

    # trimming solutions, not projection
    trimming_start = time.time()
    if len(sampling_set) != len(variable_ordering_list):
        samples = trim_samples(samples, sampling_set)
    trimming_end = time.time()

    pobdd_output_file = smooth_obdd_file + '.pobdd'
    pobdd_utils.write_output_pobdd_file(node_list, root_node, variable_ordering_list, pobdd_output_file)
    # if exist from previous run, remove
    try:
        os.remove(pobdd_output_file + '.samples.pickle')
    except OSError:
        pass
    pickle.dump(samples, open(pobdd_output_file + '.samples.pickle', 'wb'))

    compile_time = compile_end - compile_start
    smoothing_time = smoothing_end - smoothing_start
    preprocessing_time = preprocessing_end - preprocessing_start
    sampling_time = sampling_end - sampling_start
    trimming_time = trimming_end - trimming_start

    # append to previous log file if any, should not
    with open(pobdd_output_file + '.log.txt', 'a+') as f:
        f.write('Time taken to compile OBDD[AND] in seconds: ' + str(compile_time) + '\n')
        f.write('Time taken to smooth and clean up OBDD[AND] in seconds: ' + str(smoothing_time) + '\n')
        f.write('Time taken to preprocess in seconds: ' + str(preprocessing_time) + '\n')
        f.write('Time taken to sample in seconds: ' + str(sampling_time) + '\n')
        f.write('Time taken to trim samples to sampling set in seconds: ' + str(trimming_time) + '\n')
        f.write('Total time taken: ' + str(compile_time + smoothing_time + preprocessing_time + sampling_time + trimming_time) + '\n')

