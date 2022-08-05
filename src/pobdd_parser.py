# Copyright 2022 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.

# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

from pobdd_nodes import PObddNode
from toposort import toposort

def get_variable_order(diagram_filename):
    '''
    Reads the variable ordering line, the first line of file
    For both obdd and pobdd files
    Returns a list representing the ordering of variables
    '''
    variable_order_list = []
    
    with open(diagram_filename) as f:
        # read the first line of pobdd/obdd file, containing variable order
        line = f.readline()
        variable_order_list = line.split()
        variable_order_list = variable_order_list[2:]

    return variable_order_list

def get_num_nodes(diagram_filename):
    '''
    Reads the number of nodes in pobdd/obdd file, 2nd line of file
    Returns number of nodes
    '''
    num_nodes = -1

    with open(diagram_filename) as f:
        f.readline()
        line = f.readline()
        num_nodes = line.split()[-1]

    return num_nodes

def get_root_node_id(diagram_filename):
    '''
    Reads the root node id in the pobdd/obdd file, 3rd line of file
    Returns the root node id
    '''
    root_node_id = -1

    with open(diagram_filename) as f:
        f.readline()
        f.readline()
        line = f.readline()
        root_node_id = line.split()[-1]
    
    return root_node_id


def parse_obdd(obdd_filename):
    '''
    This function takes in a smoothed obdd file, and parses it.
    
    Returns:
    list representing variable ordering  
    string number of nodes  
    string root node id  
    list of PObddNode objects - with default parameters 0.5, 0.5
    '''
    variable_order_list = get_variable_order(obdd_filename)
    num_nodes = get_num_nodes(obdd_filename)
    root_node_id = get_root_node_id(obdd_filename)

    node_list = []

    with open(obdd_filename) as f:
        # skip first 3 lines
        f.readline()
        f.readline()
        f.readline()

        # read and create nodes, append to node list
        for line in f:
            # current line looks like this: ['19:', 'C', '14', '18', '0']
            current_line_list = line.split()
            # remove ':' for node id
            current_line_list[0] = current_line_list[0][:-1]
            # remove 0 for delimiter
            current_line_list = current_line_list[:-1]
            current_node = create_pobdd_node_from_obdd(current_line_list)
            node_list.append(current_node)
        
    return variable_order_list, num_nodes, root_node_id, node_list

def create_pobdd_node_from_obdd(node_detail_list):
    '''
    Input is a list containing details of node from obdd file
    For example: ['19', 'C', '14', '18'] --> [node id, type, lo, high, others]
    First position is node id, 2nd position is node type.
    Remaining position is lo, hi child, in case of decision node, and child in case of conjunction nodes
    As it is from obdd, we first set the pobdd decision node (type 'D') parameters to be 0.5 and counts to be 0
    '''
    node = PObddNode()
    node.node_id = int(node_detail_list[0])

    current_node_type = node_detail_list[1]

    if current_node_type != 'C' and current_node_type != 'T' and current_node_type != 'F':
        node.node_type = 'D'
        node.variable = int(node_detail_list[1])
        # set parameters to 0.5 and counts to 0
        node.branch_parameters = [0.5, 0.5]
        node.branch_counts = [0, 0]
    else:
        node.node_type = current_node_type
    # if there is no child, in cases of T and F, the following line results in []
    node.child_list = [int(x) for x in node_detail_list[2:]]
    
    return node

def parse_pobdd(obdd_filename):
    '''
    This function takes in a pobdd file, and parses it.
    
    Returns:
    list representing variable ordering
    string number of nodes
    string root node id
    list of PObddNode objects - with default parameters 0.5, 0.5
    '''
    variable_order_list = get_variable_order(obdd_filename)
    num_nodes = get_num_nodes(obdd_filename)
    root_node_id = get_root_node_id(obdd_filename)

    node_list = []

    with open(obdd_filename) as f:
        # skip first 3 lines
        f.readline()
        f.readline()
        f.readline()

        # read and create nodes, append to node list
        for line in f:
            # current line looks like this: ['19:', 'C', '14', '18', '0']
            current_line_list = line.split()
            # remove ':' for node id
            current_line_list[0] = current_line_list[0][:-1]
            # remove 0 that is used to indicate end of line
            current_line_list = current_line_list[:-1]
            # create node for each line in file
            current_node = create_pobdd_node(current_line_list)
            node_list.append(current_node)
        
    return variable_order_list, num_nodes, root_node_id, node_list

def create_pobdd_node(pobdd_detail_list):
    '''
    Input is a list containing details of node from pobdd file
    First position is node id, 2nd position is node type
    For True and False nodes: [node id, type]
    For Decision nodes:
    For example: ['7', '1', '14', '0.6', '6', '12', '0.4', '4']
    Format: [node id, type, lo id, lo param, lo count, hi id, hi param, hi count]
    For conjunction nodes: 
    node id, type, <child ids>...]
    '''
    node = PObddNode()
    node.node_id = int(pobdd_detail_list[0])

    current_node_type = pobdd_detail_list[1]

    if current_node_type != 'C' and current_node_type != 'T' and current_node_type != 'F':
        node.node_type = 'D'
        node.variable = int(pobdd_detail_list[1])
        # set parameters to 0.5 and counts to 0
        node.branch_parameters = [float(pobdd_detail_list[3]), float(pobdd_detail_list[6])]
        node.branch_counts = [int(pobdd_detail_list[4]), int(pobdd_detail_list[7])]
        node.child_list = [int(pobdd_detail_list[2]), int(pobdd_detail_list[5])]
    else:
        node.node_type = current_node_type
        node.child_list = [int(x) for x in pobdd_detail_list[2:]]
    
    return node

def parse_pobdd_backedges(node_list):
    '''
    Function to create the backward edges of POBDD, from bottom to top
    Takes in a list of POBDD nodes
    Returns a dictionary of node id : set of node ids, child node id : set of parent node ids
    '''
    reverse_adj_dict = {}
    for pobdd_node in node_list:
        current_node_id = pobdd_node.node_id
        current_child_list = pobdd_node.child_list
        if len(current_child_list) == 0:
            # no child so no edges to add
            continue
        for child_id in current_child_list:
            # there are child nodes
            if child_id in reverse_adj_dict:
                # add to the set
                reverse_adj_dict[child_id].add(current_node_id)
            else:
                reverse_adj_dict[child_id] = set([current_node_id])
    return reverse_adj_dict 

def parse_pobdd_forwardedges(node_list):
    '''
    Function to create the forward edges of POBDD, from top to bottom
    Takes in a list of POBDD nodes
    Returns a dictionary of node id : set of node ids, parent node id : set of child node ids
    '''
    adj_dict = dict()
    for node in node_list:
        # if either true or false node, continue as they have no child
        if node.node_type == 'T' or node.node_type == 'F':
            continue
        else:
            # either conjunction or decision node
            for child_id in node.child_list:
                if node.node_id in adj_dict:
                    adj_dict[node.node_id].add(child_id)
                else:
                    adj_dict[node.node_id] = {child_id}
    return adj_dict

def parse_pobdd_zero_surpassed_forwardedges(node_list):
    '''
    Function to create the forward edges of POBDD, from top to bottom, ignoring edges that lead to false node
    In the process, we edit the false_child_branch of each POBDD node
    Takes in a list of POBDD nodes
    Returns a dictionary of node id : set of node ids, parent node id : set of child node ids
    NOTE: this is not using zdd, it is zdd like in a sense that all unsat edges (leading to false node is surpressed)
    '''
    adj_dict = dict()
    for node in node_list:
        if node.node_type == 'T' or node.node_type == 'F':
            continue
        else:
            # either conjunction or decision node
            for i in range(len(node.child_list)):
                child_id = node.child_list[i]
                # ignore the false nodes
                if node_list[child_id].node_type == 'F':
                    # here we only handle decision node, assumption is conjunction nodes do not have false node as child as it does not make sense to AND False, since it can be replaced with False.
                    # assumption here is also we do not have a decision node with both branches = False node
                    node.false_child_branch = i
                    continue

                if node.node_id in adj_dict:
                    adj_dict[node.node_id].add(child_id)
                else:
                    adj_dict[node.node_id] = {child_id}
    return adj_dict

def parse_pobdd_reduced_forwardedges(node_list):
    '''
    Wrapper for getting an adjacency dictionary for toposort
    '''
    return parse_pobdd_zero_surpassed_forwardedges(node_list)

def perform_toposort(pobdd_reverse_adj_dict):
    '''
    Performs toposort on the back edges of pobdd, so from bottom to top, so as to know what nodes can be lumped together in matrix multiplication calculation later on
    Takes in - adj dict of backedges
    Returns - sets of node ids that can be together in a 'layer'
    From toposort docs - Returns an iterator describing the dependencies among nodes in the input data. Each returned item will be a set. Each member of this set has no dependencies in this set, or in any set previously returned.
    '''
    sorted_list_set = list(toposort(pobdd_reverse_adj_dict))
    # could be the case that the true and false nodes are in different layers
    return sorted_list_set