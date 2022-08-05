# Copyright 2022 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.

# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

from obdd_nodes import ObddNode

def test_function():
    '''test function to print test'''
    print('test')

def get_variable_order(obdd_filename):
    '''
    Reads the variable ordering line, the first line of file
    Returns a list representing the ordering of variables
    '''
    variable_order_list = []
    
    with open(obdd_filename) as f:
        # read the first line of obdd file, containing variable order
        line = f.readline()
        # setting variable order list to be what is read from file
        variable_order_list = line.split()
        variable_order_list = variable_order_list[2:]

    return variable_order_list

def get_num_nodes(obdd_filename):
    '''
    Reads the number of nodes in obdd file, 2nd line of file
    Returns number of nodes
    '''
    num_nodes = -1

    with open(obdd_filename) as f:
        # skip the first line
        f.readline()
        # read the second line of obdd file, containing number of nodes
        line = f.readline()
        # setting variable order list to be what is read from file
        num_nodes = line.split()[-1]

    return num_nodes

def parse_nodes(obdd_filename):
    '''
    Reads the given obdd file, creates node object for each of the node lines in file
    Returns a list of ObddNode objects.
    '''
    node_list = []

    with open(obdd_filename) as f:
        # skip first 2 lines
        f.readline()
        f.readline()

        # read and create nodes, append to node list
        for line in f:
            # current line looks like this: ['19:', 'C', '14', '18', '0']
            # current_line_list = f.readline().split()
            current_line_list = line.split()
            # remove ':' for node id
            current_line_list[0] = current_line_list[0][:-1]
            # remove 0 that is used to indicate end of line
            current_line_list = current_line_list[:-1]
            # create node for each line in file, parent node list is not populated
            current_node = create_obdd_node(current_line_list)
            node_list.append(current_node)

    return node_list

def create_obdd_node(node_detail_list):
    '''
    Input is a list containing details of node
    For example: ['19', 'C', '14', '18'] --> [node id, type, lo, high, others]
    First position is node id, 2nd position is node type.
    Remaining position is lo, hi child, in case of decision node, and child in case of conjunction nodes
    Root is always at the last line of the file
    '''
    node = ObddNode()
    node.node_id = int(node_detail_list[0])

    current_node_type = node_detail_list[1]

    if current_node_type != 'C' and current_node_type != 'T' and current_node_type != 'F':
        node.node_type = 'D'
        node.variable = int(node_detail_list[1])
    else:
        node.node_type = current_node_type
    # if there is no child, in cases of T and F, the following line results in []
    node.child_list = [int(x) for x in node_detail_list[2:]]
    
    return node
