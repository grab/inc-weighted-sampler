# Copyright 2022 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.

# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

from collections import deque

def generate_node_string(node):
    '''
    Function to generate string representation of node, to be written to file.
    '''
    node_string = ''
    node_string += str(node.node_id) + ':' + '\t'
    if node.node_type == 'F' or node.node_type == 'T':
        node_string += node.node_type + ' 0' + '\n'
        return node_string
    elif node.node_type == 'C':
        node_string += node.node_type
        for child_id in node.child_list:
            node_string += ' ' + str(child_id)
        node_string += ' 0' + '\n'
        return node_string
    else:
        # decision node
        node_string += str(node.variable) + ' ' + str(node.child_list[0]) + ' ' + str(node.child_list[1]) + ' 0' + '\n'
        return node_string

def write_obdd_to_file(node_list, root_node, variable_order_list, filename):
    '''
    Writing obdd to file, in the original format
    with additional line indicating which node is root node
    '''
    with open(filename, 'a+') as f:
        # generate and write first line
        variable_order_string = 'Variable order:'
        for variable in variable_order_list:
            variable_order_string = variable_order_string + ' ' + variable
        variable_order_string = variable_order_string + '\n'
        f.write(variable_order_string)

        # generate second line: ''Number of nodes: 20''
        f.write('Number of nodes: ' + str(len(node_list)) + '\n')

        f.write('Root node: ' + str(root_node.node_id) + '\n')

        for node in node_list:
            current_line = generate_node_string(node)
            f.write(current_line)

    return

def clean_up_unused_nodes(node_list, root_node):
    '''
    Call to clean up the unused node from node_list, 
    all nodes that are unreachable from root_node will be removed
    '''

    reachable_set = set()
    traversal_stack = deque()
    traversal_stack.append(root_node.node_id)

    while len(traversal_stack) > 0:
        current_node_id = traversal_stack.pop()
        reachable_set.add(current_node_id)
        for child_id in node_list[current_node_id].child_list:
            traversal_stack.append(child_id)

    # now we have a set of reachable nodes from the root node, all others can be cleaned
    new_node_list = [node for node in node_list if node.node_id in reachable_set]

    return new_node_list

def reassign_node_ids(node_list):
    '''
    Reassigns the id of the nodes in the list, after unnecessary nodes are removed
    Ideally nodes id should correspond to their position in node list for ease,
    both when writing out to file and for access in other parts of the program

    Call after clean up

    Takes in list of nodes, remaps the node ids, including the child nodes
    '''

    old_id_to_new_id_mapping = dict()
    new_id_to_old_child_list_mapping = dict()

    # get an old id to new id mapping look up
    for i in range(len(node_list)):
        current_node = node_list[i]
        old_id_to_new_id_mapping[current_node.node_id] = i
        if current_node.node_type == 'C' or current_node.node_type == 'D':
            new_id_to_old_child_list_mapping[i] = current_node.child_list.copy()

    # might have issue that if we loop the node list, we could have assigned a new node id to child first. Take for example n1 is child of n2. We assign n2 to id 20, then we process n1. Now child of n1 is 20 (already reassigned) and not the original n2 (which is the old one)
    for node in node_list:
        old_node_id = node.node_id
        new_node_id = old_id_to_new_id_mapping[old_node_id]
        node.node_id = new_node_id
        # only need to process child list for conjunction and decision nodes
        if node.node_type == 'C' or node.node_type == 'D':
            new_child_list = []
            for old_child_id in new_id_to_old_child_list_mapping[new_node_id]:
                new_child_list.append(old_id_to_new_id_mapping[old_child_id])
            node.child_list = new_child_list
    
    return node_list
