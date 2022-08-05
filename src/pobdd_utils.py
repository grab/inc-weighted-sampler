# Copyright 2022 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.

# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

END_LINE = '0\n'

def get_output_line(node):
    '''
    This function returns the line representing the node, when writing to file
    '''
    node_type = node.node_type
    if node_type == 'F':
        return str(node.node_id) + ':\t' + 'F' + ' ' + END_LINE
    elif node_type == 'T':
        return str(node.node_id) + ':\t' + 'T' + ' ' + END_LINE
    elif node_type == 'C':
        output_line = str(node.node_id) + ':\t' + 'C'
        for child_id in node.child_list:
            output_line = output_line + ' ' + str(child_id)
        output_line = output_line + ' ' + END_LINE
        return output_line
    else:
        output_line = str(node.node_id) + ':\t' + str(node.variable) + ' ' + str(node.child_list[0]) + ' ' + str(node.branch_parameters[0]) + ' ' + str(node.branch_counts[0]) + ' ' + str(node.child_list[1]) + ' ' + str(node.branch_parameters[1]) + ' ' + str(node.branch_counts[1]) + ' ' + END_LINE
        return output_line

def write_output_pobdd_file(node_list, root_node, variable_order_list, output_filename):
    '''
    Function to write the pobdd to an output file.
    node_list - list of pobdd nodes
    variable_order_listring - variable ordering list
    root_node - root node
    output_filename - filename of the file to write to
    '''
    with open(output_filename, 'a+') as f:
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
            current_line = get_output_line(node)
            f.write(current_line)
    return


def write_assignemnt_data_to_file(assignment_list, file_object):
    line = ''
    for variable in assignment_list:
        line = line + str(variable) + ' '
    line = line + '0\n'
    file_object.write(line)

def get_true_false_node_id(node_list):
    '''
    Function takes in list of nodes of POBDD, to return the node ids for the true node and false node in the list
    NOTE: we assume that there is only 1 true node and 1 false node in the given list.
    '''
    true_node_id = -1
    false_node_id = -1
    for node in node_list:
        if node.node_type == 'T':
            true_node_id = node.node_id
        if node.node_type == 'F':
            false_node_id = node.node_id
        # check if can terminate early
        if true_node_id != -1 and false_node_id != -1:
            break
    return (true_node_id, false_node_id)
