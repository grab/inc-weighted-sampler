# Copyright 2022 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.

# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

from collections import deque
from toposort import toposort
from obdd_nodes import ObddNode
import obdd_smoother_utilities


def parse_obdd_forwardedges(node_list, root_node):
    '''
    Function to create the forward edges of OBDD, from top to bottom
    Takes in a list of OBDD nodes and root_node
    We use root_node to perform clean up in the process, do not include unreachable nodes in adj list
    Returns a dictionary of node id : set of node ids, parent node id : set of child node ids
    '''
    queue = deque()
    visited = set()
    adj_dict = dict()
    queue.append(root_node)
    while len(queue) > 0:
        current_node = queue.popleft()
        visited.add(current_node.node_id)
        if current_node.node_type == 'T' or current_node.node_type == 'F':
            continue
        else:
            for child_id in current_node.child_list:
                if current_node.node_id in adj_dict:
                    adj_dict[current_node.node_id].add(child_id)
                else:
                    adj_dict[current_node.node_id] = {child_id}
                queue.append(node_list[child_id])
    return adj_dict, visited

def smooth(node_list, root_node, true_node):
    '''
    Function first perform toposort to get bottom up order and performs smoothing in bottom up order. 
    A new node_list is returned, this is after the removal of redundant nodes
    '''
    adj_dict, visited_set = parse_obdd_forwardedges(node_list, root_node)
    sorted_list_set = list(toposort(adj_dict))

    node_variable_set_list = []
    for i in range(len(node_list)):
        node_variable_set_list.append(set())
    
    # keep track of decision nodes added during smoothing (so that we do not add more than 1 for each variable)
    smooth_decision_node_variable_to_id_dict = dict()

    next_node_id_counter = len(node_list)
    # smooth + compress the toposorted nodes in bottom up order:
    for layer in sorted_list_set:
        for node_id in layer:
            current_node = node_list[node_id]
            if current_node.node_type == 'C':
                for child_id in current_node.child_list:
                    node_variable_set_list[current_node.node_id] = node_variable_set_list[current_node.node_id].union(node_variable_set_list[child_id])
            
            elif current_node.node_type == 'D':
                node_variable_set_list[current_node.node_id].add(current_node.variable)
                lo_child_node = node_list[current_node.child_list[0]]
                hi_child_node = node_list[current_node.child_list[1]]
                lo_child_variable_set = node_variable_set_list[lo_child_node.node_id]
                hi_child_variable_set = node_variable_set_list[hi_child_node.node_id]
                
                if lo_child_variable_set == hi_child_variable_set:
                    node_variable_set_list[current_node.node_id] = node_variable_set_list[current_node.node_id].union(lo_child_variable_set)
                else:
                    if len(lo_child_variable_set - hi_child_variable_set) > 0:
                        extra_variable_set = lo_child_variable_set - hi_child_variable_set

                        new_conjunction_node = ObddNode(node_type='C', node_id=next_node_id_counter)
                        next_node_id_counter += 1
                        new_conjunction_node_id = new_conjunction_node.node_id
                        node_list.append(new_conjunction_node)
                        visited_set.add(new_conjunction_node.node_id)
                        node_variable_set_list.append(extra_variable_set)
                        new_conjunction_node.child_list.append(hi_child_node.node_id)
                        
                        for missing_variable in extra_variable_set:
                            if missing_variable in smooth_decision_node_variable_to_id_dict:
                                new_conjunction_node.child_list.append(smooth_decision_node_variable_to_id_dict[missing_variable])
                            else:
                                new_variable_node = ObddNode(node_type='D', variable=missing_variable, node_id=next_node_id_counter)
                                next_node_id_counter += 1
                                node_variable_set_list.append(set([missing_variable]))
                                new_variable_node.child_list = [true_node.node_id, true_node.node_id]
                                node_list.append(new_variable_node)
                                new_conjunction_node.child_list.append(new_variable_node.node_id)
                                visited_set.add(new_variable_node.node_id)
                                smooth_decision_node_variable_to_id_dict[missing_variable] = new_variable_node.node_id
                        
                        current_node.child_list[1] = new_conjunction_node_id
                    
                    if len(hi_child_variable_set - lo_child_variable_set) > 0:
                        extra_variable_set = hi_child_variable_set - lo_child_variable_set
                        new_conjunction_node = ObddNode(node_type='C', node_id=next_node_id_counter)
                        next_node_id_counter += 1
                        new_conjunction_node_id = new_conjunction_node.node_id
                        node_list.append(new_conjunction_node)
                        new_conjunction_node.child_list.append(lo_child_node.node_id)
                        visited_set.add(new_conjunction_node_id)
                        node_variable_set_list.append(extra_variable_set)
                        
                        for missing_variable in extra_variable_set:
                            if missing_variable in smooth_decision_node_variable_to_id_dict:
                                new_conjunction_node.child_list.append(smooth_decision_node_variable_to_id_dict[missing_variable])
                            else:
                                new_variable_node = ObddNode(node_type='D', variable=missing_variable, node_id=next_node_id_counter)
                                next_node_id_counter += 1
                                node_variable_set_list.append(set([missing_variable]))
                                new_variable_node.child_list = [true_node.node_id, true_node.node_id]
                                node_list.append(new_variable_node)
                                new_conjunction_node.child_list.append(new_variable_node.node_id)
                                visited_set.add(new_variable_node.node_id)
                                smooth_decision_node_variable_to_id_dict[missing_variable] = new_variable_node.node_id
                        
                        current_node.child_list[0] = new_conjunction_node_id
                    node_variable_set_list[current_node.node_id] = node_variable_set_list[current_node.node_id].union(lo_child_variable_set)
                    node_variable_set_list[current_node.node_id] = node_variable_set_list[current_node.node_id].union(hi_child_variable_set)
            else:
                # true and false node
                pass
    node_list = [node for node in node_list if node.node_id in visited_set]

    node_list = obdd_smoother_utilities.reassign_node_ids(node_list)
    return node_list