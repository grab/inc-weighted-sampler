# Copyright 2022 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.

# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

from collections import deque
import numpy as np
from gmpy2 import mpq

def get_probability_traversal_seq(assignment_set, node_list, root_node):
    '''
    Return a post order traversal from leaf node to root node in a stack
    adapted from: https://www.geeksforgeeks.org/iterative-postorder-traversal/
    '''
    output_stack = deque()
    temp_stack = deque()

    temp_stack.append(root_node.node_id)
    while len(temp_stack) > 0:
        current_node_id = temp_stack.pop()
        output_stack.append(current_node_id)

        current_node = node_list[current_node_id]
        current_node_type = current_node.node_type

        if current_node_type == 'T' or current_node_type == 'F':
            # leaf node no child, do not do anything
            continue
        elif current_node_type == 'C':
            # need to traverse all child of conjunction node
            for child_id in current_node.child_list:
                temp_stack.append(child_id)
        else:
            # decision node
            current_variable = current_node.variable
            pos_literal = current_variable
            neg_literal = -1 * current_variable
            # assuming assignment set is sound - no pos and neg literal appear at same time
            if pos_literal in assignment_set:
                # push hi branch into stack
                temp_stack.append(current_node.child_list[1])
            elif neg_literal in assignment_set:
                # push lo branch into stack
                temp_stack.append(current_node.child_list[0])
            else:
                # unassigned, need to push both branches into stack
                for child_id in current_node.child_list:
                    temp_stack.append(child_id)

    return output_stack


#######################
# below is for INC sampling, bottom up, only from true node (false node is invalid anyways)
#######################

def prepare_layer_node_lists(topo_node_list, node_list):
    '''
    This function takes in the toposorted node ids, seperates the nodes in each layer into conjunction and decision node lists. Refer to prepare layer matrices, this function just do not return the matrices, just the node lists for each layer.
    The first layer should only consist of true node
    Last layer should only consists of root node
    Returning a list of tuples (layer_decision_node_list, layer_conjunction_node_list)
    '''
    # checking if the toposort is corret
    assert(len(topo_node_list[0]) == 1)
    assert(node_list[list(topo_node_list[0])[0]].node_type == 'T')

    layer_node_list = []

    for i in range(1, len(topo_node_list)):
        conjunction_list = []
        decision_list = []
        for node_id in topo_node_list[i]:
            if node_list[node_id].node_type == 'C':
                conjunction_list.append(node_id)
            elif node_list[node_id].node_type == 'D':
                decision_list.append(node_id)
            else:
                print('Met a node that is neither conjunction or decision in middle layers, probabbly something wrong happened in toposort')
                break
        conjunction_list.sort()
        decision_list.sort()
        layer_node_list.append((decision_list, conjunction_list))
    return layer_node_list


def calculate_logprob(assignment_set, node_list, root_node, true_node, layer_lists):
    '''
    Function takes in:
    assignment_set - variable literals are ints in the set and both positive and negative literals should be included in the set. If variable does not appear in the set, it means they are not assigned.
    node_list - list of pobdd nodes
    root_node - root pobdd node, where to start traversing from
    true_node - true_node, to initialize the probabtility cache
    layer_lists - toposorted (decision node list, conjunction node list) for each layer in toposort, for bottom up traversal
    Function returns the log probability of the partial assignment based on the parameters of the pobdd nodes. 

    NOTE: if a complete assignment is passed in, then the calculated log probability is the log probability of that assignment.
    if empty assignment is passed in, the total probability of an assignment being a satisfying assignment is calculated
    '''

    joint_logprob_dict = {}
    joint_logprob_dict[true_node.node_id] = np.log(1.0)

    for i in range(len(layer_lists)):
        decision_node_list, conjunction_node_list = layer_lists[i]

        if len(conjunction_node_list) > 0:
            # process the conjunction nodes
            for conjunction_node_id in conjunction_node_list:
                # assignment_list = []
                current_logprob = np.log(1.0)
                current_node = node_list[conjunction_node_id]
                for child_id in current_node.child_list:
                    # product of child probabilties, in log space is sum
                    # if a child is not valid, its log prob is -np.inf, anything add to -np.inf still gets -np.inf
                    current_logprob += joint_logprob_dict[child_id]
                joint_logprob_dict[conjunction_node_id] = current_logprob

        if len(decision_node_list) > 0:
            # process the decision nodes
            for decision_node_id in decision_node_list:
                decision_node = node_list[decision_node_id]
                pos_literal = decision_node.variable
                neg_literal = -1 * pos_literal

                if decision_node.false_child_branch != None:
                    # if one child is false
                    if decision_node.false_child_branch == 0:
                        # lo child is false node
                        if joint_logprob_dict[decision_node.child_list[1]] == -np.inf:
                            # invalid child
                            joint_logprob_dict[decision_node_id] = np.log(0)
                            continue
                        
                        if neg_literal in assignment_set:
                            # invalid conditioning
                            joint_logprob_dict[decision_node_id] = np.log(0)
                        else:
                            # calculate joint prob
                            # branch parameter x joint prob of child
                            joint_logprob_dict[decision_node_id] = joint_logprob_dict[decision_node.child_list[1]] + np.log(decision_node.branch_parameters[1])
                    else:
                        # hi child is false node
                        if joint_logprob_dict[decision_node.child_list[0]] == -np.inf:
                            # invalid child
                            joint_logprob_dict[decision_node_id] = np.log(0)
                            continue

                        if pos_literal in assignment_set:
                            # invalid conditioning
                            joint_logprob_dict[decision_node_id] = np.log(0)
                        else:
                            joint_logprob_dict[decision_node_id] = joint_logprob_dict[decision_node.child_list[0]] + np.log(decision_node.branch_parameters[0])
                else:
                    # decision node both child not false
                    if joint_logprob_dict[decision_node.child_list[0]] == -np.inf and joint_logprob_dict[decision_node.child_list[1]] == -np.inf:
                        joint_logprob_dict[decision_node_id] = np.log(0)
                        continue

                    if neg_literal in assignment_set:
                        # we do not have to detect if lo child invalid because if invalid the joint prob value in dict should be -np.inf which is also what is used to indicate invalid
                        joint_logprob_dict[decision_node_id] = joint_logprob_dict[decision_node.child_list[0]] + np.log(decision_node.branch_parameters[0])
                    elif pos_literal in assignment_set:
                        joint_logprob_dict[decision_node_id] = joint_logprob_dict[decision_node.child_list[1]] + np.log(decision_node.branch_parameters[1])
                    else:
                        lo_jointprob, hi_jointprob = np.log(decision_node.branch_parameters)
                        lo_jointprob = lo_jointprob + joint_logprob_dict[decision_node.child_list[0]]
                        hi_jointprob = hi_jointprob + joint_logprob_dict[decision_node.child_list[1]]
                        total_jointprob = np.logaddexp(lo_jointprob, hi_jointprob)
                        joint_logprob_dict[decision_node.node_id] = total_jointprob
    
    return joint_logprob_dict[root_node.node_id]

def calculate_prob_hp(assignment_set, node_list, root_node, true_node, layer_lists):
    '''
    Function takes in:
    assignment_set - variable literals are ints in the set and both positive and negative literals should be included in the set. If variable does not appear in the set, it means they are not assigned.
    node_list - list of pobdd nodes
    root_node - root pobdd node, where to start traversing from
    true_node - true_node, to initialize the probabtility cache
    layer_lists - toposorted (decision node list, conjunction node list) for each layer in toposort, for bottom up traversal
    Function returns the probability of the partial assignment based on the parameters of the pobdd nodes. 

    NOTE: if a complete assignment is passed in, then the calculated probability is the probability of that assignment.
    if empty assignment is passed in, the total probability of an assignment being a satisfying assignment is calculated

    This method uses arbitrary precision math to prevent numerical underflow, returns mpq number
    '''

    prob_dict = {}
    prob_dict[true_node.node_id] = mpq(1)

    for i in range(len(layer_lists)):
        decision_node_list, conjunction_node_list = layer_lists[i]

        if len(conjunction_node_list) > 0:
            # process the conjunction nodes
            for conjunction_node_id in conjunction_node_list:
                current_prob = mpq(1)
                current_node = node_list[conjunction_node_id]
                for child_id in current_node.child_list:
                    current_prob = current_prob * prob_dict[child_id]
                prob_dict[conjunction_node_id] = current_prob

        if len(decision_node_list) > 0:
            # process the decision nodes
            for decision_node_id in decision_node_list:
                decision_node = node_list[decision_node_id]
                pos_literal = decision_node.variable
                neg_literal = -1 * pos_literal

                if decision_node.false_child_branch != None:
                    # if one child is false
                    if decision_node.false_child_branch == 0:
                        # lo child is false node
                        if prob_dict[decision_node.child_list[1]] == mpq(0):
                            prob_dict[decision_node_id] = mpq(0)
                            continue
                        
                        if neg_literal in assignment_set:
                            # invalid conditioning
                            prob_dict[decision_node_id] = mpq(0)
                        else:
                            # calculate joint prob
                            # branch parameter x joint prob of child
                            prob_dict[decision_node_id] = mpq(decision_node.branch_parameters[1]) * prob_dict[decision_node.child_list[1]]
                    else:
                        # hi child is false node
                        if prob_dict[decision_node.child_list[0]] == mpq(0):
                            prob_dict[decision_node_id] = mpq(0)
                            continue

                        if pos_literal in assignment_set:
                            # invalid conditioning
                            prob_dict[decision_node_id] = mpq(0)
                        else:
                            prob_dict[decision_node_id] = mpq(decision_node.branch_parameters[0]) * prob_dict[decision_node.child_list[0]]
                else:
                    # decision node both child not false
                    if prob_dict[decision_node.child_list[0]] == mpq(0) and prob_dict[decision_node.child_list[1]] == mpq(0):
                        prob_dict[decision_node_id] = mpq(0)
                        continue

                    if neg_literal in assignment_set:
                        # we do not have to detect if lo child invalid because if invalid the joint prob value in dict should be -np.inf which is also what is used to indicate invalid
                        prob_dict[decision_node_id] = prob_dict[decision_node.child_list[0]] * mpq(decision_node.branch_parameters[0])
                    elif pos_literal in assignment_set:
                        prob_dict[decision_node_id] = prob_dict[decision_node.child_list[1]] * mpq(decision_node.branch_parameters[1])
                    else:
                        lo_prob = prob_dict[decision_node.child_list[0]] * mpq(decision_node.branch_parameters[0])
                        hi_prob = prob_dict[decision_node.child_list[1]] * mpq(decision_node.branch_parameters[1])
                        total_prob = lo_prob + hi_prob
                        prob_dict[decision_node.node_id] = total_prob
    return prob_dict[root_node.node_id]

def sample(initial_partial_assignment_set, node_list, root_node, true_node, layer_lists, num_samples=1, seed=None, hp=False):
    '''
    Function to simplify the api, calls real sampling function underneath
    '''
    if hp:
        return sample_assignment_pzddc_hp(initial_partial_assignment_set, node_list, root_node, true_node, layer_lists, num_samples, num_processes=1, seed=seed)
    else:
        return sample_assignment_pzddc(initial_partial_assignment_set, node_list, root_node, true_node, layer_lists, num_samples, num_processes=1, seed=seed)


def sample_assignment_pzddc(initial_partial_assignment_set, node_list, root_node, true_node, layer_lists, num_samples=1, num_processes=1, seed=None):

    if len(initial_partial_assignment_set) == 0:
        # empty set, use more optimized function
        return sample_assignment_pzddc_empty(node_list, root_node, true_node, layer_lists, num_samples=num_samples, seed=seed)

    # set seed for reproducibility
    if seed != None:
        np.random.seed(seed)
    
    node_assignment_output = {}
    # we will check for true node by checking if type is int
    node_assignment_output[true_node.node_id] = np.zeros([1,1])

    assignment_set = initial_partial_assignment_set

    # need a dictionary to store log(natual log) joint prob from dynamic annotation 
    # NOTE we also use log sum exp trick to avoid need for high precision math (exp the log probabilities and sum them (decision nodes))
    joint_logprob_dict = {}
    joint_logprob_dict[true_node.node_id] = np.log(1.0)


    for i in range(len(layer_lists)):
        decision_node_list, conjunction_node_list = layer_lists[i]

        if len(conjunction_node_list) > 0:
            for conjunction_node_id in conjunction_node_list:
                assignment_list = []
                current_logprob = np.log(1.0)
                current_node = node_list[conjunction_node_id]
                has_invalid_child = False

                for child_id in current_node.child_list:
                    if type(node_assignment_output[child_id]) == int:
                        current_logprob = np.log(0.0)
                        has_invalid_child = True
                        break
                    elif node_list[child_id].node_type == 'T':
                        continue
                    else:
                        assignment_list.append(node_assignment_output[child_id])
                        current_logprob = current_logprob + joint_logprob_dict[child_id]
                if has_invalid_child:
                    node_assignment_output[conjunction_node_id] = 0
                    joint_logprob_dict[conjunction_node_id] = np.log(0.0)
                else:
                    current_assignment = np.concatenate(assignment_list, axis=1)
                    node_assignment_output[conjunction_node_id] = current_assignment
                    joint_logprob_dict[conjunction_node_id] = current_logprob
        # process decision nodes
        if len(decision_node_list) > 0:
            # process the decision nodes in parallel (might want to)
            for decision_node_id in decision_node_list:
                decision_node = node_list[decision_node_id]
                pos_literal = decision_node.variable
                neg_literal = -1 * pos_literal
                if decision_node.false_child_branch != None:
                    # if decision node has false node in one of its branches, false_child_branch should only be 0 or 1
                    if decision_node.false_child_branch == 0:
                        if type(node_assignment_output[decision_node.child_list[1]]) == int:
                            node_assignment_output[decision_node_id] = 0
                            joint_logprob_dict[decision_node_id] = np.log(0)
                            continue
                        # lo branch is false node originally
                        if neg_literal in assignment_set:
                            # incompatible
                            node_assignment_output[decision_node_id] = 0
                            joint_logprob_dict[decision_node_id] = np.log(0)
                        else:
                            # compatible
                            current_assignment = np.full([num_samples, 1], pos_literal, dtype=np.int)
                            if node_list[decision_node.child_list[1]].node_type == 'T':
                                pass
                            else:
                                current_assignment = np.hstack((current_assignment, node_assignment_output[decision_node.child_list[1]]))
                            node_assignment_output[decision_node_id] = current_assignment
                            # branch parameter x joint prob of child
                            joint_logprob_dict[decision_node_id] = joint_logprob_dict[decision_node.child_list[1]] + np.log(decision_node.branch_parameters[1])
                    else:
                        # positive branch is false node
                        if type(node_assignment_output[decision_node.child_list[0]]) == int:
                            node_assignment_output[decision_node_id] = 0
                            joint_logprob_dict[decision_node_id] = np.log(0)
                            continue

                        if pos_literal in assignment_set:
                            # incompatible
                            node_assignment_output[decision_node_id] = 0
                            joint_logprob_dict[decision_node_id] = np.log(0)
                        else:
                            # compatible
                            current_assignment = np.full([num_samples, 1], neg_literal, dtype=np.int)
                            if node_list[decision_node.child_list[0]].node_type == 'T':
                                # if child is true node we do not do anything
                                pass
                            else:
                                current_assignment = np.hstack((current_assignment, node_assignment_output[decision_node.child_list[0]]))
                            node_assignment_output[decision_node_id] = current_assignment
                            joint_logprob_dict[decision_node_id] = joint_logprob_dict[decision_node.child_list[0]] + np.log(decision_node.branch_parameters[0])
                else:
                    # not a node with one of its branch leading to false node
                    # should not have invalid nodes because everything starts at true node
                    # both child nodes invalid
                    if type(node_assignment_output[decision_node.child_list[0]]) == int and type(node_assignment_output[decision_node.child_list[1]]) == int:
                        node_assignment_output[decision_node.node_id] = 0
                        joint_logprob_dict[decision_node.node_id] = np.log(0)
                        continue
                    if neg_literal in assignment_set:
                        # check if assignment is compatible
                        if type(node_assignment_output[decision_node.child_list[0]]) == int:
                            # not compatible
                            node_assignment_output[decision_node.node_id] = 0
                            joint_logprob_dict[decision_node.node_id] = np.log(0)
                        else:
                            # compatible
                            current_assignment = np.full([num_samples, 1], neg_literal, dtype=np.int)
                            if node_list[decision_node.child_list[0]].node_type == 'T':
                                pass
                            else:
                                current_assignment = np.hstack((current_assignment, node_assignment_output[decision_node.child_list[0]]))
                            node_assignment_output[decision_node_id] = current_assignment
                            joint_logprob_dict[decision_node.node_id] = joint_logprob_dict[decision_node.child_list[0]] + np.log(decision_node.branch_parameters[0])
                    elif pos_literal in assignment_set:
                        if type(node_assignment_output[decision_node.child_list[1]]) == int:
                            node_assignment_output[decision_node.node_id] = 0
                            joint_logprob_dict[decision_node.node_id] = np.log(0)
                        else:
                            current_assignment = np.full([num_samples, 1], pos_literal, dtype=np.int)
                            if node_list[decision_node.child_list[0]].node_type == 'T':
                                pass
                            else:
                                current_assignment = np.hstack((current_assignment, node_assignment_output[decision_node.child_list[1]]))
                            node_assignment_output[decision_node_id] = current_assignment
                            joint_logprob_dict[decision_node.node_id] = joint_logprob_dict[decision_node.child_list[1]] + np.log(decision_node.branch_parameters[1])
                    else:
                        # unassigned
                        if type(node_assignment_output[decision_node.child_list[0]]) == int:
                            current_assignment = np.full([num_samples, 1], pos_literal, dtype=np.int)
                            if node_list[decision_node.child_list[1]].node_type == 'T':
                                pass
                            else:
                                current_assignment = np.hstack((current_assignment, node_assignment_output[decision_node.child_list[1]]))
                            node_assignment_output[decision_node_id] = current_assignment
                            joint_logprob_dict[decision_node_id] = np.log(decision_node.branch_parameters[1]) + joint_logprob_dict[decision_node.child_list[1]]
                        elif type(node_assignment_output[decision_node.child_list[1]]) == int:
                            current_assignment = np.full([num_samples, 1], neg_literal, dtype=np.int)
                            if node_list[decision_node.child_list[0]].node_type == 'T':
                                pass
                            else:
                                current_assignment = np.hstack((current_assignment, node_assignment_output[decision_node.child_list[0]]))
                            node_assignment_output[decision_node_id] = current_assignment
                            joint_logprob_dict[decision_node_id] = np.log(decision_node.branch_parameters[0]) + joint_logprob_dict[decision_node.child_list[0]]
                        else:
                            # compute the joint probabilities
                            lo_jointprob, hi_jointprob = np.log(decision_node.branch_parameters)
                            lo_jointprob = lo_jointprob + joint_logprob_dict[decision_node.child_list[0]]
                            hi_jointprob = hi_jointprob + joint_logprob_dict[decision_node.child_list[1]]
                            # sample
                            # need to renormalize the joint prob because might have lost density for invalid assignments
                            # (a / (a + b)) = exp( log(a) - log(a+b) ) -- log(a + b) is where logsumexp comes in
                            total_jointprob = np.logaddexp(lo_jointprob, hi_jointprob)
                            joint_logprob_dict[decision_node.node_id] = total_jointprob
                            if (hi_jointprob - total_jointprob) == -np.inf:
                                print('Potential underflow, please use high precision version to double check')
                            normalized_joint_hi_prob = np.exp(hi_jointprob - total_jointprob)
                            current_variable = decision_node.variable
                            sampled_random_vals = np.random.binomial(1, normalized_joint_hi_prob, num_samples)
                            if node_list[decision_node.child_list[1]].node_type == 'T':
                                hi_assignments = np.full([num_samples, 1], current_variable, dtype=np.int)
                            else:
                                pos_assignment_array = np.full([num_samples, 1], current_variable, dtype=np.int)
                                hi_assignments = node_assignment_output[decision_node.child_list[1]]
                                hi_assignments = np.hstack((pos_assignment_array, hi_assignments))
                            if node_list[decision_node.child_list[0]].node_type == 'T':
                                lo_assignments = np.full([num_samples, 1], -1 * current_variable, dtype=np.int)
                            else:
                                lo_assignment_array = np.full([num_samples, 1], -1 * current_variable, dtype=np.int)
                                lo_assignments = node_assignment_output[decision_node.child_list[0]]
                                lo_assignments = np.hstack((lo_assignment_array, lo_assignments))
                            hi_assignments = hi_assignments[sampled_random_vals == 1]
                            lo_assignments = lo_assignments[sampled_random_vals == 0]
                            # should not have a case where both sides is 0
                            if lo_assignments.size == 0:
                                # nothing coming from lo assignments, hi assignment should have num_samples instances
                                node_assignment_output[decision_node.node_id] = hi_assignments
                            elif hi_assignments.size == 0:
                                node_assignment_output[decision_node.node_id] = lo_assignments
                            else:
                                current_samples = np.vstack((hi_assignments, lo_assignments))
                                # fast shuffle
                                idx = np.random.permutation(num_samples)
                                current_samples = current_samples[idx, :]
                                node_assignment_output[decision_node.node_id] = current_samples
    return node_assignment_output[root_node.node_id]

# create a version that samples from an empty assignment, reducing number of checks
def sample_assignment_pzddc_empty(node_list, root_node, true_node, layer_lists, num_samples=1, seed=None):
    '''
    Function to sample num_samples assignments from an OBDD[and] that is false surpassed and represented as a layer_list after toposort
    NOTE: this version is to be used when the initial assignment is empty
    '''
    #set seed for reproducibility
    if seed != None:
        np.random.seed(seed)
    
    node_assignment_output = {}
    # we will check for true node by checking if type is int
    node_assignment_output[true_node.node_id] = np.zeros([1,1])

    # need a dictionary to store log(natual log) joint prob from dynamic annotation 
    # NOTE we also use log sum exp trick to avoid need for high precision math (exp the log probabilities and sum them (decision nodes)) 
    # (the log sum exp trick is used when performing joint prob calculations at decision nodes in the form w_lo * lo_child_prob + w_hi * hi_child_prob -- where now lo/hi child prob is in log space)
    joint_logprob_dict = {}
    joint_logprob_dict[true_node.node_id] = np.log(1.0)

    for i in range(len(layer_lists)):
        decision_node_list, conjunction_node_list = layer_lists[i]

        if len(conjunction_node_list) > 0:
            # process the conjunction nodes
            for conjunction_node_id in conjunction_node_list:
                assignment_list = []
                current_logprob = np.log(1.0)
                current_node = node_list[conjunction_node_id]
                for i in range(len(current_node.child_list)):
                    if node_list[current_node.child_list[i]].node_type == 'T':
                        continue
                    assignment_list.append(node_assignment_output[current_node.child_list[i]])
                    # multiplying probabilities in log space
                    current_logprob = current_logprob + joint_logprob_dict[current_node.child_list[i]]
                    # should not have invalid assignments
                current_assignment = np.concatenate(assignment_list, axis=1)
                node_assignment_output[current_node.node_id] = current_assignment
                joint_logprob_dict[current_node.node_id] = current_logprob
        
        if len(decision_node_list) > 0:
            for decision_node_id in decision_node_list:
                decision_node = node_list[decision_node_id]
                pos_literal = decision_node.variable
                neg_literal = -1*pos_literal
                if decision_node.false_child_branch == 1:
                    # if node is zero-surpassed node, with lo child
                    current_assignment = np.full([num_samples, 1], neg_literal, dtype=np.int)
                    if node_list[decision_node.child_list[0]].node_type != 'T':
                        current_assignment = np.hstack((current_assignment, node_assignment_output[decision_node.child_list[0]]))
                    node_assignment_output[decision_node_id] = current_assignment
                    # lo param X lo child prob (because the other side is unsat)
                    joint_logprob_dict[decision_node_id] = np.log(decision_node.branch_parameters[0]) + joint_logprob_dict[decision_node.child_list[0]]
                elif decision_node.false_child_branch == 0:
                    # if node is zero-surpassed node, with hi child
                    current_assignment = np.full([num_samples, 1], pos_literal, dtype=np.int)
                    if node_list[decision_node.child_list[1]].node_type != 'T':
                        current_assignment = np.hstack((current_assignment, node_assignment_output[decision_node.child_list[1]]))
                    node_assignment_output[decision_node_id] = current_assignment
                    joint_logprob_dict[decision_node_id] = np.log(decision_node.branch_parameters[1]) + joint_logprob_dict[decision_node.child_list[1]]
                else:
                    # both sides are valid
                    # compute the joint probabilities
                    lo_jointprob, hi_jointprob = np.log(decision_node.branch_parameters)
                    lo_jointprob = lo_jointprob + joint_logprob_dict[decision_node.child_list[0]]
                    hi_jointprob = hi_jointprob + joint_logprob_dict[decision_node.child_list[1]]
                    # need to renormalize the joint prob because might have lost density for invalid assignments
                    # (a / (a + b)) = exp( log(a) - log(a+b) ) -- log(a + b) is where logsumexp comes in
                    total_jointprob = np.logaddexp(lo_jointprob, hi_jointprob)
                    joint_logprob_dict[decision_node.node_id] = total_jointprob
                    if (hi_jointprob - total_jointprob) == -np.inf:
                        print('Potential underflow, please use high precision version to double check')
                    normalized_joint_hi_prob = np.exp(hi_jointprob - total_jointprob)
                    current_variable = decision_node.variable

                    sampled_random_vals = np.random.binomial(1, normalized_joint_hi_prob, num_samples)
                    num_hi = int(np.sum(sampled_random_vals[sampled_random_vals == 1]))
                    if node_list[decision_node.child_list[1]].node_type == 'T':
                        hi_assignments = np.full([num_samples, 1], current_variable, dtype=np.int)
                    else:
                        pos_assignment_array = np.full([num_samples, 1], current_variable, dtype=np.int)
                        hi_assignments = node_assignment_output[decision_node.child_list[1]]
                        hi_assignments = np.hstack((pos_assignment_array, hi_assignments))
                    if node_list[decision_node.child_list[0]].node_type == 'T':
                        lo_assignments = np.full([num_samples, 1], -1 * current_variable, dtype=np.int)
                    else:
                        lo_assignment_array = np.full([num_samples, 1], -1 * current_variable, dtype=np.int)
                        lo_assignments = node_assignment_output[decision_node.child_list[0]]
                        lo_assignments = np.hstack((lo_assignment_array, lo_assignments))
                    hi_assignments = hi_assignments[sampled_random_vals == 1]
                    lo_assignments = lo_assignments[sampled_random_vals == 0]
                    # should not have a case where both sides is 0
                    if lo_assignments.size == 0:
                        # nothing coming from lo assignments, hi assignment should have num_samples instances
                        node_assignment_output[decision_node.node_id] = hi_assignments
                    elif hi_assignments.size == 0:
                        node_assignment_output[decision_node.node_id] = lo_assignments
                    else:
                        current_samples = np.vstack((hi_assignments, lo_assignments))
                        # fast shuffle
                        idx = np.random.permutation(num_samples)
                        current_samples = current_samples[idx, :]
                        node_assignment_output[decision_node.node_id] = current_samples
    return node_assignment_output[root_node.node_id]

# arbitrary precision versions
def sample_assignment_pzddc_hp(initial_partial_assignment_set, node_list, root_node, true_node, layer_lists, num_samples=1, num_processes=1, seed=None):
    if len(initial_partial_assignment_set) == 0:
        # empty set, use more optimized function
        return sample_assignment_pzddc_empty_hp(node_list, root_node, true_node, layer_lists, num_samples=num_samples, seed=seed)
    # set seed for reproducibility
    if seed != None:
        np.random.seed(seed)
    
    node_assignment_output = {}
    # we will check for true node by checking if type is int
    node_assignment_output[true_node.node_id] = np.zeros([1,1])

    assignment_set = initial_partial_assignment_set

    # need a dictionary to store joint prob from dynamic annotation 

    prob_dict = {}
    prob_dict[true_node.node_id] = mpq(1)

    for i in range(len(layer_lists)):
        decision_node_list, conjunction_node_list = layer_lists[i]

        if len(conjunction_node_list) > 0:
            # process the conjunction nodes in parallel
            for conjunction_node_id in conjunction_node_list:
                assignment_list = []
                current_prob = mpq(1)
                current_node = node_list[conjunction_node_id]
                for i in range(len(current_node.child_list)):
                    if type(node_assignment_output[current_node.child_list[i]]) == int:
                        # invalid child
                        current_assignment = 0
                        current_prob = mpq(0)
                        break
                    elif node_list[current_node.child_list[i]].node_type == 'T':
                        # apparently compiler seems to produce AND true
                        continue
                    else:
                        assignment_list.append(node_assignment_output[current_node.child_list[i]])
                        current_prob = current_prob * prob_dict[current_node.child_list[i]]
                if type(current_assignment) != int:
                    current_assignment = np.concatenate(assignment_list, axis=1)
                node_assignment_output[current_node.node_id] = current_assignment
                prob_dict[current_node.node_id] = current_prob

        if len(decision_node_list) > 0:
            # process the decision nodes
            for decision_node_id in decision_node_list:
                decision_node = node_list[decision_node_id]
                pos_literal = decision_node.variable
                neg_literal = -1 * pos_literal
                if decision_node.false_child_branch != None:
                    # if decision node has false node as a child, false_child_branch should only be 0 or 1
                    if decision_node.false_child_branch == 0:
                        if type(node_assignment_output[decision_node.child_list[1]]) == int:
                            node_assignment_output[decision_node_id] = 0
                            prob_dict[decision_node_id] = mpq(0)
                            continue
                        # lo branch is false node originally
                        if neg_literal in assignment_set:
                            # incompatible
                            node_assignment_output[decision_node_id] = 0
                            prob_dict[decision_node_id] = mpq(0)
                        else:
                            # compatible
                            current_assignment = np.full([num_samples, 1], pos_literal, dtype=np.int)
                            if node_list[decision_node.child_list[1]].node_type == 'T':
                                # if child is true node we do not do anything
                                pass
                            else:
                                current_assignment = np.hstack((current_assignment, node_assignment_output[decision_node.child_list[1]]))
                            node_assignment_output[decision_node_id] = current_assignment
                            # joint prob of child
                            prob_dict[decision_node_id] = mpq(decision_node.branch_parameters[1]) * prob_dict[decision_node.child_list[1]]
                    else:
                        if type(node_assignment_output[decision_node.child_list[0]]) == int:
                            node_assignment_output[decision_node_id] = 0
                            prob_dict[decision_node_id] = mpq(0)
                            continue

                        if pos_literal in assignment_set:
                            # incompatible
                            node_assignment_output[decision_node_id] = 0
                            prob_dict[decision_node_id] = mpq(0)
                        else:
                            # compatible
                            current_assignment = np.full([num_samples, 1], neg_literal, dtype=np.int)
                            if node_list[decision_node.child_list[0]].node_type == 'T':
                                # if child is true node we do not do anything
                                pass
                            else:
                                current_assignment = np.hstack((current_assignment, node_assignment_output[decision_node.child_list[0]]))
                            node_assignment_output[decision_node_id] = current_assignment
                            prob_dict[decision_node_id] = mpq(decision_node.branch_parameters[0]) * prob_dict[decision_node.child_list[0]]
                else:
                    # both child nodes invalid
                    if type(node_assignment_output[decision_node.child_list[0]]) == int and type(node_assignment_output[decision_node.child_list[1]]) == int:
                        node_assignment_output[decision_node.node_id] = 0
                        prob_dict[decision_node_id] = mpq(0)
                        continue
                    if neg_literal in assignment_set:
                        # check if assignment is compatible
                        if type(node_assignment_output[decision_node.child_list[0]]) == int:
                            # not compatible
                            node_assignment_output[decision_node.node_id] = 0
                            prob_dict[decision_node_id] = mpq(0)
                        else:
                            # compatible
                            current_assignment = np.full([num_samples, 1], neg_literal, dtype=np.int)
                            if decision_node.child_list[0].node_type == 'T':
                                pass
                            else:
                                current_assignment = np.hstack((current_assignment, node_assignment_output[decision_node.child_list[0]]))
                            node_assignment_output[decision_node_id] = current_assignment
                            prob_dict[decision_node.node_id] = mpq(decision_node.branch_parameters[0]) * prob_dict[decision_node.child_list[0]]
                    elif pos_literal in assignment_set:
                        if type(node_assignment_output[decision_node.child_list[1]]) == int:
                            node_assignment_output[decision_node.node_id] = 0
                            prob_dict[decision_node_id] = mpq(0)
                        else:
                            current_assignment = np.full([num_samples, 1], pos_literal, dtype=np.int)
                            if decision_node.child_list[0].node_type == 'T':
                                pass
                            else:
                                current_assignment = np.hstack((current_assignment, node_assignment_output[decision_node.child_list[1]]))
                            node_assignment_output[decision_node_id] = current_assignment
                            prob_dict[decision_node.node_id] = mpq(decision_node.branch_parameters[1]) * prob_dict[decision_node.child_list[1]]
                    else:
                        # unassigned
                        if type(node_assignment_output[decision_node.child_list[0]]) == int:
                            current_assignment = np.full([num_samples, 1], pos_literal, dtype=np.int)
                            if decision_node.child_list[1].node_type == 'T':
                                pass
                            else:
                                current_assignment = np.hstack((current_assignment, node_assignment_output[decision_node.child_list[1]]))
                            node_assignment_output[decision_node_id] = current_assignment
                            prob_dict[decision_node_id] = mpq(decision_node.branch_parameters[1]) * prob_dict[decision_node.child_list[1]]
                        elif type(node_assignment_output[decision_node.child_list[1]]) == int:
                            current_assignment = np.full([num_samples, 1], neg_literal, dtype=np.int)
                            if decision_node.child_list[0].node_type == 'T':
                                pass
                            else:
                                current_assignment = np.hstack((current_assignment, node_assignment_output[decision_node.child_list[0]]))
                            node_assignment_output[decision_node_id] = current_assignment
                            prob_dict[decision_node_id] = mpq(decision_node.branch_parameters[0]) * prob_dict[decision_node.child_list[0]]
                        else:
                            # compute the joint probabilities
                            lo_prob = mpq(decision_node.branch_parameters[0]) * prob_dict[decision_node.child_list[0]]
                            hi_prob = mpq(decision_node.branch_parameters[1]) * prob_dict[decision_node.child_list[1]]
                            total_prob = lo_prob + hi_prob
                            norm_hi_prob = hi_prob / total_prob
                            prob_dict[decision_node.node_id] = total_prob

                            current_variable = decision_node.variable
                            sampled_random_vals = np.random.binomial(1, norm_hi_prob, num_samples)
                            num_hi = int(np.sum(sampled_random_vals[sampled_random_vals == 1]))
                            if node_list[decision_node.child_list[1]].node_type == 'T':
                                hi_assignments = np.full([num_samples, 1], current_variable, dtype=np.int)
                            else:
                                pos_assignment_array = np.full([num_samples, 1], current_variable, dtype=np.int)
                                hi_assignments = node_assignment_output[decision_node.child_list[1]]
                                hi_assignments = np.hstack((pos_assignment_array, hi_assignments))
                            if node_list[decision_node.child_list[0]].node_type == 'T':
                                lo_assignments = np.full([num_samples, 1], -1 * current_variable, dtype=np.int)
                            else:
                                lo_assignment_array = np.full([num_samples, 1], -1 * current_variable, dtype=np.int)
                                lo_assignments = node_assignment_output[decision_node.child_list[0]]
                                lo_assignments = np.hstack((lo_assignment_array, lo_assignments))
                            hi_assignments = hi_assignments[sampled_random_vals == 1]
                            lo_assignments = lo_assignments[sampled_random_vals == 0]
                            # should not have a case where both sides is 0
                            if lo_assignments.size == 0:
                                # nothing coming from lo assignments, hi assignment should have num_samples instances
                                node_assignment_output[decision_node.node_id] = hi_assignments
                            elif hi_assignments.size == 0:
                                node_assignment_output[decision_node.node_id] = lo_assignments
                            else:
                                current_samples = np.vstack((hi_assignments, lo_assignments))
                                # fast shuffle
                                idx = np.random.permutation(num_samples)
                                current_samples = current_samples[idx, :]
                                node_assignment_output[decision_node.node_id] = current_samples
    return node_assignment_output[root_node.node_id]

def sample_assignment_pzddc_empty_hp(node_list, root_node, true_node, layer_lists, num_samples=1, seed=None):
    '''
    Function to sample num_samples assignments from an OBDD[and] that is zero surpassed and represented as a layer_list after toposort
    NOTE: this version is to be used when the initial assignment is empty
    '''
    #set seed for reproducibility
    if seed != None:
        np.random.seed(seed)
    
    node_assignment_output = {}
    # we will check for true node by checking if type is int
    node_assignment_output[true_node.node_id] = np.zeros([1,1])

    # need a dictionary to store joint prob from dynamic annotation 
    prob_dict = {}
    prob_dict[true_node.node_id] = mpq('1')

    for i in range(len(layer_lists)):
        decision_node_list, conjunction_node_list = layer_lists[i]

        if len(conjunction_node_list) > 0:
            # process the conjunction nodes
            for conjunction_node_id in conjunction_node_list:
                assignment_list = []
                current_prob = mpq(1)
                current_node = node_list[conjunction_node_id]
                for i in range(len(current_node.child_list)):
                    if node_list[current_node.child_list[i]].node_type == 'T':
                        continue
                    assignment_list.append(node_assignment_output[current_node.child_list[i]])
                    current_prob = current_prob * mpq(prob_dict[current_node.child_list[i]])
                current_assignment = np.concatenate(assignment_list, axis=1)
                node_assignment_output[current_node.node_id] = current_assignment
                prob_dict[current_node.node_id] = current_prob
        
        if len(decision_node_list) > 0:
            for decision_node_id in decision_node_list:
                decision_node = node_list[decision_node_id]
                pos_literal = decision_node.variable
                neg_literal = -1*pos_literal
                if decision_node.false_child_branch == 1:
                    # if node has false child, with lo child
                    current_assignment = np.full([num_samples, 1], neg_literal, dtype=np.int)
                    if node_list[decision_node.child_list[0]].node_type != 'T':
                        current_assignment = np.hstack((current_assignment, node_assignment_output[decision_node.child_list[0]]))
                    node_assignment_output[decision_node_id] = current_assignment
                    # lo param X lo child prob (because the other side is unsat)
                    prob_dict[decision_node_id] = mpq(decision_node.branch_parameters[0]) * prob_dict[decision_node.child_list[0]]
                elif decision_node.false_child_branch == 0:
                    # if node has false child, with hi child
                    current_assignment = np.full([num_samples, 1], pos_literal, dtype=np.int)
                    if node_list[decision_node.child_list[1]].node_type != 'T':
                        current_assignment = np.hstack((current_assignment, node_assignment_output[decision_node.child_list[1]]))
                    node_assignment_output[decision_node_id] = current_assignment
                    prob_dict[decision_node_id] = mpq(decision_node.branch_parameters[1]) * prob_dict[decision_node.child_list[1]]
                else:
                    # both sides are valid
                    # compute the joint probabilities
                    lo_prob = mpq(decision_node.branch_parameters[0]) * prob_dict[decision_node.child_list[0]]
                    hi_prob = mpq(decision_node.branch_parameters[1]) * prob_dict[decision_node.child_list[1]]
                    total_prob = lo_prob + hi_prob
                    prob_dict[decision_node.node_id] = total_prob
                    norm_hi_prob = hi_prob / total_prob
                    current_variable = decision_node.variable

                    sampled_random_vals = np.random.binomial(1, norm_hi_prob, num_samples)
                    num_hi = int(np.sum(sampled_random_vals[sampled_random_vals == 1]))
                    if node_list[decision_node.child_list[1]].node_type == 'T':
                        hi_assignments = np.full([num_samples, 1], current_variable, dtype=np.int)
                    else:
                        pos_assignment_array = np.full([num_samples, 1], current_variable, dtype=np.int)
                        hi_assignments = node_assignment_output[decision_node.child_list[1]]
                        hi_assignments = np.hstack((pos_assignment_array, hi_assignments))
                    if node_list[decision_node.child_list[0]].node_type == 'T':
                        lo_assignments = np.full([num_samples, 1], -1 * current_variable, dtype=np.int)
                    else:
                        lo_assignment_array = np.full([num_samples, 1], -1 * current_variable, dtype=np.int)
                        lo_assignments = node_assignment_output[decision_node.child_list[0]]
                        lo_assignments = np.hstack((lo_assignment_array, lo_assignments))
                    hi_assignments = hi_assignments[sampled_random_vals == 1]
                    lo_assignments = lo_assignments[sampled_random_vals == 0]
                    if lo_assignments.size == 0:
                        # nothing coming from lo assignments, hi assignment should have num_samples instances
                        node_assignment_output[decision_node.node_id] = hi_assignments
                    elif hi_assignments.size == 0:
                        node_assignment_output[decision_node.node_id] = lo_assignments
                    else:
                        current_samples = np.vstack((hi_assignments, lo_assignments))
                        # fast shuffle
                        idx = np.random.permutation(num_samples)
                        current_samples = current_samples[idx, :]
                        node_assignment_output[decision_node.node_id] = current_samples
    return node_assignment_output[root_node.node_id]
