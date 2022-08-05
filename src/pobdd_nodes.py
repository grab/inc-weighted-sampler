# Copyright 2022 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.

# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

class PObddNode:
    '''
    Encapsulation for probabilistic obdd nodes
    4 types of nodes: T - true, F - false, C - conjunction, D - decision
    Child list - lo, hi in case of decision nodes, if conjunction then does not matter
    '''
    def __init__(self, node_type = None, variable = -1, node_id = -1):
        self.node_type = node_type
        self.variable = variable
        self.node_id = node_id
        # supposedly lo child, hi child for decision nodes
        self.child_list = []
        # probability parameters for lo child, hi child for decision nodes
        self.branch_parameters = []
        # lo - 0, hi - 1 counts for which to get the parameters from, when normalizing
        self.branch_counts=[]
        self.false_child_branch = None

    def normalize_param(self):
        '''
        Function to normalize the branch params according to the counts
        New scheme, f(x) / (f(x) + f(y)) used, where x and y are branch counts and f(x) = x + 1
        When there is little data, it is not exactly according to data because of the  + 1, but solves the issue where 1 side count is 0.
        If both branch counts are 0 then the parameters are 0.5, which means equal chance to happen
        '''
        lo_count = self.branch_counts[0]
        hi_count = self.branch_counts[1]

        self.branch_parameters[0] = (lo_count + 1) / (lo_count + hi_count + 2)
        self.branch_parameters[1] = (hi_count + 1) / (lo_count + hi_count + 2)
    
    def normalize_weights(self):
        '''
        Function to normalize the branch params based on what is the current weights
        This is to handle the case where the weights are directly defined
        '''
        total = self.branch_parameters[0] + self.branch_parameters[1]
        self.branch_parameters[0] = self.branch_parameters[0] / total
        self.branch_parameters[1] = 1.0 - self.branch_parameters[0]
