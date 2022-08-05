# Copyright 2022 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.

# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

class ObddNode:
    '''
    Encapsulation for obdd nodes
    4 types of nodes: T - true, F - false, C - conjunction, D - decision
    Child list - lo, hi in case of decision nodes, if conjunction then does not matter
    '''
    def __init__(self, node_type = None, variable = -1, node_id = -1):
        self.node_type = node_type
        self.variable = variable
        self.node_id = node_id
        self.child_list = []
        self.parent_list = []