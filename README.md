# INC: incremental weighted sampler

INC, INCremental weighted sampler, draws samples from the set of satisfying assignments of a set of boolean contraints. The sampling distribution is defined by a literal-weighted weight function. INC takes CNF in DIMACS format and operates by compiling the CNF to OBDD[AND] diagram, performing smoothing and augmenting the diagram with weights to form the PROB that is used to perform sampling.

This repository contains code for our paper *INC: A Scalable Incremental Weighted Sampler*

---
## Installation

1. INC depends on KCBox compiler to compile CNF into OBDD[AND] diagram. Please follow the instructions [here](https://github.com/meelgroup/KCBox) to compile a copy of KCBox.

2. Follow the instructions to install INC from its root directorty as follows:

```
sudo apt-get install graphviz libgmp-dev libmpfr-dev libmpc-dev time
pip install --upgrade build
python -m build
cd dist
pip install inc-0.0.1-py3-none-any.whl
```

If preference is to use from directory, then please cd to `src` folder and perform the following:
```
sudo apt-get install graphviz libgmp-dev libmpfr-dev libmpc-dev time
pip install -r requirements.txt
```

---

## Using INC


Example CNF `F = (X1 OR X2) AND (-X1 OR X3)`:
```
p cnf 3 2
1 2 0
-1 3 0
```

Start by importing all the relevant components:
```
import pobdd_sampler_func, pobdd_inference, pobdd_parser, pobdd_utils
import numpy as np
```

Compile the CNF file into OBDD[AND] format using KCBox
```
compile_success, obdd_file = pobdd_sampler_func.compile_cnf_file('/path/to/KCBox', '/path/to/example.cnf')
```

Next we perform smoothing on the OBDD[AND]:
```
smooth_success, smooth_obdd_file = pobdd_sampler_func.call_smoother(obdd_file)
```

Parse the smoothed OBDD[AND] into memory as a PROB, default weights are 0.5 for both positive and negative literals:
```
variable_ordering_list, num_nodes, root_node_id, node_list = pobdd_parser.parse_obdd(smooth_obdd_file)
root_node = node_list[int(root_node_id)]
```

Define a weight function and apply the weights to PROB (0.75 for all positive literals and 0.25 for negative literals):
```
# only required to specify weights of positive literals if the positive weights are normalized
weight_dict = {1:0.75, 2:0.75, 3:0.75}
pobdd_sampler_func.apply_weights(node_list, weight_dict)
```
A bit of preprocessing before sampling:
```
# figure out which nodes to consider (ignore false node and edges leading to false node)
forward_edge_adj_dict = pobdd_parser.parse_pobdd_reduced_forwardedges(node_list)

# perform toposort on reduced PROB to get bottom up ordering information
toposort_result_sets = pobdd_parser.perform_toposort(forward_edge_adj_dict)

# split each layer into decision and conjuction nodes
layer_node_list = pobdd_inference.prepare_layer_node_lists(toposort_result_sets, node_list)

true_node_id, false_node_id = pobdd_utils.get_true_false_node_id(node_list)
true_node = node_list[true_node_id]
```

Perform weighted sampling:
```
# returns a numpy 2d array, axis 0 is of num_samples, axis 1 is the assignment for variables (not necessarily ordered)

# if high_precision=True, arbitrary precision math is used, else log-space computation is used
# conditional_assignment_set can include literals for conditional weighted sampling
conditional_assignment_set = set()

# drawing 10000 samples at once
samples = pobdd_sampler_func.sample_assignments(conditional_assignment_set, node_list, root_node, true_node, layer_node_list, num_samples=10000, seed=None, high_precision=False)

# sort the samples according to the variables
samples = samples[np.arange(samples.shape[0])[:, None], np.abs(samples).argsort()]
```

As the PROB is in memory, we can simply use `pobdd_sampler_func.apply_weights` to apply new weights and sample again.

The PROB, along with weights, can be saved into a file:
```
# save PROB to file
pobdd_output_file = smooth_obdd_file + '.pobdd'
pobdd_utils.write_output_pobdd_file(node_list, root_node, variable_ordering_list, pobdd_output_file)

# load a PROB file
variable_order_list, num_nodes, root_node_id, node_list = pobdd_parser.parse_pobdd(pobdd_output_file)
```

---

## License

This project is licensed under the terms of the MIT license, see the LICENSE file for details.