"""
This script builds and runs a graph with pyflow.

There is no need to change anything to solve this quiz!

However, feel free to play with the network! Can you also
build a network that solves the equation below?

(x + y) + y
"""

from pyflow import *

# x, y, z = Input(value=10), Input(value=5), Input(value=50)
# f = Add(Mul(x, y, z),y)
# sorted_nodes = topological_sort([x,y,z])
# Sorted nodes is where calculation will occur, while output nodes will have retain the value of the final calculation
# output = forward_pass(output_node=f, sorted_nodes=sorted_nodes)

# NOTE: because topological_sort sets the values for the `Input` nodes we could also access
# the value for x with x.value (same goes for y).
# print("({} * {} * {}) + {}  = {} (according to pylow)".format(x, y, z, y, f))

"""
NOTE: Here we're using an Input node for more than a scalar.
In the case of weights and inputs the value of the Input node is
actually a python list!

In general, there's no restriction on the values that can be passed to an Input node.
"""

# task1 

# the input
inp1, inp2 = Input(value=6), Input(value=0.5)

# the function
func1 = Mul(inp1, inp2)

# the grpah
graph1 = topological_sort([inp1, inp2])

# the output
output1 = forward_pass(func1, graph1)

# result
# print(output1, graph1)


# task2

# the input
inputs, weights, bias = Input(value=[6, 14, 3]), Input(value=[0.5, 0.25, 1.4]), Input(value=[2])

# the function
f = Linear(inputs, weights, bias)

# the graph
graph = topological_sort([inputs, weights, bias])

# the output
output = forward_pass(f, graph)

print(output, graph) # should be 12.7 with this example