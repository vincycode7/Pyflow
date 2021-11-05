"""
This script builds and runs a graph with pyflow.

There is no need to change anything to solve this quiz!

However, feel free to play with the network! Can you also
build a network that solves the equation below?

(x + y) + y
"""

from pyflow import *

x, y = Input(value=10), Input(value=5)
f = Add(x, y)
sorted_nodes = topological_sort([x,y])

# Sorted nodes is where calculation will occur, while output nodes will have retain the value of the final calculation
output = forward_pass(output_node=f, sorted_nodes=sorted_nodes)

# NOTE: because topological_sort sets the values for the `Input` nodes we could also access
# the value for x with x.value (same goes for y).
print("{} + {} = {} (according to pylow)".format(x, y, f))