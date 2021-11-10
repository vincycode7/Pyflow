"""
This script builds and runs a graph with pyflow.

There is no need to change anything to solve this quiz!

However, feel free to play with the network! Can you also
build a network that solves the equation below?

(x + y) + y
"""

from tasks import *

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
output, graph = task1()

# result
print("Task 1 Result: ",output,"\n", "Graph 1 Result: ",graph, "\n") # Output should be: 3 with this example


# task2
output, graph = task2()

# result
print("Task 2 Result: ",output, "\n", "Graph 2 Result: ",graph, "\n") # Output should be: 12.7 with this example


# tasks 3
output, graph = task3()

# result
print("Task 3 Result: ",output, "\n", "Graph 3 Result: ",graph, "\n") # Output should be: [[-9.  4.][-9.  4.]] with this example


# tasks 4
output, graph = task4()

# result
print("Task 4 Result: ",output, "\n", "Graph 4 Result: ",graph, "\n") # Output should be: [[  1.23394576e-04   9.82013790e-01] [  1.23394576e-04   9.82013790e-01]] with this example

# tasks 5
output, graph = task5()

# result
print("Task 5 Result: ",output, "\n", "Graph 5 Result: ",graph, "\n") # Output should be: 23.4166666667 with this example