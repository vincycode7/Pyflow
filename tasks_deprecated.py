from pyflow_deprecated import *
import numpy as np

def task1():
    #task1 

    # the input
    inp1, inp2 = Input(value=6), Input(value=0.5)

    # the function
    func1= Mul(inp1, inp2)

    # the grpah
    graph = topological_sort([inp1, inp2])

    # the output
    forward_pass(graph)
    return func1.value, graph

def task2():
    # the input
    inputs, weights, bias = Input(value=[6, 14, 3]), Input(value=[0.5, 0.25, 1.4]), Input(value=[2])

    # the function
    f = Linear(inputs, weights, bias)

    # the graph
    graph = topological_sort([inputs, weights, bias])

    # the output
    forward_pass(graph)
    return f.value, graph

def task2():
    # the input
    inputs, weights, bias = Input(value=[6, 14, 3]), Input(value=[0.5, 0.25, 1.4]), Input(value=[2])

    # the function
    f = Linear(inputs, weights, bias)

    # the graph
    graph = topological_sort([inputs, weights, bias])

    # the output
    forward_pass(graph)
    return f.value,  graph

def task3():
    # the input 
    X, W, b = Input(value=np.array([[-1., -2.], [-1, -2]])), Input(value=np.array([[2., -3], [2., -3]])), Input(value=np.array([-3., -5]))

    # the linear function
    f = Linear(X, W, b)

    # the graph
    graph= topological_sort([X, W, b])

    # the output
    forward_pass(graph)
    return f.value, graph

def task4():
    X, W, b = Input(value=np.array([[-1., -2.], [-1, -2]])), Input(value=np.array([[2., -3], [2., -3]])), Input(value=np.array([-3., -5]))

    f = Linear(X, W, b)
    g = Sigmoid(f)

    # the graph
    graph= topological_sort([X, W, b])

    # the output
    forward_pass(graph)

    """
    Output should be:
    [[  1.23394576e-04   9.82013790e-01]
    [  1.23394576e-04   9.82013790e-01]]
    """
    return g.value, graph

def task5():
    y, a = Input(value=np.array([1, 2, 3])), Input(value=np.array([4.5, 5, 10]))
    cost = MSE(y, a)

    # the graph
    graph= topological_sort([y, a])

    # forward pass
    forward_pass(graph)

    """
    Expected output

    23.4166666667
    """
    return cost.value, graph
