from pyflow import *
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
    output = forward_pass(func1, graph)
    return output, graph

def task2():
    # the input
    inputs, weights, bias = Input(value=[6, 14, 3]), Input(value=[0.5, 0.25, 1.4]), Input(value=[2])

    # the function
    f = Linear(inputs, weights, bias)

    # the graph
    graph = topological_sort([inputs, weights, bias])

    # the output
    output = forward_pass(f, graph)
    return output, graph

def task2():
    # the input
    inputs, weights, bias = Input(value=[6, 14, 3]), Input(value=[0.5, 0.25, 1.4]), Input(value=[2])

    # the function
    f = Linear(inputs, weights, bias)

    # the graph
    graph = topological_sort([inputs, weights, bias])

    # the output
    output = forward_pass(f, graph)
    return output, graph

def task3():
    # the input 
    X, W, b = Input(value=np.array([[-1., -2.], [-1, -2]])), Input(value=np.array([[2., -3], [2., -3]])), Input(value=np.array([-3., -5]))

    # the linear function
    f = Linear(X, W, b)

    # the graph
    graph= topological_sort([X, W, b])

    # the output
    output = forward_pass(f, graph)
    return output, graph