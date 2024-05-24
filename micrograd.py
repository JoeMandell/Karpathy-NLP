import math
import numpy as np
import matplotlib.pyplot as plt


class Value:
    def __init__(self, data, children=(), op='', label=''):
        self.data = data
        self.grad = 0.0
        self.backward = lambda: None
        self.prev = set(children)
        self.op = op
        self.children = children
    
    def __repr__(self):
        return f"Value(data=",self.data,")"
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        def backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad
        out.backward = backward
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        def backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        out.backward = backward
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        def backward():
            self.grad = (1 - t**2) * out.grad
        out.backward = backward
        return out

#write a value class

#this class stores data, 


#this class has methods to add, multiply, and activate (say, tanh activation)

