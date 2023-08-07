
#plotting all eight 3d points using matplotlib

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize = (4,4))
ax = fig.add_subplot(111, projection = '3d')

ax.scatter(1,1,1)
ax.scatter(1,1,-1)
ax.scatter(1,-1,1)
ax.scatter(1,-1,-1)
ax.scatter(-1,1,1)
ax.scatter(-1,1,-1)
ax.scatter(-1,-1,1)
ax.scatter(-1,-1,-1)

plt.show()

#perceptron learning
#If using in colab try to change the cells to avoid errors

import numpy as np
import pandas as pd

X = np.array([[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]])
Y = np.array([-1,-1,-1,-1,-1,-1,-1,1])

w = np.random.rand(3,1)
dot = np.dot(X[0],w)
print(w)
print(dot[0])

def perceptron(X, y):
    X = np.insert(X, 0, 1, axis=1)
    w = np.zeros(X.shape[1])
    alpha = 0.1
    max_iterations = 100
    for iteration in range(max_iterations):
        error_count = 0
        for i in range(X.shape[0]):
            y_pred = np.sign(np.dot(X[i], w))
            if y_pred != y[i]:
                w += alpha * y[i] * X[i]
                error_count += 1
        if error_count == 0:
            break
    return error_count

X = np.array([[0,0],[0,1],[1,0],[1,1]])

target_functions = [
    lambda x: x[0] and x[1],
    lambda x: x[0] or x[1],
    lambda x: x[0] != x[1],
    lambda x: x[0] == x[1],
    lambda x: x[0],
    lambda x: x[1],
    lambda x: not x[0],                  
    lambda x: not x[1],                  
    lambda x: x[0] and not x[1],         
    lambda x: not(x[0] or x[1]),        
    lambda x: x[0] or not x[1],          
    lambda x: not x[0] or x[1],          
    lambda x: x[0] and x[1] and not x[1],
    lambda x: not x[0] or not x[1],      
    lambda x: x[0] == 0 and x[1] == 0,   
    lambda x: x[0] == 1 and x[1] == 1
]

targets = np.array([[target_functions[j](x) for j in range(16)] for x in X])

print(targets)

X1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

Y3 = np.array([1,-1,-1,1])
perceptron(X1,Y3)

target = np.array([list(map(int, np.binary_repr(i, width=4))) for i in range(2**4)])
combinations=np.transpose(target)
print(target)

counter1 = 0
counter2 = 0

for i in range(len(target)):
  count_errors = perceptron(X,target[i])
  if(count_errors == 0):
    counter1 += 1
  else:
    counter2 += 1
print(counter1)
print(counter2)

