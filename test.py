import numpy as np

a = np.loadtxt("a.csv", delimiter=",")
b = np.loadtxt("b.csv", delimiter=",")
c0 = np.loadtxt('c.csv', delimiter=',')
c1 = np.loadtxt('c2.csv', delimiter=',')
c2 = np.loadtxt('c3.csv', delimiter=',')
print(b)
c = a @ b
print("C:  ")
print(c)
print("C0:  ")
print((np.abs(c - c0))[0, :].mean())
print("C1:  ")
print((np.abs(c - c1))[0, :].mean())
print("C2:  ")
print((np.abs(c - c2))[0, :].mean())
