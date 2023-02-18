import matplotlib.pyplot as plt
import numpy as np



def func1():
    x = np.linspace(0,10,100)
    y = x*x + 20*np.exp(-1*x)
    plt.plot(x,y)
    plt.show()

def func2():
    x = np.linspace(0,1,100)
    y = .1*np.exp(-10*x)
    plt.plot(x,y)
    plt.show()

def func3():
    x = np.linspace(0,10,100)
    y = np.exp(-x)
    plt.plot(x,y)
    plt.show()


func3()

