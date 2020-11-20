# ============================================================================
# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
#import pywt
import sys
# ============================================================================

def gamma(x, x0, alpha, ymax=3):
    #y = np.zeros(len(x))
    y0 = np.random.rand(len(x))
    y = np.array(y0)
    #
    y[np.where(x <= x0)] = y0[np.where(x <= x0)]
    y[np.where(x > x0)] = (x[np.where(x > x0)] - x0)**alpha
    y[y>=ymax] = y0[y>=ymax]
    #y[np.where(x > np.mean(x))] = (y[-1]**(1/alpha) - x[np.where(x > np.mean(x))] -x0)**alpha
    #
    return y

def waveform(x, w0):
    #return np.exp(1j*w0*x) * np.exp(-0.5*(x**2))
    # equivalent frequency in fourier domain for a given scale
    fourier_period = (4 * np.pi) / (w0 + np.sqrt(2 + w0 ** 2)) 
    psi = np.sin(2*np.pi*w0*x) * np.exp(-0.5*(x**2))/(np.sqrt(2)*(np.pi**0.25))
    return psi

# ============================================================================

def signal1(function=waveform, plot=False):
    x = np.linspace(-3*np.pi, 3*np.pi, 500)
    y1 = waveform(x, w0=3)
    y2 = waveform(x, w0=2)
    y3 = waveform(x, w0=1)
    y = (y1+y2+y3)
    #
    if plot == True:
        plt.subplot(4,1,1)
        plt.plot(x, y1, 'k')
        plt.subplot(4,1,2)
        plt.plot(x, y2, 'k')
        plt.subplot(4,1,3)
        plt.plot(x, y3, 'k')
        plt.subplot(4,1,4)
        plt.plot(x, y, 'k')
        plt.show()
    #
    return x, y


def signal2(function=waveform, plot=False):
    x = np.linspace(-3*np.pi, 3*np.pi, 500)
    y1 = waveform(x-2, w0=3)
    y2 = waveform(x-4, w0=2)
    y3 = waveform(x-6, w0=1)
    y = (y1+y2+y3)
    #
    if plot == True:
        plt.subplot(4,1,1)
        plt.plot(x, y1, 'k')
        plt.subplot(4,1,2)
        plt.plot(x, y2, 'k')
        plt.subplot(4,1,3)
        plt.plot(x, y3, 'k')
        plt.subplot(4,1,4)
        plt.plot(x, y, 'k')
        plt.show()
    #
    return x, y

def signal_f():
    x1 = np.linspace(0, 10, 100)
    y1 = gamma(x1, 5, -1)
    x2 = np.linspace(10, 20, 100)
    y2 = gamma(x2, 15, 1)
    x3 = np.linspace(20, 30, 100)
    y3 = gamma(x3, 25, 0)
    x4 = np.linspace(30, 40, 100)
    y4 = gamma(x4, 35, 2)
    y = np.concatenate([y1, y2])
    y = np.concatenate([y, y3])
    y = np.concatenate([y, y4])
    x = np.concatenate([x1,x2])
    x = np.concatenate([x,x3])
    x = np.concatenate([x,x4])
    #
    time = np.linspace(0,40,400)
    #
    return time, y
    
def signal_r(N=10000):
    y = np.random.rand(N)
    x = 10*np.sin(2*np.linspace(-np.pi,np.pi,3000))+5*np.cos(5*np.linspace(-np.pi,np.pi,3000))
    y[:len(x)] += x
    x = 10*np.cos(2*np.linspace(-np.pi,np.pi,3000))+5*np.sin(5*np.linspace(-np.pi,np.pi,3000))
    y[len(y)-len(x)-len(y)//5:len(y)-len(y)//5] += x
    time = np.arange(len(y))
    return time, y

def signal_me(N = 50, dt= .1):
    x = np.arange(0, N, dt)
    y1 = np.cos(2*np.pi*x/20)
    y2 = np.cos(2*np.pi*x/10)
    y=np.concatenate([y1,y2])
    time = np.arange(0, len(y))*dt
    return time, y


def signal_prof():
    x = np.linspace(0, 5, 1000)
    y = np.ones(len(x))
    for ind, val in enumerate(x):
        if 0 <= val and val < 1:
            y[ind] = 1
        elif 1 <= val and val < 3:
            y[ind] = 3
        elif 3 <= val and val < 4:
            y[ind] = -2*val + 9
        else:
            y[ind] = 1
    time = np.array(x)
    return time, y
        
