# ============================================================================
# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import scipy.signal
import sys
# ============================================================================


# ============================================================================
def um_terco_Simpson(X, Y):
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    # definindo passo h
    h = abs(X[0] - X[1])
    # somando termos de x_1 ate x_(n-1)
    termos_2 = np.arange(2,len(Y)-1,2)
    termos_4 = np.arange(1,len(Y)-1,2)
    #
    integral_2 = np.sum(Y[termos_2])
    integral_4 = np.sum(Y[termos_4])
    integral = (h/3.)*(Y[0] + 4*integral_4 + 2*integral_2 + Y[-1])
    #
    return integral


def cond1(wave):
    # X, Y
    x = np.linspace(-4*np.pi, 4*np.pi, 100000)
    y = wave(x)
    # Integration:
    res = um_terco_Simpson(x, np.real(y))
    #
    print('Zero average?')
    print(res)
    return #np.abs(np.sum(w))*dx/2


def cond2(wave):
    # X, Y
    x = np.linspace(-4*np.pi, 4*np.pi, 100000)
    y = wave(x)
    y_mod_squared = np.abs(y)**2
    # Integration:
    res = um_terco_Simpson(x, y_mod_squared)
    #
    print('Unitary Energy?')
    print(res)
    return #np.abs(np.sum(w))*dx/2


# ============================================================================
def morlet(x, w0=6, norm=(np.pi**0.25)):
  '''Morlet wavelet'''
  return np.exp(1j*w0*x) * np.exp(-0.5*(x**2)) / norm


def ft_morlet(x, w0=6, norm=(np.pi**0.25)):
  '''Fourier transform of morlet wavelet'''
  return np.exp(-0.5 * (x - w0) ** 2.) / norm


def Morlet(x, mode=None, w0=6, norm=(np.pi**0.25)):
  
  # Morlet params as of Torrence and Compo (1998), Table 1
  fourier_period = (4 * np.pi) / (w0 + np.sqrt(2 + w0 ** 2))
  reconstruction_factor = .776
  morlet_params = [reconstruction_factor, norm, fourier_period]
    
  if mode==None:
    # Returning morlet wavelet
    return morlet(x, w0, norm)
  
  elif mode=='fourier':
    # Returning fourier transform of Morlet wavelet
    return ft_morlet(x, w0, norm)
  
  elif mode=='params':
    # Returning parameters (relevant for inverse cwt)
    return morlet_params
  
  else:
    raise Exception("input 'mode' is not valid; options are: None (default), 'fourier', 'params'.")

