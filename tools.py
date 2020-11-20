# ============================================================================
# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import scipy.signal
import sys
from wavelets import *
# ============================================================================


# ============================================================================
def S(W, window_scale_size):
    #
    n_rows, n_cols = np.shape(W)
    R = np.zeros(np.shape(W))
    #
    for j in range(n_cols):
        window = np.ones(window_scale_size)/window_scale_size
        R[:, j] = np.convolve(W[:, j], window, 'same')
    #
    for i in range(n_rows):
        window = np.ones(i+1)/(i+1)
        R[i, :] = np.convolve(W[i, :], window, 'same')
    #
    return R


# ============================================================================
def cwt(y, dt, wave=Morlet, lNorm=2, scale_list=None, dj=1/12.):
  '''
  TODO: describe better
  '''
  #----------------------------------------------------------
  # Important definitions
  
  # Scales considered for the transform 
  if type(scale_list)==type(None):
    # as defined by Torrence and Compo (1998)
    s0 = 2*dt
    J = int((1/dj)*np.log2(len(y)*dt/s0))
    scaleList = s0 * 2 ** (np.arange(0, J+1) * dj)
  else:
    # using input provided scales 
    scaleList = scale_list
    
  # Setting Norm
  if lNorm==1:
    norm_exp = 1.
  elif lNorm==2:
    norm_exp = 0.5
  else:
    raise Exception("Transform could not be normed, L"+str(lNorm))
 
  #----------------------------------------------------------
  # Applying wavelet transform
  
  # fixing number of terms for fft (forcing power of 2)
  nn = 2 ** (int(np.log2(len(y))) + 1)
  
  # fft over input signal
  fy = np.fft.fft(y, nn)
  
  # wavelet coeff matrix
  wt = np.zeros([len(scaleList), nn], dtype=complex)
  
  # frequencies used for the psi_bar function
  w_k = np.fft.fftfreq(nn, dt) * 2 * np.pi
  
  # Multiscale loop
  for ind, scale in enumerate(scaleList):
    # Normalizing as of Eq. (6) of Torrence and Compo (1998)
    norm = (2 * np.pi * scale / dt) ** norm_exp
    # Heavside step function
    # H=np.ones(len(w_k)) 
    # H[np.where(w_k<=0)]=0
    # Fourier transform of wavelet function
    w_fft = wave(scale * w_k, 'fourier')
    # Normalizing wavelet and applying conjugate
    w_fft_bar = norm * np.conj(w_fft)
    # Wavelet tranform via the convolution theorem
    wt[ind, :] = np.fft.ifft(fy * w_fft_bar, nn) 
  
  # Returning outputs (truncating wavelet time info until len(input)
  return wt[:, :len(y)], scaleList


def i_cwt(w, x, scale_list, dj=1/12., wave=Morlet):
  '''Inverse CWT'''
  #----------------------------------------------------------
  # Important definitions
  w = np.real(w)
  dt = x[1] - x[0]

  #----------------------------------------------------------
  # Wavelet parameters for inverse transform
  wave_params = wave(x, 'params')
  reconstruction_factor = wave_params[0]
  norm = wave_params[1]

  #----------------------------------------------------------
  # Calculating inverse w as of Eq. (11) of Torrence and Compo (1998)
  scales = np.ones([1, np.shape(w)[1]]) * scale_list[:, None]
  sum_arr = w / np.sqrt(scales)
  sum_arr = np.sum(sum_arr, axis=0)
  iw = ( (dj * np.sqrt(dt)) / (reconstruction_factor * (1/norm)) ) * sum_arr
  
  return iw


# ============================================================================
def cwt_phase(y, dt, wave=Morlet, lNorm=2, scale_list=None, dj=1/12.):
    out_cwt, yaxis = cwt(y, dt, wave, lNorm, scale_list, dj)
    out_phase = np.angle(out_cwt)
    return out_phase, yaxis


def cwt_ridge(y, dt, wave=Morlet, lNorm=2, scale_list=None, dj=1/12.):
    out_phase, yaxis = cwt_phase(y, dt, wave, lNorm, scale_list, dj)
    out_ridge = np.zeros(np.shape(out_phase))
    delta_scale = yaxis[1] - yaxis[0]
    out_ridge[1:-1, :] = (out_phase[2:, :]-out_phase[:-2, :])/(2*delta_scale)
    return out_ridge, yaxis


def wavelet_entropy(w):
    # Eq. w/o number, Section 8 of Explorando a transf. wavelet cont.
    P_num = np.abs(w)**2
    P_den = np.sum(P_num, axis=0)
    #
    P_num[np.where(P_num==0)]=1 
    P_den[np.where(P_den==0)]=1 
    #
    P = P_num/P_den
    #
    P[np.where(np.isnan(P))]=1 # removing nan
    #
    entropy = np.sum( (-P*np.log(P)), axis=0)
    #
    entropy[0] = entropy[-1] = np.mean(entropy) # avoiding border problems
    #
    return entropy
    



# ============================================================================

def cross_cwt(y_1, y_2, dt=1, wave=Morlet, lNorm=2, scale_list=None, dj=1/12.):
    cwt_1, yaxis = cwt(y_1, dt, wave, lNorm, scale_list, dj)
    cwt_2, yaxis = cwt(y_2, dt, wave, lNorm, scale_list, dj)
    x_cwt = cwt_1 * np.conj(cwt_2)
    return x_cwt, yaxis 


def cross_phase(y_1, y_2, dt=1, wave=Morlet, lNorm=2, scale_list=None, dj=1/12.):
    cwt_1, yaxis = cwt(y_1, dt, wave, lNorm, scale_list, dj)
    cwt_2, yaxis = cwt(y_2, dt, wave, lNorm, scale_list, dj)
    x_cwt = cwt_1 * np.conj(cwt_2)
    x_phase = np.angle(x_cwt)
    return x_phase, yaxis 


def cross_spectrum(y_1, y_2, dt=1, wave=Morlet, lNorm=2, scale_list=None, dj=1/12.):
    cwt_1, yaxis = cwt(y_1, dt, wave, lNorm, scale_list, dj)
    cwt_2, yaxis = cwt(y_2, dt, wave, lNorm, scale_list, dj)
    # Eq. w/o number, Section 9.1.2 of Explorando a transf. wavelet cont.
    x_cwt = cwt_1 * np.conj(cwt_2)
    cross_spec = np.sum(x_cwt, axis=0)
    return np.real(cross_spec)


def cwt_coherence(y_1, y_2, window_scale_size, dt=1, wave=Morlet, lNorm=2, scale_list=None, dj=1/12.):
    cwt_1, yaxis = cwt(y_1, dt, wave, lNorm, scale_list, dj)
    cwt_2, yaxis = cwt(y_2, dt, wave, lNorm, scale_list, dj)
    # 
    x_cwt = cwt_1 * np.conj(cwt_2)
    scales = np.ones([1, len(y_1)]) * yaxis[:, None]
    # Eq. (20) of Explorando a transf. wavelet cont.
    num = np.abs( S(x_cwt.real/scales, window_scale_size) )**2
    den1 =  S((np.abs(cwt_1)**2)/scales, window_scale_size)
    den2 =  S((np.abs(cwt_2)**2)/scales, window_scale_size)
    return num / (den1 * den2), yaxis


def scale_corr(y_1, y_2, dt=1, wave=Morlet, lNorm=2, scale_list=None, dj=1/12.):
    # Eq. (23) of Explorando a transf. wavelet cont.
    cwt_1, yaxis = cwt(y_1, dt, wave, lNorm, scale_list, dj)
    cwt_2, yaxis = cwt(y_2, dt, wave, lNorm, scale_list, dj)
    # Solving Eq. (24) for each cwt
    # 1
    cwt_1_tempAvg = np.mean(cwt_1, axis=1)
    cwt_1_tempAvg = np.ones([1, len(y_1)]) * cwt_1_tempAvg[:, None]
    ww_1 = cwt_1 - cwt_1_tempAvg
    # 2
    cwt_2_tempAvg = np.mean(cwt_2, axis=1)
    cwt_2_tempAvg = np.ones([1, len(y_2)]) * cwt_2_tempAvg[:, None]
    ww_2 = cwt_2 - cwt_2_tempAvg
    # Solving Eq. (19) for each cwt (global scalogram)
    S_1 = np.sum(np.abs(cwt_1)**2, axis=1)
    S_2 = np.sum(np.abs(cwt_2)**2, axis=1)
    #
    # Finally, Eq. (23):
    num = ww_2 * np.conj(ww_1)
    num = np.sum(num, axis=1)
    den = S_1 * S_2
    return num/np.sqrt(den), yaxis
