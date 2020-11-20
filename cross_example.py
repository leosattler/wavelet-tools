# ============================================================================
# Imports
import numpy as np
import matplotlib.pyplot as plt
from tools import *
from signals import *
# ============================================================================


# ============================================================================
#                                     SIGNALS
# ----------------------------------------------------------------------------
x, y_1 = signal1(plot=False)
x, y_2 = signal2(plot=False)
dt = x[1]-x[0]
name='2freq'

# ============================================================================
#                                      WAVELET
# ----------------------------------------------------------------------------
wave = Morlet
lNorm=2

# ============================================================================
#                                   PLOT PARAMS
# ----------------------------------------------------------------------------
n_x = 2
n_y = 5
cmap_scal = plt.cm.viridis
cmap_phase = plt.cm.Spectral
cmap_ridge = plt.cm.gist_gray
cmap_ridge = cmap_ridge.reversed()

# ============================================================================
#                             x1 - CWTs & CROSS CWT
# ----------------------------------------------------------------------------
w1, scales = cwt(y_1, dt, wave, lNorm)
w2, scales = cwt(y_2, dt, wave, lNorm)
x_w, scales = cross_cwt(y_1, y_2, dt, wave, lNorm)
x_w_2 = np.abs(x_w)**2

# ============================================================================
#                                  x2 - CROSS WAVELET PHASE
# ----------------------------------------------------------------------------
x_w_phase, scales = cross_phase(y_1, y_2, dt, wave, lNorm)

# ============================================================================
#                                 x3 - CROSS WAVELET SPECTRUM
# ----------------------------------------------------------------------------
x_w_spec = cross_spectrum(y_1, y_2, dt, wave, lNorm)

# ============================================================================
#                                 x4 - WAVELETS COHERENCE
# ----------------------------------------------------------------------------
window_scale_size = 4
C2, scales  = cwt_coherence(y_1, y_2, window_scale_size, dt, wave, lNorm)

# ============================================================================
#                             x5 - SCALE CORRELATION
# ----------------------------------------------------------------------------
C_scales, scales  = scale_corr(y_1, y_2, dt, wave, lNorm)

# ============================================================================
#                                    PLOT 1
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(4, 1, figsize=(9,7))
ax1, ax2, ax3, ax4 = ax
# ---------------------------------------------------------------------------- ax1
ax1.plot(y_1, 'k')
#
ax1.set_title('Signal 1')
ax1.set_xlim(0,len(y_1))
ax1.set_xticks([])
# ---------------------------------------------------------------------------- ax2
ax2.plot(y_2, 'k')
#
ax2.set_title('Signal 2')
ax2.set_xlim(0,len(y_2))
ax2.set_xticks([])
# ---------------------------------------------------------------------------- ax3
img3 = ax3.imshow(np.log2(x_w_2), origin='upper', aspect='auto', cmap=cmap_scal)
cbaxes = fig.add_axes([.86, 0.33, 0.015, 0.14])
cb3 = fig.colorbar(img3, ax=ax3, cax=cbaxes)
#
ax3.set_title('Cross Scalogram')
cb3.set_label('log(power)', size=14)
ax3.set_ylabel('scale [unit of scale]')
ax3.set_xticks([])
# ---------------------------------------------------------------------------- ax4
img4 = ax4.imshow(x_w_phase, origin='upper', aspect='auto', cmap=cmap_phase)
cbaxes = fig.add_axes([.86, 0.09, 0.015, 0.14])
cb4 = fig.colorbar(img4, ax=ax4, cax=cbaxes)
#
ax4.set_title('Cross Phase')
ax4.set_ylabel('scale [unit of scale]')
ax4.set_xlabel("time [unit of time]")
# ----------------------------------------------------------------------------
fig.subplots_adjust(left=.1, bottom=.08, right=.82, top=.95, wspace=None, hspace=0.4)
plt.savefig('Cross_'+name+'_1.jpg', dpi=400)
plt.show()


# ============================================================================
#                                    PLOT 2
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(4, 1, figsize=(9,9))
ax1, ax2, ax3, ax4 = ax
# ---------------------------------------------------------------------------- ax1
ax1.set_title('Signals')
ax1.plot(y_1, 'k--', label='Signal 1')
ax1.plot(y_2, 'r', label='Signal 2')
#
ax1.set_xlabel("time [unit of time]")
ax1.set_xlim(0,len(y_1))
ax1.legend(loc='upper left')
# ---------------------------------------------------------------------------- ax2
img2 = ax2.imshow(C2, origin='upper', aspect='auto', cmap=cmap_scal)
cbaxes = fig.add_axes([.899, 0.56, 0.015, 0.14])
cb2 = fig.colorbar(img2, ax=ax2, cax=cbaxes)
#
ax2.set_title('Wavelet Coherence')
ax2.set_ylabel('scale [unit of scale]')
ax2.set_xlabel("time [unit of time]")
# ---------------------------------------------------------------------------- ax3
ax3.plot(scales, np.real(C_scales), 'k-')
ax3.set_xlim(scales[0],scales[-1])
#
ax3.set_title('Scale correlation - coefficients')
ax3.set_xlabel("Scales")
# ---------------------------------------------------------------------------- ax4
ax4.plot(scales, np.angle(C_scales), 'k-')
ax4.set_xlim(scales[0],scales[-1])
#
ax4.set_title('Scale correlation - phase')
ax4.set_xlabel("Scales")
# ----------------------------------------------------------------------------
fig.subplots_adjust(left=.1, bottom=.08, right=.86, top=.95, wspace=None, hspace=0.7)
plt.savefig('Cross_'+name+'_2.jpg', dpi=400)
plt.show()

