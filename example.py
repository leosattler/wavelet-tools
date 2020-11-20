# ============================================================================
# Imports
import numpy as np
import matplotlib.pyplot as plt
from tools import *
from signals import *
# ============================================================================


# ============================================================================
#                                      SIGNAL
# ----------------------------------------------------------------------------
x, y = signal_prof()
dt = x[1] - x[0]
name='p'

# ============================================================================
#                                      WAVELET
# ----------------------------------------------------------------------------
wave = Morlet
lNorm=2

# ============================================================================
#                                   PLOT PARAMS
# ---------------------------------------------------------------------------
n_x = 2
n_y = 5
cmap_scal = plt.cm.viridis
cmap_phase = plt.cm.Spectral
cmap_ridge = plt.cm.gist_gray
cmap_ridge = cmap_ridge.reversed()

# ============================================================================
#                                    1 - CWT
# ----------------------------------------------------------------------------
w, scales = cwt(y, dt, wave, lNorm)
w_2 = np.abs(w)**2

# ============================================================================
#                                2 - INVERSE CWT
# ----------------------------------------------------------------------------
i_y = i_cwt(w, x, scales)

# ============================================================================
#                                  3 - PHASE
# ----------------------------------------------------------------------------
w_phase, scales = cwt_phase(y, dt, wave, lNorm)

# ============================================================================
#                                  4 - RIGE
# ----------------------------------------------------------------------------
w_ridge, scales = cwt_ridge(y, dt, wave, lNorm)

# ============================================================================
#                                   5 - ENTROPY
# ----------------------------------------------------------------------------
entropy = wavelet_entropy(w)

# ============================================================================
#                                    PLOT 1
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(4, 1, figsize=(9,7))
ax1, ax2, ax3, ax4 = ax
# ---------------------------------------------------------------------------- ax1
ax1.plot(y, 'k')
#
ax1.set_title('Signal')
ax1.set_xlim(0,len(y))
ax1.set_xticks([])
# ---------------------------------------------------------------------------- ax2
img2 = ax2.imshow(np.log2(w_2), origin='upper', aspect='auto', cmap=cmap_scal)
cbaxes = fig.add_axes([.86, 0.56, 0.015, 0.14])
cb2 = fig.colorbar(img2, ax=ax2, cax=cbaxes)
#
ax2.set_title('Scalogram')
cb2.set_label('log(power)', size=14)
ax2.set_ylabel('scale [unit of scale]')
ax2.set_xticks([])
# ---------------------------------------------------------------------------- ax3
img3 = ax3.imshow(w_phase, origin='upper', aspect='auto', cmap=cmap_phase)
cbaxes = fig.add_axes([.86, 0.33, 0.015, 0.14])
cb3 = fig.colorbar(img3, ax=ax3, cax=cbaxes)
#
ax3.set_title('Phase')
ax3.set_ylabel('scale [unit of scale]')
ax3.set_xticks([])
# ---------------------------------------------------------------------------- ax4
img4 = ax4.imshow(w_ridge, origin='upper', aspect='auto', vmin=0, vmax=np.max(w_ridge), cmap=cmap_ridge)
cbaxes = fig.add_axes([.86, 0.09, 0.015, 0.14])
cb4 = fig.colorbar(img4, ax=ax4, cax=cbaxes)
#
ax4.set_title('Ridge')
ax4.set_ylabel('scale [unit of scale]')
ax4.set_xlabel("time [unit of time]")
# ----------------------------------------------------------------------------
fig.subplots_adjust(left=.1, bottom=.08, right=.82, top=.95, wspace=None, hspace=0.4)
plt.savefig(name+'_1.jpg', dpi=400)
plt.show()


# ============================================================================
#                                    PLOT 1
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(9,5))
ax1, ax2 = ax
# ---------------------------------------------------------------------------- ax1
ax1.plot(y, 'k')
#
ax1.set_title('Signal')
ax1.set_xlim(0,len(y))
ax1.set_xticks([])
# ---------------------------------------------------------------------------- ax2
ax2.plot(entropy, 'k')
#
ax2.set_title('Entropy')
ax2.set_xlim(0,len(y))
ax2.set_ylabel('scale [unit of scale]')
ax2.set_xlabel("time [unit of time]")
# ----------------------------------------------------------------------------
#fig.subplots_adjust(left=.1, bottom=.08, right=.82, top=.95, wspace=None, hspace=0.4)
plt.savefig(name+'_2.jpg', dpi=400, bbox_inches='tight')
plt.show()

