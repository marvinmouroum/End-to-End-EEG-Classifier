"""End-to-End EEG Classifier package.

Deep-learning classification of motor-imagery EEG (Kaya et al. 2018,
CLA experiment): raw 22-channel EEG is reconstructed into 3D voxel
videos around the electrode positions and classified with a 3D CNN
(optionally with LSTM head) extracted from the original Colab notebook.
"""

from .models import SimpleLSTM, eeg_shallow_CNN, get_net

__version__ = "0.1.0"

__all__ = ["SimpleLSTM", "__version__", "eeg_shallow_CNN", "get_net"]
