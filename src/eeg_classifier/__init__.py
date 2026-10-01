"""End-to-End EEG Classifier package.

Deep-learning classification of motor-imagery EEG (Kaya et al. 2018,
CLA experiment): raw 22-channel EEG is reconstructed into 3D voxel
videos around the electrode positions and classified with a 3D CNN
(optionally with LSTM head) extracted from the original Colab notebook.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .models import SimpleLSTM, eeg_shallow_CNN, get_net

__version__ = "0.1.0"

__all__ = ["SimpleLSTM", "__version__", "eeg_shallow_CNN", "get_net"]


def __getattr__(name: str):  # noqa: ANN202 - PEP 562 lazy exports
    if name in {"SimpleLSTM", "eeg_shallow_CNN", "get_net"}:
        # Lazy import: keeps numpy/matplotlib-only entry points
        # (e.g. eeg_classifier.visualize) usable without torch.
        from . import models

        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
