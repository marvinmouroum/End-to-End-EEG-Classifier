"""PyTorch models extracted from ``notebooks/01_original_colab.ipynb``.

Both classes are ported functionally identical: layer shapes, parameter
counts and forward semantics match the notebook so historical results
stay comparable. Known quirks of the original design (22-layer LSTM,
softmax fed into CrossEntropyLoss, per-frame LSTM head) are documented
on the classes instead of being silently changed.
"""

import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"


class SimpleLSTM(nn.Module):
    """Softmax LSTM classifier over 22-channel EEG feature sequences.

    The input is a sequence ``[seq_len, batch, input_size]`` (22 electrode
    features per frame, ``batch_first=False``). A linear layer maps every
    timestep to ``output_size`` class scores and a softmax is applied over
    the class dimension, so the output is ``[seq_len, batch, output_size]``
    probabilities (not log-probabilities, and not raw logits — the original
    notebook combined these outputs with ``CrossEntropyLoss``).

    Quirk kept from the notebook: ``nn.LSTM(input_size, 22, hidden_size)``
    passes ``hidden_size`` as the *num_layers* argument, so with the
    defaults this is a 22-layer LSTM with hidden size 22, and
    ``init_hidden``/``init_cell`` hardcode the leading dimension 22
    (num_layers). Changing ``hidden_size`` alone breaks the hidden-state
    shape, exactly as in the original.
    """

    def __init__(self, input_size: int = 22, hidden_size: int = 22,
                 output_size: int = 4, device: str = "cuda:0") -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # NOTE: third positional arg of nn.LSTM is num_layers (notebook quirk).
        self.i2h = nn.LSTM(input_size, 22, hidden_size, bias=False, batch_first=False)
        self.i2o = nn.Linear(hidden_size, output_size)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor = None,
                cell: torch.Tensor = None, device: str | None = None) -> torch.Tensor:
        """Run the LSTM over ``input`` and return softmax class probabilities.

        ``device`` defaults to the device of the module's parameters
        (deviation from the notebook, which hardcoded ``cuda:0`` and could
        therefore only run on CUDA); passing an explicit device keeps the
        original behavior.
        """
        device = device if device is not None else next(self.parameters()).device

        if hidden is None:
            hidden = self.init_hidden(input.shape[1], device)

        if cell is None:
            cell = self.init_hidden(input.shape[1], device)

        output, (_, _) = self.i2h(input, (hidden, cell))

        output = self.i2o(output)

        return F.softmax(output, dim=2)

    def init_hidden(self, shape: int = 1, device: str = "cuda:0") -> torch.Tensor:
        """Zero initial hidden state ``[22, batch, hidden_size]``."""
        return torch.zeros(22, shape, self.hidden_size).to(device)

    def init_cell(self, shape: int = 1, device: str = "cuda:0") -> torch.Tensor:
        """Zero initial cell state ``[22, batch, hidden_size]``."""
        return torch.zeros(22, shape, self.hidden_size).to(device)


class eeg_shallow_CNN(nn.Module):
    """3D-CNN feature extractor with an LSTM head (the notebook's CNN-LSTM).

    Input: voxel video frames ``[batch, 1, 12, 16, 22]`` produced by
    :mod:`eeg_classifier.data` (asymmetric grid: 12 z x 16 x x 22 y slices
    around the reconstructed electrode positions).

    Convolutional stack (all Conv3d, stride 1 unless noted):

    - conv1: 6 filters (2, 2, 2), padding (2, 2, 2) + BatchNorm3d + ReLU
    - conv2: 12 filters (2, 2, 2), padding (1, 1, 1) + BatchNorm3d + ReLU
    - max pool (2, 2, 2), stride (1, 1, 1)
    - conv3: 17 filters (2, 2, 2), padding (1, 1, 0) + BatchNorm3d + ReLU
    - conv4: 22 filters (1, 1, 1) + BatchNorm3d + ReLU
    - max pool (2, 3, 2), stride (1, 1, 1)
    - conv5: 11 filters (3, 4, 4), stride (2, 2, 1), padding (2, 2, 2) + BN + ReLU
    - conv6: 1 filter (3, 4, 1), stride (2, 2, 1), padding (0, 1, 0) + BN + ReLU
    - max pool (4, 5, 3), stride (2, 1, 1) -> ``[batch, 1, 1, 1, 22]``

    The result is reshaped to ``[batch, timesteps=1, 22]`` and passed through
    the embedded :class:`SimpleLSTM`, returning ``[batch, 1, 4]`` class
    probabilities. Because ``SimpleLSTM`` uses ``batch_first=False``, the
    LSTM actually treats the batch dimension as the sequence axis with
    batch size 1 — per-frame semantics are unaffected since LSTM steps are
    independent. The notebook has no separate pure-CNN classifier: this
    class *is* the CNN-LSTM combination, and it classifies each voxel frame
    independently (the training loop later re-groups 200 consecutive frames
    via a reshape and evaluates the frame at each sequence end).

    Bookkeeping helpers (``save``, ``next_epoch``, ``reset``) mirror the
    notebook; ``save`` writes a pickle of the whole module to a local
    directory (default ``results/checkpoints/``) instead of Google Drive.
    Loading requires this package to be importable (see :func:`get_net`).
    """

    def __init__(self, name: str = "shallow_network") -> None:
        super().__init__()
        self.T = 120

        self.name = name

        self.training_epochs = 0

        self.lr_history = []
        self.accuracy_history = []
        self.loss_history = []

        self.conv1 = nn.Conv3d(1, 6, (2, 2, 2), stride=(1, 1, 1), padding=(2, 2, 2), dilation=1)
        self.batchnorm1 = nn.BatchNorm3d(6)

        self.conv2 = nn.Conv3d(6, 12, (2, 2, 2), stride=(1, 1, 1), padding=(1, 1, 1), dilation=1)
        self.batchnorm2 = nn.BatchNorm3d(12)

        self.pooling1 = nn.MaxPool3d((2, 2, 2), stride=(1, 1, 1), dilation=1)

        self.conv3 = nn.Conv3d(12, 17, (2, 2, 2), stride=(1, 1, 1), padding=(1, 1, 0), dilation=1)
        self.batchnorm3 = nn.BatchNorm3d(17)

        self.conv4 = nn.Conv3d(17, 22, (1, 1, 1), stride=(1, 1, 1), dilation=1)
        self.batchnorm4 = nn.BatchNorm3d(22)

        self.pooling2 = nn.MaxPool3d((2, 3, 2), stride=(1, 1, 1), dilation=1)

        self.conv5 = nn.Conv3d(22, 11, (3, 4, 4), stride=(2, 2, 1), padding=(2, 2, 2), dilation=1)
        self.batchnorm5 = nn.BatchNorm3d(11)
        self.conv6 = nn.Conv3d(11, 1, (3, 4, 1), stride=(2, 2, 1), padding=(0, 1, 0), dilation=1)
        self.batchnorm6 = nn.BatchNorm3d(1)

        self.pooling3 = nn.MaxPool3d((4, 5, 3), stride=(2, 1, 1), dilation=1)

        self.fc1 = torch.nn.Linear(in_features=1 * 1 * 22, out_features=11)
        self.fc2 = torch.nn.Linear(in_features=11, out_features=4)

        self.rnn = SimpleLSTM()

        torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        torch.nn.init.xavier_normal_(self.conv3.weight)
        torch.nn.init.xavier_normal_(self.conv4.weight)
        torch.nn.init.xavier_normal_(self.conv5.weight)
        torch.nn.init.xavier_normal_(self.conv6.weight)

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Voxel frames ``[batch, 1, 12, 16, 22]`` -> class probabilities ``[batch, 1, 4]``."""
        # first set of CNNs and then a max pool
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)

        x = self.pooling1(x)

        # second set

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = F.relu(x)

        x = self.pooling2(x)

        # set 3

        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = F.relu(x)

        x = self.pooling3(x)

        batch_size, timesteps, _C, _H, W = x.size()

        x = x.view(batch_size, timesteps, W)

        return self.rnn(x)

    def save(self, root: Path = DEFAULT_CHECKPOINT_DIR) -> Path:
        """Pickle the whole network (incl. training history) to ``root/<name>.pickle``."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)

        path = root / (self.name + ".pickle")

        with open(path, "wb") as f:
            pickle.dump(self, f)

        return path

    def next_epoch(self, lr: float, acc_hist: float, loss_hist: float) -> None:
        """Append one epoch of learning rate, accuracy and loss to the history."""
        self.training_epochs += 1
        self.lr_history.append(lr)
        self.accuracy_history.append(acc_hist)
        self.loss_history.append(loss_hist)

    def reset(self) -> None:
        """Clear epoch count and training history."""
        self.training_epochs = 0
        self.lr_history = []
        self.accuracy_history = []
        self.loss_history = []


def get_net(path: Path) -> nn.Module:
    """Load a pickled network saved by :meth:`eeg_shallow_CNN.save`.

    Checkpoints store whole module instances, so ``eeg_classifier.models``
    must be importable (and the PyTorch version should match the one used
    for saving). Notebooks saved to ``__main__`` cannot be loaded here.
    """
    with open(path, "rb") as f:
        result = pickle.load(f)

    return result
