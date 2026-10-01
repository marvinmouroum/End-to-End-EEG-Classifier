"""Data loading, cleaning and 3D voxel reconstruction.

Local-filesystem port of notebook cells 5, 6, 8 and 9: raw Kaya et al.
(2018) CLA ``.mat`` sessions (MATLAB struct ``o`` with ``data`` [nS x 22]
and ``marker`` [nS x 1]) are cleaned of service markers, combined into
frames of 22 electrode values plus one label, and reconstructed into
asymmetric ``12 x 16 x 22`` voxel grids by placing each electrode on the
grid and interpolating with 1/r falloff around it. Preprocessed batches
are stored as pickles like in the original workflow.

No Google Drive / pydrive / colab dependencies: everything operates on
local paths under ``data/`` by default (relative to the project root,
overridable via parameters).
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np

# NOTE: torch and scipy.io are imported lazily inside the functions that
# need them so that visualization-only usage (numpy + matplotlib) does
# not require the heavy dependencies.

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_VIDEO_DIR = DEFAULT_DATA_DIR / "VideoBatches"

# Electrode positions [x, y, z] on the reconstructed voxel grid, ported
# 1:1 from notebook cell 8. The positions conceptually come from mapping
# the 22 electrodes onto a head ellipsoid (a=72.5, b=100, c=56, see
# README), but the notebook hardcodes these integer grid coordinates.
# Quirk kept from the notebook: row index 2 is never assigned and stays
# at the origin [0, 0, 0].
ELECTRODE_POSITIONS = np.zeros((22, 3), dtype=int)

ELECTRODE_POSITIONS[0] = -2, 10, 0
ELECTRODE_POSITIONS[1] = 2, 10, 0
ELECTRODE_POSITIONS[3] = -4, 7, 3
ELECTRODE_POSITIONS[4] = 4, 7, 3
ELECTRODE_POSITIONS[5] = -5, 0, 4
ELECTRODE_POSITIONS[6] = 5, 0, 4
ELECTRODE_POSITIONS[7] = -4, -7, 3
ELECTRODE_POSITIONS[8] = 4, -7, 3
ELECTRODE_POSITIONS[9] = -2, -10, 0
ELECTRODE_POSITIONS[10] = 2, -10, 0
ELECTRODE_POSITIONS[11] = -7, 0, -5
ELECTRODE_POSITIONS[12] = 7, 0, 5
ELECTRODE_POSITIONS[13] = -6, 6, 0
ELECTRODE_POSITIONS[14] = 6, 6, 0
ELECTRODE_POSITIONS[15] = -7, 0, 0
ELECTRODE_POSITIONS[16] = 7, 0, 0
ELECTRODE_POSITIONS[17] = -6, -6, 0
ELECTRODE_POSITIONS[18] = 6, -6, 0
ELECTRODE_POSITIONS[19] = 0, 7, 4
ELECTRODE_POSITIONS[20] = 0, 0, 6
ELECTRODE_POSITIONS[21] = 0, -7, 4


def load_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one raw CLA ``.mat`` session and strip service-marker chunks.

    Reads the MATLAB struct ``o`` (fields ``data`` [nS x 22] and ``marker``
    [nS x 1]) and cuts out chunks containing invalid labels: the session
    start is after the last marker 99 ("initial relaxation"), breaks are
    delimited by marker 91 ("inter-session break") runs, and the session
    ends at the first marker 92 ("experiment end").

    Returns ``(data, labels)`` with all valid chunks stacked together.
    """
    path = Path(path)

    import scipy.io as sio

    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)

    o = mat["o"]
    data = o.data
    labels = o.marker

    valid_data = []
    valid_labels = []

    if np.amax(labels) <= 3:
        valid_data.append(data)
        valid_labels.append(labels)

    # cut out chunks if there were invalid labels
    else:
        start = np.where(labels == 99)
        clean_start = start[0][-1] + 1

        breaks = np.where(labels == 91)

        if breaks[0].size != 0:
            clean_breaks = [breaks[0][0]]

            for i in range(len(breaks[0]) - 1):
                if breaks[0][i] + 1 != breaks[0][i + 1]:
                    clean_breaks.append(breaks[0][i])
                    clean_breaks.append(breaks[0][i + 1])

            if clean_breaks[-1] != breaks[-1][-1]:
                clean_breaks.append(breaks[-1][-1])

            valid_data.append(data[clean_start:clean_breaks[0]])
            valid_labels.append(labels[clean_start:clean_breaks[0]])

            for b in range(1, len(clean_breaks) - 1, 2):
                valid_data.append(data[clean_breaks[b] + 1:clean_breaks[b + 1]])
                valid_labels.append(labels[clean_breaks[b] + 1:clean_breaks[b + 1]])
        else:
            print("\n !!! DATASET does NOT contain BREAK flags !!!")

        finish = np.where(labels == 92)
        if finish[0].size != 0:
            clean_finish = finish[0][0]
        else:
            clean_finish = labels.shape[0]

        if breaks[0].size != 0:
            valid_data.append(data[clean_breaks[-1] + 1:clean_finish])
            valid_labels.append(labels[clean_breaks[-1] + 1:clean_finish])
        else:
            valid_data.append(data[clean_start:clean_finish])
            valid_labels.append(labels[clean_start:clean_finish])

    newdata = np.vstack(valid_data[0:len(valid_data)])
    newlabels = np.hstack(valid_labels[0:len(valid_labels)])

    return (newdata, newlabels)


def combine_data(data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Append the per-frame labels as a 23rd column (notebook cell 6)."""
    return np.hstack((data, labels.reshape((labels.shape[0], 1))))


def make_3d_point(x: int, y: int, z: int, r: int, theta: float, phi: float) -> list[int]:
    """Spherical offset point around ``(x, y, z)`` at distance ``r``."""
    return [
        x + int(r * math.cos(theta) * math.sin(phi)),
        y + int(r * math.sin(theta) * math.sin(phi)),
        z + int(r * math.cos(phi)),
    ]


def make_3d_data(image3d_data: np.ndarray) -> np.ndarray:
    """Reconstruct one ``12 x 16 x 22`` voxel frame from electrode points.

    ``image3d_data`` rows are ``[x, y, z, value]`` electrode points (see
    :data:`ELECTRODE_POSITIONS`). Each point is shifted from the grid
    center to the lower corner (``+7, +10, +5``), written into the
    ``[z, x, y]``-indexed matrix, and interpolated around with 1/r falloff
    at distances 2 and 3 over a 45-degree spherical grid. Out-of-bounds
    points are skipped; a point only overwrites a voxel if the existing
    magnitude is smaller.
    """
    matrix = np.zeros([12, 16, 22])  # z, x, y

    for point in image3d_data:
        newpoint = [int(point[0]) + 7, int(point[1]) + 10, int(point[2]) + 5]

        if newpoint[0] >= matrix.shape[1] or newpoint[1] >= matrix.shape[2] \
                or newpoint[2] >= matrix.shape[0] or newpoint[0] < 0 \
                or newpoint[1] < 0 or newpoint[2] < 0:
            continue

        if abs(matrix[newpoint[2], newpoint[0], newpoint[1]]) <= abs(point[3]):
            matrix[newpoint[2], newpoint[0], newpoint[1]] = int(point[3])

        rho = [2, 3]  # interpolation distances around the point of interest

        for r in rho:
            for theta in range(0, 360, 45):
                for phi in range(0, 180, 45):
                    p = make_3d_point(newpoint[0], newpoint[1], newpoint[2],
                                      r, math.radians(theta), math.radians(phi))

                    if p[0] < matrix.shape[1] and p[1] < matrix.shape[2] \
                            and p[2] < matrix.shape[0] \
                            and abs(matrix[p[2], p[0], p[1]]) <= abs(point[3] / r):
                        matrix[p[2], p[0], p[1]] = int(point[3] / r)

    return matrix


def get_data(data: np.ndarray, batch_size: int = 256,
             test_batch_size: int = 256) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Split a combined session 50/50 into train/validation dataloaders.

    Quirk kept from the notebook: the first half is used for training and
    the second half (minus the final sample) for validation.
    """
    import torch

    num_samples = len(data)
    training_samples = int(num_samples * 0.5 + 1)
    num_samples - training_samples

    training_data = data[0:training_samples]
    validation_data = data[training_samples:-1]

    train_loader = torch.utils.data.DataLoader(training_data, batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(validation_data, batch_size, shuffle=False)

    return train_loader, val_loader


def get_video_data(loader: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Attach electrode positions to a batch of frames -> ``[batch, 22, 4]``.

    Returns the ``[x, y, z, value]`` electrode points per frame and the
    frame labels (column 22 of the combined data).
    """
    batchsize = loader.shape[0]

    training_data = np.empty([batchsize, 22, 4])

    train_label = np.empty(batchsize)

    for it, timestep_data in enumerate(loader):
        train3d = np.hstack((ELECTRODE_POSITIONS, timestep_data[0:22].reshape([22, 1])))
        training_data[it] = train3d
        train_label[it] = timestep_data[22]

    return (training_data, train_label)


def make_video(video_data: np.ndarray) -> np.ndarray:
    """Reconstruct the voxel grid for one frame of electrode points."""
    return make_3d_data(video_data)


def get_training_data(train: tuple[np.ndarray, np.ndarray],
                      mean: float, std: float) -> tuple[torch.Tensor, np.ndarray]:
    """Build the normalized ``[batch, 1, 12, 16, 22]`` voxel stream for a batch."""
    import torch

    stream = np.zeros([train[0].shape[0], 1, 12, 16, 22])

    for i, image in enumerate(train[0]):
        if i >= stream.shape[0]:
            break
        stream[i, 0] = make_video(image)

    tensor = torch.from_numpy(stream)

    tensor = (tensor - mean) / std

    return (tensor, train[1])


def save_video(data: np.ndarray, batch_size: int = 256, path: Path = DEFAULT_VIDEO_DIR,
               mean: float = 0, std: float = 1) -> None:
    """Pickle normalized voxel-video batches of one session to ``path``.

    Writes ``train_<n>.pickle`` for the first half of the session and
    ``val_<n>.pickle`` for the second half. Each pickle holds
    ``[tensor (batch, 1, 12, 16, 22), labels (batch,)]``.

    Deviation from the notebook (cell 8): the original iterated
    ``train_loader`` in *both* loops, so its ``val_*`` files silently
    contained training data again; here the validation loop actually
    consumes the validation split.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    print("saving video files in", path, "with batch size: ", batch_size)

    train_loader, val_loader = get_data(data, batch_size)

    print("\nsaving training data\n")

    for batch_idx, inputs in enumerate(train_loader):

        if batch_idx % 100 == 0:
            print("saving batches: ", batch_idx, "-", batch_idx + 100)

        result_data = get_video_data(inputs)
        result = get_training_data(result_data, mean, std)

        name = path / ("train_" + str(batch_idx) + ".pickle")

        with open(name, "wb") as f:
            pickle.dump([result[0], result[1]], f)

    print("\nsaving validation data\n")

    for batch_idx, inputs in enumerate(val_loader):

        if batch_idx % 100 == 0:
            print("saving batches: ", batch_idx, "-", batch_idx + 100)

        result_data = get_video_data(inputs)
        result = get_training_data(result_data, mean, std)

        name = path / ("val_" + str(batch_idx) + ".pickle")

        with open(name, "wb") as f:
            pickle.dump([result[0], result[1]], f)


def prepare_subject(mat_path: Path, subject_id: int, data_dir: Path = DEFAULT_DATA_DIR,
                    batch_size: int = 256) -> Path:
    """Preprocess one raw ``.mat`` session into voxel-batch pickles.

    Local replacement for the notebook's ``save_video_to_drive`` (cell 9):
    loads and cleans the session, computes session mean/std for
    normalization, and writes train/val pickles to
    ``<data_dir>/VideoBatches/subject<id>/``. Returns that directory.
    """
    mat_path = Path(mat_path)
    data_dir = Path(data_dir)

    print("creating data for ", mat_path.name)

    cleaned_data = load_file(mat_path)

    mean = cleaned_data[0].mean()
    std = cleaned_data[0].std()

    data = combine_data(cleaned_data[0], cleaned_data[1])

    save_dir = data_dir / "VideoBatches" / ("subject" + str(subject_id))

    save_video(data, batch_size=batch_size, path=save_dir, mean=mean, std=std)

    print("Done")

    return save_dir
