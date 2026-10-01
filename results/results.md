# Results

Per-subject accuracies and losses from the original 2021 runs, transcribed from the
tables in [docs/paper.pdf](../docs/paper.pdf) (Tables 2, 3, and 4). Nothing here was
re-measured; the values are reproduced exactly as printed in the paper, including
suspicious ones (for example subjects 6 and 8 share identical LSTM accuracy and loss
values, which look like transcription artifacts in the original).

Subject 7 is missing from every table because it was skipped in all original runs.

Comparability caveat: the CNN and LSTM classified every time step, while the CNN-LSTM
was evaluated on the final frame of 200-frame sequences only, so its numbers are not
directly comparable to the other two columns.

## Overview

| Model | Input | Accuracy per subject | Mean accuracy | Training time |
|---|---|---|---|---|
| Single 3D CNN | voxel frames, `12 x 16 x 22` | 71.0 - 76.5% | 73.5% | ~1 h per epoch |
| Single LSTM | raw 22-channel signal | 69.9 - 72.4% | 70.9% | ~3 min per epoch |
| CNN-LSTM | 4 x 200-frame voxel sequences | 43.1 - 63.8% | 52.6% | slowest of the three |

The mean values are computed from the tables below, not quoted from the paper. The 3D CNN
outperforms the LSTM on raw data by about 2.6 percentage points on average, which matches
the paper's discussion; the paper attributes this gap to the spatial contribution
outweighing the temporal one, but not by enough to justify the GPU cost of the 3D
reconstruction. The paper's CNN-LSTM summary of "50 - 60%" and "up to 76%" for the CNN
refer to the same runs tabulated below.

Training used Adam with a cyclic learning rate |cos(i) x 1e-4|, momentum 0.9, weight
decay 1e-9, mini-batches of 4 x 200-frame sequences (4 seconds of recording), and 15
epochs for the CNN-LSTM; the standalone LSTM ran 20 epochs (paper Table 1 and Results
section).

## Single 3D CNN (paper Table 2)

| Subject | Accuracy (%) | Loss |
|---|---|---|
| 0 | 71.04 | 0.00409 |
| 1 | 73.81 | 0.00391 |
| 2 | 73.49 | 0.00394 |
| 3 | 72.39 | 0.00374 |
| 4 | 73.79 | 0.00393 |
| 5 | 73.87 | 0.00378 |
| 6 | 72.40 | 0.00311 |
| 8 | 71.01 | 0.00412 |
| 9 | 74.02 | 0.00391 |
| 10 | 73.91 | 0.00388 |
| 11 | 72.41 | 0.00378 |
| 12 | 73.93 | 0.00390 |
| 13 | 74.91 | 0.00388 |
| 14 | 76.49 | 0.00324 |
| 15 | 74.59 | 0.00389 |
| 16 | 74.19 | 0.00391 |

## Single LSTM on raw data (paper Table 3)

| Subject | Accuracy (%) | Loss |
|---|---|---|
| 0 | 71.44 | 0.000637198 |
| 1 | 71.31 | 0.000637198 |
| 2 | 70.31 | 0.000668321 |
| 3 | 71.95 | 0.000644332 |
| 4 | 70.84 | 0.000666331 |
| 5 | 71.02 | 0.000666341 |
| 6 | 72.37 | 0.006632123 |
| 8 | 72.37 | 0.006632123 |
| 9 | 70.14 | 0.000723402 |
| 10 | 70.14 | 0.000621102 |
| 11 | 70.84 | 0.000637130 |
| 12 | 70.11 | 0.000737171 |
| 13 | 71.46 | 0.000658722 |
| 14 | 69.87 | 0.005666654 |
| 15 | 70.01 | 0.006586622 |
| 16 | 70.17 | 0.006371292 |

The paper notes that these LSTM results converged to a narrow 69 to 72 percent band and
were reached after at most 20 epochs, with about 3 minutes per epoch over 666,700 time
steps.

## CNN-LSTM (paper Table 4)

| Subject | Accuracy (%) | Loss |
|---|---|---|
| 0 | 46.51 | 0.31965 |
| 1 | 45.14 | 0.32307 |
| 2 | 43.06 | 0.32827 |
| 3 | 44.47 | 0.32473 |
| 4 | 43.14 | 0.32806 |
| 5 | 54.02 | 0.30086 |
| 6 | 54.18 | 0.30058 |
| 8 | 54.69 | 0.29919 |
| 9 | 54.09 | 0.30070 |
| 10 | 52.89 | 0.30369 |
| 11 | 53.91 | 0.30114 |
| 12 | 63.79 | 0.27643 |
| 13 | 54.75 | 0.29904 |
| 14 | 58.90 | 0.28867 |
| 15 | 58.47 | 0.28974 |
| 16 | 58.83 | 0.28884 |

The CNN-LSTM was trained for fewer epochs than the other two models due to time
constraints, and its accuracy counts only the sequence-end predictions (one label per
200-frame sequence), as described in the paper.

## Reproducing

The current package trains and evaluates the CNN-LSTM (`eeg_shallow_CNN`) via
`python -m eeg_classifier.train` and `python -m eeg_classifier.evaluate`. The standalone
CNN and LSTM numbers above were produced by the original notebook and have no dedicated
training path in the package; see the [README](../README.md) for the supported workflow.
