"""Training CLI for the extracted EEG models (notebook cells 22, 25, 26, 29).

Trains :class:`eeg_classifier.models.eeg_shallow_CNN` (the notebook's
CNN-LSTM) on preprocessed voxel-video pickles produced by
:func:`eeg_classifier.data.prepare_subject`. Data is loaded as sliding
windows of 4 consecutive pickles per subject; each pickle contributes a
200-frame sequence whose label is the label of its final frame.

Usage::

    PYTHONPATH=src python -m eeg_classifier.train \
        --model cnn_lstm --epochs 25 --device cpu --data-dir data
"""

import argparse
import math
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable

from .data import DEFAULT_DATA_DIR
from .models import DEFAULT_CHECKPOINT_DIR, eeg_shallow_CNN, get_net

SEQ_LEN = 200
SEQUENCES_PER_BATCH = 4
SUBJECT_SKIPLIST = (7,)


def clean_seq(data: np.ndarray, seq: np.ndarray, seq_len: int = SEQ_LEN) -> list[np.ndarray] | None:
    """Extract one clean ``seq_len`` window from a preprocessed batch.

    A window is valid when its first label is 0 (rest) and its last label
    equals the window maximum (i.e. the window ends on an imagery cue).
    Scans backwards through the batch; returns ``[data_window, labels]``
    or ``None`` if no valid window exists.
    """
    if np.amax(seq) == 0:
        return [data[0:seq_len], seq[0:seq_len]]

    seq_size = seq.size

    first_good = False
    last_good = False
    i = 0

    while not last_good:

        while not first_good:

            snippet = seq[seq_size - seq_len - i:seq_size - i]
            data_snippet = data[seq_size - seq_len - i:seq_size - i]

            first_value = snippet[0]

            if first_value.item() == 0:
                first_good = True
            elif i == 0:
                i = seq_size - seq_len
            else:
                return

        last_value = snippet[-1]

        max_value = np.amax(snippet)

        if max_value == last_value:
            last_good = True
        elif i == 0:
            first_good = False
            i = seq_size - seq_len
        elif not last_good or not first_good:
            return

    return [data_snippet, snippet]


def get_cost_function() -> torch.nn.CrossEntropyLoss:
    """Cross-entropy loss (notebook cell 22)."""
    cost_function = torch.nn.CrossEntropyLoss()
    return cost_function


def get_optimizer(net: torch.nn.Module, lr: float, wd: float, momentum: float) -> torch.optim.Adam:
    """Adam optimizer (notebook cell 22; ``momentum`` is unused, kept for parity)."""
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999),
                                 eps=1e-08, weight_decay=wd, amsgrad=False)
    return optimizer


def count_preprocessed(data_dir: Path, subject: int, split: str) -> int:
    """Number of ``<split>_<n>.pickle`` files of one subject.

    Local replacement for the notebook's gdrive-based
    ``get_preprocessed_data_count_from`` (cell 24).
    """
    subject_dir = Path(data_dir) / ("subject" + str(subject))
    return len(list(subject_dir.glob(split + "_*.pickle")))


def collect_sequences(data_dir: Path, subject: int, split: str, index: int,
                      seq: int = SEQ_LEN) -> tuple[list, int, int]:
    """Load up to 4 consecutive preprocessed batches starting at ``index``.

    Batches whose labels cannot be cleaned into a valid sequence by
    :func:`clean_seq` are skipped. Returns ``(results, skipped, kept)``
    where ``results`` holds up to 4 ``(data, labels)`` pairs.
    """
    subject_dir = Path(data_dir) / ("subject" + str(subject))
    length = count_preprocessed(data_dir, subject, split)

    results = []

    skipped = 0
    kept = 0

    tmp_skipped = 0
    j = 0

    while j < SEQUENCES_PER_BATCH + tmp_skipped and index + SEQUENCES_PER_BATCH + tmp_skipped < length:
        j += 1

        name = subject_dir / (split + "_" + str(index + j) + ".pickle")

        with open(name, "rb") as f:
            subresult = pickle.load(f)

        clean_sub = clean_seq(subresult[0], subresult[1], seq)

        if clean_sub is None:
            skipped += 1
            tmp_skipped += 1
            continue
        else:
            kept += 1

        results.append((clean_sub[0], clean_sub[1]))

    return results, skipped, kept


def stack_batch(results: list, seq: int = SEQ_LEN) -> tuple[torch.Tensor, np.ndarray]:
    """Stack 4 cleaned sequences into inputs ``[4*seq, 1, 12, 16, 22]`` + labels."""
    t_data = Variable(
        torch.from_numpy(
            np.concatenate(
                (np.concatenate((results[0][0], results[1][0]), axis=0),
                 np.concatenate((results[2][0], results[3][0]), axis=0)),
                axis=0)
        ),
        requires_grad=True
    )

    t_labels = np.concatenate(
        (np.concatenate((results[0][1], results[1][1]), axis=None),
         np.concatenate((results[2][1], results[3][1]), axis=None)),
        axis=None)

    return t_data, t_labels


def run_epoch(net: torch.nn.Module, data_type: str, optimizer: torch.optim.Optimizer | None,
              cost_function, data_dir: Path = DEFAULT_DATA_DIR, device: str = "cuda:0",
              start: int = 0, end: int = 17, training: bool = True
              ) -> tuple[float, float, int, int]:
    """One pass over subjects ``start..end-1`` (notebook cells 25/29 loop).

    Groups 4 consecutive 200-frame pickles per step, takes the label at
    each sequence end, and evaluates/trains on the reshaped outputs
    ``[4, seq, 4]``. Returns ``(loss, accuracy_percent, samples, skipped_ratio)``.
    """
    samples = 0.
    cumulative_loss = 0.
    cumulative_accuracy = 0.

    seq = SEQ_LEN

    total_skipped = 0
    total_kept = 0

    if training:
        net.train()
    else:
        net.eval()

    with torch.set_grad_enabled(training):

        for subject in range(start, end, 1):
            if subject in SUBJECT_SKIPLIST:
                continue

            print("\n{} on subject {}".format("Training" if training else "Testing", subject))

            length = count_preprocessed(data_dir, subject, data_type)

            skipped = 0
            kept = 0

            for i in range(length - 1 - SEQUENCES_PER_BATCH):

                if i % 300 == 0:
                    print("on {} batch: ".format("train" if training else "test"), i, "-", i + 300)

                results, batch_skipped, batch_kept = collect_sequences(data_dir, subject, data_type, i, seq)

                skipped += batch_skipped
                kept += batch_kept

                total_skipped += skipped
                total_kept += kept

                if len(results) < SEQUENCES_PER_BATCH:
                    print("not enough data left for rnn -> next epoch pls")
                    continue

                t_data, t_labels = stack_batch(results, seq)

                if t_data.shape[0] < SEQUENCES_PER_BATCH * seq:
                    print("sequence length too short")
                    print(t_data.shape)
                    continue

                if t_labels.max() > 3:
                    continue

                mean = t_data.mean()
                std = t_data.std()

                t_data = (t_data - mean) / std

                inputs = t_data.float().to(device)

                targets = torch.LongTensor(
                    t_labels.reshape((SEQUENCES_PER_BATCH, seq))[0:SEQUENCES_PER_BATCH, seq - 1]
                ).to(device)

                outputs = net(inputs).reshape([SEQUENCES_PER_BATCH, seq, 4])[0:SEQUENCES_PER_BATCH, seq - 1]

                loss = cost_function(outputs, targets)

                if training:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                samples += SEQUENCES_PER_BATCH

                cumulative_loss += loss.item()

                _, predicted = outputs.max(1)

                cumulative_accuracy += predicted.eq(targets).sum().item()

            if samples > 0:
                print("Accuracy after subject evaluation -> ", cumulative_accuracy / samples * 100)

            if skipped + kept > 0:
                print(int(100 * skipped / (skipped + kept)), "% of the data was ignored due to sequence cleaning")

    accuracy = cumulative_accuracy / samples * 100 if samples > 0 else 0.
    avg_loss = cumulative_loss / samples if samples > 0 else 0.

    return avg_loss, accuracy, int(samples), (int(100 * total_skipped / (total_skipped + total_kept))
                                              if total_skipped + total_kept > 0 else 0)


def train(net: torch.nn.Module, data_type: str, optimizer: torch.optim.Optimizer,
          cost_function, data_dir: Path = DEFAULT_DATA_DIR, device: str = "cuda:0",
          start: int = 0, end: int = 17) -> tuple[float, float, float]:
    """Training pass (notebook cell 25, ``train``). Returns ``(loss, accuracy, accuracy)``."""
    loss, accuracy, _, _ = run_epoch(net, data_type, optimizer, cost_function,
                                     data_dir, device, start, end, training=True)
    return loss, accuracy, accuracy


def run_training(net: torch.nn.Module | None = None,
                 netname: str = "net",
                 data_dir: Path = DEFAULT_DATA_DIR,
                 checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
                 batch_size: int = 128,
                 device: str = "cuda:0",
                 learning_rate: float = 1e-5,
                 weight_decay: float = 0.000001,
                 momentum: float = 0.9,
                 epochs: int = 25,
                 start: int = 0,
                 end: int = 17) -> torch.nn.Module:
    """Epoch loop from notebook cell 29 (its ``main``).

    Notebooks quirk kept: the learning rate is re-computed every epoch as
    ``lr + |cos(epoch/2) * lr * 10|`` and the optimizer is rebuilt with it.
    ``batch_size`` is accepted for signature parity but does not affect the
    loop, which always stacks 4 fixed 200-frame sequences per step
    (the pickle batch size is chosen at preprocessing time). A network is
    created when ``net`` is None (the notebook referenced an undefined
    ``eeg_CNN`` here; the actual class is :class:`eeg_shallow_CNN`).
    """
    print("network to device")
    if net is None:
        net = eeg_shallow_CNN(netname).to(device)
    else:
        print("continue training on net")
        net.to(device)
        net.name = netname

    print("optimizer")
    optimizer = get_optimizer(net, learning_rate, weight_decay, momentum)

    print("cost function")
    cost_function = get_cost_function()

    lr = learning_rate
    loss = 0.30160

    for e in range(epochs):
        train_loss, train_accuracy, seq_train_accuracy = train(
            net, "train", optimizer, cost_function, data_dir, device, start, end)

        dloss = 1 - train_loss / loss

        loss = train_loss

        print("change in loss is:", dloss)

        lr = learning_rate + abs(math.cos(e / 2) * learning_rate * 10)

        print("new lr ->", lr)

        optimizer = get_optimizer(net, lr, weight_decay, momentum)

        print(f'Epoch: {e + 1:d}')
        print(f'\t Training loss {train_loss:.5f}, Training accuracy {train_accuracy:.2f}, Sequence Training accuracy {seq_train_accuracy:.2f}')
        print('-----------------------------------------------------')

        net.next_epoch(lr, train_accuracy, train_loss)
        path = net.save(checkpoint_dir)
        print("saved checkpoint ->", path)

    return net


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI argument parser for the training entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m eeg_classifier.train",
        description="Train the EEG voxel-video CNN-LSTM on preprocessed pickles.")
    parser.add_argument("--model", choices=("cnn_lstm", "cnn", "lstm"), default="cnn_lstm",
                        help="cnn_lstm: eeg_shallow_CNN (3D CNN with SimpleLSTM head). "
                             "cnn: alias for cnn_lstm — the notebook's CNN always ends in the "
                             "LSTM head, no pure-CNN classifier exists. lstm: not trainable "
                             "standalone (no training path in the notebook).")
    parser.add_argument("--name", default="CNN-LSTM-1", help="checkpoint name (net.name).")
    parser.add_argument("--net", type=Path, default=None,
                        help="pickle checkpoint to continue training from (overrides --model).")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128,
                        help="accepted for notebook parity; the loop always stacks "
                             "4 x 200-frame sequences per step.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None,
                        help="defaults to cuda when available, else cpu.")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--start", type=int, default=0, help="first subject index.")
    parser.add_argument("--end", type=int, default=17, help="one past last subject index "
                        "(subject 7 is always skipped).")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="folder containing subject<N>/<split>_<n>.pickle batches.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR,
                        help="where <name>.pickle checkpoints are written.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.model == "lstm":
        parser.error("--model lstm: SimpleLSTM has no standalone training path in the "
                     "notebook (it is only used as the head of eeg_shallow_CNN); "
                     "use --model cnn_lstm.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = device if ":" in device else device + ":0"

    net = get_net(args.net) if args.net else None

    run_training(net=net,
                 netname=args.name,
                 data_dir=args.data_dir,
                 checkpoint_dir=args.checkpoint_dir,
                 batch_size=args.batch_size,
                 device=device,
                 learning_rate=args.lr,
                 weight_decay=args.weight_decay,
                 momentum=args.momentum,
                 epochs=args.epochs,
                 start=args.start,
                 end=args.end)


if __name__ == "__main__":
    main()
