"""Evaluation CLI: notebook cell 25 ``test`` + per-subject accuracy + confusion matrix.

Loads a checkpoint saved by :meth:`eeg_classifier.models.eeg_shallow_CNN.save`,
evaluates sequence-end predictions per subject and prints a sklearn
confusion matrix over the CLA classes (1: left hand, 2: right hand,
3: passive).

Usage::

    PYTHONPATH=src python -m eeg_classifier.evaluate \
        --checkpoint results/checkpoints/CNN-LSTM-1.pickle --device cpu
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from .data import DEFAULT_DATA_DIR
from .models import get_net
from .train import (
    SEQ_LEN,
    SEQUENCES_PER_BATCH,
    SUBJECT_SKIPLIST,
    collect_sequences,
    count_preprocessed,
    get_cost_function,
    stack_batch,
)

CLASS_NAMES = {1: "left hand", 2: "right hand", 3: "passive"}


def test(net: torch.nn.Module, data_type: str, cost_function,
         data_dir: Path = DEFAULT_DATA_DIR, device: str = "cuda:0",
         start: int = 0, end: int = 17) -> dict:
    """Evaluate ``net`` on subjects ``start..end-1`` (notebook cell 25).

    ``data_type`` selects the pickle split ("train" or "val"). Every step
    stacks 4 consecutive 200-frame sequences, normalizes them by their own
    mean/std, takes the class prediction at each sequence end and
    accumulates cross-entropy loss and accuracy.

    Returns a dict with overall ``loss``, ``accuracy`` (percent),
    per-subject ``subjects`` accuracies, and the collected ``targets``
    and ``predictions``.
    """
    samples = 0.
    cumulative_loss = 0.
    cumulative_accuracy = 0.

    seq = SEQ_LEN

    subject_accuracies: dict[int, float] = {}
    all_targets: list[int] = []
    all_predictions: list[int] = []

    net.eval()

    with torch.no_grad():

        for subject in range(start, end, 1):
            if subject in SUBJECT_SKIPLIST:
                continue

            print("\nTesting on subject ", subject)

            length = count_preprocessed(data_dir, subject, data_type)

            subject_samples = 0.
            subject_loss = 0.
            subject_accuracy = 0.

            for i in range(length - 1 - SEQUENCES_PER_BATCH):

                if i % 300 == 0:
                    print("on test batch: ", i, "-", i + 300)

                results, _, _ = collect_sequences(data_dir, subject, data_type, i, seq)

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

                samples += SEQUENCES_PER_BATCH

                cumulative_loss += loss.item()

                _, predicted = outputs.max(1)

                cumulative_accuracy += predicted.eq(targets).sum().item()

                subject_samples += SEQUENCES_PER_BATCH
                subject_loss += loss.item()
                subject_accuracy += predicted.eq(targets).sum().item()

                all_targets.extend(targets.tolist())
                all_predictions.extend(predicted.tolist())

            if subject_samples > 0:
                subject_accuracies[subject] = subject_accuracy / subject_samples * 100
                print("Accuracy after subject evaluation -> ",
                      subject_accuracies[subject])
                print("subject loss -> ", subject_loss / subject_samples)

    accuracy = cumulative_accuracy / samples * 100 if samples > 0 else 0.
    loss = cumulative_loss / samples if samples > 0 else 0.

    return {
        "loss": loss,
        "accuracy": accuracy,
        "subjects": subject_accuracies,
        "targets": np.asarray(all_targets, dtype=int),
        "predictions": np.asarray(all_predictions, dtype=int),
    }


def print_confusion_matrix(targets: np.ndarray, predictions: np.ndarray) -> None:
    """Print the sklearn confusion matrix with CLA class names."""
    if targets.size == 0:
        print("\nno predictions collected - nothing to confuse")
        return

    labels = sorted(set(targets.tolist()) | set(predictions.tolist()))
    matrix = confusion_matrix(targets, predictions, labels=labels)

    header = "pred ->  " + "  ".join(
        "{} ({})".format(label, CLASS_NAMES.get(label, "?")) for label in labels)
    print("\nConfusion matrix (rows = target, cols = prediction)")
    print(header)
    for label, row in zip(labels, matrix):
        print("true {}: {}".format(label, "  ".join(str(int(v)) for v in row)))


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI argument parser for the evaluation entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m eeg_classifier.evaluate",
        description="Evaluate a pickled EEG CNN-LSTM checkpoint on preprocessed voxel batches.")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="pickle saved by eeg_shallow_CNN.save / train.py.")
    parser.add_argument("--split", choices=("train", "val"), default="val",
                        help="which preprocessed split to evaluate (notebook data_type).")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None,
                        help="defaults to cuda when available, else cpu.")
    parser.add_argument("--start", type=int, default=0, help="first subject index.")
    parser.add_argument("--end", type=int, default=17, help="one past last subject index "
                        "(subject 7 is always skipped).")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="folder containing subject<N>/<split>_<n>.pickle batches.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = device if ":" in device else device + ":0"

    net = get_net(args.checkpoint)
    net.to(device)

    result = test(net, args.split, get_cost_function(),
                  data_dir=args.data_dir, device=device, start=args.start, end=args.end)

    print("\n==================== evaluation summary ====================")
    print("checkpoint ->", args.checkpoint)
    print("split      ->", args.split)
    print("loss {:.5f}, accuracy {:.2f}%".format(result["loss"], result["accuracy"]))

    if result["subjects"]:
        print("\nper-subject accuracy:")
        for subject, accuracy in sorted(result["subjects"].items()):
            print(f"  subject {subject:>2}: {accuracy:.2f}%")

    print_confusion_matrix(result["targets"], result["predictions"])


if __name__ == "__main__":
    main()
