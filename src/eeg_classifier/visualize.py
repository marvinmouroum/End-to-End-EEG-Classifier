"""3D visualization of reconstructed EEG voxel frames (notebook cells 14/15).

Renders one voxel frame (electrode positions plus the 1/r interpolated
points from :func:`eeg_classifier.data.make_3d_data`) as a grayscale 3D
scatter plot.

Usage::

    PYTHONPATH=src python -m eeg_classifier.visualize --demo \
        --output assets/brain_activation.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering; must run before pyplot import

import matplotlib.pyplot as plt
import numpy as np

from .data import ELECTRODE_POSITIONS, PROJECT_ROOT, make_3d_data

DEFAULT_OUTPUT = PROJECT_ROOT / "assets" / "brain_activation.png"

DEMO_ACTIVATION = np.round(np.sin(np.linspace(0.0, 3.0, 22)) * 50, 1)


def plot_brain_activation(activation: np.ndarray,
                          output_path: Path | str = DEFAULT_OUTPUT,
                          electrode_positions: np.ndarray = ELECTRODE_POSITIONS,
                          *,
                          figsize: tuple[float, float] = (8.0, 8.0),
                          dpi: int = 150,
                          title: str | None = "Brain activation (voxel reconstruction)",
                          show_axes: bool = False,
                          view: tuple[float, float] | None = (22, -55),
                          point_size: float = 40.0) -> Path:
    """Render one voxel frame as a grayscale 3D scatter plot.

    ``activation`` holds one value per electrode (shape ``(22,)``). The
    frame is reconstructed with :func:`eeg_classifier.data.make_3d_data`,
    which places the electrode voxels plus their 1/r interpolated
    neighbors. All filled voxels are drawn in grayscale shades by
    magnitude (``Greys`` colormap); the electrode centers themselves are
    overlaid as darker markers.

    Writes a PNG to ``output_path`` and returns it.
    """
    activation = np.asarray(activation, dtype=float).reshape(22)
    positions = np.asarray(electrode_positions)

    frame = make_3d_data(np.hstack((positions, activation.reshape(22, 1))))

    points = []
    intensities = []

    for z in range(frame.shape[0]):
        for x in range(frame.shape[1]):
            for y in range(frame.shape[2]):
                if abs(frame[z, x, y]) > 0:
                    points.append([x, y, z])
                    intensities.append(abs(frame[z, x, y]))

    points = np.asarray(points)
    intensities = np.asarray(intensities)

    max_intensity = intensities.max() if intensities.size else 1.0

    # electrode centers shifted into the voxel grid (same +7, +10, +5 as make_3d_data)
    shifted = positions + np.array([7, 10, 5])
    in_bounds = ((shifted[:, 0] < 16) & (shifted[:, 1] < 22) & (shifted[:, 2] < 12)
                 & (shifted.min(axis=1) >= 0))
    electrode_points = shifted[in_bounds]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    if points.size:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=intensities / max_intensity, cmap="Greys",
                   alpha=0.8, edgecolors="none", s=point_size * 0.75)

    if electrode_points.size:
        ax.scatter(electrode_points[:, 0], electrode_points[:, 1], electrode_points[:, 2],
                   c="0.15", edgecolors="none", s=point_size * 1.5)

    if view is not None:
        ax.view_init(elev=view[0], azim=view[1])
    ax.set_box_aspect((16, 22, 12))

    if show_axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    else:
        ax.set_axis_off()

    if title:
        ax.set_title(title, fontsize=13, color="0.25", pad=12)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor="white", transparent=False)
    plt.close(fig)

    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI argument parser for the visualization entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m eeg_classifier.visualize",
        description="Render a reconstructed EEG voxel frame as a grayscale 3D scatter PNG.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="target PNG path (default assets/brain_activation.png).")
    parser.add_argument("--demo", action="store_true",
                        help="render a synthetic activation pattern instead of real data.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.demo:
        parser.error("no activation data available yet; pass --demo for a synthetic pattern.")

    path = plot_brain_activation(DEMO_ACTIVATION, args.output)
    print("wrote", path)


if __name__ == "__main__":
    main()
