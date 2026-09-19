"""
GP posterior vs DeiT vs ResNet for the LST latent state.

LST-defaulted alias for :mod:`experiments.compare_encoders`. It forces
``--modality lst`` and delegates; every flag of the underlying script is
accepted unchanged, so this is exactly::

    python -m experiments.compare_encoders --modality lst <same args>

The implementation is shared on purpose -- see :mod:`experiments._lst_entry` for
why an LST copy of the pipeline would be a liability rather than a feature.

Usage
-----
    python -m experiments.compare_encoders_lst --encoders gp deit resnet \\
        --lst-dir /content/lst --weather-csv <csv> --r 256 --epochs 1000
"""
from __future__ import annotations

from experiments._lst_entry import run
from experiments.compare_encoders import main as _main


def main() -> None:
    """Delegate to :func:`experiments.compare_encoders.main` with ``--modality lst``."""
    run(_main)


if __name__ == "__main__":
    main()
