"""
Train an LST DB-WM v4 model and checkpoint it.

LST-defaulted alias for :mod:`experiments.train_v4`. It forces
``--modality lst`` and delegates; every flag of the underlying script is
accepted unchanged, so this is exactly::

    python -m experiments.train_v4 --modality lst <same args>

The implementation is shared on purpose -- see :mod:`experiments._lst_entry` for
why an LST copy of the pipeline would be a liability rather than a feature.

Usage
-----
    python -m experiments.train_lst_v4 --lst-dir /content/lst \\
        --weather-csv <csv> --r 256 --memory-order 7 --horizon 6 \\
        --epochs 1000 --ckpt-dir ./_v4_ckpt_lst
"""
from __future__ import annotations

from experiments._lst_entry import run
from experiments.train_v4 import main as _main


def main() -> None:
    """Delegate to :func:`experiments.train_v4.main` with ``--modality lst``."""
    run(_main)


if __name__ == "__main__":
    main()
