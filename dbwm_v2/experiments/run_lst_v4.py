"""
Run the whole DB-WM v4 LST experiment in one process (all ten stages).

LST-defaulted alias for :mod:`experiments.run_ndvi_v4`. It forces
``--modality lst`` and delegates; every flag of the underlying script is
accepted unchanged, so this is exactly::

    python -m experiments.run_ndvi_v4 --modality lst <same args>

The implementation is shared on purpose -- see :mod:`experiments._lst_entry` for
why an LST copy of the pipeline would be a liability rather than a feature.

Usage
-----
    python -m experiments.run_lst_v4 --smoke

    python -m experiments.run_lst_v4 \\
        --lst-dir     /content/lst \\
        --weather-csv /content/drive/MyDrive/Historical_Dataset_SWR_VPD_Ta_P/sayedanwala_historical_weather_2022_2026.csv \\
        --r 256 --memory-order 7 --horizon 6 --epochs 1000 --subspace-energy 0.99
"""
from __future__ import annotations

from experiments._lst_entry import run
from experiments.run_ndvi_v4 import main as _main


def main() -> None:
    """Delegate to :func:`experiments.run_ndvi_v4.main` with ``--modality lst``."""
    run(_main)


if __name__ == "__main__":
    main()
