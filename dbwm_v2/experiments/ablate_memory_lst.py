"""
Memory-order ablation (L) on the LST record.

LST-defaulted alias for :mod:`experiments.ablate_memory`. It forces
``--modality lst`` and delegates; every flag of the underlying script is
accepted unchanged, so this is exactly::

    python -m experiments.ablate_memory --modality lst <same args>

The implementation is shared on purpose -- see :mod:`experiments._lst_entry` for
why an LST copy of the pipeline would be a liability rather than a feature.

Usage
-----
    python -m experiments.ablate_memory_lst --orders 2 4 6 7 \\
        --lst-dir /content/lst --weather-csv <csv>
"""
from __future__ import annotations

from experiments._lst_entry import run
from experiments.ablate_memory import main as _main


def main() -> None:
    """Delegate to :func:`experiments.ablate_memory.main` with ``--modality lst``."""
    run(_main)


if __name__ == "__main__":
    main()
