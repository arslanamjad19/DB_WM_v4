"""
Operational LST forecast: one origin date in, t+1..t+6 out.

LST-defaulted alias for :mod:`experiments.forecast_v4`. This one takes the
modality from the **checkpoint** rather than from a flag -- which is right,
since the basis, the operator and the normalisation all belong to the
archive the model was identified against -- so instead of forcing anything
it *verifies*: an NDVI checkpoint passed here is refused before any work
starts, rather than quietly producing NDVI results from a command that
says LST. Every flag of the underlying script is accepted unchanged.

The implementation is shared on purpose -- see :mod:`experiments._lst_entry` for
why an LST copy of the pipeline would be a liability rather than a feature.

Usage
-----
    python -m experiments.forecast_lst_v4 --ckpt <pkl> --date 2026-05-02 \\
        --lst-dir /content/lst --weather-csv <csv> \\
        --context-days 7 --export-geotiff
"""
from __future__ import annotations

from experiments._lst_entry import run_for_checkpoint
from experiments.forecast_v4 import main as _main


def main() -> None:
    """Delegate to :func:`experiments.forecast_v4.main`, checking the checkpoint."""
    run_for_checkpoint(_main)


if __name__ == "__main__":
    main()
