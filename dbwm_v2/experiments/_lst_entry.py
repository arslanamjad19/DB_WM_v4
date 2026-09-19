"""
LST-defaulted aliases for the v4 entry points.

Why these are aliases and not copies
------------------------------------
The v4 pipeline is one implementation with a ``--modality`` switch: the GP
posterior, the memory lift, the horizon family and the lifted filter do not know
or care whether a pixel is a vegetation index or a temperature, and the things
that *do* differ (units, display range, ramp, thresholds, validity gate) are data
in :mod:`dbwm.data.modality`, not code.

So a second copy of ``run_ndvi_v4.py`` with ``lst`` substituted would be 1,100
duplicated lines that start identical and end different -- and the difference
would be silent, because both would still run. Every fix to the shared
mathematics would then have to be made twice, and the LST results would slowly
stop being comparable to the NDVI ones, which is the entire point of running
both.

What these modules give instead is the *command surface* -- ``python -m
experiments.run_lst_v4 ...`` alongside ``run_ndvi_v4`` -- over the identical
implementation. Each sets ``--modality lst`` when it is not already present and
delegates. Passing ``--modality ndvi`` to an ``_lst_`` entry point is refused
rather than honoured: an ``run_lst_v4`` invocation that quietly produced NDVI
results would be the worst outcome available here.

The equivalence is exact and is pinned by ``tests/test_lst_modality.py``:
``run_lst_v4 <args>`` and ``run_ndvi_v4 --modality lst <args>`` build the same
parsed arguments.
"""
from __future__ import annotations

import sys
from typing import Callable, List, Optional, Sequence


def force_modality(
    argv: Optional[Sequence[str]], modality: str = "lst"
) -> List[str]:
    """
    Return ``argv`` with ``--modality <modality>`` guaranteed present.

    :param argv: the argument list, or ``None`` for ``sys.argv[1:]``.
    :param modality: the modality this entry point is for.
    :return: the adjusted argument list.
    :raises SystemExit: if the caller asked for a *different* modality, which
        would make the entry point's own name a lie.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    for i, a in enumerate(args):
        got = None
        if a == "--modality" and i + 1 < len(args):
            got = args[i + 1]
        elif a.startswith("--modality="):
            got = a.split("=", 1)[1]
        if got is None:
            continue
        if got.lower() != modality:
            raise SystemExit(
                "experiments.{entry}_* is the {mod} entry point, but --modality "
                "{got} was passed. Running it would write {got} results under "
                "{mod} filenames. Use `python -m experiments.run_ndvi_v4 "
                "--modality {got} ...` (or the matching _{got} alias) "
                "instead.".format(entry=modality, mod=modality, got=got.lower())
            )
        return args
    return ["--modality", modality] + args


def run(main: Callable[[], None], argv: Optional[Sequence[str]] = None,
        modality: str = "lst") -> None:
    """
    Invoke a v4 entry point's ``main()`` with the modality forced.

    ``sys.argv`` is rewritten rather than passed through, because the delegated
    ``main()`` functions call ``parse_args()`` with no arguments -- that is how
    every one of them is written, and changing all six signatures to thread an
    ``argv`` through would be a larger edit to shared code than this is worth.

    :param main: the ``main()`` of the underlying entry point.
    :param argv: arguments, or ``None`` for ``sys.argv[1:]``.
    :param modality: the modality to force.
    """
    sys.argv = [sys.argv[0]] + force_modality(argv, modality)
    main()


def run_for_checkpoint(
    main: Callable[[], None], argv: Optional[Sequence[str]] = None,
    modality: str = "lst",
) -> None:
    """
    Invoke a checkpoint-driven entry point, verifying the checkpoint's modality.

    ``infer_v4`` and ``forecast_v4`` take the modality from the **checkpoint**,
    not from a flag -- rightly, since the model was identified against one
    archive and scoring it against the other is meaningless. So forcing
    ``--modality`` here would achieve nothing: ``infer_v4`` registers the flag
    (through ``add_data_args``) but never reads it, and ``forecast_v4`` does not
    register it at all.

    What is worth doing instead is checking. ``infer_lst_v4 --ckpt <an NDVI
    checkpoint>`` would otherwise run happily and produce NDVI results from a
    command that says LST -- exactly the confusion the separate entry points
    exist to prevent. The checkpoint is peeked at before any work starts.

    :param main: the ``main()`` of the underlying entry point.
    :param argv: arguments, or ``None`` for ``sys.argv[1:]``.
    :param modality: the modality this entry point is for.
    :raises SystemExit: if the checkpoint was trained on another modality.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    _assert_ckpt_modality(args, modality)
    sys.argv = [sys.argv[0]] + args
    main()


def _assert_ckpt_modality(args: Sequence[str], modality: str) -> None:
    """
    Refuse a checkpoint trained on a different modality.

    Failures to *read* the checkpoint are deliberately ignored: a missing or
    malformed path is the delegated script's error to report, with its own
    message, and duplicating that here would only make it worse.

    :param args: the argument list.
    :param modality: the modality this entry point is for.
    :raises SystemExit: on a confirmed mismatch.
    """
    path = None
    for i, a in enumerate(args):
        if a == "--ckpt" and i + 1 < len(args):
            path = args[i + 1]
        elif a.startswith("--ckpt="):
            path = a.split("=", 1)[1]
    if not path:
        return
    try:
        import pickle

        with open(path, "rb") as fh:
            got = pickle.load(fh)["config"]["data"]["modality"]
    except Exception:
        return
    if str(got).lower() != modality:
        raise SystemExit(
            "{path} was trained on {got}, but this is the {mod} entry point. "
            "Scoring a model against the other archive is not an ablation, it is "
            "a category error -- the basis, the operator and the normalisation "
            "all belong to {got}. Run this checkpoint through the plain entry "
            "point instead (the {got} one), e.g. `python -m "
            "experiments.infer_v4 --ckpt {path} ...`.".format(
                path=path, got=str(got).lower(), mod=modality,
            )
        )

