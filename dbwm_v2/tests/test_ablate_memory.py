"""
Tests for the memory-order ablation and the calendar widening it depends on.

The ablation's whole validity rests on two properties that are easy to break and
invisible when broken:

**Every order is scored on identical origins.** ``purge_boundary_windows`` drops
the first ``L-1`` test dates, so purging per order would hand ``L = 2`` five more
origins than ``L = 7``. The columns would then differ by their evaluation set as
well as by their model, and the difference would be reported as a memory effect.

**Parameter counts are comparable.** v3 Sec. 2.2.4's argument is about counts --
an unstructured order-7 kernel at ``r = 256`` is 4.6e5 parameters against ~1,200
transitions, while S2 is ``rL``. A table of scores without the counts invites
reading a structured win as an order effect.
"""
import datetime as dt

import numpy as np
import pytest

from dbwm.data.seasons import purge_boundary_windows
from dbwm.dynamics.memory import identify_memory
from tests.test_memory import make_s2_system, simulate


def test_shared_origins_are_identical_across_orders():
    """
    Purging with max(L) gives every order the same evaluation set.

    This is the property the ablation script relies on; purging per order is the
    natural thing to write and silently invalidates the comparison.
    """
    train = np.arange(0, 100)
    test = np.arange(100, 140)
    orders = [2, 4, 6, 7]
    shared = purge_boundary_windows(train, test, max(orders))[1]
    per_order = [purge_boundary_windows(train, test, l)[1] for l in orders]
    # Per order they differ; sharing max(L) makes them one set.
    assert len({len(p) for p in per_order}) > 1
    assert all(set(shared).issubset(set(p)) for p in per_order)
    assert len(shared) == len(test) - max(orders) + 1


def test_parameter_count_matches_the_s2_claim():
    """
    S2 costs ``rL`` real parameters -- the count v3 Sec. 2.2.4 quotes.

    Each modal block carries one complex AR(L) coefficient vector; a real mode
    uses ``L`` of the ``2L`` reals, a conjugate pair uses all ``2L`` for the
    pair, so the total is ``rL`` either way.
    """
    from experiments.ablate_memory import _parameter_count

    truth = make_s2_system(order=3)
    w = simulate(truth, 900, seed=3, noise=0.4)
    op, _ = identify_memory(w, 4, None, parameterization="s2", ridge_mu=1e-6)
    assert _parameter_count(op) == op.r * op.order

    un, _ = identify_memory(w, 4, None, parameterization="unstructured",
                            ridge_mu=1e-6)
    # Unstructured is L r^2 -- the count that makes structure mandatory.
    assert _parameter_count(un) == op.order * op.r**2
    assert _parameter_count(un) > _parameter_count(op)


def test_default_orders_are_the_reported_set():
    """The thesis reports 2, 4, 6 and 7; the default must not drift from that."""
    from experiments.ablate_memory import DEFAULT_ORDERS

    assert DEFAULT_ORDERS == (2, 4, 6, 7)


def test_extend_archive_end_covers_the_requested_origin_and_horizon():
    """
    An operational origin sits past the archive; the calendar must widen for it.

    Refusing would make "forecast the next six days from today" impossible,
    since today is always past the end of a fixed archive window.
    """
    from experiments.infer_v4 import _extend_archive_end

    got = _extend_archive_end("2026-04-30", ["2026-05-02"], 6)
    assert dt.date.fromisoformat(got) == dt.date(2026, 5, 8)
    # An origin already inside the window changes nothing.
    assert _extend_archive_end("2026-04-30", ["2026-01-01"], 6) == "2026-04-30"
    # The widest requested origin wins.
    got = _extend_archive_end(
        "2026-04-30", ["2026-05-02", "2026-06-01", "2026-05-10"], 3
    )
    assert dt.date.fromisoformat(got) == dt.date(2026, 6, 4)


def test_extend_archive_end_ignores_unparseable_dates():
    """
    A malformed date is left to ``resolve_origin_dates``, which reports it properly.

    Raising here would give the user a message about the calendar when the
    actual problem is a typo in their date.
    """
    from experiments.infer_v4 import _extend_archive_end

    assert _extend_archive_end("2026-04-30", ["not-a-date"], 6) == "2026-04-30"
