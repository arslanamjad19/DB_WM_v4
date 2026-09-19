"""
Tests for the cropping-season calendar and the train/test protocol.

The split is a *quantitative* claim ("~3 Kharif / 3.5 Rabi / 3.5 Zaid in
training"), so it is pinned here rather than trusted.
"""
import datetime as dt

import numpy as np
import pytest

from dbwm.data import seasons as S

ARCHIVE_START = dt.date(2022, 1, 1)
ARCHIVE_END = dt.date(2026, 4, 30)
SPLIT = dt.date(2025, 4, 15)


def test_seasons_tile_the_year():
    """Every day of a year belongs to exactly one season, and spans sum to 365."""
    counts = {s: 0 for s in S.SEASONS}
    d = dt.date(2023, 1, 1)
    while d.year == 2023:
        counts[S.season_of(d)] += 1
        d += dt.timedelta(days=1)
    assert sum(counts.values()) == 365
    assert counts[S.KHARIF] == 122  # Jun+Jul+Aug+Sep = 30+31+31+30
    assert counts[S.ZAID] == 92     # Mar+Apr+May = 31+30+31
    assert counts[S.RABI] == 151    # Jan+Feb+Oct+Nov+Dec


def test_rabi_straddles_the_year_boundary():
    """January belongs to the Rabi instance that started the PREVIOUS October.

    This is the classic off-by-one in agronomic calendars: labelling 15 Jan 2022
    as "Rabi 2022" would merge two different crop cycles.
    """
    assert S.instance_of(dt.date(2022, 1, 15)) == S.SeasonInstance(S.RABI, 2021)
    assert S.instance_of(dt.date(2022, 10, 15)) == S.SeasonInstance(S.RABI, 2022)
    assert S.instance_of(dt.date(2022, 2, 28)) == S.SeasonInstance(S.RABI, 2021)
    assert S.instance_of(dt.date(2022, 3, 1)) == S.SeasonInstance(S.ZAID, 2022)


def test_leap_february_lengthens_rabi():
    """Rabi 2023/24 ends on 29 Feb 2024 and is one day longer than its neighbours."""
    leap = S.SeasonInstance(S.RABI, 2023)
    normal = S.SeasonInstance(S.RABI, 2022)
    assert leap.end == dt.date(2024, 2, 29)
    assert leap.nominal_days == 152
    assert normal.nominal_days == 151


def test_archive_instance_counts_match_the_thesis_enumeration():
    """4 Kharif, 5 Rabi (1 partial) and 5 Zaid (1 partial) over the archive."""
    inst = S.enumerate_instances(ARCHIVE_START, ARCHIVE_END)
    by_season = {s: [i for i in inst if i.name == s] for s in S.SEASONS}
    assert len(by_season[S.KHARIF]) == 4
    assert len(by_season[S.RABI]) == 5
    assert len(by_season[S.ZAID]) == 5
    assert inst == sorted(inst, key=lambda i: i.start)


def test_default_split_hits_the_season_targets():
    """The 2025-04-15 cut realises 3.00 / 3.39 / 3.49 seasons in training."""
    table = S.coverage_table(ARCHIVE_START, ARCHIVE_END, SPLIT)
    assert table["train"][S.KHARIF] == pytest.approx(3.00, abs=0.01)
    assert table["train"][S.RABI] == pytest.approx(3.39, abs=0.01)
    assert table["train"][S.ZAID] == pytest.approx(3.49, abs=0.01)
    # Every instance is fully accounted for across the two splits.
    for s in S.SEASONS:
        total = table["train"][s] + table["test"][s]
        assert total == pytest.approx(
            sum(
                S.overlap_days(i, ARCHIVE_START, ARCHIVE_END) / i.nominal_days
                for i in S.enumerate_instances(ARCHIVE_START, ARCHIVE_END)
                if i.name == s
            ),
            abs=1e-9,
        )


def test_assert_split_targets_accepts_default_and_rejects_bad_cut():
    """The guard passes on the chosen cut and fails on an obviously wrong one."""
    targets = {S.KHARIF: 3.0, S.RABI: 3.5, S.ZAID: 3.5}
    S.assert_split_targets(ARCHIVE_START, ARCHIVE_END, SPLIT, targets, tol=0.15)
    with pytest.raises(ValueError, match="misses the training season targets"):
        S.assert_split_targets(
            ARCHIVE_START, ARCHIVE_END, dt.date(2023, 1, 1), targets, tol=0.15
        )


def test_split_is_strictly_chronological():
    """Every training date precedes every test date -- no leakage by construction."""
    axis = [ARCHIVE_START + dt.timedelta(days=i) for i in range(1581)]
    tr, te = S.split_indices(axis, SPLIT)
    assert tr.size == 1200 and te.size == 381
    assert max(axis[i] for i in tr) < min(axis[i] for i in te)


def test_split_indices_rejects_unordered_dates():
    """A shuffled axis is a bug, not something to silently accept."""
    axis = [dt.date(2022, 1, 3), dt.date(2022, 1, 1), dt.date(2022, 1, 2)]
    with pytest.raises(ValueError, match="chronologically ordered"):
        S.split_indices(axis, dt.date(2022, 1, 2))


def test_purge_boundary_windows_drops_exactly_L_minus_1_origins():
    """Test origins whose memory tail reaches into training are not scored."""
    tr = np.arange(100)
    te = np.arange(100, 140)
    _, scorable = S.purge_boundary_windows(tr, te, memory_order=7)
    assert scorable.size == te.size - 6
    assert scorable[0] == 106  # first origin whose 7-frame history is all test-side
