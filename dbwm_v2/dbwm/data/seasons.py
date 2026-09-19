"""
Cropping-season calendar for the Lahore AOI, and the train/test protocol.

Three agricultural seasons partition the calendar year:

===========  ==================================  ============
Season       Span                                Nominal days
===========  ==================================  ============
Kharif       1 Jun  --  30 Sep  (same year)      122
Rabi         1 Oct  --  end Feb (**next** year)  151 / 152
Zaid         1 Mar  --  31 May  (same year)      92
===========  ==================================  ============

(122 + 151 + 92 = 365, so the three spans tile the year exactly.)

Season *instances*
------------------
A season **instance** is one occurrence of a season, keyed by ``(name, start_year)``
where ``start_year`` is the calendar year the instance *begins* in. Rabi straddles a
year boundary, so 15 Jan 2022 belongs to instance ``Rabi 2021`` (Oct 2021 - Feb 2022),
not ``Rabi 2022``. Getting this wrong is the classic off-by-one in agronomic
calendars: it would silently merge two different crop cycles into one label.

Fractional coverage
-------------------
The archive (2022-01-01 .. 2026-04-30) truncates the first and last instances, so a
split does not contain whole numbers of seasons. :func:`coverage_table` therefore
reports **fractional** counts -- for each instance, the number of its days falling in
the split divided by its *nominal* length. This is what makes the thesis target
("~3 Kharif / 3.5 Rabi / 3.5 Zaid in training") a checkable quantity rather than a
hand-wave, and :func:`assert_split_targets` checks it.

Why a single chronological cut
------------------------------
The memory lift (v3 Def. 2.3) makes each training example depend on the six
preceding frames, so any split whose blocks interleave in time lets a training
window straddle a held-out block. A single cut keeps the test set strictly in the
future and leaves only ``L-1`` boundary-adjacent windows to drop
(:func:`purge_boundary_windows`).
"""
from __future__ import annotations

import calendar
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

KHARIF = "kharif"
RABI = "rabi"
ZAID = "zaid"

SEASONS: Tuple[str, str, str] = (KHARIF, RABI, ZAID)

#: Human-readable aliases used in plots and report headings.
SEASON_LABELS: Dict[str, str] = {
    KHARIF: "Summer (Kharif)",
    RABI: "Winter (Rabi)",
    ZAID: "Spring (Zaid)",
}

#: (start_month, start_day) -> the month a season begins on.
_KHARIF_MONTHS = (6, 7, 8, 9)
_ZAID_MONTHS = (3, 4, 5)
# Rabi is everything else: Oct, Nov, Dec, Jan, Feb.


def season_of(d: dt.date) -> str:
    """
    Return the season name a date falls in.

    :param d: a calendar date.
    :return: one of :data:`KHARIF`, :data:`RABI`, :data:`ZAID`.
    """
    if d.month in _KHARIF_MONTHS:
        return KHARIF
    if d.month in _ZAID_MONTHS:
        return ZAID
    return RABI


def instance_start_year(d: dt.date) -> int:
    """
    Return the year the season instance containing ``d`` *begins* in.

    Only Rabi differs from ``d.year``: January and February belong to the Rabi
    instance that started the previous October.

    :param d: a calendar date.
    :return: the instance's start year.
    """
    if season_of(d) == RABI and d.month <= 2:
        return d.year - 1
    return d.year


@dataclass(frozen=True)
class SeasonInstance:
    """
    One occurrence of a cropping season.

    :ivar name: season name (:data:`KHARIF` / :data:`RABI` / :data:`ZAID`).
    :ivar start_year: calendar year the instance begins in.
    """

    name: str
    start_year: int

    @property
    def start(self) -> dt.date:
        """First calendar day of the instance (ignoring archive truncation)."""
        if self.name == KHARIF:
            return dt.date(self.start_year, 6, 1)
        if self.name == ZAID:
            return dt.date(self.start_year, 3, 1)
        return dt.date(self.start_year, 10, 1)

    @property
    def end(self) -> dt.date:
        """Last calendar day of the instance (inclusive)."""
        if self.name == KHARIF:
            return dt.date(self.start_year, 9, 30)
        if self.name == ZAID:
            return dt.date(self.start_year, 5, 31)
        end_year = self.start_year + 1
        return dt.date(end_year, 2, calendar.monthrange(end_year, 2)[1])

    @property
    def nominal_days(self) -> int:
        """Full length of the instance in days (152 for a leap-February Rabi)."""
        return (self.end - self.start).days + 1

    @property
    def label(self) -> str:
        """Display label, e.g. ``'Winter (Rabi) 2023/24'``."""
        if self.name == RABI:
            return "{} {}/{:02d}".format(
                SEASON_LABELS[self.name], self.start_year, (self.start_year + 1) % 100
            )
        return "{} {}".format(SEASON_LABELS[self.name], self.start_year)

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.label


def instance_of(d: dt.date) -> SeasonInstance:
    """
    Return the season instance a date belongs to.

    :param d: a calendar date.
    :return: the containing :class:`SeasonInstance`.
    """
    return SeasonInstance(season_of(d), instance_start_year(d))


def enumerate_instances(start: dt.date, end: dt.date) -> List[SeasonInstance]:
    """
    List every season instance overlapping ``[start, end]``, chronologically.

    Instances truncated by the archive boundaries are included -- their partial
    nature is expressed through :func:`coverage_table`, not by omission.

    :param start: first date of the archive.
    :param end: last date of the archive (inclusive).
    :return: chronologically ordered instances.
    """
    if end < start:
        raise ValueError("end {} precedes start {}".format(end, start))
    seen, out = set(), []
    d = start
    while d <= end:
        inst = instance_of(d)
        if inst not in seen:
            seen.add(inst)
            out.append(inst)
        d += dt.timedelta(days=1)
    out.sort(key=lambda i: i.start)
    return out


def overlap_days(inst: SeasonInstance, start: dt.date, end: dt.date) -> int:
    """
    Number of days of ``inst`` falling inside ``[start, end]``.

    :param inst: the season instance.
    :param start: window start (inclusive).
    :param end: window end (inclusive).
    :return: overlap in days (0 if disjoint).
    """
    lo, hi = max(inst.start, start), min(inst.end, end)
    return max(0, (hi - lo).days + 1)


def coverage_table(
    start: dt.date, end: dt.date, cut: dt.date
) -> Dict[str, Dict[str, float]]:
    """
    Fractional season counts on each side of a chronological cut.

    For every instance overlapping the archive, its days in the split are divided
    by its *nominal* length and summed per season. A value of ``3.4`` means "three
    full Rabi seasons plus 40% of a fourth".

    :param start: first archive date.
    :param end: last archive date (inclusive).
    :param cut: first date of the **test** split (train is ``[start, cut)``).
    :return: ``{"train": {season: fraction}, "test": {season: fraction}}``.
    """
    if not start <= cut <= end + dt.timedelta(days=1):
        raise ValueError("cut {} lies outside [{}, {}]".format(cut, start, end))
    train_end = cut - dt.timedelta(days=1)
    out = {
        "train": {s: 0.0 for s in SEASONS},
        "test": {s: 0.0 for s in SEASONS},
    }
    for inst in enumerate_instances(start, end):
        nominal = float(inst.nominal_days)
        out["train"][inst.name] += overlap_days(inst, start, train_end) / nominal
        out["test"][inst.name] += overlap_days(inst, cut, end) / nominal
    return out


def assert_split_targets(
    start: dt.date,
    end: dt.date,
    cut: dt.date,
    targets: Dict[str, float],
    tol: float = 0.15,
) -> Dict[str, Dict[str, float]]:
    """
    Verify that a cut realises the intended fractional season counts in training.

    :param start: first archive date.
    :param end: last archive date (inclusive).
    :param cut: first date of the test split.
    :param targets: intended training fractions, e.g. ``{KHARIF: 3.0, RABI: 3.5,
                    ZAID: 3.5}``.
    :param tol: allowed absolute deviation per season, in units of seasons.
    :return: the realised :func:`coverage_table`.
    :raises ValueError: if any season deviates from its target by more than ``tol``.
    """
    table = coverage_table(start, end, cut)
    bad = {
        s: (table["train"][s], t)
        for s, t in targets.items()
        if abs(table["train"][s] - t) > tol
    }
    if bad:
        raise ValueError(
            "Cut {} misses the training season targets (tol={}): {}".format(
                cut, tol, {s: "got {:.2f}, want {:.2f}".format(g, w) for s, (g, w) in bad.items()}
            )
        )
    return table


def split_indices(
    dates: Sequence[dt.date], cut: dt.date
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a chronologically ordered date list at ``cut``.

    :param dates: chronologically ordered dates (the model's time axis).
    :param cut: first date of the **test** split.
    :return: ``(train_idx, test_idx)`` integer index arrays.
    """
    arr = np.asarray([d < cut for d in dates], dtype=bool)
    if not np.all(arr[:-1] >= arr[1:]):
        raise ValueError("dates are not chronologically ordered")
    idx = np.arange(len(dates))
    return idx[arr], idx[~arr]


def purge_boundary_windows(
    train_idx: np.ndarray, test_idx: np.ndarray, memory_order: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Drop the test windows whose memory tail reaches back into training.

    With the memory lift, predicting from origin ``t`` consumes
    ``w_t, ..., w_{t-L+1}``. The first ``L-1`` test origins therefore see training
    frames in their history. Those origins are dropped from the *evaluation* set so
    every scored forecast is conditioned only on test-side observations; the frames
    themselves stay in the filter so it can warm up across the boundary.

    :param train_idx: training indices.
    :param test_idx: test indices.
    :param memory_order: ``L``.
    :return: ``(train_idx, scorable_test_idx)``.
    """
    if memory_order < 1:
        raise ValueError("memory_order must be >= 1")
    return train_idx, test_idx[memory_order - 1 :]


def season_labels(dates: Iterable[dt.date]) -> np.ndarray:
    """
    Season name for each date, as an array of strings.

    :param dates: dates.
    :return: ``(n,)`` array of season names.
    """
    return np.asarray([season_of(d) for d in dates], dtype=object)


def instance_labels(dates: Iterable[dt.date]) -> np.ndarray:
    """
    Season *instance* for each date (crop-cycle identity, not just the season).

    :param dates: dates.
    :return: ``(n,)`` array of :class:`SeasonInstance`.
    """
    return np.asarray([instance_of(d) for d in dates], dtype=object)


def season_masks(dates: Sequence[dt.date]) -> Dict[str, np.ndarray]:
    """
    Boolean mask per season over a date axis.

    :param dates: dates.
    :return: ``{season: (n,) bool array}``.
    """
    labels = season_labels(dates)
    return {s: (labels == s) for s in SEASONS}


def format_coverage(table: Dict[str, Dict[str, float]]) -> str:
    """
    Render a :func:`coverage_table` as a readable block for logs / reports.

    :param table: output of :func:`coverage_table`.
    :return: multi-line string.
    """
    lines = ["{:<18} {:>8} {:>8}".format("season", "train", "test")]
    for s in SEASONS:
        lines.append(
            "{:<18} {:>8.2f} {:>8.2f}".format(
                SEASON_LABELS[s], table["train"][s], table["test"][s]
            )
        )
    lines.append(
        "{:<18} {:>8.2f} {:>8.2f}".format(
            "total", sum(table["train"].values()), sum(table["test"].values())
        )
    )
    return "\n".join(lines)
