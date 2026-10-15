"""Reconstruct acquisition identities for the released Study L NV observations."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qml_microrobot.study_l_nv import _inventory_member, author_features


# Directory and filename fields jointly bind each raw observation to one session
# and to its motor-stage coordinates while preserving archive order.
RAW_PATTERN = re.compile(
    r"Fig\.S4/Data/(?P<session>20\d\d-\d\d-\d\d \d+chip\d+ \d+mW)/"
    r".*?x=(?P<x>[-\d.]+)um, y=(?P<y>[-\d.]+)um raw data\.npy$"
)


@dataclass(frozen=True)
class NvSessionData:
    """Observation-level features, micrometre targets and acquisition metadata."""

    features: np.ndarray
    positions_um: np.ndarray
    session_ids: np.ndarray
    acquisition_dates: np.ndarray
    raw_members: np.ndarray
    coordinate_ids: np.ndarray
    has_prior_session_repeat: np.ndarray


def load_l1_session_data(root: Path) -> NvSessionData:
    """Load 819 released rows in archive order with session and repeat metadata.

    The returned arrays contain ``[819, 24]`` log features, ``[819, 2]``
    positions in micrometres, directory-defined session identifiers and a flag
    indicating whether the same coordinate appeared in an earlier session.
    """

    parameters = np.load(_inventory_member(root, "para_list.npy"), allow_pickle=False)
    parameter_std = np.load(
        _inventory_member(root, "para_std_list.npy"), allow_pickle=False
    )
    x = np.load(_inventory_member(root, "x_pos.npy"), allow_pickle=False).astype(float)
    y = np.load(_inventory_member(root, "y_pos.npy"), allow_pickle=False).astype(float)
    members = []
    inventory = root / "results" / "study_l1_archive_member_structure.csv"
    with inventory.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            match = RAW_PATTERN.search(row["member"])
            if match:
                members.append(
                    (
                        match.group("session"),
                        float(match.group("x")),
                        float(match.group("y")),
                        row["member"],
                    )
                )
    if len(members) != len(parameters) or len(members) != 819:
        raise ValueError("L1 session member count drift")
    member_positions = np.asarray([(row[1], row[2]) for row in members])
    positions = np.column_stack([x, y])
    if not np.array_equal(member_positions, positions):
        raise ValueError("derived L1 rows do not match archive member order")
    sessions = np.asarray([row[0] for row in members])
    dates = np.asarray([row[0].split()[0] for row in members])
    coordinate_ids = np.asarray(
        [f"x{int(px)}_y{int(py)}" for px, py in positions]
    )
    prior = np.zeros(len(members), dtype=bool)
    seen: dict[str, set[str]] = {}
    for index, (coordinate, session) in enumerate(zip(coordinate_ids, sessions)):
        prior[index] = bool(seen.get(coordinate, set()) - {session})
        seen.setdefault(coordinate, set()).add(session)
    return NvSessionData(
        features=author_features(parameters, parameter_std),
        positions_um=positions,
        session_ids=sessions,
        acquisition_dates=dates,
        raw_members=np.asarray([row[3] for row in members]),
        coordinate_ids=coordinate_ids,
        has_prior_session_repeat=prior,
    )
