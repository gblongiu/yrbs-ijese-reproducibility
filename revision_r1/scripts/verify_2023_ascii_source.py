from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_ROWS = 20103
EXPECTED_LINE_WIDTH = 421
EXPECTED_SHA256 = "0a83cba11dc8a0cdf0043aacf0b80478f86cf93a91a8ecb81a276518e486241a"

FIELDS = {
    "q1": (17, 17, "int"),
    "q2": (18, 18, "int"),
    "q3": (19, 19, "int"),
    "QN24": (201, 201, "int"),
    "QN25": (202, 202, "int"),
    "QN26": (203, 203, "int"),
    "weight": (388, 397, "float"),
    "stratum": (398, 400, "int"),
    "psu": (401, 406, "int"),
    "raceeth": (414, 415, "int"),
}


def field(line: str, start: int, end: int) -> str:
    return line[start - 1 : end].strip()


def number(value: str, kind: str) -> float:
    if value == "":
        return np.nan
    return float(value) if kind == "float" else int(value)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_ascii(path: Path) -> tuple[pd.DataFrame, dict[int, int]]:
    rows = []
    widths: dict[int, int] = {}
    with path.open("r", encoding="latin-1", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\r\n")
            widths[len(line)] = widths.get(len(line), 0) + 1
            row = {}
            for name, (start, end, kind) in FIELDS.items():
                row[name] = number(field(line, start, end), kind)
            rows.append(row)
    return pd.DataFrame(rows), widths


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the reconstructed 2023 analysis fields against the national YRBS ASCII source.")
    parser.add_argument("ascii_file", type=Path)
    parser.add_argument("--workbook", type=Path, default=Path("data/raw/YRBS_2023_MH_subset.xlsx"))
    parser.add_argument("--out", type=Path, default=Path("revision_r1/outputs/tables/yrbs2023_ascii_source_equivalence.csv"))
    args = parser.parse_args()

    observed_hash = sha256(args.ascii_file)
    parsed, widths = parse_ascii(args.ascii_file)
    retained = pd.read_excel(args.workbook, usecols=list(FIELDS))

    if len(parsed) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS:,} records, found {len(parsed):,}.")
    if widths != {EXPECTED_LINE_WIDTH: EXPECTED_ROWS}:
        raise RuntimeError(f"Unexpected fixed-width layout: {widths}")
    if observed_hash != EXPECTED_SHA256:
        raise RuntimeError(f"Unexpected SHA-256: {observed_hash}")
    if len(retained) != EXPECTED_ROWS:
        raise RuntimeError(f"Retained workbook has {len(retained):,} rows, expected {EXPECTED_ROWS:,}.")

    records = []
    for name in FIELDS:
        a = parsed[name]
        b = retained[name]
        if name == "weight":
            match = np.isclose(a, b, rtol=0.0, atol=0.0, equal_nan=True)
        else:
            match = (a.isna() & b.isna()) | (a == b)
            match = match.fillna(False).to_numpy()
        records.append(
            {
                "field": name,
                "ascii_start": FIELDS[name][0],
                "ascii_end": FIELDS[name][1],
                "records_compared": len(match),
                "records_matching": int(np.sum(match)),
                "records_mismatching": int(len(match) - np.sum(match)),
            }
        )

    result = pd.DataFrame(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)

    if result["records_mismatching"].sum() != 0:
        raise RuntimeError("One or more retained fields differ from the CDC ASCII source.")

    print(f"SHA-256: {observed_hash}")
    print(f"Rows: {len(parsed):,}")
    print(f"Fixed width: {EXPECTED_LINE_WIDTH}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
