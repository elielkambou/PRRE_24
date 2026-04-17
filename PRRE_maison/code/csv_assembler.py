"""Utilities for loading and assembling multiple CBOE option CSV files."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd

FRENCH_MONTHS = {
    "janvier": 1,
    "fevrier": 2,
    "f\u00e9vrier": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "aout": 8,
    "ao\u00fbt": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "decembre": 12,
    "d\u00e9cembre": 12,
}

CBOE_COLUMNS = [
    "expiration",
    "call_symbol",
    "call_last",
    "call_net",
    "call_bid",
    "call_ask",
    "call_volume",
    "call_iv",
    "call_delta",
    "call_gamma",
    "call_oi",
    "strike",
    "put_symbol",
    "put_last",
    "put_net",
    "put_bid",
    "put_ask",
    "put_volume",
    "put_iv",
    "put_delta",
    "put_gamma",
    "put_oi",
]

NUMERIC_COLUMNS = [
    column
    for column in CBOE_COLUMNS
    if column not in {"expiration", "call_symbol", "put_symbol"}
]

CONTRACT_ID_COLUMNS = ["expiration", "strike", "call_symbol", "put_symbol"]


@dataclass(frozen=True)
class CsvMetadata:
    source_file: Path
    quote_date: pd.Timestamp
    spot: float
    spot_last: float
    spot_bid: float | None
    spot_ask: float | None


@dataclass
class AssembledOptionData:
    quote_date: pd.Timestamp
    spot: float
    option_chain: pd.DataFrame
    file_metadata: pd.DataFrame
    input_files: list[Path]
    duplicates_removed: int


def discover_csv_files(directory: str | Path, pattern: str = "*.csv") -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory.")

    csv_files = sorted(path for path in directory.glob(pattern) if path.is_file())
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files matching pattern {pattern!r} were found in {directory}."
        )
    return csv_files


def _read_csv_lines(csv_path: str | Path) -> list[str]:
    return Path(csv_path).read_text(encoding="utf-8").splitlines()


def _extract_metadata_lines(lines: list[str]) -> tuple[str, str]:
    non_empty_lines = [line for line in lines if line.strip()]
    if len(non_empty_lines) < 3:
        raise ValueError("The CSV header does not contain enough metadata lines.")
    return non_empty_lines[0], non_empty_lines[1]


def _find_table_header_row(lines: list[str]) -> int:
    header_prefix = "Expiration Date,Calls,Last Sale"
    for index, line in enumerate(lines):
        if line.startswith(header_prefix):
            return index
    raise ValueError("Could not find the option table header row in the CSV.")


def parse_cboe_metadata(csv_path: str | Path) -> CsvMetadata:
    csv_path = Path(csv_path)
    lines = _read_csv_lines(csv_path)
    market_line, quote_line = _extract_metadata_lines(lines)

    last_match = re.search(r"Last:\s*([0-9.]+)", market_line)
    bid_match = re.search(r"Bid:\s*([0-9.]+)", quote_line)
    ask_match = re.search(r"Ask:\s*([0-9.]+)", quote_line)
    if last_match is None:
        raise ValueError(f"Could not parse spot Last from {csv_path}.")

    date_match = re.search(
        r"Date:\s*(\d{1,2})\s+([A-Za-z\u00e9\u00fb\u00f4\u00ee\u00e0\u00f9]+)\s+(\d{4})",
        quote_line,
    )
    if date_match is None:
        raise ValueError(f"Could not parse quote date from {csv_path}.")

    day = int(date_match.group(1))
    month_name = date_match.group(2).lower()
    year = int(date_match.group(3))
    if month_name not in FRENCH_MONTHS:
        raise ValueError(f"Unknown French month name {month_name!r} in {csv_path}.")

    quote_date = pd.Timestamp(
        year=year,
        month=FRENCH_MONTHS[month_name],
        day=day,
    )

    spot_last = float(last_match.group(1))
    spot_bid = float(bid_match.group(1)) if bid_match else None
    spot_ask = float(ask_match.group(1)) if ask_match else None
    spot = 0.5 * (spot_bid + spot_ask) if spot_bid is not None and spot_ask is not None else spot_last

    return CsvMetadata(
        source_file=csv_path,
        quote_date=quote_date,
        spot=spot,
        spot_last=spot_last,
        spot_bid=spot_bid,
        spot_ask=spot_ask,
    )


def _add_quote_quality_columns(option_chain: pd.DataFrame) -> pd.DataFrame:
    option_chain = option_chain.copy()

    option_chain["call_mid"] = 0.5 * (option_chain["call_bid"] + option_chain["call_ask"])
    option_chain["put_mid"] = 0.5 * (option_chain["put_bid"] + option_chain["put_ask"])
    option_chain["call_bid_ask_spread"] = (
        option_chain["call_ask"] - option_chain["call_bid"]
    ).clip(lower=0.0)
    option_chain["put_bid_ask_spread"] = (
        option_chain["put_ask"] - option_chain["put_bid"]
    ).clip(lower=0.0)
    option_chain["quote_completeness"] = option_chain[
        ["call_bid", "call_ask", "put_bid", "put_ask", "call_iv", "put_iv"]
    ].notna().sum(axis=1)
    option_chain["total_volume"] = option_chain[["call_volume", "put_volume"]].fillna(0.0).sum(axis=1)
    option_chain["total_open_interest"] = option_chain[["call_oi", "put_oi"]].fillna(0.0).sum(axis=1)
    option_chain["mean_bid_ask_spread"] = option_chain[
        ["call_bid_ask_spread", "put_bid_ask_spread"]
    ].mean(axis=1, skipna=True)
    option_chain["mean_bid_ask_spread"] = option_chain["mean_bid_ask_spread"].fillna(np.inf)

    return option_chain


def load_single_option_chain(csv_path: str | Path) -> tuple[pd.DataFrame, CsvMetadata]:
    csv_path = Path(csv_path)
    csv_lines = _read_csv_lines(csv_path)
    table_header_row = _find_table_header_row(csv_lines)
    metadata = parse_cboe_metadata(csv_path)

    option_chain = pd.read_csv(csv_path, skiprows=table_header_row + 1, names=CBOE_COLUMNS)
    option_chain = option_chain.dropna(how="all").copy()

    for column in NUMERIC_COLUMNS:
        option_chain[column] = pd.to_numeric(option_chain[column], errors="coerce")

    option_chain["expiration"] = pd.to_datetime(
        option_chain["expiration"],
        format="%a %b %d %Y",
        errors="raise",
    )
    option_chain["quote_date"] = metadata.quote_date
    option_chain["ttm_days"] = (option_chain["expiration"] - metadata.quote_date).dt.days.astype(float)
    option_chain["ttm_years"] = option_chain["ttm_days"] / 365.0
    option_chain["source_file"] = str(csv_path)
    option_chain["source_spot"] = metadata.spot

    option_chain = _add_quote_quality_columns(option_chain)
    option_chain = option_chain.sort_values(["expiration", "strike", "call_symbol", "put_symbol"]).reset_index(drop=True)

    return option_chain, metadata


def _metadata_to_frame(
    metadata_list: Sequence[CsvMetadata],
    option_chains: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for metadata, option_chain in zip(metadata_list, option_chains):
        row = asdict(metadata)
        row["source_file"] = str(metadata.source_file)
        row["row_count"] = int(len(option_chain))
        row["min_expiration"] = option_chain["expiration"].min()
        row["max_expiration"] = option_chain["expiration"].max()
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["quote_date", "source_file"]).reset_index(drop=True)


def deduplicate_option_chain(option_chain: pd.DataFrame) -> pd.DataFrame:
    if option_chain.empty:
        return option_chain.copy()

    deduped_chain = option_chain.sort_values(
        [
            "expiration",
            "strike",
            "quote_completeness",
            "total_open_interest",
            "total_volume",
            "mean_bid_ask_spread",
            "source_file",
        ],
        ascending=[True, True, False, False, False, True, True],
    )
    deduped_chain = deduped_chain.drop_duplicates(subset=CONTRACT_ID_COLUMNS, keep="first")
    deduped_chain = deduped_chain.sort_values(["expiration", "strike", "call_symbol", "put_symbol"]).reset_index(drop=True)
    return deduped_chain


def assemble_option_chains(
    csv_paths: Sequence[str | Path],
    require_same_quote_date: bool = True,
) -> AssembledOptionData:
    if not csv_paths:
        raise ValueError("csv_paths is empty. Provide at least one CSV file.")

    input_files = [Path(csv_path) for csv_path in csv_paths]
    single_chains: list[pd.DataFrame] = []
    metadata_list: list[CsvMetadata] = []

    for csv_path in input_files:
        option_chain, metadata = load_single_option_chain(csv_path)
        single_chains.append(option_chain)
        metadata_list.append(metadata)

    quote_dates = sorted({metadata.quote_date for metadata in metadata_list})
    if require_same_quote_date and len(quote_dates) != 1:
        details = ", ".join(
            f"{metadata.source_file.name}: {metadata.quote_date.date()}"
            for metadata in metadata_list
        )
        raise ValueError(
            "All CSV files must share the same quote date. "
            f"Received: {details}"
        )

    assembled_quote_date = quote_dates[0]
    assembled_spot = float(np.median([metadata.spot for metadata in metadata_list]))
    raw_option_chain = pd.concat(single_chains, ignore_index=True)
    deduped_option_chain = deduplicate_option_chain(raw_option_chain)
    file_metadata = _metadata_to_frame(metadata_list, single_chains)
    duplicates_removed = int(len(raw_option_chain) - len(deduped_option_chain))

    return AssembledOptionData(
        quote_date=assembled_quote_date,
        spot=assembled_spot,
        option_chain=deduped_option_chain,
        file_metadata=file_metadata,
        input_files=input_files,
        duplicates_removed=duplicates_removed,
    )


def load_option_chains_from_directory(
    directory: str | Path,
    pattern: str = "*.csv",
    require_same_quote_date: bool = True,
) -> AssembledOptionData:
    csv_files = discover_csv_files(directory=directory, pattern=pattern)
    return assemble_option_chains(
        csv_paths=csv_files,
        require_same_quote_date=require_same_quote_date,
    )

