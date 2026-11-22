"""Aggregate the two Azure LLM request traces into one-minute workload data."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from project_utils import PROJECT_ROOT, resolve_from_root, write_json


EXPECTED_RAW_COLUMNS = ["TIMESTAMP", "ContextTokens", "GeneratedTokens"]
DEFAULT_CODE_FILE = PROJECT_ROOT / "data" / "raw" / "AzureLLMInferenceTrace_code_1week.csv"
DEFAULT_CONVERSATION_FILE = (
    PROJECT_ROOT / "data" / "raw" / "AzureLLMInferenceTrace_conv_1week.csv"
)
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "azure_llm_one_minute.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-file", default=str(DEFAULT_CODE_FILE))
    parser.add_argument("--conversation-file", default=str(DEFAULT_CONVERSATION_FILE))
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Number of raw request rows read at once (default: 1,000,000).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing processed dataset.",
    )
    return parser.parse_args()


def validate_raw_header(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Raw trace not found: {path}")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if header != EXPECTED_RAW_COLUMNS:
        raise ValueError(
            f"Unexpected columns in {path.name}. "
            f"Expected {EXPECTED_RAW_COLUMNS}, found {header}."
        )


def aggregate_trace(
    path: Path,
    service: str,
    chunk_size: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Stream one raw event file and aggregate counts and token sums by minute."""
    validate_raw_header(path)
    aggregate: pd.DataFrame | None = None
    raw_rows = 0
    first_timestamp: pd.Timestamp | None = None
    last_timestamp: pd.Timestamp | None = None

    reader = pd.read_csv(
        path,
        usecols=EXPECTED_RAW_COLUMNS,
        dtype={"TIMESTAMP": "string", "ContextTokens": "int64", "GeneratedTokens": "int64"},
        chunksize=chunk_size,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        # Azure mixes timestamps with and without fractional seconds.
        timestamps = pd.to_datetime(
            chunk["TIMESTAMP"], format="mixed", utc=True, errors="coerce"
        )
        if timestamps.isna().any():
            bad_count = int(timestamps.isna().sum())
            raise ValueError(f"{path.name} contains {bad_count} invalid timestamps.")
        if (chunk[["ContextTokens", "GeneratedTokens"]] < 0).any().any():
            raise ValueError(f"{path.name} contains negative token counts.")

        raw_rows += len(chunk)
        chunk_min = timestamps.min()
        chunk_max = timestamps.max()
        first_timestamp = chunk_min if first_timestamp is None else min(first_timestamp, chunk_min)
        last_timestamp = chunk_max if last_timestamp is None else max(last_timestamp, chunk_max)

        working = pd.DataFrame(
            {
                "minute": timestamps.dt.floor("min"),
                "context_tokens": chunk["ContextTokens"].to_numpy(),
                "generated_tokens": chunk["GeneratedTokens"].to_numpy(),
            }
        )
        grouped = working.groupby("minute", sort=True).agg(
            request_count=("context_tokens", "size"),
            context_tokens=("context_tokens", "sum"),
            generated_tokens=("generated_tokens", "sum"),
        )

        if aggregate is None:
            aggregate = grouped
        else:
            aggregate = pd.concat([aggregate, grouped]).groupby(level=0, sort=True).sum()

        if chunk_number == 1 or chunk_number % 5 == 0:
            print(f"  {service}: processed {raw_rows:,} requests")

    if aggregate is None or aggregate.empty:
        raise ValueError(f"No rows were read from {path}.")

    aggregate = aggregate.sort_index().astype("int64")
    aggregate.columns = [f"{service}_{column}" for column in aggregate.columns]
    metadata = {
        "file": str(path),
        "file_size_bytes": path.stat().st_size,
        "raw_request_rows": raw_rows,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "one_minute_rows_before_intersection": len(aggregate),
    }
    return aggregate, metadata


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    code_file = resolve_from_root(args.code_file)
    conversation_file = resolve_from_root(args.conversation_file)
    output_file = resolve_from_root(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists() and not args.force:
        raise FileExistsError(
            f"Processed dataset already exists: {output_file}\n"
            "Use --force to rebuild it from the raw traces."
        )

    print("Aggregating code-completion requests...")
    code, code_metadata = aggregate_trace(code_file, "code", args.chunk_size)
    print("Aggregating conversational requests...")
    conversation, conversation_metadata = aggregate_trace(
        conversation_file, "conversation", args.chunk_size
    )

    common_start = max(code.index.min(), conversation.index.min())
    common_end = min(code.index.max(), conversation.index.max())
    if common_start > common_end:
        raise ValueError("The code and conversation traces do not overlap in time.")

    common_minutes = pd.date_range(common_start, common_end, freq="min", tz="UTC")
    code = code.reindex(common_minutes, fill_value=0)
    conversation = conversation.reindex(common_minutes, fill_value=0)

    combined = pd.concat([code, conversation], axis=1)
    combined.index.name = "minute"
    combined["total_request_count"] = (
        combined["code_request_count"] + combined["conversation_request_count"]
    )
    combined["total_context_tokens"] = (
        combined["code_context_tokens"] + combined["conversation_context_tokens"]
    )
    combined["total_generated_tokens"] = (
        combined["code_generated_tokens"] + combined["conversation_generated_tokens"]
    )
    combined["total_tokens"] = (
        combined["total_context_tokens"] + combined["total_generated_tokens"]
    )

    ordered_columns = [
        "code_request_count",
        "conversation_request_count",
        "code_context_tokens",
        "conversation_context_tokens",
        "code_generated_tokens",
        "conversation_generated_tokens",
        "total_request_count",
        "total_context_tokens",
        "total_generated_tokens",
        "total_tokens",
    ]
    combined = combined[ordered_columns].astype("int64")
    if combined.isna().any().any():
        raise ValueError("Unexpected missing values remain after one-minute aggregation.")

    combined.reset_index().to_csv(output_file, index=False)
    metadata = {
        "description": "Azure LLM code and conversation traces aggregated to one-minute UTC intervals",
        "code_trace": code_metadata,
        "conversation_trace": conversation_metadata,
        "common_window_start": common_minutes[0],
        "common_window_end": common_minutes[-1],
        "sampling_frequency": "1 minute",
        "processed_rows": len(combined),
        "processed_numerical_columns": len(ordered_columns),
        "zero_request_minutes": {
            "code": int((combined["code_request_count"] == 0).sum()),
            "conversation": int((combined["conversation_request_count"] == 0).sum()),
            "combined": int((combined["total_request_count"] == 0).sum()),
        },
        "output_file": str(output_file),
        "output_sha256": sha256_file(output_file),
    }
    metadata_path = output_file.with_name(f"{output_file.stem}_metadata.json")
    write_json(metadata_path, metadata)

    print("\nDataset build complete")
    print(f"  Common interval: {common_minutes[0]} through {common_minutes[-1]}")
    print(f"  One-minute observations: {len(combined):,}")
    print(f"  Output: {output_file}")
    print(f"  Metadata: {metadata_path}")


if __name__ == "__main__":
    main()

