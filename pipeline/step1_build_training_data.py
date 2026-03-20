"""
Step 1 — Build Training Data from Multiple Overture Releases
=============================================================

What this script does
---------------------
1. Scans `overture_releases/` for parquet files matching the naming convention:
       YYYY-MM-DD.N_<optional-label>.parquet
   and sorts them chronologically (oldest → newest).

2. For every consecutive pair of releases (R_i → R_{i+1}), performs a FULL OUTER
   JOIN on the stable Overture place `id` field:

     - Place in R_i, MISSING in R_{i+1}        → label 0 (Closed / churned)
     - Place in R_{i+1} with operating_status='closed' → label 0 (Closed / explicit)
     - Place in R_{i+1}, operating_status not 'closed' → label 1 (Open)

3. For each labelled row, stores:
     - The "base" snapshot   (columns from R_i,     prefixed base_*)
     - The "current" snapshot (columns from R_{i+1}, no prefix)
     - Metadata: release_pair, release_index, release_date_base, release_date_current

4. Concatenates all pairs into one raw training dataframe and deduplicates so that
   the SAME place appearing across many consecutive windows only contributes once
   per unique window (place_id + release_pair).

Output
------
  pipeline_output/01_training_data_raw.parquet

Usage
-----
  python pipeline/step1_build_training_data.py [options]

  Options:
    --releases-dir  DIR   Folder containing parquet release files (default: overture_releases/)
    --output-dir    DIR   Where to write outputs (default: pipeline_output/)
    --max-open      N     Max open samples per release pair (default: 9000)
    --max-closed    N     Max closed samples per release pair (default: 3000)
    --no-downsample       Do not downsample; use all rows (useful for large datasets)
"""

import argparse
import json
import os
import re
import sys

import duckdb
import pandas as pd


# ─── helpers ────────────────────────────────────────────────────────────────

RELEASE_PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2}\.\d+)"  # date + revision  e.g. 2026-01-21.0
)


def _discover_releases(releases_dir: str) -> list:
    """
    Return sorted list of (release_tag, abs_filepath) tuples.
    Files must match YYYY-MM-DD.N<anything>.parquet
    """
    entries = []
    for fname in os.listdir(releases_dir):
        if not fname.endswith(".parquet"):
            continue
        m = RELEASE_PATTERN.match(fname)
        if not m:
            print(
                f"  ⚠️  Skipping '{fname}' — does not match YYYY-MM-DD.N prefix convention"
            )
            continue
        tag = m.group(1)  # e.g. "2026-01-21.0"
        entries.append((tag, os.path.join(releases_dir, fname)))

    entries.sort(key=lambda x: x[0])  # lexicographic == chronological for ISO dates
    return entries


def _to_json_str(x):
    """Normalise a column value to a JSON string (or None)."""
    if x is None:
        return None
    if isinstance(x, float) and pd.isna(x):
        return None
    if isinstance(x, (dict, list)):
        return json.dumps(x)
    return str(x)


def _get_churned_ids(path_base: str, path_curr: str) -> set:
    """Return IDs present in base but absent from curr (will be label=0 for this pair)."""
    con = duckdb.connect()
    try:
        result = con.execute(f"""
            SELECT base.id
            FROM read_parquet('{path_base}') AS base
            LEFT JOIN read_parquet('{path_curr}') AS curr ON base.id = curr.id
            WHERE curr.id IS NULL
        """).df()
    finally:
        con.close()
    return set(result["id"].tolist())


def _get_open_sample_ids(path_base: str, path_curr: str, n: int, seed: int = 42) -> set:
    """Return a random sample of IDs that are OPEN (present in both base and curr)."""
    con = duckdb.connect()
    try:
        result = con.execute(f"""
            SELECT curr.id
            FROM read_parquet('{path_curr}') AS curr
            INNER JOIN read_parquet('{path_base}') AS base ON base.id = curr.id
            USING SAMPLE {n} ROWS (bernoulli, {seed})
        """).df()
    except Exception:
        # Fallback: no USING SAMPLE, sample in pandas
        result = con.execute(f"""
            SELECT curr.id
            FROM read_parquet('{path_curr}') AS curr
            INNER JOIN read_parquet('{path_base}') AS base ON base.id = curr.id
        """).df()
        result = result.sample(min(n, len(result)), random_state=seed)
    finally:
        con.close()
    return set(result["id"].tolist())


def _build_pair(
    tag_base: str,
    path_base: str,
    tag_curr: str,
    path_curr: str,
    release_index: int,
    max_open: int,
    max_closed: int,
    no_downsample: bool,
    must_include_ids: "set | None" = None,
) -> "pd.DataFrame | None":
    """
    Join two consecutive releases and produce labelled rows.

    Columns in output
    -----------------
    id, names, categories, websites, socials, phones, addresses, brand,
    sources, emails, confidence,
    base_id, base_names, base_categories, base_websites, base_socials,
    base_phones, base_addresses, base_brand, base_sources, base_emails,
    base_confidence,
    label,                    # 0=Closed / 1=Open
    release_pair,             # "2026-01-21.0→2026-02-18.0"
    release_index,            # 0-based index of this pair in the sorted list
    release_date_base,        # "2026-01-21.0"
    release_date_current,     # "2026-02-18.0"
    """
    pair_label = f"{tag_base}→{tag_curr}"
    print(f"\n  {'─'*60}")
    print(f"  Pair [{release_index}]: {pair_label}")
    print(f"  {'─'*60}")

    con = duckdb.connect()

    # Note: DuckDB handles nested JSON as objects; we cast to VARCHAR so pandas
    # receives them as plain JSON strings (consistent with process_data_v5.py).
    query = f"""
        WITH base AS (
            SELECT
                id,
                CAST(names      AS VARCHAR) AS names,
                CAST(categories AS VARCHAR) AS categories,
                CAST(websites   AS VARCHAR) AS websites,
                CAST(socials    AS VARCHAR) AS socials,
                CAST(phones     AS VARCHAR) AS phones,
                CAST(emails     AS VARCHAR) AS emails,
                CAST(addresses  AS VARCHAR) AS addresses,
                CAST(brand      AS VARCHAR) AS brand,
                CAST(sources    AS VARCHAR) AS sources,
                CAST(confidence AS DOUBLE)  AS confidence,
                try_cast(operating_status AS VARCHAR) AS operating_status,
                CAST(_city      AS VARCHAR) AS city
            FROM read_parquet('{path_base}')
        ),
        curr AS (
            SELECT
                id,
                CAST(names      AS VARCHAR) AS names,
                CAST(categories AS VARCHAR) AS categories,
                CAST(websites   AS VARCHAR) AS websites,
                CAST(socials    AS VARCHAR) AS socials,
                CAST(phones     AS VARCHAR) AS phones,
                CAST(emails     AS VARCHAR) AS emails,
                CAST(addresses  AS VARCHAR) AS addresses,
                CAST(brand      AS VARCHAR) AS brand,
                CAST(sources    AS VARCHAR) AS sources,
                CAST(confidence AS DOUBLE)  AS confidence,
                try_cast(operating_status AS VARCHAR) AS operating_status,
                CAST(_city      AS VARCHAR) AS city
            FROM read_parquet('{path_curr}')
        )
        SELECT
            -- Current "best available" snapshot (COALESCE keeps row for churned places)
            COALESCE(curr.id,         base.id)         AS id,
            COALESCE(curr.names,      base.names)       AS names,
            COALESCE(curr.categories, base.categories)  AS categories,
            COALESCE(curr.websites,   base.websites)    AS websites,
            COALESCE(curr.socials,    base.socials)     AS socials,
            COALESCE(curr.phones,     base.phones)      AS phones,
            COALESCE(curr.emails,     base.emails)      AS emails,
            COALESCE(curr.addresses,  base.addresses)   AS addresses,
            COALESCE(curr.brand,      base.brand)       AS brand,
            COALESCE(curr.sources,    base.sources)     AS sources,
            curr.confidence                             AS confidence,
            COALESCE(curr.city,       base.city)        AS city,

            -- Previous / baseline snapshot
            base.id         AS base_id,
            base.names      AS base_names,
            base.categories AS base_categories,
            base.websites   AS base_websites,
            base.socials    AS base_socials,
            base.phones     AS base_phones,
            base.emails     AS base_emails,
            base.addresses  AS base_addresses,
            base.brand      AS base_brand,
            base.sources    AS base_sources,
            base.confidence AS base_confidence,

            -- Label
            CASE
                WHEN curr.id IS NULL                      THEN 0   -- churned = Closed
                WHEN curr.operating_status = 'closed'     THEN 0   -- explicit Closed
                ELSE 1                                              -- Open
            END AS label

        FROM base
        FULL OUTER JOIN curr ON base.id = curr.id
        -- Keep rows where at least one side existed (excludes true nulls)
        WHERE base.id IS NOT NULL OR curr.id IS NOT NULL
    """

    try:
        df = con.execute(query).df()
    except Exception as exc:
        print(f"  ❌ DuckDB error for pair {pair_label}: {exc}")
        return None
    finally:
        con.close()

    n_open   = int((df["label"] == 1).sum())
    n_closed = int((df["label"] == 0).sum())
    print(f"  Raw rows: {len(df):,}  (Open: {n_open:,} / Closed: {n_closed:,})")

    # ── Optional downsampling ────────────────────────────────────────────────
    if not no_downsample:
        open_df   = df[df["label"] == 1]
        closed_df = df[df["label"] == 0]

        if must_include_ids and len(open_df) > max_open:
            # Always keep future churners so trajectory features can look back
            must_open  = open_df[open_df["id"].isin(must_include_ids)]
            rest_open  = open_df[~open_df["id"].isin(must_include_ids)]
            n_fill = max(0, max_open - len(must_open))
            if len(rest_open) > n_fill:
                rest_open = rest_open.sample(n=n_fill, random_state=42)
            open_df = pd.concat([must_open, rest_open])
            print(f"  Future-churner anchors kept in open set: {len(must_open):,}")
        elif len(open_df) > max_open:
            open_df = open_df.sample(n=max_open, random_state=42)

        if len(closed_df) > max_closed:
            closed_df = closed_df.sample(n=max_closed, random_state=42)

        df = pd.concat([open_df, closed_df]).copy()
        print(
            f"  Sampled: {len(df):,}  "
            f"(Open: {len(open_df):,} / Closed: {len(closed_df):,})"
        )

    # ── Metadata columns ─────────────────────────────────────────────────────
    df["release_pair"]         = pair_label
    df["release_index"]        = release_index
    df["release_date_base"]    = tag_base
    df["release_date_current"] = tag_curr

    return df


# ─── main ────────────────────────────────────────────────────────────────────

def build_training_data(
    releases_dir: str = "overture_releases",
    output_dir:   str = "pipeline_output",
    max_open:     int = 9000,
    max_closed:   int = 3000,
    no_downsample: bool = False,
) -> str:
    """
    Build the raw labelled training dataset from all consecutive release pairs.

    Returns the path to the output parquet file.
    """
    print("\n" + "=" * 70)
    print("STEP 1 — BUILD TRAINING DATA FROM OVERTURE RELEASES")
    print("=" * 70)

    # ── Discover releases ────────────────────────────────────────────────────
    releases = _discover_releases(releases_dir)

    if len(releases) < 2:
        print(
            f"\n❌  Found {len(releases)} valid release file(s) in '{releases_dir}/'.\n"
            f"    At least 2 are required to form one comparison pair.\n"
            f"    See overture_releases/README.md for the naming convention."
        )
        sys.exit(1)

    print(f"\n  Found {len(releases)} releases (sorted chronologically):")
    for i, (tag, path) in enumerate(releases):
        size_mb = os.path.getsize(path) / 1_048_576
        print(f"    [{i}] {tag}  ({size_mb:.1f} MB)  {os.path.basename(path)}")

    # ── Build each consecutive pair ──────────────────────────────────────────
    n_pairs = len(releases) - 1
    print(f"\n  Will build {n_pairs} comparison pair(s):\n")

    # Pre-compute IDs to anchor per pair:
    # - churned IDs from pair i+1  → must appear as OPEN in pair i (so pre_closure_loss works)
    # - an equal-sized open sample from pair i+1 → also anchored in pair i, so that
    #   releases_seen=2 is NOT a pure proxy for label=0 (leak prevention)
    anchor_ids_per_pair: list = []
    for i in range(n_pairs):
        _, path_base_i = releases[i]
        _, path_curr_i = releases[i + 1]
        churned = _get_churned_ids(path_base_i, path_curr_i)
        open_sample = _get_open_sample_ids(path_base_i, path_curr_i, n=max_open)
        anchor_ids_per_pair.append(churned | open_sample)
        print(f"  Pair {i} anchors: {len(churned):,} churners + {len(open_sample):,} open = {len(churned | open_sample):,}")

    all_dfs: list = []

    for i in range(n_pairs):
        tag_base, path_base = releases[i]
        tag_curr, path_curr = releases[i + 1]

        # Anchor: force-include future churners AND a matching open sample from next pair
        must_include = anchor_ids_per_pair[i + 1] if i + 1 < n_pairs else None

        df_pair = _build_pair(
            tag_base=tag_base,
            path_base=path_base,
            tag_curr=tag_curr,
            path_curr=path_curr,
            release_index=i,
            max_open=max_open,
            max_closed=max_closed,
            no_downsample=no_downsample,
            must_include_ids=must_include,
        )

        if df_pair is not None:
            all_dfs.append(df_pair)

    if not all_dfs:
        print("\n❌  No pairs could be built. Exiting.")
        sys.exit(1)

    # ── Combine & deduplicate ────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Combining all pairs …")
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total rows (before dedup): {len(combined):,}")

    # Keep each (place_id, release_pair) combination exactly once.
    deduped = combined.drop_duplicates(
        subset=["id", "release_pair"], keep="first"
    ).reset_index(drop=True)
    print(f"  Total rows (after dedup):  {len(deduped):,}")
    print(
        f"  Open: {(deduped['label']==1).sum():,} | "
        f"Closed: {(deduped['label']==0).sum():,} | "
        f"Open rate: {(deduped['label']==1).mean():.1%}"
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "01_training_data_raw.parquet")
    deduped.to_parquet(output_path, index=False)
    print(f"\n  ✅  Saved → {output_path}")

    return output_path


# ─── CLI entry-point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Step 1: Build raw labelled training data from N Overture release parquets.\n"
            "Place your release files in overture_releases/ then run this script."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--releases-dir",
        default="overture_releases",
        help="Folder containing Overture parquet release files (default: overture_releases/)",
    )
    parser.add_argument(
        "--output-dir",
        default="pipeline_output",
        help="Directory for pipeline outputs (default: pipeline_output/)",
    )
    parser.add_argument(
        "--max-open",
        type=int,
        default=9000,
        help="Max open-labelled samples per release pair (default: 9000)",
    )
    parser.add_argument(
        "--max-closed",
        type=int,
        default=3000,
        help="Max closed-labelled samples per release pair (default: 3000)",
    )
    parser.add_argument(
        "--no-downsample",
        action="store_true",
        help="Disable downsampling — use all rows (may produce large datasets)",
    )
    args = parser.parse_args()

    build_training_data(
        releases_dir=args.releases_dir,
        output_dir=args.output_dir,
        max_open=args.max_open,
        max_closed=args.max_closed,
        no_downsample=args.no_downsample,
    )
