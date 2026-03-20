"""
Step 1 — Build Training Data from Multiple Overture Releases
=============================================================

What this script does
---------------------
1. Scans `overture_releases/` for parquet files matching the naming convention:
       YYYY-MM-DD.N_<optional-label>.parquet
   and sorts them chronologically (oldest → newest).

2. Labels places using a HIGH-QUALITY CLOSED definition:
     "Present in 2 consecutive past releases, absent in the next."

   For releases R0, R1, R2:
     - High-quality closed (HQC): present in R0 AND R1, missing from R2
       These are confirmed churners with trajectory history.
     - Open: present in R2 (the latest release)
     - Pair 0 (R0→R1) contributes open rows only (no HQC in pair 0 because
       there is no prior release to establish a 2-window history).
     - Pair 1 (R1→R2) contributes ALL HQC closed rows plus open rows.

3. Globally resamples open rows to hit a 60/40 open/closed ratio (configurable
   via --target-open-rate).  ALL HQC closed rows are kept — only open rows are
   downsampled.

4. Stores dual snapshots per row:
     - Current snapshot  (columns from R_{i+1}, no prefix)
     - Base snapshot     (columns from R_i,   prefixed base_*)
     - Metadata: release_pair, release_index, release_date_base,
                 release_date_current

Output
------
  pipeline_output/01_training_data_raw.parquet

Usage
-----
  python pipeline/step1_build_training_data.py [options]

  Options:
    --releases-dir  DIR   Folder containing parquet release files (default: overture_releases/)
    --output-dir    DIR   Where to write outputs (default: pipeline_output/)
    --target-open-rate N  Target fraction of open labels, 0–1 (default: 0.6)
    --max-open-per-pair N Per-pair open ceiling before global rebalance (default: 300000)
    --no-downsample       Keep all open rows (skip global rebalance)
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


def _get_hq_closed_ids(path_prev: str, path_base: str, path_curr: str) -> set:
    """
    Return IDs that are present in BOTH path_prev (R_{i-1}) AND path_base (R_i)
    but absent from path_curr (R_{i+1}).

    These are the HIGH-QUALITY CLOSED places:
      "present in 2 consecutive past releases, absent in the next."
    """
    con = duckdb.connect()
    try:
        result = con.execute(f"""
            SELECT prev.id
            FROM read_parquet('{path_prev}') AS prev
            INNER JOIN read_parquet('{path_base}') AS base ON prev.id = base.id
            LEFT  JOIN read_parquet('{path_curr}') AS curr ON prev.id = curr.id
            WHERE curr.id IS NULL
        """).df()
    finally:
        con.close()
    return set(result["id"].tolist())


def _get_all_present_ids(path1: str, path2: str) -> set:
    """Return ALL IDs present in both path1 and path2."""
    con = duckdb.connect()
    try:
        result = con.execute(f"""
            SELECT a.id
            FROM read_parquet('{path1}') AS a
            INNER JOIN read_parquet('{path2}') AS b ON a.id = b.id
        """).df()
    finally:
        con.close()
    return set(result["id"].tolist())


def _build_pair(
    tag_base: str,
    path_base: str,
    tag_curr: str,
    path_curr: str,
    release_index: int,
    hq_closed_ids: set,
    max_open_per_pair: int,
    no_downsample: bool,
    must_include_open_ids: "set | None" = None,
) -> "pd.DataFrame | None":
    """
    Join two consecutive releases and produce labelled rows.

    Closed labels are restricted to `hq_closed_ids` only (high-quality closed).
    Pass an empty set to produce open-only rows (used for pair 0).

    ALL HQC closed rows are kept (no downsampling of closed).
    Open rows are capped at `max_open_per_pair` with priority given to
    `must_include_open_ids` (trajectory anchors).

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

            -- Label (pre-filtered to HQC below)
            CASE
                WHEN curr.id IS NULL                      THEN 0   -- churned = Closed
                WHEN curr.operating_status = 'closed'     THEN 0   -- explicit Closed
                ELSE 1                                              -- Open
            END AS label

        FROM base
        FULL OUTER JOIN curr ON base.id = curr.id
        WHERE base.id IS NOT NULL OR curr.id IS NOT NULL
    """

    try:
        df = con.execute(query).df()
    except Exception as exc:
        print(f"  ❌ DuckDB error for pair {pair_label}: {exc}")
        return None
    finally:
        con.close()

    n_raw_open   = int((df["label"] == 1).sum())
    n_raw_closed = int((df["label"] == 0).sum())
    print(f"  Raw rows: {len(df):,}  (Open: {n_raw_open:,} / Closed: {n_raw_closed:,})")

    # ── Filter closed to HQC only ─────────────────────────────────────────────
    # hq_closed_ids == empty set → pair has no HQC (e.g. pair 0), drop all closed
    closed_mask = df["label"] == 0
    if hq_closed_ids:
        hq_mask  = df["id"].isin(hq_closed_ids)
        # Keep closed rows only if they're in the HQC set
        df = df[~closed_mask | (closed_mask & hq_mask)].copy()
    else:
        # No HQC available for this pair — open rows only
        df = df[~closed_mask].copy()

    n_open_pool   = int((df["label"] == 1).sum())
    n_hq_closed   = int((df["label"] == 0).sum())
    print(f"  After HQC filter: {len(df):,}  (Open pool: {n_open_pool:,} / HQC closed: {n_hq_closed:,})")

    # ── Cap open per pair (global rebalance done in build_training_data) ──────
    if not no_downsample and n_open_pool > max_open_per_pair:
        open_df   = df[df["label"] == 1]
        closed_df = df[df["label"] == 0]

        if must_include_open_ids:
            # Always keep trajectory anchors (HQC open in pair 0 must be present
            # so that pre_closure_loss and releases_seen work correctly)
            must_open = open_df[open_df["id"].isin(must_include_open_ids)]
            rest_open = open_df[~open_df["id"].isin(must_include_open_ids)]
            n_fill = max(0, max_open_per_pair - len(must_open))
            if len(rest_open) > n_fill:
                rest_open = rest_open.sample(n=n_fill, random_state=42)
            open_df = pd.concat([must_open, rest_open])
            print(f"  Trajectory anchors in open set: {len(must_open):,}")
        else:
            open_df = open_df.sample(n=max_open_per_pair, random_state=42)

        df = pd.concat([open_df, closed_df]).copy()
        n_open_final = int((df["label"] == 1).sum())
        print(f"  After open cap:  {len(df):,}  (Open: {n_open_final:,} / Closed: {n_hq_closed:,})")

    # ── Metadata columns ─────────────────────────────────────────────────────
    df["release_pair"]         = pair_label
    df["release_index"]        = release_index
    df["release_date_base"]    = tag_base
    df["release_date_current"] = tag_curr

    return df


# ─── main ────────────────────────────────────────────────────────────────────

def build_training_data(
    releases_dir:      str   = "overture_releases",
    output_dir:        str   = "pipeline_output",
    target_open_rate:  float = 0.6,
    max_open_per_pair: int   = 300_000,
    no_downsample:     bool  = False,
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
        )
        sys.exit(1)

    print(f"\n  Found {len(releases)} releases (sorted chronologically):")
    for i, (tag, path) in enumerate(releases):
        size_mb = os.path.getsize(path) / 1_048_576
        print(f"    [{i}] {tag}  ({size_mb:.1f} MB)  {os.path.basename(path)}")

    n_pairs = len(releases) - 1
    print(f"\n  Will build {n_pairs} comparison pair(s).\n")

    # ── Pre-compute HQC IDs per pair ─────────────────────────────────────────
    # A pair has HQC only when there is a PRIOR release (pair index >= 1).
    # HQC for pair i = present in R_{i-1} AND R_i, absent from R_{i+1}.
    hq_closed_ids_per_pair: list[set] = []
    for i in range(n_pairs):
        if i == 0:
            # Pair 0 has no prior release → no HQC
            hq_closed_ids_per_pair.append(set())
            print(f"  Pair 0: no prior release → 0 HQC closed (open-only pair)")
        else:
            _, path_prev = releases[i - 1]   # R_{i-1}
            _, path_base = releases[i]        # R_i
            _, path_curr = releases[i + 1]    # R_{i+1}
            hq = _get_hq_closed_ids(path_prev, path_base, path_curr)
            hq_closed_ids_per_pair.append(hq)
            rate = len(hq) / max(1, 1_657_959)  # approximate denom
            print(f"  Pair {i}: {len(hq):,} HQC closed  (present in 2 past releases, gone in next)")

    # ── Compute trajectory anchors ────────────────────────────────────────────
    # For pair i, we force HQC places from pair (i+1) into pair i's OPEN set
    # so that pre_closure_loss and releases_seen capture their pre-closure signal.
    # We also anchor an equal-size random sample of stable-open places so that
    # releases_seen=2 is not a pure proxy for label=0.
    anchor_open_ids_per_pair: list[set] = []
    for i in range(n_pairs):
        if i + 1 < n_pairs:
            hq_next = hq_closed_ids_per_pair[i + 1]  # future HQC (to be closed in pair i+1)
            if hq_next:
                # Sample an equal number of stable-open IDs from pair i+1's open pool
                _, path_base_next = releases[i + 1]
                _, path_curr_next = releases[i + 2]
                stable_open = _get_all_present_ids(path_base_next, path_curr_next) - hq_next
                n_balance = min(len(hq_next), len(stable_open))
                import random; rng = random.Random(42)
                balanced_open_sample = set(rng.sample(sorted(stable_open), n_balance))
                anchor_open_ids_per_pair.append(hq_next | balanced_open_sample)
                print(
                    f"  Pair {i} anchors: {len(hq_next):,} future-HQC + "
                    f"{n_balance:,} stable-open = {len(hq_next)+n_balance:,}"
                )
            else:
                anchor_open_ids_per_pair.append(set())
        else:
            anchor_open_ids_per_pair.append(None)  # last pair — no next anchors

    # ── Build each consecutive pair ──────────────────────────────────────────
    all_dfs: list = []

    for i in range(n_pairs):
        tag_base, path_base = releases[i]
        tag_curr, path_curr = releases[i + 1]

        must_include = anchor_open_ids_per_pair[i]   # open anchors for this pair

        df_pair = _build_pair(
            tag_base=tag_base,
            path_base=path_base,
            tag_curr=tag_curr,
            path_curr=path_curr,
            release_index=i,
            hq_closed_ids=hq_closed_ids_per_pair[i],
            max_open_per_pair=max_open_per_pair,
            no_downsample=no_downsample,
            must_include_open_ids=must_include,
        )

        if df_pair is not None:
            all_dfs.append(df_pair)

    if not all_dfs:
        print("\n❌  No pairs could be built. Exiting.")
        sys.exit(1)

    # ── Combine ──────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Combining all pairs …")
    combined = pd.concat(all_dfs, ignore_index=True)

    # Keep each (place_id, release_pair) exactly once
    combined = combined.drop_duplicates(
        subset=["id", "release_pair"], keep="first"
    ).reset_index(drop=True)

    n_closed_total = int((combined["label"] == 0).sum())
    n_open_total   = int((combined["label"] == 1).sum())
    print(f"  After dedup: {len(combined):,}  (Open: {n_open_total:,} / Closed: {n_closed_total:,})")

    # ── Global 60/40 rebalancing ──────────────────────────────────────────────
    # ALL HQC closed are kept.  Open rows are downsampled to hit target_open_rate.
    if not no_downsample and n_closed_total > 0:
        n_open_target = int(round(n_closed_total * target_open_rate / (1.0 - target_open_rate)))
        print(
            f"\n  Rebalancing to {target_open_rate:.0%} open / {1-target_open_rate:.0%} closed …"
            f"\n    HQC closed kept:  {n_closed_total:,}"
            f"\n    Open target:      {n_open_target:,}  (from {n_open_total:,} available)"
        )
        if n_open_total > n_open_target:
            open_df   = combined[combined["label"] == 1].sample(n=n_open_target, random_state=42)
            closed_df = combined[combined["label"] == 0]
            combined  = pd.concat([open_df, closed_df]).reset_index(drop=True)
        elif n_open_total < n_open_target:
            print(f"  ⚠️  Only {n_open_total:,} open rows available; target was {n_open_target:,}.")
            actual_rate = n_open_total / (n_open_total + n_closed_total)
            print(f"      Actual open rate will be {actual_rate:.1%}.")

    # ── Save ─────────────────────────────────────────────────────────────────
    n_open_final   = int((combined["label"] == 1).sum())
    n_closed_final = int((combined["label"] == 0).sum())
    actual_open_pct = n_open_final / max(1, len(combined))
    print(
        f"\n  Final dataset: {len(combined):,} rows  "
        f"(Open: {n_open_final:,} {actual_open_pct:.1%} / "
        f"Closed: {n_closed_final:,} {1-actual_open_pct:.1%})"
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "01_training_data_raw.parquet")
    combined.to_parquet(output_path, index=False)
    print(f"\n  ✅  Saved → {output_path}")

    return output_path


# ─── CLI entry-point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Step 1: Build raw labelled training data from N Overture release parquets.\n"
            "Uses high-quality closed labels (present in 2 past releases, gone in next)\n"
            "and rebalances open/closed to a configurable ratio (default: 60/40)."
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
        "--target-open-rate",
        type=float,
        default=0.6,
        help="Target fraction of open labels after global rebalance (default: 0.6 = 60/40)",
    )
    parser.add_argument(
        "--max-open-per-pair",
        type=int,
        default=300_000,
        help="Per-pair open ceiling before global rebalance (default: 300000)",
    )
    parser.add_argument(
        "--no-downsample",
        action="store_true",
        help="Keep all open rows (skip global 60/40 rebalance)",
    )
    args = parser.parse_args()

    build_training_data(
        releases_dir=args.releases_dir,
        output_dir=args.output_dir,
        target_open_rate=args.target_open_rate,
        max_open_per_pair=args.max_open_per_pair,
        no_downsample=args.no_downsample,
    )
