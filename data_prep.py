"""
================================================================================
DATA PREPARATION — FX Customer Acquisition & Foreign Tourist Targeting
================================================================================
Run once before launching the dashboard:

    python data_prep.py              # default: scope = 18e (Paris 18th only)
    python data_prep.py --scope 18e  # explicit 18th arrondissement
    python data_prep.py --scope all  # full Paris dataset
================================================================================
"""

import argparse
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from langdetect import detect, LangDetectException
from multiprocessing import Pool, cpu_count

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
LISTINGS_FILE = os.path.join(BASE_DIR, "listings.csv.gz")
REVIEWS_FILE  = os.path.join(BASE_DIR, "reviews.csv.gz")
CALENDAR_FILE = os.path.join(BASE_DIR, "calendar.csv.gz")
OUTPUT_FILE   = os.path.join(BASE_DIR, "processed_fx_target_data.csv")

OCCUPANCY_WINDOW_DAYS = 60
MIN_REVIEWS_FOR_RATIO = 3
REVIEW_CHUNK_SIZE     = 50_000
N_WORKERS             = max(1, cpu_count() - 1)

# ---------------------------------------------------------------------------
# 18th arrondissement neighbourhood names as they appear in Inside Airbnb.
# The dataset uses Paris "quartiers" (sub-districts of the 18e).
# Verify against your data: df["neighbourhood_cleansed"].unique()
# ---------------------------------------------------------------------------
SCOPE_18E_NEIGHBOURHOODS = [
    "Buttes-Montmartre",   # sometimes used for the whole arrondissement
    "Grandes-Carrières",
    "Clignancourt",
    "La Chapelle",
    "Goutte-d'Or",
    "Montmartre",
]


# ---------------------------------------------------------------------------
# HELPER — safe language detection
# ---------------------------------------------------------------------------
def _detect_lang(text: str) -> str:
    if not isinstance(text, str) or len(text.strip()) < 10:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def _detect_batch(texts: list) -> list:
    return [_detect_lang(t) for t in texts]


# ---------------------------------------------------------------------------
# 1. LISTINGS
# ---------------------------------------------------------------------------
def process_listings(scope: str) -> pd.DataFrame:
    print("[1/4] Loading listings …")
    cols = [
        "id", "latitude", "longitude",
        "neighbourhood_cleansed", "price",
        "room_type", "availability_30", "availability_60",
        "number_of_reviews", "review_scores_rating",
    ]
    df = pd.read_csv(LISTINGS_FILE, compression="gzip", usecols=cols)

    if df["price"].dtype == object:
        df["price"] = (
            df["price"]
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .astype(float)
        )

    df["occupancy_30d_listings"] = ((30 - df["availability_30"]) / 30 * 100).round(1)
    df["occupancy_60d_listings"] = ((60 - df["availability_60"]) / 60 * 100).round(1)

    print(f"   → {len(df):,} listings loaded (full Paris).")

    if scope == "18e":
        before = len(df)
        df = df[df["neighbourhood_cleansed"].isin(SCOPE_18E_NEIGHBOURHOODS)].copy()
        print(f"   [scope=18e] Kept {len(df):,} / {before:,} listings.")
        if df.empty:
            avail = pd.read_csv(
                LISTINGS_FILE, compression="gzip", usecols=["neighbourhood_cleansed"]
            )["neighbourhood_cleansed"].dropna().unique()
            print("   WARNING: 0 listings matched. Available neighbourhood names:")
            print("  ", list(avail[:30]))
            sys.exit(1)

    return df


# ---------------------------------------------------------------------------
# 2. CALENDAR → occupancy
# ---------------------------------------------------------------------------
def process_calendar(listing_ids=None) -> pd.DataFrame:
    print(f"[2/4] Loading calendar (next {OCCUPANCY_WINDOW_DAYS} days) …")
    cal = pd.read_csv(
        CALENDAR_FILE,
        compression="gzip",
        usecols=["listing_id", "date", "available"],
        parse_dates=["date"],
    )
    if listing_ids is not None:
        cal = cal[cal["listing_id"].isin(listing_ids)]

    start = cal["date"].min()
    cal   = cal[(cal["date"] >= start) & (cal["date"] < start + pd.Timedelta(days=OCCUPANCY_WINDOW_DAYS))]
    cal["is_booked"] = (cal["available"] == "f").astype(int)

    occ = (
        cal.groupby("listing_id")
        .agg(total_days=("is_booked", "count"), booked_days=("is_booked", "sum"))
        .reset_index()
    )
    occ["occupancy_rate_pct"] = (occ["booked_days"] / occ["total_days"] * 100).round(1)
    print(f"   → Occupancy computed for {len(occ):,} listings.")
    return occ[["listing_id", "occupancy_rate_pct"]]


# ---------------------------------------------------------------------------
# 3. REVIEWS → language detection + foreign tourist ratio
# ---------------------------------------------------------------------------
def process_reviews(listing_ids=None) -> pd.DataFrame:
    print("[3/4] Loading reviews & running NLP language detection …")
    print(f"      Workers: {N_WORKERS}")

    t0 = time.time()
    results = []
    total_processed = 0

    reader = pd.read_csv(
        REVIEWS_FILE,
        compression="gzip",
        usecols=["listing_id", "comments"],
        chunksize=REVIEW_CHUNK_SIZE,
    )

    for chunk_num, chunk in enumerate(reader, start=1):
        chunk = chunk.dropna(subset=["comments"])
        if listing_ids is not None:
            chunk = chunk[chunk["listing_id"].isin(listing_ids)]
        if chunk.empty:
            continue

        texts = chunk["comments"].astype(str).tolist()

        if N_WORKERS > 1 and len(texts) > 1000:
            batch_size = max(1, len(texts) // N_WORKERS)
            batches    = [texts[i: i + batch_size] for i in range(0, len(texts), batch_size)]
            with Pool(N_WORKERS) as pool:
                mapped = pool.map(_detect_batch, batches)
            langs = [lang for sub in mapped for lang in sub]
        else:
            langs = _detect_batch(texts)

        chunk = chunk.copy()
        chunk["lang"] = langs
        results.append(chunk[["listing_id", "lang"]])

        total_processed += len(chunk)
        elapsed = time.time() - t0
        speed   = total_processed / elapsed if elapsed > 0 else 0
        print(f"      Chunk {chunk_num}: {total_processed:>10,} reviews  ({speed:,.0f}/s)", end="\r")

    print()

    all_reviews = pd.concat(results, ignore_index=True)
    print(f"   → Language detected for {len(all_reviews):,} reviews in {time.time()-t0:.1f}s")

    all_reviews["is_foreign"] = (~all_reviews["lang"].isin(["fr", "unknown"])).astype(int)

    agg = (
        all_reviews.groupby("listing_id")
        .agg(total_reviews=("lang", "count"), foreign_reviews=("is_foreign", "sum"))
        .reset_index()
    )
    agg["foreign_tourist_ratio_pct"] = (agg["foreign_reviews"] / agg["total_reviews"] * 100).round(1)

    foreign_only = all_reviews[all_reviews["is_foreign"] == 1]
    if len(foreign_only) > 0:
        dominant = (
            foreign_only.groupby("listing_id")["lang"]
            .agg(lambda x: x.value_counts().index[0] if len(x) > 0 else "n/a")
            .reset_index()
            .rename(columns={"lang": "dominant_foreign_lang"})
        )
        agg = agg.merge(dominant, on="listing_id", how="left")
    else:
        agg["dominant_foreign_lang"] = "n/a"

    agg.loc[agg["total_reviews"] < MIN_REVIEWS_FOR_RATIO, "foreign_tourist_ratio_pct"] = np.nan
    print(f"   → Ratio computed for {agg['foreign_tourist_ratio_pct'].notna().sum():,} listings.")
    return agg


# ---------------------------------------------------------------------------
# 4. MERGE & EXPORT
# ---------------------------------------------------------------------------
def merge_and_export(listings, calendar_occ, reviews_agg) -> None:
    print("[4/4] Merging & exporting …")

    df = listings.merge(calendar_occ, left_on="id", right_on="listing_id", how="left")
    df["occupancy_rate_pct"] = df["occupancy_rate_pct"].fillna(df["occupancy_60d_listings"])
    df.drop(columns=["listing_id"], inplace=True, errors="ignore")

    df = df.merge(reviews_agg, left_on="id", right_on="listing_id", how="left")
    df.drop(columns=["listing_id"], inplace=True, errors="ignore")

    df["foreign_tourist_ratio_pct"] = df["foreign_tourist_ratio_pct"].fillna(0)

    df.drop(
        columns=[c for c in ["availability_30", "availability_60",
                              "occupancy_30d_listings", "occupancy_60d_listings"]
                 if c in df.columns],
        inplace=True,
    )
    df.sort_values("foreign_tourist_ratio_pct", ascending=False, inplace=True)
    df.to_csv(OUTPUT_FILE, index=False)

    size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
    print(f"\n{'='*60}")
    print(f"  OUTPUT : {OUTPUT_FILE}")
    print(f"  Rows   : {len(df):,}")
    print(f"  Size   : {size_mb:.1f} MB")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FX Target Data Preparation")
    parser.add_argument(
        "--scope",
        choices=["all", "18e"],
        default="18e",
        help="'18e' = Paris 18th arrondissement only (default)  |  'all' = full Paris",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"  FX DATA PREP — scope={args.scope}")
    print("=" * 60)
    t_start = time.time()

    for f, label in [(LISTINGS_FILE, "listings"), (REVIEWS_FILE, "reviews"), (CALENDAR_FILE, "calendar")]:
        if not os.path.isfile(f):
            print(f"ERROR: {label} file not found at {f}")
            sys.exit(1)

    listings     = process_listings(args.scope)
    listing_ids  = set(listings["id"]) if args.scope == "18e" else None
    calendar_occ = process_calendar(listing_ids)
    reviews_agg  = process_reviews(listing_ids)

    merge_and_export(listings, calendar_occ, reviews_agg)

    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")
    print("Done.  streamlit run app.py")


if __name__ == "__main__":
    main()
