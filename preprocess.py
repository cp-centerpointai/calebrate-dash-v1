#!/usr/bin/env python3
"""
Data preprocessing script for store distribution analysis dashboard.
Transforms raw CSV data into optimized parquet format and generates lookup tables.

Run this script once when data changes:
    python preprocess.py
"""

import pandas as pd
import json
from pathlib import Path


def parse_comma_delimited(value: str) -> list[str]:
    """
    Parse a comma-delimited string into a list of stripped strings.
    Returns empty list for null/empty values.
    """
    if pd.isna(value) or value == "":
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def main():
    # Paths
    input_csv = Path("sample.csv")
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv, dtype=str)
    print(f"Loaded {len(df):,} rows")

    # Parse brands column: convert comma-delimited string to list
    print("Parsing all_brands column...")
    df["all_brands"] = df["all_brands"].apply(parse_comma_delimited)

    # Parse products column: same treatment
    print("Parsing all_products column...")
    df["all_products"] = df["all_products"].apply(parse_comma_delimited)

    # Add brand count column
    print("Adding brand_count column...")
    df["brand_count"] = df["all_brands"].apply(len)

    # Add brands_display column for PyDeck tooltips (comma-joined string)
    print("Adding brands_display column...")
    df["brands_display"] = df["all_brands"].apply(lambda x: ", ".join(x) if x else "None")

    # Handle nulls in category fields
    print("Handling null category values...")
    category_cols = ["main_category", "subcategory", "detailed_category"]
    for col in category_cols:
        df[col] = df[col].fillna("Uncategorized")
        df[col] = df[col].replace("", "Uncategorized")

    # Convert latitude/longitude to float (they were read as string)
    print("Converting coordinate columns...")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Drop source_file column if present (not needed for dashboard)
    if "source_file" in df.columns:
        df = df.drop(columns=["source_file"])

    # Create brand lookup table: distinct brands sorted alphabetically
    print("Creating brand lookup table...")
    all_brands_set = set()
    for brands_list in df["all_brands"]:
        all_brands_set.update(brands_list)
    brands_sorted = sorted(all_brands_set, key=str.lower)
    print(f"Found {len(brands_sorted)} distinct brands")

    # Create subcategory lookup table: distinct subcategories sorted alphabetically
    print("Creating subcategory lookup table...")
    subcategories = df["subcategory"].unique().tolist()
    subcategories = [s for s in subcategories if s != "Uncategorized"]
    subcategories_sorted = sorted(subcategories, key=str.lower)
    # Add "All" as first option, then "Uncategorized" if it exists
    if "Uncategorized" in df["subcategory"].values:
        subcategories_sorted.append("Uncategorized")
    subcategories_with_all = ["All"] + subcategories_sorted
    print(f"Found {len(subcategories_sorted)} distinct subcategories")

    # Save outputs
    print("\nSaving outputs...")

    # Save parquet with pyarrow for list column support
    parquet_path = output_dir / "stores.parquet"
    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    print(f"  Saved: {parquet_path}")

    # Save brands JSON
    brands_path = output_dir / "brands.json"
    with open(brands_path, "w") as f:
        json.dump(brands_sorted, f, indent=2)
    print(f"  Saved: {brands_path}")

    # Save subcategories JSON
    subcategories_path = output_dir / "subcategories.json"
    with open(subcategories_path, "w") as f:
        json.dump(subcategories_with_all, f, indent=2)
    print(f"  Saved: {subcategories_path}")

    print("\nPreprocessing complete!")
    print(f"  Total stores: {len(df):,}")
    print(f"  Distinct brands: {len(brands_sorted)}")
    print(f"  Distinct subcategories: {len(subcategories_sorted)}")


if __name__ == "__main__":
    main()
