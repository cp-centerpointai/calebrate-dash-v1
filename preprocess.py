#!/usr/bin/env python3
"""
Data preprocessing script for Coors Edge distribution analysis dashboard.
Transforms raw CSV data into optimized parquet format with store categorization.

This dashboard analyzes Coors Edge distribution vs Athletic Brewing competition
within the Coors distribution network.

Run this script once when data changes:
    python preprocess.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Fixed brand configuration for this dashboard
FOCUS_BRAND = "Coors Edge"
COMPETITOR_BRAND = "Athletic Brewing"
# Brand name as it appears in the data
COMPETITOR_BRAND_DATA_NAME = "Athletic Brew"
# Brands to exclude from the dataset
EXCLUDE_BRANDS = ["High Noon"]


def parse_comma_delimited(value: str) -> list[str]:
    """
    Parse a comma-delimited string into a list of stripped strings.
    Returns empty list for null/empty values.
    """
    if pd.isna(value) or value == "":
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def assign_quintiles(df: pd.DataFrame, col: str, bracket_col: str, format_func) -> pd.DataFrame:
    """
    Assign quintile labels for any numeric column.
    Each quintile has ~20% of stores with valid data.
    """
    valid_data = df[col].dropna()
    if len(valid_data) == 0:
        df[bracket_col] = "Unknown"
        return df

    quintiles = np.percentile(valid_data, [20, 40, 60, 80])

    def get_label(value):
        if pd.isna(value):
            return "Unknown"
        if value < quintiles[0]:
            return f"Q1 ({format_func(None, quintiles[0], 'lt')})"
        elif value < quintiles[1]:
            return f"Q2 ({format_func(quintiles[0], quintiles[1], 'range')})"
        elif value < quintiles[2]:
            return f"Q3 ({format_func(quintiles[1], quintiles[2], 'range')})"
        elif value < quintiles[3]:
            return f"Q4 ({format_func(quintiles[2], quintiles[3], 'range')})"
        else:
            return f"Q5 ({format_func(quintiles[3], None, 'gt')})"

    df[bracket_col] = df[col].apply(get_label)
    return df, quintiles


def format_income(low, high, mode):
    """Format income values for bracket labels."""
    if mode == 'lt':
        return f"< ${high/1000:.0f}K"
    elif mode == 'gt':
        return f"> ${low/1000:.0f}K"
    else:
        return f"${low/1000:.0f}K-${high/1000:.0f}K"


def format_percent(low, high, mode):
    """Format percentage values for bracket labels."""
    if mode == 'lt':
        return f"< {high:.0f}%"
    elif mode == 'gt':
        return f"> {low:.0f}%"
    else:
        return f"{low:.0f}%-{high:.0f}%"


def main():
    # Paths
    input_csv = Path("sample_with_census_ids.csv")
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

    # Remove excluded brands (High Noon) from brand lists
    print(f"Removing excluded brands: {EXCLUDE_BRANDS}...")
    def filter_brands(brands_list):
        return [b for b in brands_list if b not in EXCLUDE_BRANDS]
    df["all_brands"] = df["all_brands"].apply(filter_brands)

    # Add brand count column (after filtering)
    print("Adding brand_count column...")
    df["brand_count"] = df["all_brands"].apply(len)

    # Add brands_display column for tooltips (comma-joined string)
    print("Adding brands_display column...")
    df["brands_display"] = df["all_brands"].apply(lambda x: ", ".join(x) if x else "None")

    # Add store categorization flags for Coors Edge vs Athletic Brewing analysis
    print("Adding store categorization flags...")
    df["has_coors_edge"] = df["all_brands"].apply(lambda x: FOCUS_BRAND in x)
    df["has_athletic"] = df["all_brands"].apply(lambda x: COMPETITOR_BRAND_DATA_NAME in x)

    # Create store category for easy filtering/coloring
    # Categories: "coors_edge_only", "athletic_only", "both", "neither"
    def categorize_store(row):
        has_focus = row["has_coors_edge"]
        has_competitor = row["has_athletic"]
        if has_focus and has_competitor:
            return "both"
        elif has_focus:
            return "coors_edge_only"
        elif has_competitor:
            return "athletic_only"
        else:
            return "neither"

    df["store_category"] = df.apply(categorize_store, axis=1)

    # Print category distribution
    category_counts = df["store_category"].value_counts()
    print(f"Store categories:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count:,} ({count/len(df)*100:.1f}%)")

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

    # Join census demographic data for stratified analytics
    print("Joining census demographic data...")
    census_path = Path("data/census_tracts.parquet")
    if census_path.exists():
        # Load all demographic columns (skip geometry for performance)
        census_df = pd.read_parquet(
            census_path,
            columns=["GEOID", "median_hh_income", "pct_pop_21_34", "pct_college_educated"]
        )

        # Format tract_geoid to 11-char string for join
        df["tract_geoid"] = df["tract_geoid"].apply(
            lambda x: str(int(float(x))).zfill(11) if pd.notna(x) and x != "" else None
        )

        # Merge on tract_geoid = GEOID
        df = df.merge(census_df, left_on="tract_geoid", right_on="GEOID", how="left")
        df = df.drop(columns=["GEOID"], errors="ignore")

        matched = df["median_hh_income"].notna().sum()
        print(f"  Matched {matched:,} stores ({matched/len(df)*100:.1f}%) with census data")

        # Assign quintiles for all three demographics
        print("Assigning demographic quintiles...")

        # Income brackets
        df, income_q = assign_quintiles(df, "median_hh_income", "income_bracket", format_income)
        print(f"  Income thresholds: {[f'${q/1000:.0f}K' for q in income_q]}")

        # Age (21-34) brackets
        df, age_q = assign_quintiles(df, "pct_pop_21_34", "age_bracket", format_percent)
        print(f"  Age 21-34 thresholds: {[f'{q:.1f}%' for q in age_q]}")

        # Education brackets
        df, edu_q = assign_quintiles(df, "pct_college_educated", "education_bracket", format_percent)
        print(f"  College educated thresholds: {[f'{q:.1f}%' for q in edu_q]}")

    else:
        print("Warning: Census data not found. Skipping demographic bracket assignment.")
        df["tract_geoid"] = None
        df["median_hh_income"] = None
        df["pct_pop_21_34"] = None
        df["pct_college_educated"] = None
        df["income_bracket"] = "Unknown"
        df["age_bracket"] = "Unknown"
        df["education_bracket"] = "Unknown"

    # Drop columns not needed for dashboard
    drop_cols = ["source_file", "state_fips", "county_fips", "block_geoid", "block_group", "block", "Unnamed: 0"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Create category hierarchy lookup tables
    print("Creating category hierarchy lookup tables...")

    # Main categories sorted alphabetically (with "All" at start, "Uncategorized" at end if present)
    main_categories = df["main_category"].unique().tolist()
    main_categories = [c for c in main_categories if c != "Uncategorized"]
    main_categories_sorted = sorted(main_categories, key=str.lower)
    if "Uncategorized" in df["main_category"].values:
        main_categories_sorted.append("Uncategorized")
    main_categories_with_all = ["All"] + main_categories_sorted
    print(f"Found {len(main_categories_sorted)} distinct main categories")

    # Map: main_category -> list of subcategories
    main_to_sub = df.groupby("main_category")["subcategory"].unique().apply(list).to_dict()
    # Sort each list alphabetically, move "Uncategorized" to end
    for key in main_to_sub:
        subs = main_to_sub[key]
        subs = [s for s in subs if s != "Uncategorized"]
        subs = sorted(subs, key=str.lower)
        if "Uncategorized" in df[df["main_category"] == key]["subcategory"].values:
            subs.append("Uncategorized")
        main_to_sub[key] = ["All"] + subs
    print(f"Created main_category -> subcategory mapping")

    # Map: subcategory -> list of detailed_categories
    sub_to_detailed = df.groupby("subcategory")["detailed_category"].unique().apply(list).to_dict()
    # Sort each list alphabetically, move "Uncategorized" to end
    for key in sub_to_detailed:
        details = sub_to_detailed[key]
        details = [d for d in details if d != "Uncategorized"]
        details = sorted(details, key=str.lower)
        if "Uncategorized" in df[df["subcategory"] == key]["detailed_category"].values:
            details.append("Uncategorized")
        sub_to_detailed[key] = ["All"] + details
    print(f"Created subcategory -> detailed_category mapping")

    # Legacy: subcategory lookup (for backward compatibility)
    subcategories = df["subcategory"].unique().tolist()
    subcategories = [s for s in subcategories if s != "Uncategorized"]
    subcategories_sorted = sorted(subcategories, key=str.lower)
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

    # Save main_categories JSON
    main_categories_path = output_dir / "main_categories.json"
    with open(main_categories_path, "w") as f:
        json.dump(main_categories_with_all, f, indent=2)
    print(f"  Saved: {main_categories_path}")

    # Save main_to_subcategories mapping JSON
    main_to_sub_path = output_dir / "main_to_subcategories.json"
    with open(main_to_sub_path, "w") as f:
        json.dump(main_to_sub, f, indent=2)
    print(f"  Saved: {main_to_sub_path}")

    # Save subcategory_to_detailed mapping JSON
    sub_to_detailed_path = output_dir / "subcategory_to_detailed.json"
    with open(sub_to_detailed_path, "w") as f:
        json.dump(sub_to_detailed, f, indent=2)
    print(f"  Saved: {sub_to_detailed_path}")

    # Save subcategories JSON (legacy, for backward compatibility)
    subcategories_path = output_dir / "subcategories.json"
    with open(subcategories_path, "w") as f:
        json.dump(subcategories_with_all, f, indent=2)
    print(f"  Saved: {subcategories_path}")

    print("\nPreprocessing complete!")
    print(f"  Total stores: {len(df):,}")
    print(f"  Focus brand: {FOCUS_BRAND}")
    print(f"  Competitor brand: {COMPETITOR_BRAND} (data name: {COMPETITOR_BRAND_DATA_NAME})")
    print(f"  Excluded brands: {EXCLUDE_BRANDS}")
    print(f"  Distinct subcategories: {len(subcategories_sorted)}")


if __name__ == "__main__":
    main()
