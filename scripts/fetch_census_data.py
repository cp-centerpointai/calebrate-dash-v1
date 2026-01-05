#!/usr/bin/env python3
"""
Census Data Prep Script

Downloads Census tract geometries and ACS 5-year estimates, then joins them
into a single GeoJSON file for use in the Streamlit dashboard.

Required packages (install with pip):
    pip install geopandas requests python-dotenv pyarrow

Usage:
    python scripts/fetch_census_data.py

Output:
    data/census_tracts.parquet (GeoParquet format for smaller file size)
"""

import os
import sys
import ssl
import tempfile
from pathlib import Path
import requests
import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv

# Fix SSL certificate issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables from .env file
load_dotenv()

# Census API key
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")
if not CENSUS_API_KEY:
    print("Error: CENSUS_API_KEY not found in environment variables.")
    print("Please add CENSUS_API_KEY=your_key to your .env file")
    sys.exit(1)

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

# State FIPS codes (all 50 states + DC)
STATE_FIPS = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
    "06": "California", "08": "Colorado", "09": "Connecticut", "10": "Delaware",
    "11": "District of Columbia", "12": "Florida", "13": "Georgia", "15": "Hawaii",
    "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa",
    "20": "Kansas", "21": "Kentucky", "22": "Louisiana", "23": "Maine",
    "24": "Maryland", "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
    "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska",
    "32": "Nevada", "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico",
    "36": "New York", "37": "North Carolina", "38": "North Dakota", "39": "Ohio",
    "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania", "44": "Rhode Island",
    "45": "South Carolina", "46": "South Dakota", "47": "Tennessee", "48": "Texas",
    "49": "Utah", "50": "Vermont", "51": "Virginia", "53": "Washington",
    "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming"
}

# State FIPS to abbreviation mapping
STATE_ABBREV = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY"
}


def download_tract_geometries() -> gpd.GeoDataFrame:
    """
    Download Census tract geometries from Census TIGER/Cartographic Boundary Files.
    Uses the national file for all tracts.
    """
    print("Downloading Census tract geometries...")

    # Use 2022 Cartographic Boundary Files (500k resolution for balance of detail/size)
    # National file containing all tracts
    url = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_tract_500k.zip"

    print(f"  Fetching from: {url}")
    gdf = gpd.read_file(url)

    # Filter to just 50 states + DC (exclude territories)
    gdf = gdf[gdf["STATEFP"].isin(STATE_FIPS.keys())]

    # Create GEOID column if not present (should be STATEFP + COUNTYFP + TRACTCE)
    if "GEOID" not in gdf.columns:
        gdf["GEOID"] = gdf["STATEFP"] + gdf["COUNTYFP"] + gdf["TRACTCE"]

    # Add state name and abbreviation
    gdf["state_fips"] = gdf["STATEFP"]
    gdf["state_name"] = gdf["STATEFP"].map(STATE_FIPS)
    gdf["state_abbrev"] = gdf["STATEFP"].map(STATE_ABBREV)

    print(f"  Downloaded {len(gdf):,} tracts across {gdf['state_fips'].nunique()} states")

    return gdf


def fetch_acs_data_for_state(state_fips: str) -> pd.DataFrame:
    """
    Fetch ACS 5-year estimates for a single state.

    Variables:
    - B19013_001E: Median Household Income
    - B01001_*: Age cohorts for population 21-34
    - B15003_*: Educational attainment for bachelor's degree or higher
    """

    # Age variables for 21-34 (male and female)
    # B01001_010E: Male 22-24, B01001_011E: Male 25-29, B01001_012E: Male 30-34
    # B01001_034E: Female 22-24, B01001_035E: Female 25-29, B01001_036E: Female 30-34
    # Note: 21 is included in 20-21 cohort which we'll use as approximation
    # B01001_009E: Male 20-21, B01001_033E: Female 20-21
    age_vars = [
        "B01001_009E", "B01001_010E", "B01001_011E", "B01001_012E",  # Male 20-34
        "B01001_033E", "B01001_034E", "B01001_035E", "B01001_036E",  # Female 20-34
        "B01001_001E"  # Total population
    ]

    # Education variables - Bachelor's degree or higher (ages 25+)
    # B15003_022E: Bachelor's, B15003_023E: Master's, B15003_024E: Professional, B15003_025E: Doctorate
    # B15003_001E: Total population 25+
    edu_vars = [
        "B15003_001E",  # Total pop 25+
        "B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E"  # Bachelor's+
    ]

    # Income variable
    income_vars = ["B19013_001E"]

    all_vars = income_vars + age_vars + edu_vars
    vars_str = ",".join(all_vars)

    # ACS 5-year 2022 estimates
    url = (
        f"https://api.census.gov/data/2022/acs/acs5"
        f"?get=NAME,{vars_str}"
        f"&for=tract:*"
        f"&in=state:{state_fips}"
        f"&key={CENSUS_API_KEY}"
    )

    response = requests.get(url)

    if response.status_code != 200:
        print(f"  Warning: Failed to fetch ACS data for state {state_fips}: {response.status_code}")
        return pd.DataFrame()

    data = response.json()

    # First row is headers
    headers = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns=headers)

    # Create GEOID (state + county + tract)
    df["GEOID"] = df["state"] + df["county"] + df["tract"]

    # Convert numeric columns
    for col in all_vars:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Calculate derived metrics
    # Population ages 21-34 (using 20-34 as approximation)
    df["pop_21_34"] = (
        df["B01001_009E"] + df["B01001_010E"] + df["B01001_011E"] + df["B01001_012E"] +
        df["B01001_033E"] + df["B01001_034E"] + df["B01001_035E"] + df["B01001_036E"]
    )
    df["total_pop"] = df["B01001_001E"]
    df["pct_pop_21_34"] = (df["pop_21_34"] / df["total_pop"] * 100).round(1)

    # % Bachelor's degree or higher
    df["pop_bachelors_plus"] = (
        df["B15003_022E"] + df["B15003_023E"] + df["B15003_024E"] + df["B15003_025E"]
    )
    df["pop_25_plus"] = df["B15003_001E"]
    df["pct_college_educated"] = (df["pop_bachelors_plus"] / df["pop_25_plus"] * 100).round(1)

    # Median household income (negative values are Census codes for missing data)
    df["median_hh_income"] = df["B19013_001E"]
    df.loc[df["median_hh_income"] < 0, "median_hh_income"] = pd.NA

    # Select only needed columns
    result = df[["GEOID", "median_hh_income", "pct_pop_21_34", "pct_college_educated"]].copy()

    return result


def fetch_all_acs_data() -> pd.DataFrame:
    """Fetch ACS data for all states."""
    print("Fetching ACS 5-year estimates...")

    all_data = []
    states = list(STATE_FIPS.keys())

    for i, state_fips in enumerate(states):
        state_name = STATE_FIPS[state_fips]
        print(f"  [{i+1}/{len(states)}] {state_name}...")

        df = fetch_acs_data_for_state(state_fips)
        if not df.empty:
            all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    print(f"  Fetched data for {len(combined):,} tracts")

    return combined


def main():
    """Main function to download and process Census data."""
    print("=" * 60)
    print("Census Data Prep Script")
    print("=" * 60)

    # Download tract geometries
    tracts_gdf = download_tract_geometries()

    # Fetch ACS data
    acs_df = fetch_all_acs_data()

    # Join ACS data to tract geometries
    print("\nJoining ACS data to tract geometries...")
    merged = tracts_gdf.merge(acs_df, on="GEOID", how="left")

    # Report join statistics
    acs_matched = merged["median_hh_income"].notna().sum()
    print(f"  Matched {acs_matched:,} of {len(merged):,} tracts with ACS data")

    # Keep only necessary columns
    columns_to_keep = [
        "GEOID", "state_fips", "state_name", "state_abbrev",
        "median_hh_income", "pct_pop_21_34", "pct_college_educated",
        "geometry"
    ]
    merged = merged[columns_to_keep]

    # Save as GeoParquet (much smaller than GeoJSON)
    output_path = OUTPUT_DIR / "census_tracts.parquet"
    print(f"\nSaving to {output_path}...")
    merged.to_parquet(output_path, index=False)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")

    print("\nDone!")
    print("=" * 60)

    # Summary statistics
    print("\nSummary Statistics:")
    print(f"  Total tracts: {len(merged):,}")
    print(f"  States covered: {merged['state_fips'].nunique()}")
    print(f"\n  Median HH Income:")
    print(f"    Min: ${merged['median_hh_income'].min():,.0f}")
    print(f"    Max: ${merged['median_hh_income'].max():,.0f}")
    print(f"    Mean: ${merged['median_hh_income'].mean():,.0f}")
    print(f"\n  % Population 21-34:")
    print(f"    Min: {merged['pct_pop_21_34'].min():.1f}%")
    print(f"    Max: {merged['pct_pop_21_34'].max():.1f}%")
    print(f"    Mean: {merged['pct_pop_21_34'].mean():.1f}%")
    print(f"\n  % College Educated:")
    print(f"    Min: {merged['pct_college_educated'].min():.1f}%")
    print(f"    Max: {merged['pct_college_educated'].max():.1f}%")
    print(f"    Mean: {merged['pct_college_educated'].mean():.1f}%")


if __name__ == "__main__":
    main()
