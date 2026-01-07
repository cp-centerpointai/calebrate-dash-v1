# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Venue Distribution Analysis Dashboard - A Streamlit application for analyzing brand distribution across retail venues. Users select a focus brand and competitor brands, then view an interactive map showing venue distribution with color-coded markers and dynamic viewport statistics.

## Commands

```bash
# Activate virtual environment (required before running any Python)
source .venv/bin/activate  # or: source venv/bin/activate

# Run the Streamlit dashboard
streamlit run app.py

# Reprocess venue data (only needed when sample.csv changes)
python preprocess.py

# Fetch Census demographic data (requires CENSUS_API_KEY in .env)
python scripts/fetch_census_data.py
```

## Key Dependencies

- `streamlit` + `streamlit-folium`: Dashboard framework and Folium map integration
- `folium` + `branca`: Interactive mapping with MarkerCluster and color scales
- `pandas` + `pyarrow`: Data handling with parquet support for list columns
- `geopandas`: Census tract geometry processing

## Architecture

### Data Flow

1. **Raw Data**: `sample.csv` (~188k venues with brand/product data)
2. **Preprocessing**: `preprocess.py` transforms CSV → parquet + JSON lookup tables
3. **Census Data**: `scripts/fetch_census_data.py` downloads tract geometries and ACS demographics
4. **Runtime**: `app.py` loads only preprocessed files (no CSV parsing at runtime)

### Key Files

- `preprocess.py` - One-time data transformation for venue data
- `scripts/fetch_census_data.py` - Census tract geometry + ACS demographic data fetcher
- `app.py` - Streamlit dashboard with two-phase UI (selection → map)
- `data/stores.parquet` - Processed venue data with list columns for brands/products
- `data/brands.json` - Distinct brand names for dropdowns
- `data/subcategories.json` - Venue subcategory options
- `data/census_tracts.parquet` - GeoParquet with tract geometries and demographics

### Application Flow

The app uses a two-phase UI controlled by `st.session_state.analysis_phase`:

1. **Selection Phase** (`render_selection_screen`): User selects focus brand, competitor brands, optional subcategory filter, and optional state filter. Clicking "Analyze" transitions to map phase.
2. **Map Phase** (`render_map_view`): Interactive Folium map displays venues with:
   - Color-coded CircleMarkers (green=focus brand, red=competitor only, gray=whitespace)
   - MarkerCluster for performance (clusters at `maxClusterRadius: 30`, unclusters at zoom 12+)
   - Optional Census choropleth overlays (requires state selection due to data volume)
   - Sidebar controls for census overlay selection and "Reset Filters" button
   - Right-side panel showing viewport-filtered statistics that update on pan/zoom

### Venue Categorization Logic

Venues are classified into three mutually exclusive groups:
- **With focus brand** (green): Venue carries the selected focus brand
- **Competitor only** (red): Venue carries competitor brand(s) but NOT the focus brand
- **Neither/whitespace** (gray): Venue carries neither focus nor competitor brands

### Census Overlay Options

When a state is selected, the sidebar offers demographic overlays:
- `median_hh_income`: Median Household Income from ACS
- `pct_pop_21_34`: Percentage of population ages 21-34
- `pct_college_educated`: Percentage with Bachelor's degree or higher

### Session State Keys

- `analysis_phase`: "selection" or "map"
- `focus_brand`, `competitor_brands`: User selections
- `subcategory`, `selected_state_code`: Optional filters
- `census_overlay_option`: Selected demographic overlay ("None" by default)

### Data Schema

The `all_brands` and `all_products` columns are stored as Python lists in parquet (via pyarrow). The `brands_display` column is a pre-joined string for tooltip display.

**Venue columns**: `store_id`, `name`, `address`, `city`, `state`, `postal_code`, `latitude`, `longitude`, `chain_name`, `subcategory`, `detailed_category`, `all_brands`, `brands_display`, `brand_count`

**Census columns**: `GEOID`, `state_fips`, `state_name`, `state_abbrev`, `median_hh_income`, `pct_pop_21_34`, `pct_college_educated`, `geometry`

## Environment Variables

Create a `.env` file with:
- `CENSUS_API_KEY`: Required for `scripts/fetch_census_data.py` (get from https://api.census.gov/data/key_signup.html)
