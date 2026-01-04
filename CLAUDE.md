# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Store Distribution Analysis Dashboard - A Streamlit application for analyzing brand distribution across retail stores. Users select a focus brand and competitor brands, then view an interactive map showing store distribution with color-coded markers and dynamic viewport statistics.

## Commands

```bash
# Activate virtual environment (required before running any Python)
source venv/bin/activate

# Run the Streamlit dashboard
streamlit run app.py

# Reprocess data (only needed when sample.csv changes)
python preprocess.py
```

## Architecture

### Data Flow

1. **Raw Data**: `sample.csv` (~188k stores with brand/product data)
2. **Preprocessing**: `preprocess.py` transforms CSV → parquet + JSON lookup tables
3. **Runtime**: `app.py` loads only preprocessed files (no CSV parsing at runtime)

### Key Files

- `preprocess.py` - One-time data transformation script
- `app.py` - Streamlit dashboard with two-phase UI (selection → map)
- `data/stores.parquet` - Processed store data with list columns for brands/products
- `data/brands.json` - Distinct brand names for dropdowns
- `data/subcategories.json` - Store subcategory options

### Application Flow

1. **Selection Phase**: User selects focus brand, competitor brands, optional subcategory filter, and optional state filter
2. **Map Phase**: Interactive Folium map displays stores with:
   - Color-coded CircleMarkers (green=focus brand, red=competitor only, gray=whitespace)
   - MarkerCluster for performance (neutral dark blue clusters, unclusters at zoom level 12)
   - Dynamic metrics that update based on current map viewport bounds via `st_folium` returned bounds

### Store Categorization Logic

Stores are classified into three mutually exclusive groups:
- **With focus brand** (green): Store carries the selected focus brand
- **Competitor only** (red): Store carries competitor brand(s) but NOT the focus brand
- **Neither/whitespace** (gray): Store carries neither focus nor competitor brands

### Session State Keys

- `analysis_phase`: "selection" or "map"
- `focus_brand`, `competitor_brands`: User selections
- `subcategory`, `selected_state_code`: Optional filters

### Data Schema

The `all_brands` and `all_products` columns are stored as Python lists in parquet (via pyarrow). The `brands_display` column is a pre-joined string for tooltip display.

Key columns: `store_id`, `name`, `address`, `city`, `state`, `postal_code`, `latitude`, `longitude`, `chain_name`, `subcategory`, `detailed_category`, `all_brands`, `brands_display`, `brand_count`
