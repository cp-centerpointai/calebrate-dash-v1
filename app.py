"""
Store Distribution Analysis Dashboard - Map Visualization with Folium

Run with:
    streamlit run app.py
"""

import json
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from pathlib import Path
import branca.colormap as cm


# Page configuration
st.set_page_config(
    page_title="Store Distribution Analysis",
    layout="wide",
)


@st.cache_data
def load_data():
    """Load preprocessed data files."""
    data_dir = Path("data")

    # Load parquet store data
    stores_df = pd.read_parquet(data_dir / "stores.parquet")

    # Load lookup tables
    with open(data_dir / "brands.json") as f:
        brands = json.load(f)

    with open(data_dir / "subcategories.json") as f:
        subcategories = json.load(f)

    return stores_df, brands, subcategories


@st.cache_data
def load_census_data():
    """Load Census tract data with geometries."""
    data_dir = Path("data")
    census_path = data_dir / "census_tracts.parquet"

    if not census_path.exists():
        return None

    gdf = gpd.read_parquet(census_path)
    return gdf


# Census overlay options
CENSUS_OVERLAYS = {
    "None": None,
    "Median Household Income": "median_hh_income",
    "% Population 21-34": "pct_pop_21_34",
    "% College Educated": "pct_college_educated"
}


def filter_stores(df: pd.DataFrame, subcategory: str) -> pd.DataFrame:
    """Filter stores by subcategory."""
    if subcategory == "All":
        return df
    return df[df["subcategory"] == subcategory]


def get_store_color(store_brands, focus_brand: str, competitor_brands: list) -> str:
    """
    Returns color string based on brand presence.
    - Green: Store carries the focus brand
    - Red: Store carries competitor brand(s) but NOT focus brand
    - Grey: Store carries neither (whitespace)
    """
    has_focus = focus_brand in store_brands
    has_competitor = any(comp in store_brands for comp in competitor_brands)

    if has_focus:
        return "green"
    elif has_competitor:
        return "red"
    else:
        return "gray"


def add_color_column(df: pd.DataFrame, focus_brand: str, competitor_brands: list) -> pd.DataFrame:
    """Add color column based on brand categorization."""
    df = df.copy()
    df["color"] = df["all_brands"].apply(
        lambda brands: get_store_color(brands, focus_brand, competitor_brands)
    )
    return df


def categorize_stores(
    df: pd.DataFrame,
    focus_brand: str,
    competitor_brands: list[str]
) -> dict:
    """
    Categorize stores into three groups:
    - With focus brand
    - Competitor only (has competitor brand(s) but not focus brand)
    - Neither (whitespace - no focus or competitor brands)
    """
    def has_focus(brands_list):
        return focus_brand in brands_list

    def has_any_competitor(brands_list):
        return any(comp in brands_list for comp in competitor_brands)

    has_focus_mask = df["all_brands"].apply(has_focus)
    has_competitor_mask = df["all_brands"].apply(has_any_competitor)

    with_focus = has_focus_mask.sum()
    competitor_only = (~has_focus_mask & has_competitor_mask).sum()
    neither = (~has_focus_mask & ~has_competitor_mask).sum()

    return {
        "total": len(df),
        "with_focus": with_focus,
        "competitor_only": competitor_only,
        "neither": neither,
    }


def add_census_choropleth(m: folium.Map, census_gdf: gpd.GeoDataFrame, overlay_column: str):
    """Add a Census choropleth layer to the map."""
    # Filter out null values for the overlay column
    valid_data = census_gdf[census_gdf[overlay_column].notna()].copy()

    if len(valid_data) == 0:
        return

    # Get min/max for color scale
    vmin = valid_data[overlay_column].quantile(0.05)  # Use 5th percentile to reduce outlier impact
    vmax = valid_data[overlay_column].quantile(0.95)  # Use 95th percentile

    # Create colormap based on overlay type
    if overlay_column == "median_hh_income":
        caption = "Median Household Income ($)"
        colormap = cm.LinearColormap(
            colors=["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"],
            vmin=vmin,
            vmax=vmax,
            caption=caption
        )
        format_value = lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
    elif overlay_column == "pct_pop_21_34":
        caption = "% Population Ages 21-34"
        colormap = cm.LinearColormap(
            colors=["#feebe2", "#fbb4b9", "#f768a1", "#c51b8a", "#7a0177"],
            vmin=vmin,
            vmax=vmax,
            caption=caption
        )
        format_value = lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
    else:  # pct_college_educated
        caption = "% College Educated (Bachelor's+)"
        colormap = cm.LinearColormap(
            colors=["#f7fcf5", "#c7e9c0", "#74c476", "#31a354", "#006d2c"],
            vmin=vmin,
            vmax=vmax,
            caption=caption
        )
        format_value = lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"

    # Create style function
    def style_function(feature):
        value = feature["properties"].get(overlay_column)
        if value is None or pd.isna(value):
            return {
                "fillColor": "#cccccc",
                "color": "#666666",
                "weight": 0.5,
                "fillOpacity": 0.3
            }
        # Clamp value to colormap range
        clamped_value = max(vmin, min(vmax, value))
        return {
            "fillColor": colormap(clamped_value),
            "color": "#666666",
            "weight": 0.5,
            "fillOpacity": 0.5
        }

    # Add GeoJson layer (no tooltip for census tracts)
    folium.GeoJson(
        valid_data,
        name="Census Overlay",
        style_function=style_function,
    ).add_to(m)

    # Add colormap legend to map
    colormap.add_to(m)


def create_folium_map(
    stores_df: pd.DataFrame,
    census_gdf: gpd.GeoDataFrame = None,
    census_overlay: str = None,
):
    """Create a Folium map with store markers and optional Census overlay."""

    center_lat = stores_df["latitude"].mean()
    center_lon = stores_df["longitude"].mean()

    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles="cartodbpositron",
    )

    # Add Census choropleth layer if selected (BEFORE store markers)
    if census_gdf is not None and census_overlay is not None:
        add_census_choropleth(m, census_gdf, census_overlay)

    # Custom cluster icon function - neutral dark blue color
    icon_create_function = """
    function(cluster) {
        var count = cluster.getChildCount();
        var size = count < 10 ? 30 : count < 50 ? 40 : count < 100 ? 50 : 60;
        return L.divIcon({
            html: '<div style="background-color: #2c3e50; color: white; border-radius: 50%; width: ' + size + 'px; height: ' + size + 'px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 12px; border: 2px solid white; box-shadow: 0 2px 5px rgba(0,0,0,0.3);">' + count + '</div>',
            className: 'custom-cluster-icon',
            iconSize: L.point(size, size)
        });
    }
    """

    # Add markers using MarkerCluster for performance
    marker_cluster = MarkerCluster(
        name="Stores",
        overlay=True,
        control=False,
        icon_create_function=icon_create_function,
        options={
            "maxClusterRadius": 30,  # Tighter radius - markers must be very close to cluster
            "spiderfyOnMaxZoom": True,
            "zoomToBoundsOnClick": True,
            "disableClusteringAtZoom": 12,  # Show individual markers at zoom 12+
            "spiderfyDistanceMultiplier": 1.5,
        }
    )

    # Add individual markers
    for _, row in stores_df.iterrows():
        # Create tooltip content (shown on hover)
        tooltip_html = f"""
        <b>{row['name']}</b><br>
        {row['address']}, {row['city']}, {row['state']} {row['postal_code']}<br>
        <b>Category:</b> {row['subcategory']} &gt; {row['detailed_category']}<br>
        <b>Brands:</b> {row['brands_display']}
        """

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6,
            color=row["color"],
            fill=True,
            fill_color=row["color"],
            fill_opacity=0.7,
            tooltip=folium.Tooltip(tooltip_html),
        ).add_to(marker_cluster)

    marker_cluster.add_to(m)

    # Fit bounds to data on initial load
    if len(stores_df) > 0:
        sw = [stores_df["latitude"].min(), stores_df["longitude"].min()]
        ne = [stores_df["latitude"].max(), stores_df["longitude"].max()]
        m.fit_bounds([sw, ne])

    return m


def render_selection_screen(stores_df, brands, subcategories):
    """Render the brand/competitor selection screen with state selection."""

    st.title("Store Distribution Analysis")
    st.subheader("Selection Controls")

    col1, col2 = st.columns(2)

    with col1:
        focus_brand = st.selectbox(
            "Select Focus Brand",
            options=[""] + brands,
            index=0,
            key="focus_brand_select",
            help="Required: Select the primary brand to analyze"
        )

        if focus_brand:
            st.session_state.focus_brand = focus_brand
        else:
            st.session_state.focus_brand = None

    with col2:
        available_competitors = [b for b in brands if b != focus_brand]
        competitor_brands = st.multiselect(
            "Select Competitor Brands",
            options=available_competitors,
            default=[],
            key="competitor_brands_select",
            help="Required: Select at least one competitor brand"
        )
        st.session_state.competitor_brands = competitor_brands

    col3, col4 = st.columns(2)

    with col3:
        subcategory = st.selectbox(
            "Filter by Store Subcategory (optional)",
            options=subcategories,
            index=0,
            key="subcategory_select",
            help="Optional: Filter stores by subcategory"
        )
        st.session_state.subcategory = subcategory

    with col4:
        # State selection - compute state options based on current subcategory filter
        filtered_df = filter_stores(stores_df, subcategory)
        state_clusters = filtered_df.groupby("state").agg(
            count=("store_id", "count"),
        ).reset_index()
        state_clusters = state_clusters.sort_values("count", ascending=False)

        state_options = ["All States"] + [
            f"{row['state']} ({row['count']:,} stores)"
            for _, row in state_clusters.iterrows()
        ]

        selected_state = st.selectbox(
            "Select State (optional)",
            options=state_options,
            index=0,
            key="state_selector",
            help="Optional: Focus on a specific state"
        )
        st.session_state.selected_state = selected_state

    # Validation
    focus_brand_set = bool(st.session_state.focus_brand)
    competitors_set = len(st.session_state.competitor_brands) >= 1
    is_valid = focus_brand_set and competitors_set

    if not focus_brand_set:
        st.warning("Please select a focus brand")
    elif not competitors_set:
        st.warning("Please select at least one competitor brand")

    st.divider()

    analyze_clicked = st.button(
        "Analyze",
        disabled=not is_valid,
        type="primary",
        use_container_width=True
    )

    if analyze_clicked and is_valid:
        # Set selected state for filtering
        if selected_state != "All States":
            state_code = selected_state.split(" (")[0]
            st.session_state.selected_state_code = state_code
        else:
            st.session_state.selected_state_code = None

        st.session_state.analysis_phase = "map"
        st.rerun()


def render_map_view(stores_df, census_gdf):
    """Render the map visualization with individual stores."""

    # Sidebar controls
    with st.sidebar:
        st.header("Analysis Controls")

        st.markdown(f"""
**Focus Brand:** {st.session_state.focus_brand}

**Competitors:** {", ".join(st.session_state.competitor_brands)}

**Subcategory:** {st.session_state.subcategory}

**State:** {st.session_state.get("selected_state", "All States")}
        """)

        st.divider()

        # Census overlay selectbox - use session state to avoid triggering map resets
        census_available = census_gdf is not None
        if census_available:
            census_overlay_option = st.selectbox(
                "Census Overlay",
                options=list(CENSUS_OVERLAYS.keys()),
                index=list(CENSUS_OVERLAYS.keys()).index(st.session_state.census_overlay_option),
                key="census_overlay_select",
                help="Overlay Census tract data on the map"
            )
            # Update session state (does not trigger rerun)
            st.session_state.census_overlay_option = census_overlay_option
            census_overlay_column = CENSUS_OVERLAYS[census_overlay_option]
        else:
            st.info("Census data not available. Run scripts/fetch_census_data.py to enable overlays.")
            census_overlay_column = None

        st.divider()

        if st.button("Restart Analysis", type="secondary", use_container_width=True):
            st.session_state.analysis_phase = "selection"
            st.session_state.selected_state_code = None
            st.rerun()

        # Legend
        st.divider()
        st.markdown("**Legend**")
        st.markdown("🟢 Has focus brand")
        st.markdown("🔴 Competitor only")
        st.markdown("⚫ Neither (whitespace)")

    # Filter stores by subcategory
    filtered_df = filter_stores(stores_df, st.session_state.subcategory)

    # Filter by state if selected
    selected_state_code = st.session_state.get("selected_state_code")
    if selected_state_code:
        filtered_df = filtered_df[filtered_df["state"] == selected_state_code]

    # Filter census data by state if available and a state is selected
    state_census_gdf = None
    if census_gdf is not None and census_overlay_column is not None:
        if selected_state_code:
            state_census_gdf = census_gdf[census_gdf["state_abbrev"] == selected_state_code]
        else:
            # If no state filter, don't show census overlay (too much data)
            st.sidebar.warning("Select a state to enable Census overlay")
            census_overlay_column = None

    # Add color column based on brand categorization
    view_df = add_color_column(
        filtered_df,
        st.session_state.focus_brand,
        st.session_state.competitor_brands
    )

    # Main content
    st.title("Store Distribution Map")

    # Create placeholders for metrics that will update
    metrics_container = st.container()

    st.markdown("Hover over stores to see details. Click on clusters to zoom in.")

    # Use all census tracts for the selected state
    filtered_census_gdf = state_census_gdf
    if filtered_census_gdf is not None:
        st.sidebar.caption(f"Showing {len(filtered_census_gdf):,} tracts")

    # Create the map
    m = create_folium_map(view_df, filtered_census_gdf, census_overlay_column)

    # Display map
    # Use returned_objects=[] to prevent automatic Streamlit reruns on map interaction
    # This fixes the issue where clicking a cluster would zoom in then immediately snap back
    # because st_folium was triggering a rerun with the pre-animation bounds
    map_data = st_folium(
        m,
        use_container_width=True,
        height=600,
        returned_objects=[],  # No reruns on map interaction
        key="store_map",
    )

    # Show stats for all visible stores (entire filtered dataset)
    # Since we can't get real-time bounds without triggering reruns, show totals
    viewport_df = view_df

    # Calculate stats for stores in current viewport
    results = categorize_stores(
        viewport_df,
        st.session_state.focus_brand,
        st.session_state.competitor_brands
    )

    # Display metrics in the container above the map
    with metrics_container:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Stores", f"{results['total']:,}")
        col2.metric("With Focus Brand", f"{results['with_focus']:,}")
        col3.metric("Competitor Only", f"{results['competitor_only']:,}")
        col4.metric("Whitespace", f"{results['neither']:,}")


def main():
    # Load data
    stores_df, brands, subcategories = load_data()
    census_gdf = load_census_data()

    # Initialize session state
    if "focus_brand" not in st.session_state:
        st.session_state.focus_brand = None
    if "competitor_brands" not in st.session_state:
        st.session_state.competitor_brands = []
    if "subcategory" not in st.session_state:
        st.session_state.subcategory = "All"
    if "analysis_phase" not in st.session_state:
        st.session_state.analysis_phase = "selection"
    if "selected_state" not in st.session_state:
        st.session_state.selected_state = "All States"
    if "selected_state_code" not in st.session_state:
        st.session_state.selected_state_code = None
    if "census_overlay_option" not in st.session_state:
        st.session_state.census_overlay_option = "None"

    # Route to appropriate view
    if st.session_state.analysis_phase == "selection":
        render_selection_screen(stores_df, brands, subcategories)
    else:
        render_map_view(stores_df, census_gdf)


if __name__ == "__main__":
    main()
