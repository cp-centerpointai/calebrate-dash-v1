"""
Coors Edge Distribution Analysis Dashboard

Analyzes Coors Edge distribution vs Athletic Brewing competition
within the Coors distribution network.

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

# Fixed brand configuration
FOCUS_BRAND = "Coors Edge"
COMPETITOR_BRAND = "Athletic Brewing"

# Store category colors
CATEGORY_COLORS = {
    "both": "#9b59b6",           # Purple - has both brands
    "coors_edge_only": "#27ae60", # Green - Coors Edge only
    "athletic_only": "#e74c3c",   # Red - Athletic only
    "neither": "#95a5a6",         # Gray - whitespace
}

# Store category display names (matching legend)
CATEGORY_DISPLAY_NAMES = {
    "both": "Both Brands",
    "coors_edge_only": "Coors Edge Only",
    "athletic_only": "Athletic Only",
    "neither": "No NA Beer",
}

# Page configuration
st.set_page_config(
    page_title="Coors Edge Distribution Analysis",
    layout="wide",
)


@st.cache_data
def load_data():
    """Load preprocessed data files."""
    data_dir = Path("data")

    # Load parquet store data (includes pre-computed store_category)
    stores_df = pd.read_parquet(data_dir / "stores.parquet")

    # Load category hierarchy lookups
    with open(data_dir / "main_categories.json") as f:
        main_categories = json.load(f)

    with open(data_dir / "main_to_subcategories.json") as f:
        main_to_sub = json.load(f)

    with open(data_dir / "subcategory_to_detailed.json") as f:
        sub_to_detailed = json.load(f)

    return stores_df, main_categories, main_to_sub, sub_to_detailed


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
    "% Population 25-44": "pct_pop_25_44",
    "Median Age": "median_age",
    "% College Educated": "pct_college_educated",
    "% Non-Family Households": "pct_nonfamily_hh",
    "Population Density": "pop_density"
}


def filter_stores(
    df: pd.DataFrame,
    main_category: str = "All",
    subcategory: str = "All",
    detailed_category: str = "All"
) -> pd.DataFrame:
    """Filter stores by category hierarchy."""
    result = df
    if main_category != "All":
        result = result[result["main_category"] == main_category]
    if subcategory != "All":
        result = result[result["subcategory"] == subcategory]
    if detailed_category != "All":
        result = result[result["detailed_category"] == detailed_category]
    return result


def add_color_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add color column based on pre-computed store_category."""
    df = df.copy()
    df["color"] = df["store_category"].map(CATEGORY_COLORS)
    return df


def get_category_stats(df: pd.DataFrame) -> dict:
    """
    Get store counts by category using pre-computed store_category column.
    Returns counts for: both, coors_edge_only, athletic_only, neither
    """
    counts = df["store_category"].value_counts()
    return {
        "total": len(df),
        "both": counts.get("both", 0),
        "coors_edge_only": counts.get("coors_edge_only", 0),
        "athletic_only": counts.get("athletic_only", 0),
        "neither": counts.get("neither", 0),
    }


def categorize_stores_by_bracket(
    df: pd.DataFrame,
    bracket_col: str = "income_bracket"
) -> pd.DataFrame:
    """
    Categorize stores by demographic bracket with counts and percentages.
    Uses pre-computed store_category column.
    """
    if bracket_col not in df.columns:
        return pd.DataFrame()

    # Map store_category to analysis categories
    # For the table, we show: coors_edge (both + coors_edge_only), athletic_only, neither
    df = df.copy()
    df["has_coors"] = df["store_category"].isin(["both", "coors_edge_only"])
    df["athletic_only_flag"] = df["store_category"] == "athletic_only"
    df["neither_flag"] = df["store_category"] == "neither"

    # Group by bracket
    grouped = df.groupby(bracket_col).agg(
        total=("store_id", "count"),
        with_coors_edge=("has_coors", "sum"),
        with_both=("store_category", lambda x: (x == "both").sum()),
        athletic_only=("athletic_only_flag", "sum"),
        neither=("neither_flag", "sum"),
    ).reset_index()

    grouped = grouped.rename(columns={bracket_col: "bracket"})

    # Calculate percentages
    grouped["coors_pct"] = (grouped["with_coors_edge"] / grouped["total"] * 100).round(1)
    grouped["both_pct"] = (grouped["with_both"] / grouped["total"] * 100).round(1)
    grouped["athletic_pct"] = (grouped["athletic_only"] / grouped["total"] * 100).round(1)
    grouped["no_na_beer_pct"] = (grouped["neither"] / grouped["total"] * 100).round(1)

    # Sort by quintile prefix (Q1, Q2, ..., then Unknown)
    def sort_key(label):
        if label == "Unknown":
            return (1, "")
        return (0, label)

    grouped = grouped.sort_values("bracket", key=lambda x: x.map(sort_key))
    return grouped


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
    elif overlay_column == "pct_pop_25_44":
        caption = "% Population Ages 25-44"
        colormap = cm.LinearColormap(
            colors=["#feebe2", "#fbb4b9", "#f768a1", "#c51b8a", "#7a0177"],
            vmin=vmin,
            vmax=vmax,
            caption=caption
        )
        format_value = lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
    elif overlay_column == "median_age":
        caption = "Median Age"
        colormap = cm.LinearColormap(
            colors=["#f7fcf5", "#c7e9c0", "#74c476", "#31a354", "#006d2c"],
            vmin=vmin,
            vmax=vmax,
            caption=caption
        )
        format_value = lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
    elif overlay_column == "pct_college_educated":
        caption = "% College Educated (Bachelor's+)"
        colormap = cm.LinearColormap(
            colors=["#f7fcf5", "#c7e9c0", "#74c476", "#31a354", "#006d2c"],
            vmin=vmin,
            vmax=vmax,
            caption=caption
        )
        format_value = lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
    elif overlay_column == "pct_nonfamily_hh":
        caption = "% Non-Family Households"
        colormap = cm.LinearColormap(
            colors=["#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84", "#fc8d59"],
            vmin=vmin,
            vmax=vmax,
            caption=caption
        )
        format_value = lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
    elif overlay_column == "pop_density":
        caption = "Population Density (per sq mi)"
        colormap = cm.LinearColormap(
            colors=["#edf8fb", "#b3cde3", "#8c96c6", "#8856a7", "#810f7c"],
            vmin=vmin,
            vmax=vmax,
            caption=caption
        )
        format_value = lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A"
    else:
        # Fallback for any other column
        caption = overlay_column
        colormap = cm.LinearColormap(
            colors=["#f7fcf5", "#c7e9c0", "#74c476", "#31a354", "#006d2c"],
            vmin=vmin,
            vmax=vmax,
            caption=caption
        )
        format_value = lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"

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


def filter_by_bounds(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """Filter dataframe to stores within the given map bounds."""
    if not bounds:
        return df

    south = bounds["_southWest"]["lat"]
    north = bounds["_northEast"]["lat"]
    west = bounds["_southWest"]["lng"]
    east = bounds["_northEast"]["lng"]

    mask = (
        (df["latitude"] >= south) &
        (df["latitude"] <= north) &
        (df["longitude"] >= west) &
        (df["longitude"] <= east)
    )
    return df[mask]


def create_folium_map(
    stores_df: pd.DataFrame,
    census_gdf: gpd.GeoDataFrame = None,
    census_overlay: str = None,
    center: list = None,
    zoom: int = None,
):
    """Create a Folium map with store markers and optional Census overlay."""

    # Use provided center/zoom or calculate from data
    if center is not None and zoom is not None:
        center_lat, center_lon = center
        zoom_start = zoom
    else:
        center_lat = stores_df["latitude"].mean()
        center_lon = stores_df["longitude"].mean()
        zoom_start = 7

    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
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
        # Get lead score (integer 0-100)
        lead_score = row.get('lead_score', 0) if pd.notna(row.get('lead_score')) else 0

        # Create tooltip content (shown on hover)
        tooltip_html = f"""
        <b>{row['name']}</b><br>
        {row['address']}, {row['city']}, {row['state']} {row['postal_code']}<br>
        <b>Category:</b> {row['subcategory']} &gt; {row['detailed_category']}<br>
        <b>Lead Score:</b> {lead_score}
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

    # Only fit bounds on initial load (when no center/zoom provided)
    if center is None and zoom is None and len(stores_df) > 0:
        sw = [stores_df["latitude"].min(), stores_df["longitude"].min()]
        ne = [stores_df["latitude"].max(), stores_df["longitude"].max()]
        m.fit_bounds([sw, ne])

    return m


def render_drilldown_table(df: pd.DataFrame, state_code: str = None):
    """Render the drill-down table with store details."""
    # Get viewport bounds for filtering if available
    viewport_bounds = st.session_state.get("viewport_bounds")

    # Filter by viewport if bounds are set
    if viewport_bounds:
        table_df = filter_by_bounds(df, viewport_bounds)
        filter_label = "stores in current view"
    else:
        table_df = df
        if state_code:
            filter_label = f"stores in {state_code}"
        else:
            filter_label = "stores"

    # Header with close button
    header_col, close_col = st.columns([4, 1])
    with header_col:
        st.subheader(f"Store Details ({len(table_df):,} {filter_label})")
    with close_col:
        if st.button("✕ Close", key="close_drilldown", use_container_width=True):
            st.session_state.show_drilldown = False
            st.rerun()

    if len(table_df) == 0:
        st.info("No stores in the current view. Try adjusting filters or viewport.")
        return

    # Prepare display dataframe
    display_df = table_df[[
        "name",
        "address",
        "city",
        "state",
        "postal_code",
        "main_category",
        "subcategory",
        "detailed_category",
        "store_category",
        "lead_score",
    ]].copy()

    # Map store_category to display names
    display_df["store_type"] = display_df["store_category"].map(CATEGORY_DISPLAY_NAMES)

    # Format address column
    display_df["full_address"] = (
        display_df["address"] + ", " +
        display_df["city"] + ", " +
        display_df["state"] + " " +
        display_df["postal_code"].astype(str)
    )

    # Create final display dataframe with renamed columns
    final_df = display_df[[
        "name",
        "full_address",
        "main_category",
        "subcategory",
        "detailed_category",
        "store_type",
        "lead_score",
    ]].copy()

    final_df.columns = [
        "Store Name",
        "Address",
        "Category",
        "Subcategory",
        "Detailed Category",
        "Store Type",
        "Lead Score",
    ]

    # Sort by lead score descending by default
    final_df = final_df.sort_values("Lead Score", ascending=False)

    # Display table with sorting enabled
    st.dataframe(
        final_df,
        hide_index=True,
        use_container_width=True,
        height=400,
        column_config={
            "Store Name": st.column_config.TextColumn("Store Name", width="medium"),
            "Address": st.column_config.TextColumn("Address", width="large"),
            "Category": st.column_config.TextColumn("Category", width="small"),
            "Subcategory": st.column_config.TextColumn("Subcategory", width="small"),
            "Detailed Category": st.column_config.TextColumn("Detailed Category", width="small"),
            "Store Type": st.column_config.TextColumn("Store Type", width="small"),
            "Lead Score": st.column_config.NumberColumn(
                "Lead Score",
                format="%d",
                width="small",
                help="Score (0-100) indicating store potential for NA beer. Based on: store category, demographic profile (income, age, education), and competitor presence."
            ),
        }
    )

    # Store type filter hint
    st.caption("Click column headers to sort. Use sidebar filters to narrow results.")


def reset_filters():
    """Callback to reset all filters before widgets are instantiated."""
    # Reset category hierarchy
    st.session_state.main_category = "All"
    st.session_state.subcategory = "All"
    st.session_state.detailed_category = "All"
    # Reset widget keys for category selectors
    st.session_state.main_category_select = "All"
    st.session_state.subcategory_select = "All"
    st.session_state.detailed_category_select = "All"
    # Reset state filter
    st.session_state.selected_state = "All States"
    st.session_state.selected_state_code = None
    st.session_state.state_selector = "All States"
    # Reset census overlay
    st.session_state.census_overlay_option = "None"
    st.session_state.census_overlay_select = "None"
    # Reset map state
    st.session_state.show_map = False
    st.session_state.show_drilldown = False
    st.session_state.viewport_bounds = None


def render_map_view(stores_df, main_categories, main_to_sub, sub_to_detailed, census_gdf):
    """Render the map visualization with individual stores."""

    # Handle reset request before widgets are instantiated
    if st.session_state.get("reset_requested", False):
        reset_filters()
        st.session_state.reset_requested = False

    # Sidebar controls
    with st.sidebar:
        st.header("Coors Edge Analysis")

        st.markdown(f"""
**Focus:** {FOCUS_BRAND}

**Competitor:** {COMPETITOR_BRAND}
        """)

        st.divider()

        # Main Category selector
        main_category = st.selectbox(
            "Main Category",
            options=main_categories,
            index=main_categories.index(st.session_state.main_category) if st.session_state.main_category in main_categories else 0,
            key="main_category_select",
            help="Filter stores by main category"
        )
        # Reset child selections if main category changed
        if main_category != st.session_state.main_category:
            st.session_state.subcategory = "All"
            st.session_state.detailed_category = "All"
        st.session_state.main_category = main_category

        # Subcategory selector (only show if main category is selected)
        if main_category != "All":
            subcategory_options = main_to_sub.get(main_category, ["All"])
            subcategory = st.selectbox(
                "Subcategory",
                options=subcategory_options,
                index=subcategory_options.index(st.session_state.subcategory) if st.session_state.subcategory in subcategory_options else 0,
                key="subcategory_select",
                help="Filter stores by subcategory"
            )
            # Reset detailed_category if subcategory changed
            if subcategory != st.session_state.subcategory:
                st.session_state.detailed_category = "All"
            st.session_state.subcategory = subcategory
        else:
            subcategory = "All"
            st.session_state.subcategory = "All"

        # Detailed Category selector (only show if subcategory is selected)
        if subcategory != "All":
            detailed_options = sub_to_detailed.get(subcategory, ["All"])
            detailed_category = st.selectbox(
                "Detailed Category",
                options=detailed_options,
                index=detailed_options.index(st.session_state.detailed_category) if st.session_state.detailed_category in detailed_options else 0,
                key="detailed_category_select",
                help="Filter stores by detailed category"
            )
            st.session_state.detailed_category = detailed_category
        else:
            detailed_category = "All"
            st.session_state.detailed_category = "All"

        # State filter - compute state options
        filtered_for_states = filter_stores(stores_df, main_category, subcategory, detailed_category)
        state_clusters = filtered_for_states.groupby("state").agg(
            count=("store_id", "count"),
        ).reset_index()
        state_clusters = state_clusters.sort_values("count", ascending=False)

        state_options = ["All States"] + [
            f"{row['state']} ({row['count']:,} stores)"
            for _, row in state_clusters.iterrows()
        ]

        # Find current index
        current_state = st.session_state.get("selected_state", "All States")
        try:
            state_index = next(i for i, opt in enumerate(state_options) if opt.startswith(current_state.split(" (")[0]))
        except StopIteration:
            state_index = 0

        selected_state = st.selectbox(
            "State",
            options=state_options,
            index=state_index,
            key="state_selector",
            help="Focus on a specific state (required for Census overlay)"
        )

        # Parse state code and detect changes
        if selected_state != "All States":
            new_state_code = selected_state.split(" (")[0]
        else:
            new_state_code = None

        # Reset show_map and viewport bounds if state changed
        old_state_code = st.session_state.get("selected_state_code")
        if new_state_code != old_state_code:
            st.session_state.show_map = False
            st.session_state.viewport_bounds = None

        st.session_state.selected_state = selected_state
        st.session_state.selected_state_code = new_state_code

        st.divider()

        # Census overlay selectbox
        census_available = census_gdf is not None
        if census_available:
            census_overlay_option = st.selectbox(
                "Census Overlay",
                options=list(CENSUS_OVERLAYS.keys()),
                index=list(CENSUS_OVERLAYS.keys()).index(st.session_state.census_overlay_option),
                key="census_overlay_select",
                help="Overlay Census tract data on the map"
            )
            st.session_state.census_overlay_option = census_overlay_option
            census_overlay_column = CENSUS_OVERLAYS[census_overlay_option]
        else:
            st.info("Census data not available. Run scripts/fetch_census_data.py to enable overlays.")
            census_overlay_column = None

        st.divider()

        # Reset button to go back to initial state
        if st.button("Reset Filters", use_container_width=True):
            st.session_state.reset_requested = True
            st.rerun()


    # Filter stores by category hierarchy
    filtered_df = filter_stores(
        stores_df,
        st.session_state.main_category,
        st.session_state.subcategory,
        st.session_state.detailed_category
    )

    # Filter by state if selected
    selected_state_code = st.session_state.get("selected_state_code")
    if selected_state_code:
        filtered_df = filtered_df[filtered_df["state"] == selected_state_code]

    # Main content
    st.title("Coors Edge Distribution Analysis")

    # Dynamic summary title based on state selection
    if selected_state_code:
        summary_title = f"{selected_state_code} Summary"
    else:
        summary_title = "National Summary"

    # Title row with action buttons
    title_col, btn_col1, btn_col2 = st.columns([3, 1, 1])
    with title_col:
        st.subheader(summary_title)
    with btn_col1:
        # Map it button - disabled if no state selected
        map_disabled = selected_state_code is None
        if st.button(
            "Map it",
            disabled=map_disabled,
            use_container_width=True,
            help="Select a state to enable mapping" if map_disabled else "View store distribution on map"
        ):
            st.session_state.show_map = True
            st.rerun()
    with btn_col2:
        # Drill Down button - always enabled
        if st.button(
            "Drill Down",
            use_container_width=True,
            help="View detailed store list"
        ):
            st.session_state.show_drilldown = True
            st.rerun()

    # Show summary stats
    stats = get_category_stats(filtered_df)
    total = stats['total']

    if not selected_state_code:
        st.caption(f"The full dataset contains {total:,} stores across all states.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("🟣 **Both Brands**")
        st.markdown(f"### {stats['both']:,}")
        st.caption(f"{stats['both']/total*100:.1f}% of stores" if total > 0 else "0% of stores")
    with col2:
        st.markdown(f"🟢 **{FOCUS_BRAND} Only**")
        st.markdown(f"### {stats['coors_edge_only']:,}")
        st.caption(f"{stats['coors_edge_only']/total*100:.1f}% of stores" if total > 0 else "0% of stores")
    with col3:
        st.markdown(f"🔴 **{COMPETITOR_BRAND} Only**")
        st.markdown(f"### {stats['athletic_only']:,}")
        st.caption(f"{stats['athletic_only']/total*100:.1f}% of stores" if total > 0 else "0% of stores")
    with col4:
        st.markdown("⚪ **No NA Beer**")
        st.markdown(f"### {stats['neither']:,}")
        st.caption(f"{stats['neither']/total*100:.1f}% of stores" if total > 0 else "0% of stores")

    # Check if we should show drill-down or map
    show_map = st.session_state.get("show_map", False) and selected_state_code
    show_drilldown = st.session_state.get("show_drilldown", False)

    # Early return if neither map nor drilldown should be shown
    if not show_map and not show_drilldown:
        return

    st.divider()

    # Prepare data for both views
    view_df = add_color_column(filtered_df)

    # Show drill-down table if enabled
    if show_drilldown:
        render_drilldown_table(view_df, selected_state_code)

    # Only proceed with map if enabled
    if not show_map:
        return

    # Filter census data by state if available
    state_census_gdf = None
    census_overlay_column = CENSUS_OVERLAYS.get(st.session_state.census_overlay_option)
    if census_gdf is not None and census_overlay_column is not None:
        state_census_gdf = census_gdf[census_gdf["state_abbrev"] == selected_state_code]

    st.caption("Hover over stores to see details.")

    # Use all census tracts for the selected state
    filtered_census_gdf = state_census_gdf

    # Create the map - fits to state bounds on load
    m = create_folium_map(
        view_df,
        filtered_census_gdf,
        census_overlay_column,
    )

    # Side-by-side layout: map on left, stats panel on right
    map_col, stats_col = st.columns([65, 35])

    with map_col:
        # Return bounds so we can filter stats when user clicks refresh
        map_data = st_folium(
            m,
            use_container_width=True,
            height=600,
            returned_objects=["bounds"],
            key="store_map",
        )

    # Get current bounds from the map
    current_bounds = map_data.get("bounds") if map_data else None

    # Initialize viewport bounds in session state if not present
    if "viewport_bounds" not in st.session_state:
        st.session_state.viewport_bounds = None

    # Filter stores by viewport bounds if we have saved bounds
    if st.session_state.viewport_bounds:
        viewport_df = filter_by_bounds(view_df, st.session_state.viewport_bounds)
    else:
        viewport_df = view_df

    # Calculate stats for all stores in the state
    stats = get_category_stats(viewport_df)

    with stats_col:
        # Determine if we're showing viewport or full state data
        is_viewport_filtered = st.session_state.viewport_bounds is not None

        # Calculate percentages for display
        total = stats['total']
        if total > 0:
            both_pct = stats['both'] / total * 100
            coors_pct = stats['coors_edge_only'] / total * 100
            athletic_pct = stats['athletic_only'] / total * 100
            white_pct = stats['neither'] / total * 100
        else:
            both_pct = coors_pct = athletic_pct = white_pct = 0

        # Compact header with total and refresh button
        header_col, btn_col = st.columns([2, 1])
        with header_col:
            location_label = "View" if is_viewport_filtered else selected_state_code
            st.markdown(f"**{total:,}** stores in {location_label}")
        with btn_col:
            if st.button("🔄 Refresh", help="Refresh stats for current view"):
                if current_bounds:
                    st.session_state.viewport_bounds = current_bounds
                    st.rerun()

        # Reset to full state link (only show if filtered)
        if is_viewport_filtered:
            if st.button("Show all stores in state", type="tertiary", use_container_width=True):
                st.session_state.viewport_bounds = None
                st.rerun()

        # Tabs for Summary vs Census Breakdown
        tab_summary, tab_census = st.tabs(["Summary", "Census"])

        with tab_summary:
            # Compact legend-style stats using custom HTML
            legend_html = f"""
            <style>
                .legend-container {{
                    background: rgba(255,255,255,0.05);
                    border-radius: 8px;
                    padding: 12px;
                    margin-top: 8px;
                }}
                .legend-row {{
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: 6px 0;
                    border-bottom: 1px solid rgba(255,255,255,0.1);
                }}
                .legend-row:last-child {{
                    border-bottom: none;
                }}
                .legend-label {{
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }}
                .legend-dot {{
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    display: inline-block;
                }}
                .legend-value {{
                    text-align: right;
                    font-weight: 600;
                }}
                .legend-pct {{
                    color: #888;
                    font-size: 0.85em;
                    margin-left: 4px;
                }}
            </style>
            <div class="legend-container">
                <div class="legend-row">
                    <span class="legend-label">
                        <span class="legend-dot" style="background: #9b59b6;"></span>
                        Both Brands
                    </span>
                    <span class="legend-value">{stats['both']:,}<span class="legend-pct">({both_pct:.1f}%)</span></span>
                </div>
                <div class="legend-row">
                    <span class="legend-label">
                        <span class="legend-dot" style="background: #27ae60;"></span>
                        Coors Edge Only
                    </span>
                    <span class="legend-value">{stats['coors_edge_only']:,}<span class="legend-pct">({coors_pct:.1f}%)</span></span>
                </div>
                <div class="legend-row">
                    <span class="legend-label">
                        <span class="legend-dot" style="background: #e74c3c;"></span>
                        Athletic Only
                    </span>
                    <span class="legend-value">{stats['athletic_only']:,}<span class="legend-pct">({athletic_pct:.1f}%)</span></span>
                </div>
                <div class="legend-row">
                    <span class="legend-label">
                        <span class="legend-dot" style="background: #95a5a6;"></span>
                        No NA Beer
                    </span>
                    <span class="legend-value">{stats['neither']:,}<span class="legend-pct">({white_pct:.1f}%)</span></span>
                </div>
            </div>
            """
            st.markdown(legend_html, unsafe_allow_html=True)

        with tab_census:
            census_selection = st.session_state.get("census_overlay_option", "None")
            if census_selection == "None":
                st.info("Select a Census Overlay from the sidebar to see demographic breakdown.")
            else:
                overlay_config = {
                    "Median Household Income": ("income_bracket", "By Income"),
                    "% Population 21-34": ("age_bracket", "By Age (21-34)"),
                    "% College Educated": ("education_bracket", "By Education"),
                }

                bracket_col, title = overlay_config.get(census_selection, (None, None))

                if bracket_col is None or bracket_col not in viewport_df.columns:
                    st.warning("Run `python preprocess.py` to enable census analytics.")
                else:
                    st.markdown(f"**{title}**")

                    bracket_stats = categorize_stores_by_bracket(
                        viewport_df,
                        bracket_col=bracket_col
                    )

                    if len(bracket_stats) > 0:
                        display_df = bracket_stats[[
                            "bracket", "total", "coors_pct", "athletic_pct", "no_na_beer_pct"
                        ]].copy()
                        display_df.columns = ["Bracket", "Stores", "Coors %", "Athletic %", "No NA %"]

                        st.dataframe(
                            display_df,
                            hide_index=True,
                            use_container_width=True,
                        )
                    else:
                        st.info("No stores in current view.")


def main():
    # Load data
    stores_df, main_categories, main_to_sub, sub_to_detailed = load_data()
    census_gdf = load_census_data()

    # Initialize session state
    if "main_category" not in st.session_state:
        st.session_state.main_category = "All"
    if "subcategory" not in st.session_state:
        st.session_state.subcategory = "All"
    if "detailed_category" not in st.session_state:
        st.session_state.detailed_category = "All"
    if "selected_state" not in st.session_state:
        st.session_state.selected_state = "All States"
    if "selected_state_code" not in st.session_state:
        st.session_state.selected_state_code = None
    if "census_overlay_option" not in st.session_state:
        st.session_state.census_overlay_option = "None"
    if "show_map" not in st.session_state:
        st.session_state.show_map = False
    if "show_drilldown" not in st.session_state:
        st.session_state.show_drilldown = False

    # Render map view directly (no selection phase needed)
    render_map_view(stores_df, main_categories, main_to_sub, sub_to_detailed, census_gdf)


if __name__ == "__main__":
    main()
