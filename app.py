"""
Store Distribution Analysis Dashboard - Map Visualization with Folium

Run with:
    streamlit run app.py
"""

import json
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from pathlib import Path


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


def create_folium_map(stores_df: pd.DataFrame):
    """Create a Folium map with store markers."""

    # Calculate center and bounds
    center_lat = stores_df["latitude"].mean()
    center_lon = stores_df["longitude"].mean()

    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles="cartodbpositron",
    )

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
            "maxClusterRadius": 50,
            "disableClusteringAtZoom": 12,
            "spiderfyOnMaxZoom": False,
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

    # Fit bounds to data
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


def render_map_view(stores_df):
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

        if st.button("Restart Analysis", type="secondary", use_container_width=True):
            st.session_state.analysis_phase = "selection"
            st.session_state.selected_state_code = None
            st.session_state.map_bounds = None
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

    st.markdown("Hover over stores to see details. Zoom and pan to update statistics.")

    # Create and display the map
    m = create_folium_map(view_df)

    # Display map and capture viewport changes
    map_data = st_folium(
        m,
        width=None,
        height=600,
        returned_objects=["bounds"],
        key="store_map",
    )

    # Get current bounds from map interaction
    current_bounds = map_data.get("bounds") if map_data else None

    # Filter stores by current viewport bounds and calculate stats
    if current_bounds:
        viewport_df = filter_by_bounds(view_df, current_bounds)
    else:
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
        col1.metric("Stores in View", f"{results['total']:,}")
        col2.metric("With Focus Brand", f"{results['with_focus']:,}")
        col3.metric("Competitor Only", f"{results['competitor_only']:,}")
        col4.metric("Whitespace", f"{results['neither']:,}")


def main():
    # Load data
    stores_df, brands, subcategories = load_data()

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

    # Route to appropriate view
    if st.session_state.analysis_phase == "selection":
        render_selection_screen(stores_df, brands, subcategories)
    else:
        render_map_view(stores_df)


if __name__ == "__main__":
    main()
