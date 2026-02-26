"""
================================================================================
STREAMLIT DASHBOARD — FX Customer Acquisition & Foreign Tourist Targeting
================================================================================
Launch:  streamlit run app.py
================================================================================
"""

import io
import os
from datetime import datetime

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from folium.plugins import HeatMap, MarkerCluster
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FX Tourist Targeting — Paris Airbnb",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "processed_fx_target_data.csv")

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem; border-radius: 10px; margin-bottom: 1.5rem; color: #fff;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; color: #e2e2e2; }
    .main-header p  { margin: 0.3rem 0 0 0; font-size: 0.95rem; color: #a8b2d1; }
    .kpi-card {
        background: #fff; border: 1px solid #e0e0e0; border-radius: 10px;
        padding: 1.2rem 1.5rem; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .kpi-card .kpi-value { font-size: 2rem; font-weight: 800; color: #0f3460; margin: 0; }
    .kpi-card .kpi-label { font-size: 0.85rem; color: #555; margin: 0.3rem 0 0 0; }
    .section-title {
        font-size: 1.15rem; font-weight: 700; color: #0f3460;
        border-left: 4px solid #e94560; padding-left: 0.7rem;
        margin: 1.5rem 0 0.8rem 0;
    }
    [data-testid="stSidebar"] { background: #f7f9fc; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading processed data …")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["latitude", "longitude", "price",
                "foreign_tourist_ratio_pct", "occupancy_rate_pct",
                "total_reviews", "foreign_reviews"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# MAP DATA HELPERS (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def prep_map_data(df: pd.DataFrame, max_pts: int) -> pd.DataFrame:
    """Subset for map rendering, prioritise by foreign_reviews volume."""
    dm = df.dropna(subset=["latitude", "longitude"]).copy()
    dm["foreign_reviews"] = dm["foreign_reviews"].fillna(0)
    if len(dm) > max_pts:
        dm = dm.nlargest(max_pts, "foreign_reviews")
    return dm


@st.cache_data(show_spinner=False)
def prep_heatmap_data(df: pd.DataFrame) -> list:
    """[lat, lon, weight] list — weight = foreign_reviews (volume-based)."""
    h = df[["latitude", "longitude", "foreign_reviews"]].dropna()
    h = h[h["foreign_reviews"] > 0]
    return h.values.tolist()


@st.cache_data(show_spinner=False)
def prep_hotspot_grid(df: pd.DataFrame, grid_deg: float = 0.005) -> pd.DataFrame:
    """Bin listings into ~500 m grid cells and aggregate FX signals."""
    dg = df.dropna(subset=["latitude", "longitude"]).copy()
    dg["lat_bin"] = (dg["latitude"]  / grid_deg).round() * grid_deg
    dg["lon_bin"] = (dg["longitude"] / grid_deg).round() * grid_deg
    return (
        dg.groupby(["lat_bin", "lon_bin"])
        .agg(sum_fx=("foreign_reviews", "sum"),
             mean_occ=("occupancy_rate_pct", "mean"),
             n=("id", "count"))
        .reset_index()
    )


# ---------------------------------------------------------------------------
# COLOUR HELPERS
# ---------------------------------------------------------------------------
def _ratio_color(ratio: float) -> str:
    if ratio >= 75: return "#e63946"
    if ratio >= 50: return "#f4a261"
    if ratio >= 25: return "#2a9d8f"
    return "#457b9d"


def _occ_color(occ) -> str:
    if pd.isna(occ):  return "#aaaaaa"
    if occ >= 70:     return "#2dc653"
    if occ >= 40:     return "#f4a261"
    return "#e63946"


# ---------------------------------------------------------------------------
# EXPORT HELPERS
# ---------------------------------------------------------------------------
def _build_filter_summary(neighbourhoods, min_foreign_ratio, price_range,
                           min_occupancy, show_clusters, show_heatmap,
                           show_hotspots, n_listings) -> str:
    nb = (", ".join(neighbourhoods)
          if len(neighbourhoods) <= 4
          else f"{len(neighbourhoods)} neighbourhoods")
    pr = f"€{price_range[0]} – €{price_range[1]}" if price_range else "all"
    layers = " | ".join(
        x for x, flag in [("Clusters", show_clusters),
                           ("Heatmap", show_heatmap),
                           ("Hotspots", show_hotspots)] if flag
    ) or "none"
    return (
        f"Neighbourhoods : {nb}\n"
        f"Min Foreign Ratio : {min_foreign_ratio}%\n"
        f"Price Range : {pr}\n"
        f"Min Occupancy : {min_occupancy}%\n"
        f"Map layers : {layers}\n"
        f"Listings shown : {n_listings:,}"
    )


def _scatter_to_ax(ax, df_map: pd.DataFrame) -> None:
    """Draw a foreign-ratio scatter plot on a matplotlib Axes."""
    fr = df_map["foreign_tourist_ratio_pct"].fillna(0)
    bands = [
        (fr >= 75,               "#e63946", "≥ 75%"),
        ((fr >= 50) & (fr < 75), "#f4a261", "50–74%"),
        ((fr >= 25) & (fr < 50), "#2a9d8f", "25–49%"),
        (fr <  25,               "#457b9d", "< 25%"),
    ]
    for mask, color, label in bands:
        sub = df_map[mask]
        if not sub.empty:
            ax.scatter(sub["longitude"], sub["latitude"],
                       c=color, s=5, alpha=0.45, label=label,
                       linewidths=0, rasterized=True)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude",  fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(title="Foreign Ratio", fontsize=8, title_fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_facecolor("#f0f0f0")


def _make_export_fig(df_map: pd.DataFrame, filter_summary: str,
                     date_str: str, figsize=(11, 9)):
    fig = plt.figure(figsize=figsize, facecolor="white")
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[1, 4],
                   hspace=0.25, top=0.96, bottom=0.06, left=0.08, right=0.95)

    ax_txt = fig.add_subplot(gs[0])
    ax_txt.axis("off")
    ax_txt.text(0.5, 1.0, "FX Customer Acquisition — Paris Airbnb Hotspots",
                ha="center", va="top", fontsize=13, fontweight="bold",
                color="#0f3460", transform=ax_txt.transAxes)
    ax_txt.text(0.0, 0.72,
                f"Generated: {date_str}\n\n{filter_summary}",
                ha="left", va="top", fontsize=8, color="#333",
                transform=ax_txt.transAxes, linespacing=1.55)

    ax_map = fig.add_subplot(gs[1])
    _scatter_to_ax(ax_map, df_map)
    ax_map.set_title(f"Scatter map — {len(df_map):,} listings",
                     fontsize=10, fontweight="bold", color="#0f3460")
    return fig


@st.cache_data(show_spinner=False, max_entries=4)
def generate_png_bytes(df_map: pd.DataFrame, filter_summary: str,
                        date_str: str) -> bytes:
    fig = _make_export_fig(df_map, filter_summary, date_str, figsize=(11, 9))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


@st.cache_data(show_spinner=False, max_entries=4)
def generate_pdf_bytes(df_map: pd.DataFrame, df_top20: pd.DataFrame,
                        filter_summary: str, date_str: str) -> bytes:
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Page 1 — map
        fig1 = _make_export_fig(df_map, filter_summary, date_str, figsize=(11, 8.5))
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        # Page 2 — Top-20 table
        if not df_top20.empty:
            fig2, ax2 = plt.subplots(figsize=(11, 8.5))
            fig2.patch.set_facecolor("white")
            ax2.axis("off")
            ax2.set_title("Top 20 — Highest Foreign Review Volume",
                          fontsize=12, fontweight="bold", color="#0f3460",
                          loc="left", pad=12)

            display_cols = [c for c in [
                "id", "neighbourhood_cleansed", "price",
                "occupancy_rate_pct", "total_reviews",
                "foreign_reviews", "foreign_tourist_ratio_pct",
            ] if c in df_top20.columns]

            col_labels = {
                "id": "ID", "neighbourhood_cleansed": "Neighbourhood",
                "price": "Price (€)", "occupancy_rate_pct": "Occ %",
                "total_reviews": "Reviews", "foreign_reviews": "FX Rev",
                "foreign_tourist_ratio_pct": "FX %",
            }

            tdata = df_top20[display_cols].copy()
            for col in ["price", "occupancy_rate_pct", "foreign_tourist_ratio_pct"]:
                if col in tdata.columns:
                    tdata[col] = tdata[col].apply(
                        lambda x: f"{x:.1f}" if pd.notna(x) else "—")
            for col in ["total_reviews", "foreign_reviews"]:
                if col in tdata.columns:
                    tdata[col] = tdata[col].apply(
                        lambda x: f"{int(x):,}" if pd.notna(x) else "—")

            tbl = ax2.table(
                cellText=tdata.values,
                colLabels=[col_labels.get(c, c) for c in display_cols],
                loc="center", cellLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7.5)
            tbl.scale(1, 1.6)

            for j in range(len(display_cols)):
                tbl[0, j].set_facecolor("#0f3460")
                tbl[0, j].set_text_props(color="white", fontweight="bold")
            for i in range(1, len(df_top20) + 1):
                bg = "#f7f9fc" if i % 2 == 0 else "white"
                for j in range(len(display_cols)):
                    tbl[i, j].set_facecolor(bg)

            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    st.markdown(
        """
        <div class="main-header">
            <h1>FX Customer Acquisition — Foreign Tourist Targeting</h1>
            <p>Paris Airbnb Intelligence &nbsp;|&nbsp; Identify high-density
            foreign-tourist neighbourhoods for field &amp; sales operations</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not os.path.isfile(DATA_FILE):
        st.error(
            f"**Data file not found:** `{DATA_FILE}`\n\n"
            "Run `python data_prep.py` first."
        )
        st.stop()

    try:
        df_raw = load_data(DATA_FILE)
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

    # =========================================================================
    # SIDEBAR — Filters
    # =========================================================================
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/"
        "La_Tour_Eiffel_vue_de_la_Tour_Saint-Jacques%2C_Paris_ao%C3%BBt_2014_%282%29.jpg/"
        "220px-La_Tour_Eiffel_vue_de_la_Tour_Saint-Jacques%2C_Paris_ao%C3%BBt_2014_%282%29.jpg",
        use_container_width=True,
    )
    st.sidebar.markdown("### Filters")

    all_neighbourhoods = sorted(df_raw["neighbourhood_cleansed"].dropna().unique().tolist())
    selected_neighbourhoods = st.sidebar.multiselect(
        "Neighbourhood(s)", options=all_neighbourhoods, default=all_neighbourhoods,
        help="Select one or more Paris arrondissement-level neighbourhoods.",
    )

    min_foreign_ratio = st.sidebar.slider(
        "Min. Foreign Tourist Ratio (%)", 0, 100, 0, 5,
        help="Only show listings where at least this share of reviews is non-French.",
    )

    has_price = df_raw["price"].notna().any()
    if has_price:
        p_min = float(df_raw["price"].min())
        p_max = float(df_raw["price"].max())
        price_range = st.sidebar.slider(
            "Nightly Price Range (€)",
            int(p_min), int(p_max) + 1, (int(p_min), int(p_max) + 1), 10,
        )
    else:
        st.sidebar.info("Price data not available.")
        price_range = None

    min_occupancy = st.sidebar.slider(
        "Min. Occupancy Rate (%)", 0, 100, 0, 5,
        help="Filter by minimum calendar-based occupancy rate.",
    )

    # ---- Map Controls ----
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Map Controls")
    show_clusters = st.sidebar.checkbox("Show clustered markers", value=True)
    show_heatmap  = st.sidebar.checkbox("Show heatmap", value=True)
    show_hotspots = st.sidebar.checkbox("Show grid hotspots (~500 m)", value=False)
    max_points    = st.sidebar.slider("Max markers", 500, 10_000, 5_000, 500)

    # =========================================================================
    # APPLY FILTERS
    # =========================================================================
    df = df_raw.copy()
    if selected_neighbourhoods:
        df = df[df["neighbourhood_cleansed"].isin(selected_neighbourhoods)]
    df = df[df["foreign_tourist_ratio_pct"].fillna(0) >= min_foreign_ratio]
    if price_range is not None:
        df = df[
            ((df["price"] >= price_range[0]) & (df["price"] <= price_range[1]))
            | df["price"].isna()
        ]
    df = df[df["occupancy_rate_pct"].fillna(0) >= min_occupancy]

    if df.empty:
        st.warning("No listings match the current filters — please relax them.")
        st.stop()

    # ---- Prepare derived datasets ----
    df_map   = prep_map_data(df, max_points)
    df_top20 = (
        df.dropna(subset=["foreign_reviews"])
        .nlargest(20, "foreign_reviews")
        .reset_index(drop=True)
    )

    filter_summary = _build_filter_summary(
        selected_neighbourhoods, min_foreign_ratio, price_range,
        min_occupancy, show_clusters, show_heatmap, show_hotspots, len(df_map),
    )

    # ---- Export section ----
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Export")

    date_str  = datetime.now().strftime("%Y-%m-%d %H:%M")
    ts        = datetime.now().strftime("%Y%m%d_%H%M")

    png_bytes = generate_png_bytes(df_map, filter_summary, date_str)
    pdf_bytes = generate_pdf_bytes(df_map, df_top20, filter_summary, date_str)

    st.sidebar.download_button(
        "📸 Screenshot (PNG)", data=png_bytes,
        file_name=f"airbnb_fx_screenshot_{ts}.png",
        mime="image/png", use_container_width=True,
    )
    st.sidebar.download_button(
        "⬇️ Download report (PDF)", data=pdf_bytes,
        file_name=f"airbnb_fx_report_{ts}.pdf",
        mime="application/pdf", use_container_width=True,
    )

    # =========================================================================
    # KPI CARDS
    # =========================================================================
    st.markdown('<div class="section-title">Key Performance Indicators</div>',
                unsafe_allow_html=True)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    def _kpi(value, label):
        return (f'<div class="kpi-card">'
                f'<p class="kpi-value">{value}</p>'
                f'<p class="kpi-label">{label}</p></div>')

    avg_foreign  = df["foreign_tourist_ratio_pct"].mean()
    avg_price    = df["price"].mean() if has_price else None
    price_disp   = f"€{avg_price:,.0f}" if avg_price is not None and not np.isnan(avg_price) else "N/A"
    avg_occ      = df["occupancy_rate_pct"].mean()

    with kpi1: st.markdown(_kpi(f"{len(df):,}", "FX Targeted Listings"),      unsafe_allow_html=True)
    with kpi2: st.markdown(_kpi(f"{avg_foreign:.1f}%", "Avg Foreign Tourist Ratio"), unsafe_allow_html=True)
    with kpi3: st.markdown(_kpi(price_disp, "Avg Nightly Price"),              unsafe_allow_html=True)
    with kpi4: st.markdown(_kpi(f"{avg_occ:.1f}%", "Avg Occupancy Rate"),     unsafe_allow_html=True)

    # =========================================================================
    # INTERACTIVE MAP
    # =========================================================================
    st.markdown('<div class="section-title">Foreign Tourist Hotspot Map</div>',
                unsafe_allow_html=True)
    st.caption(
        f"Showing **{len(df_map):,}** of {len(df):,} filtered listings "
        f"(sorted by foreign review volume). "
        f"Adjust *Max markers* in the sidebar for performance."
    )

    m = folium.Map(
        location=[df_map["latitude"].mean(), df_map["longitude"].mean()],
        zoom_start=13,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # ---- Clustered markers ----
    if show_clusters and not df_map.empty:
        cluster = MarkerCluster(
            name="Listings",
            options={"maxClusterRadius": 50, "disableClusteringAtZoom": 17},
        )
        for _, row in df_map.iterrows():
            ratio   = float(row.get("foreign_tourist_ratio_pct", 0) or 0)
            occ     = float(row.get("occupancy_rate_pct", 0) or 0)
            fr_rev  = int(row.get("foreign_reviews", 0) or 0)
            tot_rev = int(row.get("total_reviews", 0) or 0)
            lang    = str(row.get("dominant_foreign_lang", "n/a") or "n/a")
            pval    = row.get("price", None)
            pstr    = f"€{pval:,.0f}" if pd.notna(pval) else "N/A"
            color   = _ratio_color(ratio)

            popup_html = (
                f"<div style='width:215px;font-size:12px;line-height:1.65'>"
                f"<b>ID:</b> {row['id']}<br>"
                f"<b>Neighbourhood:</b> {row['neighbourhood_cleansed']}<br>"
                f"<b>Price:</b> {pstr}<br>"
                f"<b>Occupancy:</b> {occ:.1f}%<br>"
                f"<b>Total Reviews:</b> {tot_rev:,}<br>"
                f"<b>Foreign Reviews:</b> {fr_rev:,}<br>"
                f"<b>Foreign Ratio:</b> {ratio:.1f}%<br>"
                f"<b>Top Foreign Lang:</b> {lang}"
                f"</div>"
            )
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=5, color=color,
                fill=True, fill_color=color, fill_opacity=0.75,
                popup=folium.Popup(popup_html, max_width=230),
                tooltip=f"{row['neighbourhood_cleansed']} | {ratio:.0f}% FX",
            ).add_to(cluster)

        cluster.add_to(m)

        legend_html = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
             background:white;padding:10px 14px;border-radius:8px;
             border:1px solid #ccc;font-size:11px;line-height:1.7">
            <b>Foreign Ratio</b><br>
            <span style="color:#e63946">&#9679;</span> &ge;75%<br>
            <span style="color:#f4a261">&#9679;</span> 50–74%<br>
            <span style="color:#2a9d8f">&#9679;</span> 25–49%<br>
            <span style="color:#457b9d">&#9679;</span> &lt;25%
        </div>"""
        m.get_root().html.add_child(folium.Element(legend_html))

    # ---- Heatmap (weighted by foreign_reviews volume) ----
    if show_heatmap:
        heat_data = prep_heatmap_data(df_map)
        if heat_data:
            HeatMap(
                data=heat_data,
                name="Foreign density heatmap",
                radius=16, blur=22, max_zoom=17, min_opacity=0.3,
            ).add_to(m)

    # ---- Grid hotspots ----
    if show_hotspots:
        grid = prep_hotspot_grid(df_map)
        if not grid.empty:
            max_fx  = grid["sum_fx"].max() or 1
            fg_grid = folium.FeatureGroup(name="Grid hotspots (~500 m)")
            for _, g in grid.iterrows():
                radius = max(6, min(38, int(g["sum_fx"] / max_fx * 38)))
                c      = _occ_color(g["mean_occ"])
                folium.CircleMarker(
                    location=[g["lat_bin"], g["lon_bin"]],
                    radius=radius, color=c,
                    fill=True, fill_color=c, fill_opacity=0.5,
                    tooltip=(
                        f"FX reviews (sum): {int(g['sum_fx'])}<br>"
                        f"Avg occupancy: {g['mean_occ']:.1f}%<br>"
                        f"Listings in cell: {int(g['n'])}"
                    ),
                ).add_to(fg_grid)
            fg_grid.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width=None, height=560, key="main_map")

    # =========================================================================
    # CHARTS — unchanged
    # =========================================================================
    st.markdown(
        '<div class="section-title">Top 10 Neighbourhoods — Foreign Tourist Density</div>',
        unsafe_allow_html=True,
    )
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        neigh_stats = (
            df.groupby("neighbourhood_cleansed")
            .agg(avg_foreign_ratio=("foreign_tourist_ratio_pct", "mean"),
                 listing_count=("id", "count"))
            .reset_index()
            .sort_values("avg_foreign_ratio", ascending=False)
            .head(10)
        )
        fig1 = px.bar(
            neigh_stats, x="avg_foreign_ratio", y="neighbourhood_cleansed",
            orientation="h", text="avg_foreign_ratio",
            color="avg_foreign_ratio",
            color_continuous_scale=["#457b9d", "#e63946"],
            labels={"avg_foreign_ratio": "Avg Foreign Ratio (%)",
                    "neighbourhood_cleansed": "Neighbourhood"},
            title="Avg Foreign Tourist Ratio by Neighbourhood",
        )
        fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig1.update_layout(yaxis=dict(autorange="reversed"), height=420,
                           margin=dict(l=10, r=10, t=40, b=10),
                           coloraxis_showscale=False, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with chart_col2:
        neigh_count = (
            df[df["foreign_tourist_ratio_pct"] > 0]
            .groupby("neighbourhood_cleansed")
            .agg(foreign_listing_count=("id", "count"))
            .reset_index()
            .sort_values("foreign_listing_count", ascending=False)
            .head(10)
        )
        fig2 = px.bar(
            neigh_count, x="foreign_listing_count", y="neighbourhood_cleansed",
            orientation="h", text="foreign_listing_count",
            color="foreign_listing_count",
            color_continuous_scale=["#2a9d8f", "#0f3460"],
            labels={"foreign_listing_count": "Listings with Foreign Reviews",
                    "neighbourhood_cleansed": "Neighbourhood"},
            title="Total FX-Potential Listings by Neighbourhood",
        )
        fig2.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig2.update_layout(yaxis=dict(autorange="reversed"), height=420,
                           margin=dict(l=10, r=10, t=40, b=10),
                           coloraxis_showscale=False, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Dominant Foreign Languages (Top 10)</div>',
                unsafe_allow_html=True)
    if "dominant_foreign_lang" in df.columns:
        lang_counts = (
            df[df["dominant_foreign_lang"].notna() & (df["dominant_foreign_lang"] != "n/a")]
            ["dominant_foreign_lang"].value_counts().head(10).reset_index()
        )
        lang_counts.columns = ["Language", "Listings"]
        fig3 = px.bar(
            lang_counts, x="Listings", y="Language", orientation="h",
            text="Listings", color="Listings",
            color_continuous_scale=["#a8dadc", "#1d3557"],
            title="Most Common Dominant Foreign Language per Listing",
        )
        fig3.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig3.update_layout(yaxis=dict(autorange="reversed"), height=380,
                           margin=dict(l=10, r=10, t=40, b=10),
                           coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Language breakdown not available.")

    st.markdown('<div class="section-title">Detailed Listing Data</div>',
                unsafe_allow_html=True)
    with st.expander("Show / hide data table", expanded=False):
        display_cols = ["id", "neighbourhood_cleansed", "latitude", "longitude",
                        "foreign_tourist_ratio_pct", "occupancy_rate_pct",
                        "total_reviews", "foreign_reviews"]
        if has_price:
            display_cols.insert(4, "price")
        if "dominant_foreign_lang" in df.columns:
            display_cols.append("dominant_foreign_lang")
        avail = [c for c in display_cols if c in df.columns]
        st.dataframe(df[avail].reset_index(drop=True), use_container_width=True, height=400)

    st.markdown("---")
    st.caption(
        "Data source: Inside Airbnb (Paris) — processed with NLP language detection. "
        "Dashboard built for executive FX customer acquisition targeting."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.exception(e)
