"""Interactive Streamlit dashboard for Olist e-commerce and payments analytics."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import sqlite3

from analytics import (
    ab_test,
    cohort_retention,
    database_summary,
    delivery_funnel,
    geospatial_hotspots,
    monthly_metrics,
    payment_mix,
    regional_sla,
    regression_results,
    rfm_segments,
)
from init_db import DATABASE_PATH, download_and_build

st.set_page_config(page_title="Olist Product Analytics", page_icon="🛒", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {padding-top: 1.35rem; padding-bottom: 2.4rem; max-width: 1500px;}
.hero {padding: 1.4rem 1.7rem; border-radius: 18px; color: #fff;
       background: linear-gradient(110deg, #102a43 0%, #1565c0 58%, #37a6a0 100%); margin-bottom: 1.2rem;}
.hero h1 {margin: 0; font-size: 2rem; letter-spacing: -.03em;}
.hero p {margin: .5rem 0 0; color: #e4f2ff;}
[data-testid="stMetric"] {background: transparent; color: inherit;
                           border: 1px solid rgba(128, 128, 128, .28); border-radius: 14px; padding: 12px 15px;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(128, 128, 128, .22);}
</style>
<div class="hero"><h1>Olist Product Analytics</h1>
<p>Executive performance, customer value, delivery operations and evidence-led experimentation.</p></div>
""", unsafe_allow_html=True)

PAGES = ["Executive Summary", "Cohorts & RFM", "Regression Drivers", "A/B Testing Simulator"]
page = st.sidebar.radio("Navigate", PAGES, index=0)
st.sidebar.divider()
st.sidebar.caption("Source: Olist Brazilian E-Commerce public dataset")


@st.cache_data(ttl=900, show_spinner="Loading and analyzing the Olist warehouse…")
def cached_analytics(name: str) -> Any:
    """Cache read-heavy functions between interactive reruns."""
    loaders = {
        "summary": database_summary,
        "monthly": monthly_metrics,
        "funnel": delivery_funnel,
        "payments": payment_mix,
        "sla": regional_sla,
        "map": geospatial_hotspots,
        "cohort": lambda: cohort_retention(max_months=24),
        "rfm": rfm_segments,
        "regression": regression_results,
    }
    if name not in loaders:
        raise ValueError(f"Unknown analytics request: {name}")
    return loaders[name]()


@st.cache_resource(show_spinner="Preparing the Olist analytics warehouse on first launch…")
def ensure_warehouse() -> None:
    """Build the ignored local database automatically when deployed without it."""
    if not DATABASE_PATH.is_file():
        download_and_build(DATABASE_PATH)


def filter_months(frame: pd.DataFrame, label: str = "Analysis period") -> pd.DataFrame:
    """Render a date slider and filter monthly records inclusively."""
    result = frame.copy()
    result["month"] = pd.to_datetime(result["month"], errors="coerce")
    result = result.dropna(subset=["month"])
    if result.empty:
        return result
    first, last = result["month"].min().date(), result["month"].max().date()
    if first == last:
        return result
    selected = st.sidebar.slider(label, min_value=first, max_value=last, value=(first, last), format="MMM YYYY")
    return result[result["month"].dt.date.between(selected[0], selected[1])]


def money(value: float) -> str:
    return f"R$ {value:,.0f}"


try:
    try:
        ensure_warehouse()
    except Exception as error:
        st.error(f"Could not prepare the Olist data warehouse: {error}")
        st.info("The first launch downloads the public dataset through KaggleHub. Check that the deployment has outbound internet access and adequate disk space.")
        st.stop()
    source_summary = cached_analytics("summary")
    with st.sidebar.expander("Warehouse health", expanded=False):
        for table, count in source_summary.items():
            st.caption(f"{table.replace('_', ' ').title()}: {count:,} rows")

    if page == "Executive Summary":
        monthly = cached_analytics("monthly")
        if monthly.empty:
            st.warning("There are no delivered orders available for this period.")
            st.stop()
        monthly = filter_months(monthly, "Purchase month range")
        sla = cached_analytics("sla")
        payments = cached_analytics("payments")
        funnel = cached_analytics("funnel")
        hotspots = cached_analytics("map")
        available_states = sorted(sla["state"].dropna().unique().tolist())
        selected_states = st.sidebar.multiselect("Destination states", available_states, default=available_states)
        selected_sla = sla[sla["state"].isin(selected_states)]
        selected_hotspots = hotspots[hotspots["state"].isin(selected_states)]
        latest = monthly.sort_values("month").iloc[-1]
        weighted_breach = (selected_sla["late_orders"].sum() / selected_sla["delivered_orders"].sum() * 100
                           if not selected_sla.empty and selected_sla["delivered_orders"].sum() else 0.0)
        worst_state = selected_sla.iloc[0] if not selected_sla.empty else None

        kpis = st.columns(4)
        kpis[0].metric("Latest monthly GMV", money(float(latest["gmv"])),
                       f"{latest['mom_growth_pct']:.1f}% MoM" if pd.notna(latest["mom_growth_pct"]) else "First month")
        kpis[1].metric("Delivered orders", f"{int(latest['total_orders']):,}")
        kpis[2].metric("Average order value", f"R$ {latest['aov']:,.2f}")
        kpis[3].metric("SLA breach rate", f"{weighted_breach:.1f}%",
                       f"Highest: {worst_state['state']}" if worst_state is not None else "No selected states")

        left, right = st.columns((1.6, 1))
        with left:
            metric = st.radio("Commercial trend", ["GMV", "Orders", "AOV"], horizontal=True)
            column = {"GMV": "gmv", "Orders": "total_orders", "AOV": "aov"}[metric]
            fig = px.area(monthly, x="month", y=column, markers=True, title=f"Monthly {metric}",
                          color_discrete_sequence=["#1976d2"])
            fig.update_layout(hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            payment_metric = st.selectbox("Payment mix measure", ["Transaction volume", "Payment value", "Installments"])
            payment_column = {"Transaction volume": "payment_transactions", "Payment value": "payment_value",
                              "Installments": "avg_installments"}[payment_metric]
            st.plotly_chart(px.pie(payments, names="payment_type", values=payment_column, hole=.48,
                                   title=payment_metric, color_discrete_sequence=px.colors.qualitative.Safe),
                            use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.plotly_chart(px.bar(selected_sla.sort_values("sla_breach_rate_pct"), x="sla_breach_rate_pct", y="state",
                                   orientation="h", color="avg_delay_days", color_continuous_scale="Reds",
                                   hover_data=["delivered_orders", "late_orders", "avg_delivery_days"],
                                   title="Delivery SLA breaches by state"), use_container_width=True)
        with right:
            if selected_hotspots.empty:
                st.info("No geolocation matches for the selected states.")
            else:
                geo = px.scatter_geo(selected_hotspots, lat="latitude", lon="longitude", scope="south america",
                                     size="delivered_orders", color="sla_breach_rate_pct", hover_name="state",
                                     hover_data={"avg_delay_days": ":.2f", "delivered_orders": True,
                                                 "latitude": False, "longitude": False},
                                     color_continuous_scale="YlOrRd", title="Regional delivery hotspot map")
                geo.update_geos(showland=True, landcolor="#eef3f7", showcountries=True)
                st.plotly_chart(geo, use_container_width=True)

        with st.expander("Order-to-delivery funnel and bottlenecks"):
            funnel_metric = st.selectbox("Funnel volume or duration", ["Order volume", "Stage duration (days / hours)"])
            if funnel_metric == "Order volume":
                funnel_long = funnel.melt(id_vars="month", value_vars=["purchased_orders", "approved_orders", "carrier_handoff_orders", "delivered_orders"],
                                          var_name="stage", value_name="orders")
                funnel_fig = px.line(funnel_long, x="month", y="orders", color="stage", markers=True, title="Monthly order stage volumes")
            else:
                funnel_long = funnel.melt(id_vars="month", value_vars=["approval_hours", "transit_days", "end_to_end_days"],
                                          var_name="stage", value_name="duration")
                funnel_fig = px.line(funnel_long, x="month", y="duration", color="stage", markers=True,
                                     title="Average approval and delivery durations")
            st.plotly_chart(funnel_fig, use_container_width=True)
        if worst_state is not None:
            st.info(f"Operations insight: **{worst_state['state']}** has the highest state-level breach rate at **{worst_state['sla_breach_rate_pct']:.1f}%**.")
        with st.expander("Explore monthly KPI data"):
            st.dataframe(monthly, use_container_width=True, hide_index=True)

    elif page == "Cohorts & RFM":
        months_to_show = st.sidebar.slider("Cohort horizon (months)", min_value=1, max_value=24, value=12)
        cohorts = cached_analytics("cohort")
        cohorts = cohorts[cohorts["cohort_index"] < months_to_show]
        if cohorts.empty:
            st.info("No cohort observations are available.")
        else:
            retention = cohorts.pivot(index="cohort_month", columns="cohort_index", values="retention_pct")
            retention = retention.reindex(columns=range(months_to_show))
            fig = px.imshow(retention, aspect="auto", text_auto=".1f", color_continuous_scale="Blues",
                            labels={"x": "Months since first purchase", "y": "Acquisition cohort", "color": "Retained %"},
                            title="Monthly cohort retention (unique customers)")
            st.plotly_chart(fig, use_container_width=True)
        rfm = cached_analytics("rfm")
        segments = sorted(rfm["segment"].unique().tolist())
        selected_segments = st.multiselect("RFM segments", segments, default=segments)
        filtered_rfm = rfm[rfm["segment"].isin(selected_segments)]
        summary = filtered_rfm.groupby("segment", as_index=False).agg(
            customers=("customer_unique_id", "nunique"), revenue=("monetary", "sum"),
            avg_recency_days=("recency", "mean"), avg_frequency=("frequency", "mean"),
        )
        left, right = st.columns(2)
        with left:
            st.plotly_chart(px.bar(summary, x="segment", y="customers", color="revenue", title="Customer count and segment revenue",
                                   color_continuous_scale="Teal"), use_container_width=True)
        with right:
            st.plotly_chart(px.scatter(summary, x="avg_recency_days", y="revenue", size="customers", color="segment",
                                       hover_data=["avg_frequency"], title="Segment value and recency"), use_container_width=True)
        st.dataframe(summary.sort_values("revenue", ascending=False), use_container_width=True, hide_index=True)
        with st.expander("Customer-level RFM detail"):
            st.dataframe(filtered_rfm, use_container_width=True, hide_index=True)

    elif page == "Regression Drivers":
        sample_size = st.sidebar.slider("Maximum model observations", min_value=5_000, max_value=50_000, value=30_000, step=5_000)
        ols, logit, metadata = regression_results(sample_limit=sample_size)
        st.caption(f"Order-level observations: {metadata['observations']:,} · OLS R²: {metadata['ols_r_squared']:.3f} · Logistic pseudo-R²: {metadata['logit_pseudo_r_squared']:.3f} · Low-review prevalence: {metadata['low_review_rate']:.1%}")
        st.subheader("Delivery delay drivers — robust OLS")
        st.caption("Dependent variable: actual delivery date minus estimated delivery date, in days. Predictors: approval hours and total order freight value. Negative values indicate early delivery.")
        st.dataframe(ols, use_container_width=True, hide_index=True)
        st.subheader("Low-review drivers — logistic regression")
        st.caption("Outcome: order review score ≤ 2. Odds ratios above 1 indicate greater odds of a low review.")
        st.dataframe(logit, use_container_width=True, hide_index=True)
        late = logit[logit["term"] == "is_late"]
        if not late.empty:
            odds = float(late.iloc[0]["odds_ratio"])
            st.info(f"Modelled late-delivery odds ratio: **{odds:.2f}×** (95% CI {late.iloc[0]['odds_ci_low']:.2f}–{late.iloc[0]['odds_ci_high']:.2f}). Association is not causal evidence.")
        terms = ols[ols["term"] != "const"].copy()
        st.plotly_chart(px.bar(terms, x="term", y="coefficient", error_y=terms["ci_high"] - terms["coefficient"],
                               error_y_minus=terms["coefficient"] - terms["ci_low"], title="OLS coefficients and 95% confidence intervals",
                               color="coefficient", color_continuous_scale="RdBu"), use_container_width=True)

    else:
        st.subheader("Checkout offer experiment")
        st.write("Estimate required traffic, compare conversion proportions, and make a transparent launch recommendation.")
        with st.sidebar:
            st.markdown("#### Experiment assumptions")
            baseline = st.number_input("Expected baseline conversion rate", min_value=0.001, max_value=0.95, value=0.10, step=0.005, format="%.3f")
            mde_pct = st.number_input("Minimum detectable absolute lift (pp)", min_value=0.1, max_value=30.0, value=2.0, step=0.1)
            alpha = st.select_slider("Significance level", options=[0.01, 0.05, 0.10], value=0.05)
            power = st.select_slider("Statistical power", options=[0.70, 0.80, 0.90, 0.95], value=0.80)
            st.markdown("#### Observed experiment")
            control_visitors = st.number_input("Control visitors", min_value=1, value=10_000, step=500)
            control_conversions = st.number_input("Control conversions", min_value=0, value=1_000, step=50)
            treatment_visitors = st.number_input("Treatment visitors", min_value=1, value=10_000, step=500)
            treatment_conversions = st.number_input("Treatment conversions", min_value=0, value=1_120, step=50)
        try:
            result = ab_test(int(control_visitors), int(control_conversions), int(treatment_visitors), int(treatment_conversions),
                             mde_pct / 100, alpha, power, baseline_rate=baseline)
        except ValueError as error:
            st.error(str(error))
            st.stop()
        color = "#14804a" if result.decision == "LAUNCH" else "#b54708"
        st.markdown(f"### Recommendation: <span style='color:{color}'>{result.decision}</span>", unsafe_allow_html=True)
        kpis = st.columns(5)
        kpis[0].metric("Control conversion", f"{result.control_rate:.2%}")
        kpis[1].metric("Treatment conversion", f"{result.treatment_rate:.2%}", f"{result.absolute_lift:+.2%} absolute")
        kpis[2].metric("Relative lift", f"{result.relative_lift:+.1%}" if np.isfinite(result.relative_lift) else "—")
        kpis[3].metric("Two-sided p-value", f"{result.p_value:.4f}")
        kpis[4].metric("Required / variant", f"{result.required_sample_per_variant:,}")
        variants = pd.DataFrame({"variant": ["Control", "Treatment"],
                                 "rate": [result.control_rate, result.treatment_rate],
                                 "low": [result.control_ci[0], result.treatment_ci[0]],
                                 "high": [result.control_ci[1], result.treatment_ci[1]]})
        variants["error_plus"] = variants["high"] - variants["rate"]
        variants["error_minus"] = variants["rate"] - variants["low"]
        fig = go.Figure(go.Bar(x=variants["variant"], y=variants["rate"],
                               error_y=dict(type="data", symmetric=False, array=variants["error_plus"], arrayminus=variants["error_minus"]),
                               marker_color=["#607d8b", "#1976d2"], text=variants["rate"].map(lambda value: f"{value:.2%}"), textposition="outside"))
        fig.update_layout(title=f"Observed conversion and {(1-alpha):.0%} Wilson confidence intervals", yaxis_tickformat=".0%", yaxis_title="Conversion rate")
        st.plotly_chart(fig, use_container_width=True)
        lift_low, lift_high = result.lift_ci
        st.caption(f"Two-proportion z-statistic: {result.z_statistic:.3f}. Approximate {(1-alpha):.0%} CI for absolute lift: {lift_low:+.2%} to {lift_high:+.2%}. The launch rule requires positive observed lift, statistical significance, and target sample size in both groups.")

except (FileNotFoundError, sqlite3.Error, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
    st.error(f"Unable to load the requested analytics: {error}")
    st.info("Build or refresh the local database, then rerun the dashboard.")
    st.code("python init_db.py\nstreamlit run app.py")
