"""Reusable SQLite analytics, customer modelling, and experimentation for Olist."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

DATABASE_PATH = Path(__file__).resolve().with_name("olist_analytics.db")


def query(sql: str, params: tuple[Any, ...] = (), db_path: Path = DATABASE_PATH) -> pd.DataFrame:
    """Run a parameterized read query against the local SQLite warehouse."""
    path = db_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Analytics database not found at {path}. Run python init_db.py first.")
    with sqlite3.connect(path, timeout=30) as connection:
        connection.row_factory = sqlite3.Row
        return pd.read_sql_query(sql, connection, params=params)


def database_summary() -> dict[str, int]:
    """Return core table counts used by the dashboard's source health panel."""
    tables = ("orders", "customers", "order_items", "order_payments", "order_reviews")
    return {table: int(query(f"SELECT COUNT(*) AS n FROM {table}").iloc[0]["n"]) for table in tables}


def monthly_metrics(start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    """Monthly delivered-order GMV, order count, AOV and windowed MoM growth."""
    return query(
        """WITH order_value AS (
            SELECT o.order_id, strftime('%Y-%m-01', o.order_purchase_timestamp) AS month,
                   SUM(i.price) AS gmv
            FROM orders o JOIN order_items i ON i.order_id = o.order_id
            WHERE o.order_status = 'delivered' AND o.order_purchase_timestamp IS NOT NULL
              AND (:start_date IS NULL OR date(o.order_purchase_timestamp) >= date(:start_date))
              AND (:end_date IS NULL OR date(o.order_purchase_timestamp) < date(:end_date, '+1 day'))
            GROUP BY o.order_id, strftime('%Y-%m-01', o.order_purchase_timestamp)
        ), monthly AS (
            SELECT month, SUM(gmv) AS gmv, COUNT(*) AS total_orders, AVG(gmv) AS aov
            FROM order_value GROUP BY month
        ), growth AS (
            SELECT *, LAG(gmv) OVER (ORDER BY month) AS prior_gmv FROM monthly
        )
        SELECT month, ROUND(gmv, 2) AS gmv, total_orders, ROUND(aov, 2) AS aov,
               ROUND(100.0 * (gmv - prior_gmv) / NULLIF(prior_gmv, 0), 2) AS mom_growth_pct
        FROM growth ORDER BY month""",
        {"start_date": start_date, "end_date": end_date},
    )


def delivery_funnel(start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    """Monthly order-to-delivery stage counts and elapsed-time measures."""
    return query(
        """SELECT strftime('%Y-%m-01', order_purchase_timestamp) AS month,
            COUNT(*) AS purchased_orders,
            SUM(CASE WHEN order_approved_at IS NOT NULL THEN 1 ELSE 0 END) AS approved_orders,
            SUM(CASE WHEN order_delivered_carrier_date IS NOT NULL THEN 1 ELSE 0 END) AS carrier_handoff_orders,
            SUM(CASE WHEN order_status = 'delivered' THEN 1 ELSE 0 END) AS delivered_orders,
            ROUND(AVG(CASE WHEN order_approved_at IS NOT NULL THEN
                (JULIANDAY(order_approved_at) - JULIANDAY(order_purchase_timestamp)) * 24 END), 2) AS approval_hours,
            ROUND(AVG(CASE WHEN order_delivered_customer_date IS NOT NULL AND order_delivered_carrier_date IS NOT NULL THEN
                JULIANDAY(order_delivered_customer_date) - JULIANDAY(order_delivered_carrier_date) END), 2) AS transit_days,
            ROUND(AVG(CASE WHEN order_delivered_customer_date IS NOT NULL THEN
                JULIANDAY(order_delivered_customer_date) - JULIANDAY(order_purchase_timestamp) END), 2) AS end_to_end_days
        FROM orders WHERE order_purchase_timestamp IS NOT NULL
          AND (:start_date IS NULL OR date(order_purchase_timestamp) >= date(:start_date))
          AND (:end_date IS NULL OR date(order_purchase_timestamp) < date(:end_date, '+1 day'))
        GROUP BY month ORDER BY month""",
        {"start_date": start_date, "end_date": end_date},
    )


def payment_mix() -> pd.DataFrame:
    """Payment transaction counts, monetary share and average installments."""
    return query(
        """WITH payment_totals AS (
            SELECT payment_type, COUNT(*) AS payment_transactions,
                   SUM(payment_value) AS payment_value, AVG(payment_installments) AS avg_installments
            FROM order_payments GROUP BY payment_type
        )
        SELECT payment_type, payment_transactions, ROUND(payment_value, 2) AS payment_value,
               ROUND(100.0 * payment_transactions / NULLIF(SUM(payment_transactions) OVER (), 0), 2) AS volume_share_pct,
               ROUND(100.0 * payment_value / NULLIF(SUM(payment_value) OVER (), 0), 2) AS value_share_pct,
               ROUND(avg_installments, 2) AS avg_installments
        FROM payment_totals ORDER BY payment_transactions DESC""")


def _delivery_frame() -> pd.DataFrame:
    data = query(
        """SELECT o.order_id, c.customer_state AS state, c.customer_zip_code_prefix AS zip_prefix,
                  o.order_purchase_timestamp, o.order_delivered_customer_date,
                  o.order_estimated_delivery_date
           FROM orders o JOIN customers c ON c.customer_id = o.customer_id
           WHERE o.order_status = 'delivered' AND o.order_delivered_customer_date IS NOT NULL
             AND o.order_estimated_delivery_date IS NOT NULL""")
    for column in ("order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"):
        data[column] = pd.to_datetime(data[column], errors="coerce")
    data = data.dropna(subset=["state", "order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"])
    data["delay_days"] = (data["order_delivered_customer_date"] - data["order_estimated_delivery_date"]).dt.total_seconds() / 86400
    data["delivery_days"] = (data["order_delivered_customer_date"] - data["order_purchase_timestamp"]).dt.total_seconds() / 86400
    return data


def regional_sla() -> pd.DataFrame:
    """Delivery SLA breach volume and durations by customer destination state."""
    data = _delivery_frame().assign(is_late=lambda frame: frame["delay_days"] > 0)
    result = data.groupby("state", as_index=False).agg(
        delivered_orders=("order_id", "nunique"),
        late_orders=("is_late", "sum"),
        sla_breach_rate_pct=("is_late", lambda values: 100 * values.mean()),
        avg_delay_days=("delay_days", lambda values: values[values > 0].mean()),
        median_delay_days=("delay_days", "median"),
        avg_delivery_days=("delivery_days", "mean"),
    )
    return result.replace([np.inf, -np.inf], np.nan).fillna({"avg_delay_days": 0}).round(2).sort_values("sla_breach_rate_pct", ascending=False)


def geospatial_hotspots() -> pd.DataFrame:
    """Approximate state centroids from ZIP-prefix delivery and geolocation matches."""
    geo = query(
        """SELECT geolocation_zip_code_prefix AS zip_prefix,
                  AVG(geolocation_lat) AS latitude, AVG(geolocation_lng) AS longitude
           FROM geolocation WHERE geolocation_lat BETWEEN -35 AND 6
             AND geolocation_lng BETWEEN -75 AND -30
           GROUP BY geolocation_zip_code_prefix""")
    delivery = _delivery_frame().merge(geo, on="zip_prefix", how="inner")
    centers = delivery.groupby("state", as_index=False).agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"))
    return regional_sla().merge(centers, on="state", how="left").dropna(subset=["latitude", "longitude"])


def cohort_retention(max_months: int = 12) -> pd.DataFrame:
    """Unique-customer monthly cohort retention indexed to the first delivered order."""
    if not 1 <= max_months <= 36:
        raise ValueError("max_months must be between 1 and 36.")
    data = query(
        """SELECT c.customer_unique_id, o.order_purchase_timestamp
           FROM orders o JOIN customers c ON c.customer_id = o.customer_id
           WHERE o.order_status = 'delivered' AND o.order_purchase_timestamp IS NOT NULL
             AND c.customer_unique_id IS NOT NULL""")
    if data.empty:
        return pd.DataFrame(columns=["cohort_month", "cohort_index", "active_customers", "cohort_size", "retention_pct"])
    data["month"] = pd.to_datetime(data["order_purchase_timestamp"], errors="coerce").dt.to_period("M")
    data = data.dropna(subset=["month"]).drop_duplicates(["customer_unique_id", "month"])
    data["cohort_month"] = data.groupby("customer_unique_id")["month"].transform("min")
    data["cohort_index"] = (data["month"].dt.year - data["cohort_month"].dt.year) * 12 + data["month"].dt.month - data["cohort_month"].dt.month
    data = data[data["cohort_index"].between(0, max_months - 1)]
    grouped = data.groupby(["cohort_month", "cohort_index"])["customer_unique_id"].nunique().rename("active_customers").reset_index()
    sizes = grouped[grouped["cohort_index"] == 0][["cohort_month", "active_customers"]].rename(columns={"active_customers": "cohort_size"})
    grouped = grouped.merge(sizes, on="cohort_month", how="left")
    grouped["cohort_month"] = grouped["cohort_month"].astype(str)
    grouped["retention_pct"] = (100 * grouped["active_customers"] / grouped["cohort_size"]).round(2)
    return grouped


def rfm_segments() -> pd.DataFrame:
    """Build customer-level RFM metrics and robust quartile-based segments."""
    data = query(
        """SELECT o.order_id, c.customer_unique_id, o.order_purchase_timestamp,
                  SUM(i.price + i.freight_value) AS order_value
           FROM orders o JOIN customers c ON c.customer_id = o.customer_id
           JOIN order_items i ON i.order_id = o.order_id
           WHERE o.order_status = 'delivered' AND c.customer_unique_id IS NOT NULL
             AND o.order_purchase_timestamp IS NOT NULL
           GROUP BY o.order_id, c.customer_unique_id, o.order_purchase_timestamp""")
    if data.empty:
        return pd.DataFrame(columns=["customer_unique_id", "recency", "frequency", "monetary", "r_score", "f_score", "m_score", "segment"])
    data["order_purchase_timestamp"] = pd.to_datetime(data["order_purchase_timestamp"], errors="coerce")
    data = data.dropna(subset=["order_purchase_timestamp", "order_value"])
    reference_date = data["order_purchase_timestamp"].max().normalize() + pd.Timedelta(days=1)
    rfm = data.groupby("customer_unique_id", as_index=False).agg(
        recency=("order_purchase_timestamp", lambda values: int((reference_date - values.max()).days)),
        frequency=("order_id", "nunique"), monetary=("order_value", "sum"),
    )
    # Rank first prevents qcut failures when metrics contain many tied values.
    for metric, score, reverse in (("recency", "r_score", True), ("frequency", "f_score", False), ("monetary", "m_score", False)):
        rank = rfm[metric].rank(method="first")
        bucket = np.ceil(rank / len(rfm) * 4).clip(1, 4).astype(int)
        rfm[score] = (5 - bucket if reverse else bucket).astype(int)
    rfm["segment"] = np.select(
        [rfm["r_score"].ge(3) & rfm["f_score"].ge(3) & rfm["m_score"].ge(3),
         rfm["r_score"].ge(3) & rfm["f_score"].ge(2) & rfm["m_score"].ge(2),
         rfm["r_score"].le(2) & (rfm["f_score"].ge(3) | rfm["m_score"].ge(3)),
         rfm["r_score"].eq(1) & rfm["f_score"].le(2)],
        ["Champions", "High-Value Loyal", "At Risk", "Lost"], default="Standard")
    return rfm.sort_values(["monetary", "frequency"], ascending=False).reset_index(drop=True)


def _model_data() -> pd.DataFrame:
    """Create one row per delivered order, avoiding payment/review join fanout."""
    data = query(
        """WITH payments AS (SELECT order_id, SUM(payment_value) AS payment_value FROM order_payments GROUP BY order_id),
             reviews AS (SELECT order_id, AVG(review_score) AS review_score FROM order_reviews GROUP BY order_id)
           SELECT o.order_id, o.order_purchase_timestamp, o.order_approved_at,
                  o.order_delivered_customer_date, o.order_estimated_delivery_date,
                                    p.payment_value, r.review_score, i.freight_value
           FROM orders o JOIN payments p ON p.order_id = o.order_id
           JOIN reviews r ON r.order_id = o.order_id
                     JOIN (SELECT order_id, SUM(freight_value) AS freight_value FROM order_items GROUP BY order_id) i
                         ON i.order_id = o.order_id
           WHERE o.order_status = 'delivered' AND o.order_approved_at IS NOT NULL
             AND o.order_delivered_customer_date IS NOT NULL AND o.order_estimated_delivery_date IS NOT NULL""")
    date_columns = ("order_purchase_timestamp", "order_approved_at", "order_delivered_customer_date", "order_estimated_delivery_date")
    for column in date_columns:
        data[column] = pd.to_datetime(data[column], errors="coerce")
    data = data.dropna(subset=[*date_columns, "payment_value", "review_score"])
    data["delay_days"] = (data["order_delivered_customer_date"] - data["order_estimated_delivery_date"]).dt.total_seconds() / 86400
    data["approval_hours"] = (data["order_approved_at"] - data["order_purchase_timestamp"]).dt.total_seconds() / 3600
    return data.replace([np.inf, -np.inf], np.nan).dropna(subset=["delay_days", "approval_hours", "payment_value"])


def regression_results(sample_limit: int = 50_000) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit robust OLS and logistic regression models on reproducible order samples."""
    if sample_limit < 100:
        raise ValueError("sample_limit must be at least 100.")
    data = _model_data()
    if len(data) > sample_limit:
        data = data.sample(n=sample_limit, random_state=42)
    if len(data) < 30 or data["review_score"].nunique() < 2:
        raise ValueError("Insufficient outcome variation or observations for regression models.")
    ols = sm.OLS(data["delay_days"], sm.add_constant(data[["approval_hours", "freight_value"]])).fit(cov_type="HC3")
    model = data.assign(low_review=(data["review_score"] <= 2).astype(int), is_late=(data["delay_days"] > 0).astype(int))
    if model["low_review"].nunique() < 2:
        raise ValueError("Logistic regression requires both low-review and other-review observations.")
    logit = sm.Logit(model["low_review"], sm.add_constant(model[["is_late", "payment_value"]])).fit(disp=False, maxiter=200)

    def tidy(fit: Any, odds_ratio: bool = False) -> pd.DataFrame:
        confidence = fit.conf_int()
        result = pd.DataFrame({"term": fit.params.index, "coefficient": fit.params.values,
                               "p_value": fit.pvalues.values, "ci_low": confidence.iloc[:, 0].values,
                               "ci_high": confidence.iloc[:, 1].values})
        if odds_ratio:
            result["odds_ratio"] = np.exp(np.clip(result["coefficient"], -30, 30))
            result["odds_ci_low"] = np.exp(np.clip(result["ci_low"], -30, 30))
            result["odds_ci_high"] = np.exp(np.clip(result["ci_high"], -30, 30))
        return result.round(4)

    metadata = {"observations": len(data), "ols_r_squared": float(ols.rsquared),
                "logit_pseudo_r_squared": float(logit.prsquared), "low_review_rate": float(model["low_review"].mean())}
    return tidy(ols), tidy(logit, odds_ratio=True), metadata


@dataclass(frozen=True)
class ABTestResult:
    control_rate: float
    treatment_rate: float
    absolute_lift: float
    relative_lift: float
    required_sample_per_variant: int
    z_statistic: float
    p_value: float
    control_ci: tuple[float, float]
    treatment_ci: tuple[float, float]
    lift_ci: tuple[float, float]
    decision: str


def ab_test(control_visitors: int, control_conversions: int, treatment_visitors: int,
            treatment_conversions: int, mde: float, alpha: float = 0.05,
            power: float = 0.8, baseline_rate: float | None = None) -> ABTestResult:
    """Calculate a two-sided two-proportion test and independent Wilson intervals."""
    if min(control_visitors, treatment_visitors) < 1:
        raise ValueError("Each variant must have at least one visitor.")
    if min(control_conversions, treatment_conversions) < 0 or control_conversions > control_visitors or treatment_conversions > treatment_visitors:
        raise ValueError("Conversions must be non-negative and cannot exceed visitors.")
    if not 0 < mde < 1 or not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("MDE, alpha and power must each be between 0 and 1.")
    control_rate = control_conversions / control_visitors
    treatment_rate = treatment_conversions / treatment_visitors
    p0 = control_rate if baseline_rate is None else baseline_rate
    if not 0 < p0 < 1 or p0 + mde >= 1:
        raise ValueError("Baseline rate must be between zero and one and baseline + MDE must be below one.")
    p1 = p0 + mde
    pooled_design = (p0 + p1) / 2
    numerator = norm.ppf(1 - alpha / 2) * np.sqrt(2 * pooled_design * (1 - pooled_design)) + norm.ppf(power) * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
    required = int(np.ceil(numerator**2 / mde**2))
    z_stat, p_value = proportions_ztest([treatment_conversions, control_conversions], [treatment_visitors, control_visitors])
    control_ci = proportion_confint(control_conversions, control_visitors, alpha=alpha, method="wilson")
    treatment_ci = proportion_confint(treatment_conversions, treatment_visitors, alpha=alpha, method="wilson")
    lift = treatment_rate - control_rate
    lift_se = np.sqrt(control_rate * (1 - control_rate) / control_visitors + treatment_rate * (1 - treatment_rate) / treatment_visitors)
    lift_margin = norm.ppf(1 - alpha / 2) * lift_se
    decision = "LAUNCH" if p_value < alpha and lift > 0 and min(control_visitors, treatment_visitors) >= required else "NO-LAUNCH"
    return ABTestResult(control_rate, treatment_rate, lift, lift / control_rate if control_rate else float("nan"),
                        required, float(z_stat), float(p_value), tuple(map(float, control_ci)),
                        tuple(map(float, treatment_ci)), (float(lift - lift_margin), float(lift + lift_margin)), decision)
