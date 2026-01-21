# pages/02_Forecast_Comparison.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
from sqlalchemy import text
from database_connection import get_database_connection
from sqlalchemy.exc import OperationalError, DBAPIError

st.set_page_config(page_title="Forecast Comparison", layout="wide")


REF_TABLE_NAME = "streamlit_forecast_comparison.ors_forecast"
DEMAND_TABLE_NAME = "streamlit_forecast_comparison.sku_week_demand"
CURRENT_FCST_TABLE_NAME = "streamlit_forecast_comparison.current_forecast"
horizon=12


# ---------------- DB ----------------
@st.cache_resource
def _get_engine():
    return get_database_connection()

def show_db_error_and_retry(err: Exception, *, key: str):
    st.error("Database connection was lost. Please reconnect and try again.")
    with st.expander("Details", expanded=False):
        st.code(str(err))

    if st.button("Reconnect & retry", key=key, use_container_width=True):
        # 1) chiudi pool attuale (se esiste)
        try:
            eng = _get_engine()
            eng.dispose()
        except Exception:
            pass

        # 2) forza ricreazione engine + ricarica cache dati
        _get_engine.clear()
        try:
            load_reference_table.clear()
        except Exception:
            pass
        try:
            load_demand_from_db.clear()
        except Exception:
            pass
        try:
            load_current_forecast_latest_from_db.clear()
        except Exception:
            pass

        st.rerun()

    st.stop()


@st.cache_data(ttl=300)
def load_reference_table() -> pd.DataFrame:
    engine = _get_engine()
    if engine is None:
        raise RuntimeError("Database connection is not configured or returned None.")

    q = f"""
        SELECT
            "FAMILY"         AS "FAMILY",
            "SKU ID"         AS "SKU ID",
            "week"           AS "Week",
            "ORS prediction" AS "Deda Forecast"
        FROM {REF_TABLE_NAME}
    """

    with engine.connect() as conn:
        try:
            df = pd.read_sql(text(q), conn)
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise

    df["SKU ID"] = df["SKU ID"].astype(str).str.strip()
    df["Week"] = pd.to_datetime(df["Week"], errors="coerce").dt.date
    df["Deda Forecast"] = pd.to_numeric(df["Deda Forecast"], errors="coerce")
    df = df.dropna(subset=["SKU ID", "Week"])
    return df


@st.cache_data(ttl=300)
def load_demand_from_db() -> pd.DataFrame:
    engine = _get_engine()
    if engine is None:
        return pd.DataFrame(columns=["SKU ID", "Week", "Demand"])

    q = f"""
        SELECT
            "SKU ID" AS "SKU ID",
            "Week"   AS "Week",
            "Demand" AS "Demand"
        FROM {DEMAND_TABLE_NAME}
    """

    with engine.connect() as conn:
        try:
            df = pd.read_sql(text(q), conn)
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            return pd.DataFrame(columns=["SKU ID", "Week", "Demand"])

    df["SKU ID"] = df["SKU ID"].astype(str).str.strip()
    df["Week"] = pd.to_datetime(df["Week"], errors="coerce").dt.date
    df["Demand"] = pd.to_numeric(df["Demand"], errors="coerce")
    df = df.dropna(subset=["SKU ID", "Week", "Demand"])
    df = df.groupby(["SKU ID", "Week"], as_index=False)["Demand"].mean()
    return df


@st.cache_data(ttl=300)
def load_current_forecast_latest_from_db() -> tuple[pd.DataFrame, datetime | None]:
    engine = _get_engine()
    if engine is None:
        return pd.DataFrame(columns=["SKU ID", "Week", "Current Forecast"]), None

    q = f"""
        SELECT
            "SKU ID"           AS "SKU ID",
            "Week"             AS "Week",
            "Current Forecast" AS "Current Forecast",
            "Date update"      AS "Date update"
        FROM {CURRENT_FCST_TABLE_NAME}
        WHERE "Date update" = (SELECT MAX("Date update") FROM {CURRENT_FCST_TABLE_NAME})
    """

    with engine.connect() as conn:
        try:
            df = pd.read_sql(text(q), conn)
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            return pd.DataFrame(columns=["SKU ID", "Week", "Current Forecast"]), None

    if df.empty:
        return pd.DataFrame(columns=["SKU ID", "Week", "Current Forecast"]), None

    latest_ts = pd.to_datetime(df["Date update"].max()).to_pydatetime()

    df["SKU ID"] = df["SKU ID"].astype(str).str.strip()
    df["Week"] = pd.to_datetime(df["Week"], errors="coerce").dt.date
    df["Current Forecast"] = pd.to_numeric(df["Current Forecast"], errors="coerce")
    df = df.dropna(subset=["SKU ID", "Week", "Current Forecast"])
    df = df.groupby(["SKU ID", "Week"], as_index=False)["Current Forecast"].mean()
    return df, latest_ts


# ---------------- Metrics ----------------
def compute_week_ahead_table(df_ref: pd.DataFrame,
                            df_demand: pd.DataFrame,
                            df_cf: pd.DataFrame,
                            horizon: int = 12) -> pd.DataFrame:

    if df_ref.empty:
        return pd.DataFrame()

    # "current week" = settimana precedente alla min Week in ors_forecast
    min_week = df_ref["Week"].min()
    current_week = (pd.to_datetime(min_week) - timedelta(days=7)).date()

    base = df_ref.copy()

    # merge demand + current forecast
    base = base.merge(df_demand, on=["SKU ID", "Week"], how="left")
    base = base.merge(df_cf, on=["SKU ID", "Week"], how="left")

    # week ahead integer
    base["Week_dt"] = pd.to_datetime(base["Week"])
    cw_dt = pd.to_datetime(current_week)
    base["Week Ahead"] = ((base["Week_dt"] - cw_dt).dt.days // 7).astype("Int64")

    base = base[(base["Week Ahead"] >= 1) & (base["Week Ahead"] <= horizon)].copy()

    # APE: abs(forecast - demand) / abs(demand) * 100
    # Demand==0 -> NaN (APE non definito)
    demand = pd.to_numeric(base["Demand"], errors="coerce")
    valid_demand = demand.notna() & (demand != 0)

    base["APE (Current)"] = pd.NA
    base["APE (Deda)"] = pd.NA

    base.loc[valid_demand, "APE (Current)"] = (
        (base.loc[valid_demand, "Current Forecast"] - demand[valid_demand]).abs() / demand[valid_demand].abs() * 100
    )
    base.loc[valid_demand, "APE (Deda)"] = (
        (base.loc[valid_demand, "Deda Forecast"] - demand[valid_demand]).abs() / demand[valid_demand].abs() * 100
    )

    # aggregations per week-ahead
    def _std(s: pd.Series):
        s = pd.to_numeric(s, errors="coerce").dropna()
        return s.std() if len(s) >= 2 else pd.NA

    agg = (
        base.groupby("Week Ahead")
        .agg(
            Week=("Week_dt", "min"),
            sku_required=("SKU ID", "size"),
            sku_with_demand=("Demand", lambda s: pd.to_numeric(s, errors="coerce").notna().sum()),
            ape_current=("APE (Current)", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            std_current=("APE (Current)", _std),
            ape_deda=("APE (Deda)", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            std_deda=("APE (Deda)", _std),
        )
        .reset_index()
        .sort_values("Week Ahead")
    )

    # make Week a date
    agg["Week"] = pd.to_datetime(agg["Week"], errors="coerce").dt.date

    # rename columns
    agg = agg.rename(columns={
        "ape_current": "APE (Current)",
        "std_current": "Std Dev APE (Current)",
        "ape_deda": "APE (Deda)",
        "std_deda": "Std Dev APE (Deda)",
        "sku_required": "SKU count (required)",
        "sku_with_demand": "SKU count (with demand)",
    })

    return agg 


def _std_nontrivial(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s.std() if len(s) >= 2 else pd.NA


def build_base_for_metrics(df_ref: pd.DataFrame,
                           df_demand: pd.DataFrame,
                           df_cf: pd.DataFrame,
                           horizon: int = 12) -> pd.DataFrame:
    """
    Unisce ref + demand + current forecast e calcola:
    - Week Ahead (1..horizon)
    - APE (Current) e APE (Deda)
    Usa solo Demand != 0 per le APE.
    """
    if df_ref.empty:
        return pd.DataFrame()

    min_week = df_ref["Week"].min()
    current_week = (pd.to_datetime(min_week) - pd.Timedelta(days=7)).date()

    base = df_ref.copy()
    base = base.merge(df_demand, on=["SKU ID", "Week"], how="left")
    base = base.merge(df_cf, on=["SKU ID", "Week"], how="left")

    base["Week_dt"] = pd.to_datetime(base["Week"])
    cw_dt = pd.to_datetime(current_week)
    base["Week Ahead"] = ((base["Week_dt"] - cw_dt).dt.days // 7).astype("Int64")

    # limito all'orizzonte 1..horizon (coerente con la tabella week-ahead)
    base = base[(base["Week Ahead"] >= 1) & (base["Week Ahead"] <= horizon)].copy()

    demand = pd.to_numeric(base["Demand"], errors="coerce")
    valid_demand = demand.notna() & (demand != 0)

    base["APE (Current)"] = pd.NA
    base["APE (Deda)"] = pd.NA

    base.loc[valid_demand, "APE (Current)"] = (
        (pd.to_numeric(base.loc[valid_demand, "Current Forecast"], errors="coerce") - demand[valid_demand]).abs()
        / demand[valid_demand].abs() * 100
    )
    base.loc[valid_demand, "APE (Deda)"] = (
        (pd.to_numeric(base.loc[valid_demand, "Deda Forecast"], errors="coerce") - demand[valid_demand]).abs()
        / demand[valid_demand].abs() * 100
    )

    return base


def compute_global_stats(base: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Observations (with valid demand)",
        "APE (Current)", "Std Dev APE (Current)",
        "APE (Deda)", "Std Dev APE (Deda)"
    ]

    if base is None or base.empty:
        return pd.DataFrame(columns=cols)

    demand = pd.to_numeric(base["Demand"], errors="coerce")
    valid_demand = demand.notna() & (demand != 0)

    ape_c = pd.to_numeric(base.loc[valid_demand, "APE (Current)"], errors="coerce")
    ape_d = pd.to_numeric(base.loc[valid_demand, "APE (Deda)"], errors="coerce")

    def _std_nontrivial(s: pd.Series):
        s = pd.to_numeric(s, errors="coerce").dropna()
        return s.std() if len(s) >= 2 else pd.NA

    row = {
        "Observations (with valid demand)": int(valid_demand.sum()),
        "APE (Current)": ape_c.mean(),
        "Std Dev APE (Current)": _std_nontrivial(ape_c),
        "APE (Deda)": ape_d.mean(),
        "Std Dev APE (Deda)": _std_nontrivial(ape_d),
    }
    return pd.DataFrame([row])

def compute_family_stats(base: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "FAMILY",
        "Observations (with valid demand)",
        "APE (Current)", "Std Dev APE (Current)",
        "APE (Deda)", "Std Dev APE (Deda)",
    ]
    if base is None or base.empty:
        return pd.DataFrame(columns=cols)

    if "FAMILY" not in base.columns:
        return pd.DataFrame(columns=cols)

    tmp = base.copy()

    demand = pd.to_numeric(tmp["Demand"], errors="coerce")
    valid = demand.notna() & (demand != 0)

    tmp["APE (Current)"] = pd.to_numeric(tmp["APE (Current)"], errors="coerce")
    tmp["APE (Deda)"] = pd.to_numeric(tmp["APE (Deda)"], errors="coerce")

    # APE vale solo dove demand è valida
    tmp.loc[~valid, "APE (Current)"] = pd.NA
    tmp.loc[~valid, "APE (Deda)"] = pd.NA
    tmp["_obs"] = valid.astype(int)

    agg = (
        tmp.groupby("FAMILY", dropna=False)
        .agg(
            **{
                "Observations (with valid demand)": ("_obs", "sum"),
                "APE (Current)": ("APE (Current)", "mean"),
                "Std Dev APE (Current)": ("APE (Current)", _std_nontrivial),
                "APE (Deda)": ("APE (Deda)", "mean"),
                "Std Dev APE (Deda)": ("APE (Deda)", _std_nontrivial),
            }
        )
        .reset_index()
    )

    # garantisco che tutte le FAMILY presenti in base appaiano (anche con 0 obs)
    fams = tmp[["FAMILY"]].drop_duplicates()
    out = fams.merge(agg, on="FAMILY", how="left")
    out["Observations (with valid demand)"] = out["Observations (with valid demand)"].fillna(0).astype(int)

    return out.sort_values(["Observations (with valid demand)", "FAMILY"], ascending=[False, True])


def build_ape_long(base: pd.DataFrame) -> pd.DataFrame:
    """
    Ritorna un DF lungo con colonne:
      - Model: "Current" | "Deda"
      - APE: valore APE (%)
    Usa solo righe con Demand valida (notna e != 0).
    """
    if base is None or base.empty:
        return pd.DataFrame(columns=["Model", "APE"])

    demand = pd.to_numeric(base.get("Demand"), errors="coerce")
    valid = demand.notna() & (demand != 0)

    ape_c = pd.to_numeric(base.loc[valid, "APE (Current)"], errors="coerce")
    ape_d = pd.to_numeric(base.loc[valid, "APE (Deda)"], errors="coerce")

    out = pd.concat(
        [
            pd.DataFrame({"Model": "Current", "APE": ape_c}),
            pd.DataFrame({"Model": "Deda", "APE": ape_d}),
        ],
        ignore_index=True,
    )

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["APE"])
    return out


# ---------------- UI ----------------
st.title("Forecast Comparison")


try:
    df_ref = load_reference_table()
except (OperationalError, DBAPIError) as e:
    show_db_error_and_retry(e, key="reconnect_cmp")
df_demand = load_demand_from_db()
df_cf, cf_ts = load_current_forecast_latest_from_db()

# quick status
c1, c2, c3 = st.columns(3)
c1.metric("Reference rows (ors_forecast)", f"{len(df_ref):,}")
c2.metric("Demand rows (DB)", f"{len(df_demand):,}")
with c3:
    ts = st.session_state.get("current_fcst_snapshot_ts")
    ts = pd.to_datetime(ts, errors="coerce") if ts is not None else None

    if ts is None or pd.isna(ts):
        date_str = "—"
        time_str = ""
    else:
        date_str = ts.strftime("%m/%d/%Y")   # US format
        time_str = ts.strftime("%H:%M")      # hour:minute (no ms)

    st.markdown(
        f"""
        <div style="padding:0.15rem 0;">
          <div style="font-size:0.85rem; opacity:0.65; margin-bottom:0.15rem;">
            Current Forecast snapshot
          </div>
          <div style="font-size:1.55rem; font-weight:650; line-height:1.1;">
            {date_str}
          </div>
          {"<div style='font-size:0.9rem; opacity:0.65; margin-top:0.2rem;'>" + time_str + "</div>" if time_str else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


if df_ref.empty:
    st.warning("Reference table is empty. Nothing to compare yet.")
    st.stop()

base = build_base_for_metrics(df_ref, df_demand, df_cf, horizon=horizon)

with st.sidebar:
    st.subheader("APE distribution (KDE)")

    ape_long = build_ape_long(base)

    if ape_long.empty:
        st.info("Not enough valid Demand to plot APE distributions yet.")
    else:
        # range “robusto” per evitare code estreme che schiacciano il grafico
        q_low = float(ape_long["APE"].quantile(0.01))
        q_high = float(ape_long["APE"].quantile(0.99))
        if q_low == q_high:
            q_low = max(0.0, q_low - 1.0)
            q_high = q_high + 1.0

        x_min, x_max = st.slider(
            "APE range (%)",
            min_value=float(max(0.0, q_low)),
            max_value=float(q_high),
            value=(float(max(0.0, q_low)), float(q_high)),
        )

        chart = (
            alt.Chart(ape_long)
            .transform_filter((alt.datum.APE >= x_min) & (alt.datum.APE <= x_max))
            .transform_density(
                "APE",
                groupby=["Model"],
                as_=["APE", "density"],
                extent=[x_min, x_max],
            )
            .mark_line()
            .encode(
                x=alt.X("APE:Q", title="APE (%)"),
                y=alt.Y("density:Q", title="Density"),
                stroke=alt.Stroke("Model:N", title="Model"),
            )
        )

        st.altair_chart(chart, use_container_width=True)


# --- GLOBAL TABLE ---
st.subheader("Global APE (all available observations)")

global_tbl = compute_global_stats(base)

global_fmt = {
    "APE (Current)": "{:.2f}",
    "Std Dev APE (Current)": "{:.2f}",
    "APE (Deda)": "{:.2f}",
    "Std Dev APE (Deda)": "{:.2f}",
}

if global_tbl is None or global_tbl.empty:
    st.info("No valid demand observations yet (Demand missing or Demand=0).")
else:
    st.dataframe(
        global_tbl.style.format(global_fmt, na_rep=""),
        use_container_width=True,
        hide_index=True
    )

st.divider()

# --- WEEK-AHEAD TABLE ---
table = compute_week_ahead_table(df_ref, df_demand, df_cf, horizon=horizon)

st.subheader("APE by Week Ahead")

if table.empty:
    st.info("No rows available for the selected horizon.")
    st.stop()

# format display: 2 decimals for KPI columns
fmt = {
    "APE (Current)": "{:.2f}",
    "Std Dev APE (Current)": "{:.2f}",
    "APE (Deda)": "{:.2f}",
    "Std Dev APE (Deda)": "{:.2f}",
}
sty = table.style.format(fmt, na_rep="")

st.caption(
    "Note: APE is computed only when Demand is available and Demand != 0. "
    "Weeks without sufficient demand will show blank KPI values."
)

st.dataframe(sty, use_container_width=True, hide_index=True)

# --- FAMILY TABLE ---
st.subheader("APE by Family")

family_tbl = compute_family_stats(base)

family_fmt = {
    "APE (Current)": "{:.2f}",
    "Std Dev APE (Current)": "{:.2f}",
    "APE (Deda)": "{:.2f}",
    "Std Dev APE (Deda)": "{:.2f}",
}

total_obs = int(family_tbl["Observations (with valid demand)"].sum()) if family_tbl is not None and not family_tbl.empty else 0
if total_obs == 0:
    st.info("No valid demand observations yet to compute family KPIs (Demand missing or Demand=0).")

st.dataframe(
    family_tbl.style.format(family_fmt, na_rep=""),
    use_container_width=True,
    hide_index=True
)