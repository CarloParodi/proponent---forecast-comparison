# app.py
import io
import csv
import streamlit as st
import pandas as pd
from datetime import datetime
from sqlalchemy import text
from database_connection import get_database_connection
from sqlalchemy.exc import OperationalError, DBAPIError

pd.set_option("styler.render.max_elements", 1_000_000)
st.set_page_config(page_title="Forecast Table", layout="wide")

# --- CONFIG ---
REF_TABLE_NAME = "streamlit_forecast_comparison.ors_forecast"
DEMAND_TABLE_NAME = "streamlit_forecast_comparison.sku_week_demand"
CURRENT_FCST_TABLE_NAME = "streamlit_forecast_comparison.current_forecast"

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
def load_demand_from_db() -> pd.DataFrame:
    engine = _get_engine()
    if engine is None:
        return pd.DataFrame(columns=["SKU ID", "Week", "Demand"])

    query = f"""
        SELECT
            "SKU ID" AS "SKU ID",
            "Week"   AS "Week",
            "Demand" AS "Demand"
        FROM {DEMAND_TABLE_NAME}
    """

    try:
        with engine.connect() as conn:
            try:
                df = pd.read_sql(text(query), conn)
            except Exception:
                # IMPORTANT: clear failed transaction before returning connection to pool
                try:
                    conn.rollback()
                except Exception:
                    pass
                return pd.DataFrame(columns=["SKU ID", "Week", "Demand"])
    except Exception:
        return pd.DataFrame(columns=["SKU ID", "Week", "Demand"])

    df["SKU ID"] = df["SKU ID"].astype(str).str.strip()
    df["Week"] = pd.to_datetime(df["Week"], errors="coerce").dt.date
    df["Demand"] = pd.to_numeric(df["Demand"], errors="coerce")
    df = df.dropna(subset=["SKU ID", "Week", "Demand"])
    df = df.groupby(["SKU ID", "Week"], as_index=False)["Demand"].mean()
    return df


@st.cache_data(ttl=300)
def load_reference_table() -> pd.DataFrame:
    engine = _get_engine()
    if engine is None:
        raise RuntimeError("Database connection is not configured or returned None.")

    query = f"""
        SELECT
            "FAMILY"         AS "FAMILY",
            "SKU ID"         AS "SKU ID",
            "week"           AS "Week",
            "ORS prediction" AS "Deda Forecast"
        FROM {REF_TABLE_NAME}
    """

    with engine.connect() as conn:
        try:
            df = pd.read_sql(text(query), conn)
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise

    df["SKU ID"] = df["SKU ID"].astype(str).str.strip()
    df["Week"] = pd.to_datetime(df["Week"], errors="coerce").dt.date
    df = df.dropna(subset=["SKU ID", "Week"])
    return df


# ---------------- CSV helpers ----------------
def _detect_delimiter(sample_text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return ","


def _read_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    text = raw.decode("utf-8-sig", errors="replace")
    sep = _detect_delimiter(text[:4096])
    df = pd.read_csv(io.StringIO(text), sep=sep)
    df.columns = [c.strip() for c in df.columns]
    return df


def _parse_numeric(series: pd.Series) -> pd.Series:
    """
    Handles comma decimals and thousands:
      "12,3" -> 12.3
      "1.234,56" -> 1234.56
      "1 234,56" -> 1234.56
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(" ", "", regex=False)

    both_mask = s.str.contains(r"\.", regex=True) & s.str.contains(",", regex=False)
    s.loc[both_mask] = s.loc[both_mask].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)

    only_comma_mask = (~s.str.contains(r"\.", regex=True)) & s.str.contains(",", regex=False)
    s.loc[only_comma_mask] = s.loc[only_comma_mask].str.replace(",", ".", regex=False)

    return pd.to_numeric(s, errors="coerce")


# ---------------- Validation ----------------
def validate_current_forecast_df(df_cf: pd.DataFrame, df_ref: pd.DataFrame):
    # Se non c’è nulla su DB
    if df_cf is None or df_cf.empty:
        ref_skus = set(df_ref["SKU ID"])
        return {
            "ok": True,
            "invalid_week_rows": 0,
            "invalid_value_rows": 0,
            "duplicate_key_rows": 0,
            "missing_skus": sorted(ref_skus),
            "missing_pairs": [],
            "extra_pairs": [],
        }

    df = df_cf.copy()
    df["SKU ID"] = df["SKU ID"].astype(str).str.strip()
    df["Week"] = pd.to_datetime(df["Week"], errors="coerce").dt.date
    df["Current Forecast"] = pd.to_numeric(df["Current Forecast"], errors="coerce")

    invalid_week = int(df["Week"].isna().sum())
    invalid_val = int(df["Current Forecast"].isna().sum())
    dup_count = int(df.duplicated(subset=["SKU ID", "Week"]).sum())

    ref_skus = set(df_ref["SKU ID"])
    cf_skus = set(df["SKU ID"])

    missing_skus = sorted(ref_skus - cf_skus)

    ref_keys = set(zip(df_ref["SKU ID"], df_ref["Week"]))
    cf_keys = set(zip(df["SKU ID"], df["Week"]))

    # missing weeks solo per SKUs presenti nel CF (evita ridondanza)
    common_skus = ref_skus & cf_skus
    df_ref_common = df_ref[df_ref["SKU ID"].isin(common_skus)]
    ref_keys_common = set(zip(df_ref_common["SKU ID"], df_ref_common["Week"]))

    missing_pairs = sorted(ref_keys_common - cf_keys)
    extra_pairs = sorted(cf_keys - ref_keys)

    return {
        "ok": True,
        "invalid_week_rows": invalid_week,
        "invalid_value_rows": invalid_val,
        "duplicate_key_rows": dup_count,
        "missing_skus": missing_skus,
        "missing_pairs": missing_pairs,
        "extra_pairs": extra_pairs,
    }


def validate_demand_csv(df_csv: pd.DataFrame, df_ref: pd.DataFrame):
    required_cols = ["SKU ID", "Week", "Demand"]
    missing_cols = [c for c in required_cols if c not in df_csv.columns]
    if missing_cols:
        return {"ok": False, "error": f"Missing required columns: {missing_cols}. Expected: {required_cols}"}

    df = df_csv.copy()
    df["SKU ID"] = df["SKU ID"].astype(str).str.strip()
    df["Week"] = pd.to_datetime(df["Week"], errors="coerce").dt.date
    df["Demand"] = _parse_numeric(df["Demand"])

    invalid_week = int(df["Week"].isna().sum())
    invalid_val = int(df["Demand"].isna().sum())
    dup_count = int(df.duplicated(subset=["SKU ID", "Week"]).sum())

    ref_keys = set(zip(df_ref["SKU ID"], df_ref["Week"]))
    csv_keys = set(zip(df["SKU ID"], df["Week"]))

    extra_pairs = sorted(csv_keys - ref_keys)  # flagged, not necessarily error

    # NOTE: demand csv is allowed to be partial (e.g., one future week)
    return {
        "ok": True,
        "df_parsed": df[["SKU ID", "Week", "Demand"]],
        "invalid_week_rows": invalid_week,
        "invalid_value_rows": invalid_val,
        "duplicate_key_rows": dup_count,
        "extra_pairs": extra_pairs,
    }


# ---------------- In-memory upsert ----------------
def upsert_kv(existing: pd.DataFrame | None, new_df: pd.DataFrame, key_cols, value_col) -> pd.DataFrame:
    """
    Integrates new_df into existing by (key_cols) -> keep last value.
    Only overwrites the keys present in new_df, leaving the rest intact.
    """
    if existing is None or existing.empty:
        base = new_df.copy()
    else:
        base = pd.concat([existing, new_df], ignore_index=True)

    base = base.dropna(subset=key_cols)
    base = base.sort_values(key_cols)
    base = base.drop_duplicates(subset=key_cols, keep="last")
    return base


# ---------------- Main table build ----------------
def build_main_table(df_ref: pd.DataFrame) -> pd.DataFrame:
    df = df_ref.copy()

    # Always-present columns
    df["Demand"] = pd.NA
    df["Current Forecast"] = pd.NA

    # Merge Demand (if any)
    demand_df = st.session_state.get("demand_df")
    if demand_df is not None and not demand_df.empty:
        d = demand_df.copy()
        d["SKU ID"] = d["SKU ID"].astype(str).str.strip()
        d["Week"] = pd.to_datetime(d["Week"], errors="coerce").dt.date
        d = d.groupby(["SKU ID", "Week"], as_index=False)["Demand"].mean()
        df = df.merge(d, on=["SKU ID", "Week"], how="left", suffixes=("", "_d"))
        df["Demand"] = df["Demand_d"]
        df = df.drop(columns=["Demand_d"])

    # Merge Current Forecast (if any)
    cf_df = st.session_state.get("current_fcst_df")
    if cf_df is not None and not cf_df.empty:
        c = cf_df.copy()
        c["SKU ID"] = c["SKU ID"].astype(str).str.strip()
        c["Week"] = pd.to_datetime(c["Week"], errors="coerce").dt.date
        c = c.groupby(["SKU ID", "Week"], as_index=False)["Current Forecast"].mean()
        df = df.merge(c, on=["SKU ID", "Week"], how="left", suffixes=("", "_c"))
        df["Current Forecast"] = df["Current Forecast_c"]
        df = df.drop(columns=["Current Forecast_c"])

    cols = ["FAMILY", "SKU ID", "Week", "Demand", "Deda Forecast", "Current Forecast"]
    return df[cols]


def week_coverage(df_ref: pd.DataFrame, demand_df: pd.DataFrame | None) -> pd.DataFrame:
    # Base: tutte le righe richieste (SKU,Week) dal reference
    base = df_ref[["SKU ID", "Week"]].copy()

    if demand_df is None or demand_df.empty:
        base["Demand"] = pd.NA
    else:
        d = demand_df.copy()
        d["SKU ID"] = d["SKU ID"].astype(str).str.strip()
        d["Week"] = pd.to_datetime(d["Week"], errors="coerce").dt.date
        d["Demand"] = pd.to_numeric(d["Demand"], errors="coerce")
        d = d.dropna(subset=["SKU ID", "Week"])
        d = d.groupby(["SKU ID", "Week"], as_index=False)["Demand"].mean()

        base = base.merge(d, on=["SKU ID", "Week"], how="left")

    cov = (
        base.groupby("Week")
        .agg(required_rows=("SKU ID", "size"), filled_rows=("Demand", lambda s: s.notna().sum()))
        .reset_index()
    )
    cov["completion_pct"] = (cov["filled_rows"] / cov["required_rows"]).fillna(0.0)
    cov["status"] = cov["completion_pct"].apply(
        lambda x: "Complete" if x == 1 else ("Missing" if x == 0 else "Partial")
    )
    return cov.sort_values("Week")


# ---------------- Persist Demand to DB (upsert) ----------------
def _split_schema_table(full_name: str):
    full_name = full_name.strip()
    if "." in full_name:
        schema, table = full_name.split(".", 1)
        return schema.strip('"'), table.strip('"')
    return None, full_name.strip('"')


def persist_demand_to_db(engine, demand_df: pd.DataFrame, table_name: str) -> tuple[bool, str]:
    if demand_df is None or demand_df.empty:
        return False, "No demand data to save (memory is empty)."

    from sqlalchemy import Table, Column, MetaData, Date, Float, String
    from sqlalchemy import select, func
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    df = demand_df.copy()
    df["SKU ID"] = df["SKU ID"].astype(str).str.strip()
    df["Week"] = pd.to_datetime(df["Week"], errors="coerce").dt.date
    df["Demand"] = pd.to_numeric(df["Demand"], errors="coerce")
    df = df.dropna(subset=["SKU ID", "Week", "Demand"])
    df = df.groupby(["SKU ID", "Week"], as_index=False)["Demand"].mean()

    # 👇 importante: se dopo pulizia rimane vuoto, non scriverà nulla
    if df.empty:
        return False, "Demand dataframe is empty after cleaning (no valid rows to write)."

    schema, tbl = _split_schema_table(table_name)

    md = MetaData()
    t = Table(
        tbl,
        md,
        Column("SKU ID", String(255), primary_key=True),
        Column("Week", Date, primary_key=True),
        Column("Demand", Float),
        schema=schema,
    )

    md.create_all(engine)

    rows = [
        {"SKU ID": r["SKU ID"], "Week": r["Week"], "Demand": float(r["Demand"])}
        for _, r in df.iterrows()
    ]

    chunk_size = 5000
    with engine.begin() as conn:
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i:i + chunk_size]

            insert_stmt = pg_insert(t).values(chunk)
            upsert_stmt = insert_stmt.on_conflict_do_update(
                index_elements=["SKU ID", "Week"],
                set_={"Demand": insert_stmt.excluded.Demand},
            )
            conn.execute(upsert_stmt)
        cnt = conn.execute(select(func.count()).select_from(t)).scalar_one()

    load_demand_from_db.clear()
    return True, f"Saved/updated {len(rows):,} rows into {table_name}. Table now has {cnt:,} rows."

from sqlalchemy import text

@st.cache_data(ttl=300)
def load_current_forecast_latest_from_db() -> tuple[pd.DataFrame, datetime | None]:
    engine = _get_engine()
    if engine is None:
        return pd.DataFrame(columns=["SKU ID", "Week", "Current Forecast"]), None

    q = f'''
        SELECT
            "SKU ID"           AS "SKU ID",
            "Week"             AS "Week",
            "Current Forecast" AS "Current Forecast",
            "Date update"      AS "Date update"
        FROM {CURRENT_FCST_TABLE_NAME}
        WHERE "Date update" = (SELECT MAX("Date update") FROM {CURRENT_FCST_TABLE_NAME})
    '''

    try:
        with engine.connect() as conn:
            try:
                df = pd.read_sql(text(q), conn)
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                return pd.DataFrame(columns=["SKU ID", "Week", "Current Forecast"]), None
    except Exception:
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

# ---------------- Styling ----------------
def style_missing_cols(df: pd.DataFrame, has_csv_cf: bool, has_demand: bool):
    def hl_missing(v):
        return "background-color: rgba(255, 0, 0, 0.12)" if pd.isna(v) else ""

    def hl_missing_demand(v):
        return "background-color: rgba(255, 165, 0, 0.12)" if pd.isna(v) else ""

    sty = df.style
    if has_csv_cf:
        sty = sty.applymap(hl_missing, subset=["Current Forecast"])
    if has_demand:
        sty = sty.applymap(hl_missing_demand, subset=["Demand"])
    return sty

# ---------------- SESSION INIT ----------------
if "show_only_problems" not in st.session_state:
    st.session_state["show_only_problems"] = False

if "current_fcst_df" not in st.session_state:
    st.session_state["current_fcst_df"] = None

if "demand_df" not in st.session_state:
    st.session_state["demand_df"] = None

if "val_cf" not in st.session_state:
    st.session_state["val_cf"] = None

if "val_demand" not in st.session_state:
    st.session_state["val_demand"] = None

if "current_fcst_loaded_from_db" not in st.session_state:
    st.session_state["current_fcst_loaded_from_db"] = False

if "current_fcst_snapshot_ts" not in st.session_state:
    st.session_state["current_fcst_snapshot_ts"] = None

if "demand_dirty" not in st.session_state:
    st.session_state["demand_dirty"] = False

    
def has_nonempty_df(name: str) -> bool:
    df = st.session_state.get(name)
    return df is not None and hasattr(df, "empty") and (not df.empty)

# ---------------- UI ----------------

def reject_upload(title: str, details: str):
    st.toast(f"❌ {title}", icon="❌")
    st.error(f"**{title}**\n\n{details}")
    st.stop()

st.title("Forecast Table")

with st.sidebar:
    st.header("Uploads")

    st.subheader("Demand (incremental)")
    uploaded_demand = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        accept_multiple_files=False,
        help="Required columns: SKU ID, Week, Demand. New uploads integrate (upsert keys).",
        key="uploader_demand",
    )

    st.divider()
    persist_box = st.container()

try:

    try:
        df_ref = load_reference_table()
    except (OperationalError, DBAPIError) as e:
        show_db_error_and_retry(e, key="reconnect_cmp")

    # Load demand from DB at startup (only if we don't already have demand in memory)
    if "demand_loaded_from_db" not in st.session_state:
        st.session_state["demand_loaded_from_db"] = False

    if (not st.session_state["demand_loaded_from_db"]) and (
        st.session_state.get("demand_df") is None or st.session_state["demand_df"].empty
    ):
        st.session_state["demand_df"] = load_demand_from_db()
        st.session_state["demand_loaded_from_db"] = True
        st.session_state["demand_dirty"] = False
    
    # Load Current Forecast latest snapshot from DB at startup (if not already in memory)
    if (not st.session_state["current_fcst_loaded_from_db"]) and (
        st.session_state.get("current_fcst_df") is None or st.session_state["current_fcst_df"].empty
    ):
        db_cf, db_ts = load_current_forecast_latest_from_db()
        st.session_state["current_fcst_df"] = db_cf
        st.session_state["current_fcst_snapshot_ts"] = db_ts
        st.session_state["current_fcst_loaded_from_db"] = True

    st.session_state["val_cf"] = validate_current_forecast_df(
        st.session_state.get("current_fcst_df"),
        df_ref
    )

    # --- Handle uploads (store in session_state) ---

    if uploaded_demand is not None:
        try:
            df_csv = _read_csv(uploaded_demand)
        except Exception as e:
            reject_upload(
                "Demand CSV could not be read",
                f"Parsing error: {e}"
            )
        else:
            res = validate_demand_csv(df_csv, df_ref)
            st.session_state["val_demand"] = res

            if not res.get("ok", False):
                reject_upload(
                    "Invalid Demand CSV format",
                    res.get("error", "Unknown validation error.")
                )
            else:
                st.session_state["demand_df"] = upsert_kv(
                    st.session_state.get("demand_df"),
                    res["df_parsed"],
                    key_cols=["SKU ID", "Week"],
                    value_col="Demand",
                )
                st.session_state["demand_dirty"] = True


    # --- Build Persist buttons AFTER loads + uploads ---
    with persist_box:
        st.header("Persist")

        can_save_demand = has_nonempty_df("demand_df") and st.session_state.get("demand_dirty", False)

        save_demand = st.button(
            "Save Demand to DB",
            use_container_width=True,
            disabled=not can_save_demand,
            help=None if can_save_demand else "Upload a Demand CSV to enable saving.",
            key="btn_save_demand",
        )

    # --- Persist demand ---
    if save_demand:
        ddf = st.session_state.get("demand_df")
        if ddf is None or ddf.empty:
            st.warning("No Demand data in memory. Upload a CSV first.")
        else:
            try:
                engine = _get_engine()
                ok, msg = persist_demand_to_db(engine, ddf, DEMAND_TABLE_NAME)
                (st.success if ok else st.warning)(msg)
                if ok:
                    load_demand_from_db.clear()
                    st.session_state["demand_df"] = load_demand_from_db()
                    st.session_state["demand_dirty"] = False
            except Exception as e:
                st.error("Failed to save Demand to DB.")

    # --- Demand coverage bar (top) ---
    cov = week_coverage(df_ref, st.session_state.get("demand_df"))
    total_weeks = int(cov["Week"].nunique())
    complete_weeks = int((cov["status"] == "Complete").sum())
    partial_weeks = int((cov["status"] == "Partial").sum())
    missing_weeks = int((cov["status"] == "Missing").sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Weeks (total)", total_weeks)
    m2.metric("Weeks complete", complete_weeks)
    m3.metric("Weeks partial", partial_weeks)
    m4.metric("Weeks missing", missing_weeks)

    total_required = int(cov["required_rows"].sum())
    total_filled = int(cov["filled_rows"].sum())
    st.progress(0 if total_required == 0 else total_filled/total_required,
                text=f"Demand coverage (rows): {total_filled}/{total_required}")

    with st.expander("Week-by-week coverage details", expanded=False):
        cov_view = cov.copy()
        cov_view.rename(columns={"completion_pct": "Completion (%)"}, inplace=True)
        cov_view["Completion (%)"] = (cov_view["Completion (%)"] * 100).round(2)
        st.dataframe(cov_view, use_container_width=True, hide_index=True)


    # --- MAIN TABLE (always on top) ---
    df_main = build_main_table(df_ref)
    has_cf = st.session_state.get("current_fcst_df") is not None and not st.session_state["current_fcst_df"].empty
    has_demand = st.session_state.get("demand_df") is not None and not st.session_state["demand_df"].empty

    # Header row: title left, toggle button right
    h1, h2 = st.columns([0.78, 0.22], vertical_alignment="center")
    with h1:
        st.subheader("Main table")
    with h2:
        btn_label = "Show only problematic rows" if not st.session_state["show_only_problems"] else "Show all rows"
        if st.button(btn_label, use_container_width=True, disabled=not has_cf):
            st.session_state["show_only_problems"] = not st.session_state["show_only_problems"]

    # Filters
    with st.expander("Filters", expanded=False):
        f1, f2, f3 = st.columns(3)

        family_options = sorted(df_main["FAMILY"].dropna().astype(str).unique().tolist())
        with f1:
            selected_families = st.multiselect("Family", options=family_options, default=[])

        with f2:
            sku_query = st.text_input("SKU filter (contains)", value="").strip()

        min_week = df_main["Week"].min()
        max_week = df_main["Week"].max()
        with f3:
            week_range = st.date_input(
                "Week range",
                value=(min_week, max_week),
                min_value=min_week,
                max_value=max_week,
            )

    df_view = df_main.copy()
    if selected_families:
        df_view = df_view[df_view["FAMILY"].astype(str).isin(selected_families)]
    if sku_query:
        df_view = df_view[df_view["SKU ID"].astype(str).str.contains(sku_query, case=False, na=False)]
    if isinstance(week_range, (tuple, list)) and len(week_range) == 2:
        start_w, end_w = week_range
        df_view = df_view[(df_view["Week"] >= start_w) & (df_view["Week"] <= end_w)]

    # problematic rows = missing Current Forecast (only meaningful after CF upload)
    if st.session_state["show_only_problems"] and has_cf:
        df_view = df_view[df_view["Current Forecast"].isna()].copy()

    st.caption(
        f"Rows shown: {len(df_view):,} / {len(df_main):,} • "
        f"Demand filled: {df_main['Demand'].notna().sum():,} • "
        f"Current Forecast filled: {df_main['Current Forecast'].notna().sum():,}"
    )
    ts = st.session_state.get("current_fcst_snapshot_ts")
    if ts is None:
        st.caption("Current Forecast snapshot: (not loaded from DB yet / not saved)")
    else:
        st.caption(f"Current Forecast snapshot (Date update): {ts}")

    # --- Pretty formatting for main table (display only) ---
    df_display = df_view.copy()
    for col in ["Demand", "Deda Forecast", "Current Forecast"]:
        if col in df_display.columns:
            df_display[col] = pd.to_numeric(df_display[col], errors="coerce").round(2)

    fmt = {c: "{:.2f}" for c in ["Demand", "Deda Forecast", "Current Forecast"]}

    if has_cf or has_demand:
        sty = style_missing_cols(df_display, has_csv_cf=has_cf, has_demand=has_demand)
        sty = sty.format(fmt, na_rep="")   # <-- QUESTO elimina gli zeri extra in visualizzazione
        st.dataframe(sty, use_container_width=True, hide_index=True)
    else:
        sty = df_display.style.format(fmt, na_rep="")
        st.dataframe(sty, use_container_width=True, hide_index=True)


    # --- EXCEPTIONS BELOW ---
    st.divider()
    st.subheader("Validation results")

    # Current Forecast validation
    if st.session_state.get("val_cf") is None:
        st.info("Current Forecast: not loaded yet.")
    else:
        v = st.session_state["val_cf"]
        if not v.get("ok", False):
            st.error(v.get("error", "Current Forecast validation failed."))
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CF Missing SKUs", len(v["missing_skus"]))
            c2.metric("CF Missing weeks (for SKUs present)", len(v["missing_pairs"]))
            c3.metric("CF Invalid Week rows", v["invalid_week_rows"])
            c4.metric("CF Invalid value rows", v["invalid_value_rows"])

            if v["duplicate_key_rows"] > 0:
                st.error(f"CF: Found {v['duplicate_key_rows']} duplicate (SKU ID, Week). Using mean for duplicates.")

            if len(v["missing_skus"]) > 0:
                st.error("CF: Some SKUs from DB are missing in the CSV.")
                st.dataframe(pd.DataFrame({"SKU ID": v["missing_skus"]}), use_container_width=True, hide_index=True)

            if len(v["missing_pairs"]) > 0:
                st.error("CF: Some weeks are missing for SKUs that exist in the CSV.")
                st.dataframe(pd.DataFrame(v["missing_pairs"], columns=["SKU ID", "Week"]).head(1000),
                             use_container_width=True, hide_index=True)

            if len(v["extra_pairs"]) > 0:
                st.info("CF: Extra (SKU ID, Week) pairs not present in DB (flagged).")
                st.dataframe(pd.DataFrame(v["extra_pairs"], columns=["SKU ID", "Week"]).head(1000),
                             use_container_width=True, hide_index=True)

    st.divider()

    # Demand validation
    if st.session_state.get("val_demand") is None:
        st.info("No Demand CSV uploaded yet.")
    else:
        v = st.session_state["val_demand"]
        if not v.get("ok", False):
            st.error(v.get("error", "Demand validation failed."))
        else:
            d1, d2, d3 = st.columns(3)
            d1.metric("Demand Invalid Week rows", v["invalid_week_rows"])
            d2.metric("Demand Invalid value rows", v["invalid_value_rows"])
            d3.metric("Demand Extra pairs (flagged)", len(v["extra_pairs"]))

            if v["duplicate_key_rows"] > 0:
                st.warning(f"Demand: Found {v['duplicate_key_rows']} duplicate (SKU ID, Week). Using mean for duplicates.")

            if len(v["extra_pairs"]) > 0:
                st.info("Demand: Extra (SKU ID, Week) pairs not present in DB (flagged).")
                st.dataframe(pd.DataFrame(v["extra_pairs"], columns=["SKU ID", "Week"]).head(1000),
                             use_container_width=True, hide_index=True)

except Exception as e:
    st.error("Failed to load data or process uploads.")
