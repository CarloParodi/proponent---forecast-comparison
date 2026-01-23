# pages/03_Methodology.py
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Methodology", layout="wide")

st.title("Methodology & KPI Guide")
st.caption("How to read APE-based metrics and why we segment results by horizon and family.")

# -----------------------------
# Helpers (for the interactive example only)
# -----------------------------
def round_half_up_to_int(x: float) -> float:
    """Round half-up to nearest integer (0.5 -> 1). Works also for negatives."""
    if x is None:
        return np.nan
    x = pd.to_numeric(x, errors="coerce")
    if pd.isna(x):
        return np.nan
    return float(np.sign(x) * np.floor(np.abs(x) + 0.5))

def ape(actual: float, forecast: float) -> float:
    """
    Absolute Percentage Error (APE) in % - app variant.

    We compute APE using a business rule for Actual=0:
      - if Actual != 0: |F - A| / |A| * 100
      - if Actual == 0 and F != 0: |F - A| / |F| * 100 -> 100%
      - if Actual == 0 and F == 0: 0%

    IMPORTANT: forecast is first rounded to the nearest integer (round half-up),
    consistently with the app logic.
    """
    if actual is None or forecast is None:
        return np.nan

    a = pd.to_numeric(actual, errors="coerce")
    f = round_half_up_to_int(forecast)

    if pd.isna(a) or pd.isna(f):
        return np.nan

    # Case 1: Actual != 0
    if a != 0:
        return abs(f - a) / abs(a) * 100.0

    # Case 2: Actual == 0
    if f == 0:
        return 0.0
    return abs(f - a) / abs(f) * 100.0  # = 100% when a==0 and f!=0


# -----------------------------
# Overview
# -----------------------------
st.header("1) What is APE?")
c1, c2 = st.columns([0.62, 0.38], vertical_alignment="top")

with c1:
    st.markdown(
        """
**APE (Absolute Percentage Error)** measures *how far* a forecast is from the actual demand **in percentage terms**, ignoring the sign.

It is computed for each observation (SKU, Week):

- **Actual (A)** = observed demand
- **Forecast (F)** = predicted demand (**rounded to the nearest integer** in this app)

We use the absolute value, so APE is always non-negative.
"""
    )

    st.markdown("### Formula used in this app")
    st.latex(r"""
\mathrm{APE} =
\begin{cases}
\frac{|F - A|}{|A|} \times 100 & \text{if } A \neq 0 \\
\frac{|F - A|}{|F|} \times 100 & \text{if } A = 0 \text{ and } F \neq 0 \\
0 & \text{if } A = 0 \text{ and } F = 0
\end{cases}
""")

    st.markdown(
        """
**Why this rule?**  
When **Actual = 0**, the “standard” APE would be undefined (division by zero).  
Our rule makes the metric usable and consistent with the business meaning:

- If demand is zero but forecast is positive → **APE = 100%** (full miss)
- If both demand and forecast are zero → **APE = 0%** (perfect)
"""
    )

with c2:
    st.info(
        "Interpretation:\n\n"
        "- **APE = 0%** → perfect forecast\n"
        "- **APE = 10%** → forecast is off by 10% of actual demand\n"
        "- **Lower is better** (on average)\n\n"
        "Special case:\n"
        "- **Actual = 0, Forecast > 0 → APE = 100%**\n"
        "- **Actual = 0, Forecast = 0 → APE = 0%**"
    )


st.divider()

# -----------------------------
# Dispersion / Std Dev
# -----------------------------
st.header("2) Standard deviation of APE (dispersion)")
st.markdown(
    """
In the UI we compute:

- **Mean APE**: average error magnitude (overall accuracy level)
- **Std Dev of APE**: how **spread out** the APE values are across SKUs (or across the chosen group)

Std dev answers: *are errors consistent across SKUs, or do we have many outliers?*
"""
)

st.latex(r"\sigma_{\mathrm{APE}} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\mathrm{APE}_i - \overline{\mathrm{APE}})^2}")

st.caption(
    "Note: this is standard deviation (dispersion). If you want the statistical variance, it is simply σ²."
)

st.divider()

# -----------------------------
# Interactive example
# -----------------------------
st.header("3) Small example (interactive)")
st.markdown(
    """
Use this toy example to see how APE is computed and how mean/std behave.

**Tip:** try setting Actual to **0** to see the special rule in action.
Forecasts are rounded to the nearest integer before computing APE (e.g., 0.7 → 1, 0.3 → 0).
"""
)

ex_c1, ex_c2, ex_c3 = st.columns(3)

with ex_c1:
    a1 = st.number_input("Actual (SKU A)", min_value=0.0, value=100.0, step=1.0)
    f1_deda = st.number_input("Deda Forecast (SKU A)", min_value=0.0, value=110.0, step=0.1)
    f1_curr = st.number_input("Current Forecast (SKU A)", min_value=0.0, value=95.0, step=0.1)

with ex_c2:
    a2 = st.number_input("Actual (SKU B)", min_value=0.0, value=120.0, step=1.0)
    f2_deda = st.number_input("Deda Forecast (SKU B)", min_value=0.0, value=90.0, step=0.1)
    f2_curr = st.number_input("Current Forecast (SKU B)", min_value=0.0, value=130.0, step=0.1)

with ex_c3:
    st.write("")
    st.caption("Forecast rounding (half-up):")
    st.write(f"SKU A Deda → {round_half_up_to_int(f1_deda):.0f}")
    st.write(f"SKU A Current → {round_half_up_to_int(f1_curr):.0f}")
    st.write(f"SKU B Deda → {round_half_up_to_int(f2_deda):.0f}")
    st.write(f"SKU B Current → {round_half_up_to_int(f2_curr):.0f}")

# Compute example
rows = []
for sku, act, fd, fc in [
    ("SKU A", a1, f1_deda, f1_curr),
    ("SKU B", a2, f2_deda, f2_curr),
]:
    rows.append(
        {
            "SKU": sku,
            "Actual": act,
            "Deda Forecast": fd,
            "Current Forecast": fc,
            "Deda Forecast (rounded)": round_half_up_to_int(fd),
            "Current Forecast (rounded)": round_half_up_to_int(fc),
            "APE (Deda) %": ape(act, fd),
            "APE (Current) %": ape(act, fc),
        }
    )

df_ex = pd.DataFrame(rows)

st.dataframe(
    df_ex.style.format(
        {
            "Actual": "{:.2f}",
            "Deda Forecast": "{:.2f}",
            "Current Forecast": "{:.2f}",
            "Deda Forecast (rounded)": "{:.0f}",
            "Current Forecast (rounded)": "{:.0f}",
            "APE (Deda) %": "{:.2f}",
            "APE (Current) %": "{:.2f}",
        },
        na_rep="—",
    ),
    use_container_width=True,
    hide_index=True,
)

# Summary metrics
ape_d = df_ex["APE (Deda) %"].dropna()
ape_c = df_ex["APE (Current) %"].dropna()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Mean APE (Deda)", "—" if ape_d.empty else f"{ape_d.mean():.2f}%")
m2.metric("Std Dev APE (Deda)", "—" if ape_d.empty else f"{ape_d.std(ddof=0):.2f}%")
m3.metric("Mean APE (Current)", "—" if ape_c.empty else f"{ape_c.mean():.2f}%")
m4.metric("Std Dev APE (Current)", "—" if ape_c.empty else f"{ape_c.std(ddof=0):.2f}%")

st.caption(
    "Tip: if you want *bias* you need a signed error metric (e.g., (Forecast-Actual)/Actual), because APE cannot tell over- vs under-forecast."
)

st.divider()

# -----------------------------
# Why Global / Week Ahead / Family
# -----------------------------
st.header("4) Why we show Global vs Week Ahead vs Family")

colA, colB = st.columns(2, vertical_alignment="top")

with colA:
    st.subheader("Global view (all observations)")
    st.markdown(
        """
This is the simplest summary:
- “On average, how good are we?”
- Are Current forecasts better than Deda overall?

It is useful as a headline metric, but it can **hide** important patterns (e.g., accuracy drops with longer horizons).
"""
    )

    st.subheader("Week ahead view (horizon decay)")
    st.markdown(
        """
Forecast quality usually **deteriorates** as you predict further into the future.

The *week ahead* table helps you see:
- how quickly performance drops from near-term to far-term horizons,
- whether Current improves more in short horizons vs long horizons,
- where a model is “safe” vs where it becomes unreliable.

In practice, you compute week-ahead buckets like 1..12 and aggregate APE across all SKUs **for each bucket**.
"""
    )

with colB:
    st.subheader("Family view (segment performance)")
    st.markdown(
        """
Not all product families behave the same way:
- different intermittency patterns,
- different promotion effects,
- different lead times / substitution / seasonality,
- different data quality.

Family KPIs help you:
- identify which families need different modeling strategies,
- spot systematic issues (e.g., a family always has higher dispersion),
- prioritize where to invest (data cleanup, feature engineering, model changes).
"""
    )

st.divider()

st.header("5) Practical notes & caveats")
st.markdown(
    """
- **Missing demand** → you cannot compute APE; those rows are skipped.
- **Actual = 0** → we apply the app rule:
  - if **Forecast > 0** ⇒ **APE = 100%**
  - if **Forecast = 0** ⇒ **APE = 0%**
- **Forecasts are rounded** to the nearest integer (half-up) before computing APE.
- **Outliers** can heavily impact averages; std dev helps you detect them.
- **Mean APE** tells “how wrong on average”; **Std dev APE** tells “how stable / consistent the errors are.”
"""
)

st.success("This page is informational only (no DB reads).")
