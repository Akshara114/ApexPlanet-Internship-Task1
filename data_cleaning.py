"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           DATA IMMERSION & WRANGLING — Task 1 Cleaning Script              ║
║           Dataset  : Sales / E-commerce (raw_sales_data.csv)               ║
║           Author   : Data Analytics Intern                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Steps covered
─────────────
 1. Load & initial profiling
 2. Duplicate removal
 3. Date standardisation (multi-format parsing → ISO 8601)
 4. Categorical standardisation (case, typos)
 5. Missing-value treatment (impute / flag / drop)
 6. Outlier detection & treatment (IQR method)
 7. Feature engineering
     • customer_age   (from date_of_birth)
     • age_group      (binned)
     • total_revenue  (price × qty × (1 − discount))
     • order_year / order_month / order_quarter
 8. Final schema validation & export
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings, os
warnings.filterwarnings("ignore")

SRC  = "raw_sales_data.csv"
DEST = "cleaned_sales_data.csv"
LOG  = "cleaning_log.txt"

log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)

def save_log():
    with open(LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & INITIAL PROFILE
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(SRC)
log("=" * 70)
log(f"[LOAD]  Rows: {len(df)} | Columns: {len(df.columns)}")
log("=" * 70)

log("\n── Missing Values (raw) ──")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
for col in missing[missing > 0].index:
    log(f"  {col:<20} {missing[col]:>4} missing  ({missing_pct[col]}%)")

log(f"\n── Dtypes ──\n{df.dtypes.to_string()}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. REMOVE DUPLICATES
# ══════════════════════════════════════════════════════════════════════════════
before = len(df)
df.drop_duplicates(subset="order_id", keep="first", inplace=True)
df.reset_index(drop=True, inplace=True)
log(f"\n[DUPLICATES]  Removed {before - len(df)} duplicate rows → {len(df)} remain")

# ══════════════════════════════════════════════════════════════════════════════
# 3. STANDARDISE DATES
# ══════════════════════════════════════════════════════════════════════════════
DATE_FMTS = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%d %b %Y"]

def parse_date(val):
    if pd.isna(val):
        return pd.NaT
    for fmt in DATE_FMTS:
        try:
            return datetime.strptime(str(val).strip(), fmt)
        except ValueError:
            continue
    return pd.NaT

df["order_date"]    = df["order_date"].apply(parse_date)
df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")

unparsed_dates = df["order_date"].isna().sum()
log(f"\n[DATES]  order_date standardised to ISO 8601 | Unparseable: {unparsed_dates}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. CATEGORICAL STANDARDISATION
# ══════════════════════════════════════════════════════════════════════════════
# 4a. category
CATEGORY_MAP = {
    "electronics": "Electronics", "electronics": "Electronics", "ELECTRONICS": "Electronics",
    "clothing": "Clothing", "clothng": "Clothing",
    "home and kitchen": "Home & Kitchen", "home & kitchen": "Home & Kitchen",
    "books": "Books",
    "sports": "Sports", "sports & outdoors": "Sports",
}
df["category"] = (df["category"].str.strip().str.lower()
                    .map(lambda x: CATEGORY_MAP.get(x, x.title() if pd.notna(x) else np.nan)))

# 4b. payment_method
PAYMENT_MAP = {
    "credit card": "Credit Card", "debit card": "Debit Card",
    "upi": "UPI", "net banking": "Net Banking",
    "cash on delivery": "Cash on Delivery", "cod": "Cash on Delivery",
}
df["payment_method"] = (df["payment_method"].str.strip().str.lower()
                          .map(lambda x: PAYMENT_MAP.get(x, x.title() if pd.notna(x) else np.nan)))

# 4c. gender
GENDER_MAP = {"male": "Male", "m": "Male", "female": "Female", "f": "Female", "other": "Other"}
df["gender"] = (df["gender"].str.strip().str.lower()
                  .map(lambda x: GENDER_MAP.get(x, np.nan) if pd.notna(x) else np.nan))

# 4d. return_flag  →  boolean
RF_MAP = {"yes": True, "y": True, "no": False, "n": False}
df["return_flag"] = (df["return_flag"].str.strip().str.lower()
                       .map(lambda x: RF_MAP.get(x, np.nan) if pd.notna(x) else np.nan))

# 4e. order_status  → title-case
df["order_status"] = df["order_status"].str.strip().str.title()

log("\n[CATEGORIES]  category / payment_method / gender / return_flag / order_status standardised")

# ══════════════════════════════════════════════════════════════════════════════
# 5. MISSING VALUE TREATMENT
# ══════════════════════════════════════════════════════════════════════════════
# discount_pct  → impute with median (most common business approach)
median_disc = df["discount_pct"].median()
df["discount_pct"].fillna(median_disc, inplace=True)
log(f"\n[MISSING]  discount_pct → imputed with median ({median_disc:.2f})")

# category / payment_method → mode imputation
for col in ["category", "payment_method"]:
    mode_val = df[col].mode()[0]
    df[col].fillna(mode_val, inplace=True)
    log(f"           {col} → imputed with mode ('{mode_val}')")

# order_status → fill with 'Unknown'
df["order_status"].fillna("Unknown", inplace=True)
log("           order_status → NaN filled with 'Unknown'")

# return_flag → fill with False (assume not returned if unrecorded)
df["return_flag"].fillna(False, inplace=True)
log("           return_flag → NaN filled with False")

# gender → fill with 'Unspecified'
df["gender"].fillna("Unspecified", inplace=True)
log("           gender → NaN filled with 'Unspecified'")

# rows missing customer_name/email → flag but keep
df["has_contact_info"] = (~df["email"].isna()) | (~df["phone"].isna())

# ══════════════════════════════════════════════════════════════════════════════
# 6. OUTLIER DETECTION & TREATMENT (IQR method)
# ══════════════════════════════════════════════════════════════════════════════
def iqr_bounds(series):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR    = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

for col, floor, cap_method in [
    ("unit_price", 1,    "iqr"),
    ("quantity",   1,    "iqr"),
]:
    lo, hi = iqr_bounds(df[col])
    lo = max(lo, floor)        # logical minimum
    outliers = ((df[col] < lo) | (df[col] > hi)).sum()
    df[col] = df[col].clip(lower=lo, upper=hi)
    log(f"\n[OUTLIERS] {col}: {outliers} outliers clipped to [{lo:.1f}, {hi:.1f}]")

# ══════════════════════════════════════════════════════════════════════════════
# 7. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
REFERENCE_DATE = pd.Timestamp("2025-01-01")

# customer_age
df["customer_age"] = ((REFERENCE_DATE - df["date_of_birth"])
                       .dt.days // 365).astype("Int64")

# age_group
bins   = [0, 24, 34, 44, 54, 100]
labels = ["18-24", "25-34", "35-44", "45-54", "55+"]
df["age_group"] = pd.cut(df["customer_age"], bins=bins, labels=labels, right=True)

# total_revenue
df["total_revenue"] = (df["unit_price"] * df["quantity"] *
                       (1 - df["discount_pct"])).round(2)

# order date parts
df["order_year"]    = df["order_date"].dt.year
df["order_month"]   = df["order_date"].dt.month
df["order_quarter"] = df["order_date"].dt.quarter

log("\n[FEATURES]  customer_age, age_group, total_revenue, order_year/month/quarter created")

# ══════════════════════════════════════════════════════════════════════════════
# 8. FINAL VALIDATION & EXPORT
# ══════════════════════════════════════════════════════════════════════════════
# Drop rows still missing critical business keys
critical_cols = ["order_id", "order_date", "unit_price", "quantity"]
before_drop = len(df)
df.dropna(subset=critical_cols, inplace=True)
log(f"\n[FINAL DROP]  Dropped {before_drop - len(df)} rows missing critical columns")

# Reset index
df.reset_index(drop=True, inplace=True)

# Column order
COL_ORDER = [
    "order_id", "customer_name", "email", "phone",
    "date_of_birth", "customer_age", "age_group", "gender",
    "order_date", "order_year", "order_month", "order_quarter",
    "product_name", "category",
    "unit_price", "quantity", "discount_pct", "total_revenue",
    "payment_method", "order_status", "return_flag",
    "city", "state", "pincode", "has_contact_info",
]
df = df[COL_ORDER]

df.to_csv(DEST, index=False)

log("\n" + "=" * 70)
log(f"[DONE]  Clean dataset: {len(df)} rows × {len(df.columns)} columns → {DEST}")
log(f"        Remaining nulls: {df.isnull().sum().sum()}")
log("=" * 70)

# Summary stats for key numerics
log("\n── Numeric Summary (cleaned) ──")
log(df[["unit_price", "quantity", "discount_pct", "total_revenue", "customer_age"]]
    .describe().round(2).to_string())

save_log()
print(f"\nCleaning log saved → {LOG}")
