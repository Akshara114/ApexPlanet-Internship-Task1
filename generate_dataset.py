"""
generate_dataset.py
Generates a realistic but intentionally messy Sales/E-commerce dataset.
Intentional issues introduced:
  - Missing values in key columns
  - Duplicate rows
  - Inconsistent date formats
  - Mixed-case / typo categories
  - Outlier prices and quantities
  - Phone numbers in inconsistent formats
  - Free-text gender entries
"""

import pandas as pd
import numpy as np
from faker import Faker
import random, string
from datetime import datetime, timedelta

fake = Faker("en_IN")
np.random.seed(42)
random.seed(42)

N = 500

# ── helpers ────────────────────────────────────────────────────────────────
def rand_date(start="2022-01-01", end="2024-12-31"):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    return s + timedelta(days=random.randint(0, (e - s).days))

DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%d %b %Y"]

CATEGORIES     = ["Electronics", "electronics", "ELECTRONICS",
                  "Clothing", "clothing", "Clothng",
                  "Home & Kitchen", "Home and Kitchen", "home & kitchen",
                  "Books", "books", "BOOKS",
                  "Sports", "Sports & Outdoors"]

PAYMENT        = ["Credit Card", "credit card", "CREDIT CARD",
                  "Debit Card", "UPI", "upi", "Net Banking",
                  "Cash on Delivery", "COD", "cod"]

GENDER_RAW     = ["Male", "male", "M", "m", "MALE",
                  "Female", "female", "F", "f", "FEMALE",
                  "Other", "other", None]

STATUS         = ["Delivered", "Shipped", "Pending",
                  "Returned", "Cancelled", None]

# ── base data ──────────────────────────────────────────────────────────────
records = []
for i in range(N):
    dob  = fake.date_of_birth(minimum_age=18, maximum_age=65)
    date = rand_date()
    fmt  = random.choice(DATE_FORMATS)

    records.append({
        "order_id"        : f"ORD-{10000 + i}",
        "customer_name"   : fake.name() if random.random() > 0.02 else None,
        "email"           : fake.email() if random.random() > 0.05 else None,
        "phone"           : fake.phone_number() if random.random() > 0.08 else None,
        "date_of_birth"   : dob.strftime("%Y-%m-%d"),
        "gender"          : random.choice(GENDER_RAW),
        "order_date"      : date.strftime(fmt),          # messy formats
        "product_name"    : fake.bs().title(),
        "category"        : random.choice(CATEGORIES),
        "unit_price"      : round(random.uniform(50, 5000), 2) if random.random() > 0.01
                            else random.choice([-99, 0, 999999]),   # outliers
        "quantity"        : random.randint(1, 10) if random.random() > 0.01
                            else random.choice([0, -1, 500]),       # outliers
        "discount_pct"    : round(random.uniform(0, 0.5), 2) if random.random() > 0.1
                            else None,
        "payment_method"  : random.choice(PAYMENT),
        "order_status"    : random.choice(STATUS),
        "city"            : fake.city() if random.random() > 0.04 else None,
        "state"           : fake.state(),
        "pincode"         : fake.postcode(),
        "return_flag"     : random.choice(["Yes", "No", "yes", "no", "Y", "N", None]),
    })

df = pd.DataFrame(records)

# ── inject missing values randomly ────────────────────────────────────────
for col, frac in [("discount_pct", 0.10), ("order_status", 0.05),
                  ("category", 0.03), ("payment_method", 0.04)]:
    df.loc[df.sample(frac=frac, random_state=1).index, col] = np.nan

# ── inject duplicate rows (~3 %) ──────────────────────────────────────────
dupes = df.sample(n=15, random_state=7)
df    = pd.concat([df, dupes], ignore_index=True)

# ── save ──────────────────────────────────────────────────────────────────
# TO (saves in the same folder as the script)
df.to_csv("raw_sales_data.csv", index=False)
print(f"Raw dataset saved — {len(df)} rows × {len(df.columns)} columns")
print(df.isnull().sum()[df.isnull().sum() > 0].to_string())
