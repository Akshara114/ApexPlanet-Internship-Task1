# 📊 Task 1 — Data Immersion & Wrangling
> **Internship Project | Sales / E-commerce Dataset**
> Timeline: 10 Days

---

## 🎯 Objective
Rapidly acquaint with the dataset and master the critical first step of any data analysis pipeline — **acquiring, cleaning, and preparing data** for downstream analytics.

---

## 📁 Repository Structure

```
task1_data_wrangling/
│
├── raw_sales_data.csv          ← Original dataset (messy, 515 rows × 18 cols)
├── cleaned_sales_data.csv      ← Final analysis-ready dataset (500 rows × 25 cols)
│
├── generate_dataset.py         ← Script to reproduce the raw dataset
├── data_cleaning.py            ← Full data cleaning & transformation pipeline
├── data_dictionary.csv         ← Column-level documentation
├── cleaning_log.txt            ← Auto-generated log of all cleaning actions
│
└── README.md
```

---

## 🗂️ Dataset Overview

| Property      | Raw Dataset       | Cleaned Dataset       |
|---------------|-------------------|----------------------|
| Rows          | 515               | 500                  |
| Columns       | 18                | 25 (7 engineered)    |
| Duplicates    | 15                | 0                    |
| Missing cells | 436               | ~517 (only nullable) |
| Date formats  | 4 inconsistent    | 1 (ISO 8601)         |

### Intentional Data Quality Issues (Raw)

| Issue | Column(s) | Action Taken |
|---|---|---|
| Duplicate rows | `order_id` | Dropped 15 duplicates |
| Mixed date formats | `order_date` | Multi-format parser → ISO 8601 |
| Case/typo inconsistency | `category`, `payment_method`, `gender` | Mapped to canonical values |
| Missing values | `discount_pct` (20%), `order_status` (18%), `return_flag` (15%) | Median/Mode/Constant imputation |
| Outliers | `unit_price`, `quantity` | IQR clipping (floor = 1) |
| Inconsistent booleans | `return_flag` | Mapped Yes/Y/No/N → True/False |

---

## ⚙️ How to Run

### Prerequisites
```bash
pip install pandas numpy faker openpyxl
```

### Step 1 — Generate raw dataset
```bash
python generate_dataset.py
# Output: raw_sales_data.csv (515 rows)
```

### Step 2 — Run cleaning pipeline
```bash
python data_cleaning.py
# Output: cleaned_sales_data.csv, cleaning_log.txt
```

---

## 🔧 Cleaning Steps Explained

### 1. Duplicate Removal
```python
df.drop_duplicates(subset="order_id", keep="first", inplace=True)
```

### 2. Date Standardisation
```python
DATE_FMTS = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%d %b %Y"]
df["order_date"] = df["order_date"].apply(parse_date)
```

### 3. Categorical Standardisation
```python
CATEGORY_MAP = {"electronics": "Electronics", "clothng": "Clothing", ...}
df["category"] = df["category"].str.lower().map(CATEGORY_MAP)
```

### 4. Missing Value Treatment
| Column | Strategy | Reasoning |
|---|---|---|
| `discount_pct` | Median imputation | Skew-robust |
| `category`, `payment_method` | Mode imputation | Most frequent is best estimate |
| `order_status` | Fill `"Unknown"` | Preserve row; flag for review |
| `return_flag` | Fill `False` | Conservative business assumption |
| `gender` | Fill `"Unspecified"` | Respect privacy |

### 5. Outlier Treatment (IQR)
```python
Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
IQR = Q3 - Q1
df[col] = df[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)
```

### 6. Feature Engineering
| Feature | Formula | Purpose |
|---|---|---|
| `customer_age` | `(2025-01-01 − date_of_birth) ÷ 365` | Demographics |
| `age_group` | `pd.cut(customer_age, bins)` | Cohort segmentation |
| `total_revenue` | `unit_price × quantity × (1 − discount_pct)` | Core KPI |
| `order_year/month/quarter` | `.dt.year / .dt.month / .dt.quarter` | Time series |
| `has_contact_info` | `email is not null OR phone is not null` | Reachability flag |

---

## 📖 Data Dictionary
See [`data_dictionary.csv`](data_dictionary.csv) for full column documentation including:
- Data type & format
- Business relevance
- Source notes / cleaning decisions

---

## 📈 Key Stats (Cleaned Dataset)

| Metric | Value |
|---|---|
| Average Order Revenue | ₹9,735 |
| Average Customer Age | 40 years |
| Average Discount | 26% |
| Most Common Category | Electronics |
| Most Common Payment | Cash on Delivery |

---

## 🎥 LinkedIn Video Walkthrough
*(3–5 min video — link to be added after upload)*

**Video outline:**
1. Show the raw data and describe the issues found
2. Walk through each cleaning step in the script
3. Compare raw vs cleaned stats
4. Demo a quick pivot table / chart from the cleaned data

---

## 📝 Deliverables Checklist

- [x] Data dictionary (`data_dictionary.csv`)
- [x] Cleaning script (`data_cleaning.py`)
- [x] Cleaned dataset (`cleaned_sales_data.csv`)
- [x] Cleaning log (`cleaning_log.txt`)
- [ ] LinkedIn video walkthrough *(pending upload)*

---

*Generated as part of a Data Analytics Internship — Task 1*
