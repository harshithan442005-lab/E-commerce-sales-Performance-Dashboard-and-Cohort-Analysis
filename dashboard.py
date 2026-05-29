import pandas as pd

# =========================
# 1️⃣ LOAD DATA
# =========================
df = pd.read_csv("data/Updated_sales.csv")

print("Original Shape:", df.shape)

# =========================
# 2️⃣ CLEANING
# =========================

# Fix column spaces
df.columns = df.columns.str.strip()

# Rename columns
df = df.rename(columns={
    "Order ID": "OrderID",
    "Product": "Product",
    "Quantity Ordered": "Quantity",
    "Price Each": "Price",
    "Order Date": "OrderDate",
    "Purchase Address": "Address"
})

# Remove nulls
df = df.dropna()

# Convert types
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")

# Drop invalid rows
df = df.dropna(subset=["OrderDate", "Quantity", "Price"])

print("After Cleaning:", df.shape)

# =========================
# 3️⃣ PREPROCESSING
# =========================

# Revenue
df["Revenue"] = df["Quantity"] * df["Price"]

# Time features
df["OrderMonth"] = df["OrderDate"].dt.to_period("M")

# =========================
# 4️⃣ COHORT ANALYSIS
# =========================

# Cohort month (first purchase)
df["CohortMonth"] = df.groupby("OrderID")["OrderDate"] \
    .transform("min") \
    .dt.to_period("M")

# Cohort index
def cohort_index(row):
    return (row["OrderMonth"].year - row["CohortMonth"].year) * 12 + \
           (row["OrderMonth"].month - row["CohortMonth"].month)

df["CohortIndex"] = df.apply(cohort_index, axis=1)

# Cohort table
cohort_table = df.groupby(["CohortMonth", "CohortIndex"])["OrderID"] \
    .nunique() \
    .unstack()

# =========================
# 5️⃣ SAVE FILES
# =========================

df.to_csv("data/processed_sales.csv", index=False)
cohort_table.to_csv("data/cohort_table.csv")

print("✅ Analysis Completed Successfully!")