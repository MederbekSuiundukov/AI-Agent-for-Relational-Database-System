# %% [markdown]
# # 🗄️ Text-to-SQL Project — Database Setup
# **Dataset:** Olist Brazilian E-Commerce (Kaggle)
# **Target DB:** TiDB Serverless (MySQL-compatible)
#
# **Notebook Flow:**
# 1. Install & import dependencies
# 2. Download dataset via `kagglehub`
# 3. Load & clean DataFrames
# 4. Connect to TiDB with SQLAlchemy
# 5. Create schema (7 tables, PKs, FKs)
# 6. Insert cleaned data (~500k rows total)
# 7. Add composite indexes for query performance
# 8. Create VIEW: vw_sales_by_category
# 9. Create VIEW: vw_customer_order_history
#    (replaces stored procedure — not supported on TiDB Serverless)

# %% [markdown]
# ## Cell 1 — Install Dependencies

# %%
import subprocess, sys

packages = [
    "kagglehub", "pandas", "numpy", "sqlalchemy",
    "pymysql", "cryptography", "python-dotenv"
]
subprocess.check_call([sys.executable, "-m", "pip", "install", *packages, "-q"])
print("✅ All packages installed.")

# %% [markdown]
# ## Cell 2 — Imports & Environment Variables

# %%
import os
import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
from dotenv import load_dotenv, find_dotenv

# Load .env from the project directory
load_dotenv(find_dotenv(usecwd=True))

TIDB_HOST     = os.getenv("TIDB_HOST",     "gateway01.eu-central-1.prod.aws.tidbcloud.com")
TIDB_PORT     = int(os.getenv("TIDB_PORT", "4000"))
TIDB_USER     = os.getenv("TIDB_USER",     "your_prefix.root")
TIDB_PASSWORD = os.getenv("TIDB_PASSWORD", "")
TIDB_DB       = os.getenv("TIDB_DB",       "ecommerce")

print(f"🔗 Will connect to TiDB host: {TIDB_HOST}:{TIDB_PORT}/{TIDB_DB}")

# %% [markdown]
# ## Cell 3 — Download Olist Dataset via KaggleHub

# %%
import kagglehub

# Requires ~/.kaggle/kaggle.json  OR  env vars KAGGLE_USERNAME + KAGGLE_KEY
print("⬇️  Downloading Olist Brazilian E-Commerce dataset …")
dataset_path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
print(f"✅ Dataset downloaded to: {dataset_path}")

csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
print(f"📁 Found {len(csv_files)} CSV files:")
for f in csv_files:
    print(f"   • {f}")

# %% [markdown]
# ## Cell 4 — Load Raw DataFrames

# %%
def load_csv(filename: str) -> pd.DataFrame:
    path = os.path.join(dataset_path, filename)
    df = pd.read_csv(path)
    print(f"  Loaded {filename}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df

print("📂 Loading CSVs …")
raw_orders        = load_csv("olist_orders_dataset.csv")
raw_order_items   = load_csv("olist_order_items_dataset.csv")
raw_products      = load_csv("olist_products_dataset.csv")
raw_customers     = load_csv("olist_customers_dataset.csv")
raw_sellers       = load_csv("olist_sellers_dataset.csv")
raw_order_reviews = load_csv("olist_order_reviews_dataset.csv")
raw_payments      = load_csv("olist_order_payments_dataset.csv")
raw_categories    = load_csv("product_category_name_translation.csv")
raw_geolocation   = load_csv("olist_geolocation_dataset.csv")

# %% [markdown]
# ## Cell 5 — Data Cleaning

# %%
def clean_str_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from all string/object columns."""
    str_cols = df.select_dtypes(include=["object", "str"]).columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())
    return df

def to_datetime_cols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def truncate_str(df: pd.DataFrame, col: str, max_len: int) -> pd.DataFrame:
    if col in df.columns:
        df[col] = df[col].astype(str).str[:max_len]
    return df


# ── 1. CUSTOMERS ──────────────────────────────────────────────────────────────
print("🧹 Cleaning customers …")
customers = (
    raw_customers
    .drop_duplicates(subset="customer_id")
    .dropna(subset=["customer_id", "customer_unique_id"])
    .pipe(clean_str_cols)
    .rename(columns={
        "customer_zip_code_prefix": "zip_code",
        "customer_city":            "city",
        "customer_state":           "state",
    })
)
customers = truncate_str(customers, "city",  100)
customers = truncate_str(customers, "state",  10)
print(f"  ✅ customers: {len(customers):,} rows")


# ── 2. SELLERS ────────────────────────────────────────────────────────────────
print("🧹 Cleaning sellers …")
sellers = (
    raw_sellers
    .drop_duplicates(subset="seller_id")
    .dropna(subset=["seller_id"])
    .pipe(clean_str_cols)
    .rename(columns={
        "seller_zip_code_prefix": "zip_code",
        "seller_city":            "city",
        "seller_state":           "state",
    })
)
sellers = truncate_str(sellers, "city",  100)
sellers = truncate_str(sellers, "state",  10)
print(f"  ✅ sellers: {len(sellers):,} rows")


# ── 3. PRODUCTS ───────────────────────────────────────────────────────────────
print("🧹 Cleaning products …")
products = (
    raw_products
    .drop_duplicates(subset="product_id")
    .dropna(subset=["product_id"])
    .merge(raw_categories, on="product_category_name", how="left")
    .pipe(clean_str_cols)
)
products["product_category_name_english"] = (
    products["product_category_name_english"]
    .fillna(products["product_category_name"])
    .fillna("unknown")
)
products = products.rename(columns={
    "product_category_name_english": "category",
    "product_weight_g":              "weight_g",
    "product_length_cm":             "length_cm",
    "product_height_cm":             "height_cm",
    "product_width_cm":              "width_cm",
    "product_photos_qty":            "photos_qty",
})
for col in ["weight_g", "length_cm", "height_cm", "width_cm", "photos_qty"]:
    products[col] = pd.to_numeric(products[col], errors="coerce").fillna(0)
products = truncate_str(products, "category", 100)
products = products[["product_id", "category", "weight_g",
                      "length_cm", "height_cm", "width_cm", "photos_qty"]]
print(f"  ✅ products: {len(products):,} rows")


# ── 4. ORDERS ─────────────────────────────────────────────────────────────────
print("🧹 Cleaning orders …")
date_cols = [
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
orders = (
    raw_orders
    .drop_duplicates(subset="order_id")
    .dropna(subset=["order_id", "customer_id"])
    .pipe(clean_str_cols)
    .pipe(to_datetime_cols, date_cols)
)
orders = orders[orders["customer_id"].isin(customers["customer_id"])]
orders = truncate_str(orders, "order_status", 30)
print(f"  ✅ orders: {len(orders):,} rows")


# ── 5. ORDER ITEMS ────────────────────────────────────────────────────────────
print("🧹 Cleaning order_items …")
order_items = (
    raw_order_items
    .drop_duplicates()
    .dropna(subset=["order_id", "product_id", "seller_id"])
    .pipe(clean_str_cols)
    .pipe(to_datetime_cols, ["shipping_limit_date"])
)
order_items = order_items[
    order_items["order_id"].isin(orders["order_id"]) &
    order_items["product_id"].isin(products["product_id"]) &
    order_items["seller_id"].isin(sellers["seller_id"])
]
for col in ["price", "freight_value"]:
    order_items[col] = pd.to_numeric(order_items[col], errors="coerce").fillna(0.0)
print(f"  ✅ order_items: {len(order_items):,} rows")


# ── 6. ORDER REVIEWS ──────────────────────────────────────────────────────────
print("🧹 Cleaning order_reviews …")
reviews = (
    raw_order_reviews
    .drop_duplicates(subset="review_id")
    .dropna(subset=["review_id", "order_id", "review_score"])
    .pipe(clean_str_cols)
    .pipe(to_datetime_cols, ["review_creation_date", "review_answer_timestamp"])
)
reviews = reviews[reviews["order_id"].isin(orders["order_id"])]
reviews["review_score"] = pd.to_numeric(
    reviews["review_score"], errors="coerce").fillna(3).astype(int)
reviews["review_comment_title"] = (
    reviews.get("review_comment_title", pd.Series(dtype=str)).fillna("").str[:200])
reviews["review_comment_message"] = (
    reviews.get("review_comment_message", pd.Series(dtype=str)).fillna("").str[:1000])
keep = ["review_id", "order_id", "review_score",
        "review_comment_title", "review_comment_message", "review_creation_date"]
reviews = reviews[[c for c in keep if c in reviews.columns]]
print(f"  ✅ reviews: {len(reviews):,} rows")


# ── 7. PAYMENTS ───────────────────────────────────────────────────────────────
print("🧹 Cleaning payments …")
payments = (
    raw_payments
    .drop_duplicates()
    .dropna(subset=["order_id", "payment_value"])
    .pipe(clean_str_cols)
)
payments = payments[payments["order_id"].isin(orders["order_id"])]
payments["payment_value"] = pd.to_numeric(
    payments["payment_value"], errors="coerce").fillna(0.0)
payments["payment_installments"] = pd.to_numeric(
    payments["payment_installments"], errors="coerce").fillna(1).astype(int)
payments = truncate_str(payments, "payment_type", 30)
print(f"  ✅ payments: {len(payments):,} rows")


# ── Row count summary ─────────────────────────────────────────────────────────
print("\n📊 Final row counts:")
for name, df in {
    "customers": customers, "sellers": sellers, "products": products,
    "orders": orders, "order_items": order_items,
    "reviews": reviews, "payments": payments,
}.items():
    flag = "✅" if len(df) >= 1000 else "⚠️  < 1000 rows!"
    print(f"  {flag} {name}: {len(df):,}")

# %% [markdown]
# ## Cell 6 — Connect to TiDB via SQLAlchemy

# %%
connection_url = (
    f"mysql+pymysql://{TIDB_USER}:{TIDB_PASSWORD}"
    f"@{TIDB_HOST}:{TIDB_PORT}/{TIDB_DB}"
    f"?charset=utf8mb4"
)

engine = sa.create_engine(
    connection_url,
    connect_args={"ssl": {"ssl_mode": "VERIFY_IDENTITY"}},
    pool_recycle=3600,
    echo=False,
)

with engine.connect() as conn:
    version = conn.execute(text("SELECT VERSION()")).scalar()
    print(f"✅ Connected! TiDB/MySQL version: {version}")

# %% [markdown]
# ## Cell 7 — Create Schema (DROP → CREATE with PKs & FKs)

# %%
DDL_STATEMENTS = [
    "SET FOREIGN_KEY_CHECKS = 0;",

    "DROP TABLE IF EXISTS payments;",
    "DROP TABLE IF EXISTS order_reviews;",
    "DROP TABLE IF EXISTS order_items;",
    "DROP TABLE IF EXISTS orders;",
    "DROP TABLE IF EXISTS products;",
    "DROP TABLE IF EXISTS sellers;",
    "DROP TABLE IF EXISTS customers;",

    """
    CREATE TABLE customers (
        customer_id         VARCHAR(50)  NOT NULL,
        customer_unique_id  VARCHAR(50)  NOT NULL,
        zip_code            VARCHAR(10),
        city                VARCHAR(100),
        state               VARCHAR(10),
        PRIMARY KEY (customer_id),
        INDEX idx_customers_state (state),
        INDEX idx_customers_city  (city)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,

    """
    CREATE TABLE sellers (
        seller_id   VARCHAR(50) NOT NULL,
        zip_code    VARCHAR(10),
        city        VARCHAR(100),
        state       VARCHAR(10),
        PRIMARY KEY (seller_id),
        INDEX idx_sellers_state (state)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,

    """
    CREATE TABLE products (
        product_id   VARCHAR(50)   NOT NULL,
        category     VARCHAR(100),
        weight_g     DECIMAL(10,2) DEFAULT 0,
        length_cm    DECIMAL(10,2) DEFAULT 0,
        height_cm    DECIMAL(10,2) DEFAULT 0,
        width_cm     DECIMAL(10,2) DEFAULT 0,
        photos_qty   SMALLINT      DEFAULT 0,
        PRIMARY KEY (product_id),
        INDEX idx_products_category (category)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,

    """
    CREATE TABLE orders (
        order_id                        VARCHAR(50)  NOT NULL,
        customer_id                     VARCHAR(50)  NOT NULL,
        order_status                    VARCHAR(30),
        order_purchase_timestamp        DATETIME,
        order_approved_at               DATETIME,
        order_delivered_carrier_date    DATETIME,
        order_delivered_customer_date   DATETIME,
        order_estimated_delivery_date   DATETIME,
        PRIMARY KEY (order_id),
        CONSTRAINT fk_orders_customer FOREIGN KEY (customer_id)
            REFERENCES customers(customer_id) ON DELETE CASCADE,
        INDEX idx_orders_customer_id  (customer_id),
        INDEX idx_orders_status       (order_status),
        INDEX idx_orders_purchase_ts  (order_purchase_timestamp)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,

    """
    CREATE TABLE order_items (
        order_id             VARCHAR(50)   NOT NULL,
        order_item_id        SMALLINT      NOT NULL,
        product_id           VARCHAR(50)   NOT NULL,
        seller_id            VARCHAR(50)   NOT NULL,
        shipping_limit_date  DATETIME,
        price                DECIMAL(10,2) DEFAULT 0.00,
        freight_value        DECIMAL(10,2) DEFAULT 0.00,
        PRIMARY KEY (order_id, order_item_id),
        CONSTRAINT fk_items_order   FOREIGN KEY (order_id)
            REFERENCES orders(order_id)   ON DELETE CASCADE,
        CONSTRAINT fk_items_product FOREIGN KEY (product_id)
            REFERENCES products(product_id),
        CONSTRAINT fk_items_seller  FOREIGN KEY (seller_id)
            REFERENCES sellers(seller_id),
        INDEX idx_items_product_id (product_id),
        INDEX idx_items_seller_id  (seller_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,

    """
    CREATE TABLE order_reviews (
        review_id              VARCHAR(50) NOT NULL,
        order_id               VARCHAR(50) NOT NULL,
        review_score           TINYINT     NOT NULL DEFAULT 3,
        review_comment_title   VARCHAR(200),
        review_comment_message VARCHAR(1000),
        review_creation_date   DATETIME,
        PRIMARY KEY (review_id),
        CONSTRAINT fk_reviews_order FOREIGN KEY (order_id)
            REFERENCES orders(order_id) ON DELETE CASCADE,
        INDEX idx_reviews_order_id (order_id),
        INDEX idx_reviews_score    (review_score)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,

    """
    CREATE TABLE payments (
        order_id               VARCHAR(50)   NOT NULL,
        payment_sequential     TINYINT       NOT NULL DEFAULT 1,
        payment_type           VARCHAR(30),
        payment_installments   SMALLINT      DEFAULT 1,
        payment_value          DECIMAL(10,2) DEFAULT 0.00,
        PRIMARY KEY (order_id, payment_sequential),
        CONSTRAINT fk_payments_order FOREIGN KEY (order_id)
            REFERENCES orders(order_id) ON DELETE CASCADE,
        INDEX idx_payments_type (payment_type)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,

    "SET FOREIGN_KEY_CHECKS = 1;",
]

with engine.connect() as conn:
    for stmt in DDL_STATEMENTS:
        stmt = stmt.strip()
        if stmt:
            conn.execute(text(stmt))
    conn.commit()

print("✅ Schema created successfully.")

# %% [markdown]
# ## Cell 8 — Insert Data

# %%
def insert_df(df: pd.DataFrame, table_name: str, chunksize: int = 5000):
    df = df.where(pd.notnull(df), None)
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists="append",
        index=False,
        chunksize=chunksize,
        method="multi",
    )
    print(f"  ✅ Inserted {len(df):,} rows → {table_name}")

print("📥 Inserting data …")
insert_df(customers,   "customers")
insert_df(sellers,     "sellers")
insert_df(products,    "products")
insert_df(orders,      "orders")
insert_df(order_items, "order_items")
insert_df(reviews,     "order_reviews")
insert_df(payments,    "payments")
print("\n🎉 All data inserted!")

# %% [markdown]
# ## Cell 9 — Row Count Verification

# %%
table_names = [
    "customers", "sellers", "products",
    "orders", "order_items", "order_reviews", "payments"
]

print("📊 Row counts in TiDB:\n")
with engine.connect() as conn:
    for tbl in table_names:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
        flag = "✅" if count >= 1000 else "⚠️  < 1,000 rows!"
        print(f"  {flag}  {tbl}: {count:,}")

# %% [markdown]
# ## Cell 10 — Additional Composite Indexes

# %%
EXTRA_INDEXES = [
    # Composite: status + date — covers filtered time-range queries
    "CREATE INDEX IF NOT EXISTS idx_orders_status_date ON orders (order_status, order_purchase_timestamp);",
    # Covering index for payment aggregation
    "CREATE INDEX IF NOT EXISTS idx_payments_type_value ON payments (payment_type, payment_value);",
    # Revenue by seller queries
    "CREATE INDEX IF NOT EXISTS idx_items_seller_price ON order_items (seller_id, price);",
    # Product-level analytics
    "CREATE INDEX IF NOT EXISTS idx_items_product_price ON order_items (product_id, price, freight_value);",
]

with engine.connect() as conn:
    for idx_sql in EXTRA_INDEXES:
        try:
            conn.execute(text(idx_sql.strip()))
            conn.commit()
            print(f"  ✅ Index created.")
        except Exception as e:
            print(f"  ⚠️  Skipped (may already exist): {e}")

print("Done with index creation.")

# %% [markdown]
# ## Cell 11 — Create VIEW: vw_sales_by_category

# %%
VIEW_SQL = """
CREATE OR REPLACE VIEW vw_sales_by_category AS
SELECT
    p.category,
    COUNT(DISTINCT oi.order_id)     AS total_orders,
    COUNT(oi.product_id)            AS total_items_sold,
    ROUND(SUM(oi.price), 2)         AS total_revenue,
    ROUND(AVG(oi.price), 2)         AS avg_item_price,
    ROUND(SUM(oi.freight_value), 2) AS total_freight,
    ROUND(AVG(r.review_score), 2)   AS avg_review_score,
    COUNT(DISTINCT oi.seller_id)    AS unique_sellers
FROM
    order_items oi
    JOIN products p ON oi.product_id = p.product_id
    JOIN orders   o ON oi.order_id   = o.order_id
    LEFT JOIN order_reviews r ON r.order_id = o.order_id
WHERE
    o.order_status = 'delivered'
GROUP BY
    p.category
ORDER BY
    total_revenue DESC;
"""

with engine.connect() as conn:
    conn.execute(text(VIEW_SQL))
    conn.commit()
print("✅ VIEW `vw_sales_by_category` created.")

# Quick preview
with engine.connect() as conn:
    result = conn.execute(text("SELECT * FROM vw_sales_by_category LIMIT 10"))
    df_view = pd.DataFrame(result.fetchall(), columns=result.keys())

print("\n📊 Top 10 categories by revenue:")
print(df_view.to_string(index=False))

# %% [markdown]
# ## Cell 12 — Create VIEW: vw_customer_order_history
# NOTE: TiDB Serverless does NOT support stored procedures (error 8108).
# We use a second VIEW to provide the same customer history capability.

# %%
with engine.connect() as conn:
    conn.execute(text("DROP VIEW IF EXISTS vw_customer_order_history"))
    conn.execute(text("""
        CREATE VIEW vw_customer_order_history AS
        SELECT
            c.customer_unique_id,
            o.order_id,
            o.order_status,
            o.order_purchase_timestamp,
            o.order_delivered_customer_date,
            p.category                      AS product_category,
            oi.price,
            oi.freight_value,
            pay.payment_type,
            pay.payment_value,
            r.review_score
        FROM
            customers c
            JOIN orders        o   ON c.customer_id  = o.customer_id
            JOIN order_items   oi  ON o.order_id      = oi.order_id
            JOIN products      p   ON oi.product_id   = p.product_id
            LEFT JOIN payments pay ON o.order_id      = pay.order_id
                                   AND pay.payment_sequential = 1
            LEFT JOIN order_reviews r ON o.order_id   = r.order_id
        ORDER BY
            o.order_purchase_timestamp DESC
    """))
    conn.commit()
print("✅ VIEW `vw_customer_order_history` created.")

# Test it with a real customer
with engine.connect() as conn:
    sample_uid = conn.execute(
        text("SELECT customer_unique_id FROM customers LIMIT 1")
    ).scalar()
    result = conn.execute(
        text("SELECT * FROM vw_customer_order_history "
             "WHERE customer_unique_id = :uid LIMIT 5"),
        {"uid": sample_uid}
    )
    df_test = pd.DataFrame(result.fetchall(), columns=result.keys())

print(f"\n🔎 Sample history for customer '{sample_uid}':")
print(df_test.to_string(index=False) if not df_test.empty else "No orders found.")

# %% [markdown]
# ## Cell 13 — Final Summary

# %%
print("=" * 60)
print("  DATABASE SETUP COMPLETE")
print("=" * 60)

with engine.connect() as conn:
    tables_result = conn.execute(text("SHOW TABLES")).fetchall()
    print(f"\n📋 All objects in '{TIDB_DB}':")
    for (tbl,) in tables_result:
        cnt = conn.execute(text(f"SELECT COUNT(*) FROM `{tbl}`")).scalar()
        print(f"  • {tbl:<30} {cnt:>10,} rows")

print("\n✅ Setup complete! You can now run the Streamlit app.")
print("   Make sure TIDB_DB = 'ecommerce' is set in Streamlit Cloud secrets.")
