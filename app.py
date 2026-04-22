"""
app.py — Text-to-SQL AI Agent
==============================
Streamlit UI that wraps a LangChain SQL Agent powered by the Grok (xAI) LLM,
connected to a TiDB Serverless MySQL-compatible database.

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    App → Settings → Secrets:
        TIDB_HOST     = "gateway01.eu-central-1.prod.aws.tidbcloud.com"
        TIDB_PORT     = "4000"
        TIDB_USER     = "your_prefix.root"
        TIDB_PASSWORD = "your_password"
        TIDB_DB       = "ecommerce"
        XAI_API_KEY   = "xai-..."
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import re

# ── Third-party ───────────────────────────────────────────────────────────────
import pandas as pd
import streamlit as st
import plotly.express as px
from sqlalchemy import create_engine, text

# ── LangChain ─────────────────────────────────────────────────────────────────
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SQL Brain — AI Database Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,600&display=swap');

:root {
    --bg:      #0d0f14;
    --surface: #161a24;
    --border:  #252c3d;
    --accent:  #6ee7b7;
    --accent2: #7dd3fc;
    --text:    #e2e8f0;
    --muted:   #64748b;
    --mono:    'Space Mono', monospace;
    --sans:    'DM Sans', sans-serif;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans);
}

.hero-title {
    font-family: var(--mono);
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 0.15rem;
}
.hero-sub {
    font-family: var(--sans);
    font-size: 0.88rem;
    color: var(--muted);
    margin-bottom: 1.5rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.msg-user, .msg-bot {
    padding: 0.85rem 1.1rem;
    border-radius: 12px;
    margin-bottom: 0.75rem;
    font-size: 0.95rem;
    line-height: 1.6;
    max-width: 92%;
}
.msg-user {
    background: var(--surface);
    border: 1px solid var(--border);
    margin-left: auto;
    border-bottom-right-radius: 2px;
}
.msg-bot {
    background: #0f2a22;
    border: 1px solid #1f4535;
    border-bottom-left-radius: 2px;
}
.msg-label {
    font-family: var(--mono);
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.35rem;
    opacity: 0.55;
}

.sql-block {
    background: #0a0c10;
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-family: var(--mono);
    font-size: 0.8rem;
    color: var(--accent2);
    overflow-x: auto;
    margin: 0.5rem 0 0.9rem 0;
    white-space: pre-wrap;
}

.chip-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin: 0.6rem 0 1rem 0; }
.chip {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    font-size: 0.78rem;
    font-family: var(--mono);
    color: var(--accent);
}

section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

div[data-testid="stChatInput"] textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
    font-size: 0.95rem !important;
    border-radius: 10px !important;
}

.stButton > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    border-radius: 6px !important;
    padding: 0.3rem 0.8rem !important;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

.stSpinner > div { border-top-color: var(--accent) !important; }
.stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECRETS / ENV helper
# ═══════════════════════════════════════════════════════════════════════════════

def get_secret(key: str, default: str = "") -> str:
    """Read from Streamlit secrets first, then fall back to env vars."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, default)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATABASE CONNECTION
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_engine():
    host     = get_secret("TIDB_HOST",     "localhost")
    port     = get_secret("TIDB_PORT",     "4000")
    user     = get_secret("TIDB_USER",     "root")
    password = get_secret("TIDB_PASSWORD", "")
    db       = get_secret("TIDB_DB",       "ecommerce")

    url = (
        f"mysql+pymysql://{user}:{password}"
        f"@{host}:{port}/{db}?charset=utf8mb4"
    )
    engine = create_engine(
        url,
        connect_args={"ssl": {"ssl_mode": "VERIFY_IDENTITY"}},
        pool_recycle=3600,
        pool_pre_ping=True,
    )
    return engine


@st.cache_resource(show_spinner=False)
def get_langchain_db(_engine):
    """
    Wrap SQLAlchemy engine in a LangChain SQLDatabase object.
    Includes all 7 base tables + 2 views for maximum agent context.
    """
    return SQLDatabase(
        engine=_engine,
        sample_rows_in_table_info=3,
        include_tables=[
            "customers",
            "sellers",
            "products",
            "orders",
            "order_items",
            "order_reviews",
            "payments",
            "vw_sales_by_category",
            "vw_customer_order_history",
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM + AGENT
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_agent(_db):
    """Build a LangChain SQL agent using the Grok (xAI) LLM."""
    xai_api_key = get_secret("XAI_API_KEY", "")
    if not xai_api_key:
        st.error("❌ XAI_API_KEY not set. Add it to Streamlit secrets.")
        st.stop()

    llm = ChatOpenAI(
        model="grok-3-mini",
        api_key=xai_api_key,
        base_url="https://api.x.ai/v1",
        temperature=0,
        max_tokens=2048,
    )

    toolkit = SQLDatabaseToolkit(db=_db, llm=llm)

    system_message = """You are an expert SQL analyst for an e-commerce business.
The database is the Olist Brazilian E-Commerce dataset with these objects:

TABLES:
  • customers        — buyer profiles: customer_id, customer_unique_id, city, state, zip_code
  • sellers          — seller profiles: seller_id, city, state, zip_code
  • products         — product_id, category, weight_g, length_cm, height_cm, width_cm, photos_qty
  • orders           — order_id, customer_id, order_status, order_purchase_timestamp, delivery dates
  • order_items      — order_id, product_id, seller_id, price, freight_value, shipping_limit_date
  • order_reviews    — review_id, order_id, review_score (1-5), review_comment_title, review_comment_message
  • payments         — order_id, payment_type, payment_installments, payment_value

VIEWS (pre-aggregated — prefer these for performance):
  • vw_sales_by_category      — category, total_orders, total_items_sold, total_revenue,
                                avg_item_price, total_freight, avg_review_score, unique_sellers
  • vw_customer_order_history — customer_unique_id, order_id, order_status, order_purchase_timestamp,
                                product_category, price, freight_value, payment_type,
                                payment_value, review_score

RULES:
1. Always write syntactically correct MySQL SQL.
2. Use vw_sales_by_category for category-level revenue or review questions.
3. Use vw_customer_order_history for customer history lookups.
4. Always add LIMIT 200 unless the user requests more.
5. After retrieving data, give a clear, concise business interpretation.
6. If a question is ambiguous, state your assumption before querying.
7. Never expose passwords or connection strings in your response.
"""

    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type="openai-tools",
        verbose=True,
        max_iterations=15,
        max_execution_time=60,
        handle_parsing_errors=True,
        system_message=system_message,
    )
    return agent


# ═══════════════════════════════════════════════════════════════════════════════
#  SQL EXTRACTION + SAFE EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_sql_safe(engine, query: str) -> pd.DataFrame | None:
    """Execute a SELECT query and return a DataFrame, or None on error."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            return pd.DataFrame(result.fetchall(), columns=result.keys())
    except Exception:
        return None


def extract_last_select(intermediate_steps: list) -> str | None:
    """
    Walk agent intermediate steps and return the last SELECT query executed.
    Handles both dict and string tool_input formats.
    """
    last_sql = None
    for step in intermediate_steps:
        try:
            action, _ = step
            tool_input = getattr(action, "tool_input", "")
            if isinstance(tool_input, dict):
                tool_input = tool_input.get("query", "")
            if isinstance(tool_input, str) and tool_input.strip().upper().startswith("SELECT"):
                last_sql = tool_input.strip()
        except Exception:
            continue
    return last_sql


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTO CHART
# ═══════════════════════════════════════════════════════════════════════════════

def try_auto_chart(df: pd.DataFrame):
    """Heuristically render a Plotly chart from a result DataFrame."""
    if df is None or df.empty or len(df) < 2:
        return None

    num_cols  = df.select_dtypes(include="number").columns.tolist()
    cat_cols  = df.select_dtypes(exclude="number").columns.tolist()
    date_cols = [c for c in df.columns
                 if any(kw in c.lower() for kw in ["date", "timestamp", "month", "year"])]

    # Time-series line chart
    if date_cols and num_cols:
        x_col, y_col = date_cols[0], num_cols[0]
        try:
            df[x_col] = pd.to_datetime(df[x_col])
            return px.line(
                df.sort_values(x_col), x=x_col, y=y_col,
                template="plotly_dark",
                color_discrete_sequence=["#6ee7b7"],
                title=f"{y_col} over time",
            )
        except Exception:
            pass

    # Bar chart for categorical + numeric (≤ 30 rows)
    if cat_cols and num_cols and len(df) <= 30:
        return px.bar(
            df, x=cat_cols[0], y=num_cols[0],
            template="plotly_dark",
            color_discrete_sequence=["#6ee7b7"],
            title=f"{num_cols[0]} by {cat_cols[0]}",
        )

    # Scatter for two numeric columns
    if len(num_cols) >= 2 and len(df) <= 500:
        return px.scatter(
            df, x=num_cols[0], y=num_cols[1],
            template="plotly_dark",
            color_discrete_sequence=["#6ee7b7"],
            title=f"{num_cols[0]} vs {num_cols[1]}",
        )

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  SAMPLE QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_QUESTIONS = [
    "What are the top 5 product categories by total revenue?",
    "Show me monthly order count trends for 2018.",
    "Which state has the most customers?",
    "What is the average review score per product category?",
    "List the top 10 sellers by number of items sold.",
    "What percentage of orders were delivered on time?",
    "What is the most common payment method?",
    "Show me average freight cost by seller state.",
    "How many orders were placed each month in 2017?",
    "What is the average order value by payment type?",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🧠 SQL Brain")
    st.markdown("<hr style='border-color:#252c3d;margin:0.5rem 0'>", unsafe_allow_html=True)

    st.markdown("**Dataset:** Olist Brazilian E-Commerce")
    st.markdown("**DB:** TiDB Serverless (MySQL 8)")
    st.markdown("**LLM:** Grok-3-mini · xAI")
    st.markdown("**Schema:** 7 tables + 2 views")
    st.markdown("<hr style='border-color:#252c3d;margin:0.75rem 0'>", unsafe_allow_html=True)

    st.markdown("##### 💡 Quick questions")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, key=f"sq_{q[:25]}"):
            st.session_state.pending_question = q

    st.markdown("<hr style='border-color:#252c3d;margin:0.75rem 0'>", unsafe_allow_html=True)

    show_sql   = st.toggle("Show generated SQL",  value=True)
    show_table = st.toggle("Show result table",    value=True)
    show_chart = st.toggle("Auto-render chart",    value=True)

    st.markdown("<hr style='border-color:#252c3d;margin:0.75rem 0'>", unsafe_allow_html=True)
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        "<p style='font-size:0.72rem;color:#475569;margin-top:1rem'>"
        "Built with LangChain · Streamlit · TiDB · xAI"
        "</p>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN HEADER + STATUS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="hero-title">SQL Brain</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">'
    'Natural Language → SQL → Insight · Olist E-Commerce Dataset'
    '</div>',
    unsafe_allow_html=True,
)

# Connection + key status chips
try:
    engine = get_engine()
    with engine.connect() as conn:
        tbl_count = conn.execute(
            text("SELECT COUNT(*) FROM information_schema.TABLES "
                 "WHERE TABLE_SCHEMA = DATABASE()")
        ).scalar()
    db_status = f"✅ Connected · {tbl_count} objects"
except Exception as e:
    db_status = f"❌ DB error: {e}"
    engine = None

xai_ok = bool(get_secret("XAI_API_KEY"))

st.markdown(
    f'<div class="chip-row">'
    f'<span class="chip">{db_status}</span>'
    f'<span class="chip">{"✅ XAI key set" if xai_ok else "❌ XAI key missing"}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

if engine is None:
    st.warning("Database not connected. Check your Streamlit secrets.")
    st.stop()

if not xai_ok:
    st.warning("XAI_API_KEY missing. Add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ═══════════════════════════════════════════════════════════════════════════════
#  RENDER CHAT HISTORY
# ═══════════════════════════════════════════════════════════════════════════════

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="msg-user">'
            f'<div class="msg-label">You</div>'
            f'{msg["content"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="msg-bot">'
            f'<div class="msg-label">SQL Brain</div>'
            f'{msg["content"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
        if show_sql and msg.get("sql"):
            st.markdown(
                f'<div class="sql-block">{msg["sql"]}</div>',
                unsafe_allow_html=True,
            )
        if show_table and msg.get("df") is not None:
            st.dataframe(msg["df"], use_container_width=True, height=260)
        if show_chart and msg.get("chart") is not None:
            st.plotly_chart(msg["chart"], use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CHAT INPUT
# ═══════════════════════════════════════════════════════════════════════════════

user_question = None

# Sidebar quick-question injection
if st.session_state.pending_question:
    user_question = st.session_state.pending_question
    st.session_state.pending_question = None

# Bottom chat input bar
chat_input = st.chat_input("Ask a business question about the e-commerce data …")
if chat_input:
    user_question = chat_input


# ═══════════════════════════════════════════════════════════════════════════════
#  AGENT INVOCATION
# ═══════════════════════════════════════════════════════════════════════════════

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.spinner("🤔 Thinking …"):
        try:
            lc_db = get_langchain_db(engine)
            agent = get_agent(lc_db)

            response = agent.invoke({"input": user_question})
            answer   = response.get("output", str(response))

            # Extract last SQL query executed by agent
            sql_query = extract_last_select(response.get("intermediate_steps", []))

            # Re-run SQL to get a clean DataFrame for display
            df_result = run_sql_safe(engine, sql_query) if sql_query else None

            # Auto-generate chart if possible
            chart = try_auto_chart(df_result) if df_result is not None else None

            st.session_state.messages.append({
                "role":    "assistant",
                "content": answer,
                "sql":     sql_query,
                "df":      df_result,
                "chart":   chart,
            })

        except Exception as exc:
            st.session_state.messages.append({
                "role":    "assistant",
                "content": f"⚠️ Agent error: `{type(exc).__name__}: {exc}`",
                "sql":     None,
                "df":      None,
                "chart":   None,
            })

    st.rerun()
