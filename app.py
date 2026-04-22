"""
app.py — Text-to-SQL AI Agent
Streamlit UI · LangChain SQL Agent · Grok (xAI) · TiDB Serverless
"""

import os
import re
import pandas as pd
import streamlit as st
import plotly.express as px
from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SQL Brain — AI Database Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CSS  — new amber + slate dark theme
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0a0a0f;
    --surface:   #12121a;
    --surface2:  #1a1a26;
    --border:    #2a2a3d;
    --accent:    #f59e0b;
    --accent2:   #fbbf24;
    --accentdim: #78350f;
    --blue:      #60a5fa;
    --text:      #f1f5f9;
    --muted:     #64748b;
    --success:   #34d399;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'Inter', sans-serif;
}

* { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] > div {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 1.5rem;
}
section[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

/* ── Hero ── */
.hero-wrap {
    padding: 2rem 0 1.2rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.hero-title {
    font-family: var(--mono);
    font-size: 2.6rem;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: -0.04em;
    line-height: 1;
    margin: 0;
}
.hero-title span { color: var(--text); opacity: 0.25; }
.hero-sub {
    font-family: var(--sans);
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 0.4rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 500;
}

/* ── Status chips ── */
.chip-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 1rem 0 1.5rem 0; }
.chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.3rem 0.75rem;
    font-size: 0.75rem;
    font-family: var(--mono);
    color: var(--muted);
    font-weight: 500;
}
.chip.ok   { border-color: var(--success); color: var(--success); }
.chip.warn { border-color: #f87171;        color: #f87171; }
.chip.info { border-color: var(--accent);  color: var(--accent); }

/* ── Chat messages ── */
.msg-wrap { margin-bottom: 1rem; }
.msg-user {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px 12px 4px 12px;
    padding: 0.9rem 1.1rem;
    margin-left: 15%;
    font-size: 0.93rem;
    line-height: 1.65;
}
.msg-bot {
    background: #130e05;
    border: 1px solid var(--accentdim);
    border-radius: 12px 12px 12px 4px;
    padding: 0.9rem 1.1rem;
    margin-right: 5%;
    font-size: 0.93rem;
    line-height: 1.65;
}
.msg-label {
    font-family: var(--mono);
    font-size: 0.62rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.4rem;
    opacity: 0.45;
}
.msg-user .msg-label { text-align: right; }

/* ── SQL block ── */
.sql-wrap {
    margin: 0.6rem 0 1rem 0;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border);
}
.sql-header {
    background: var(--surface2);
    padding: 0.3rem 0.8rem;
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--accent);
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
}
.sql-body {
    background: #060608;
    padding: 0.8rem 1rem;
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--blue);
    white-space: pre-wrap;
    overflow-x: auto;
    margin: 0;
}

/* ── Sidebar buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    font-family: var(--sans) !important;
    font-size: 0.78rem !important;
    border-radius: 8px !important;
    padding: 0.45rem 0.75rem !important;
    text-align: left !important;
    width: 100% !important;
    transition: all 0.15s ease !important;
    line-height: 1.4 !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: #130e05 !important;
}

/* ── Chat input ── */
div[data-testid="stChatInput"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
div[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
    font-size: 0.93rem !important;
}
div[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
}

/* ── Toggle ── */
.stToggle label { color: var(--muted) !important; font-size: 0.82rem !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Divider ── */
.sdiv {
    height: 1px;
    background: var(--border);
    margin: 0.9rem 0;
}

/* ── Sidebar logo ── */
.sb-logo {
    font-family: var(--mono);
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--accent);
    margin-bottom: 0.15rem;
}
.sb-tagline {
    font-size: 0.72rem;
    color: var(--muted);
    margin-bottom: 1rem;
}
.sb-meta {
    font-size: 0.75rem;
    color: var(--muted);
    line-height: 1.9;
}
.sb-meta b { color: var(--text); font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECRETS helper
# ═══════════════════════════════════════════════════════════════════════════════

def get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, default)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATABASE  — engine + dynamic table discovery
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
    return create_engine(
        url,
        connect_args={"ssl": {"ssl_mode": "VERIFY_IDENTITY"}},
        pool_recycle=3600,
        pool_pre_ping=True,
    )


def get_langchain_db(_engine):
    with _engine.connect() as conn:
        rows = conn.execute(text("SHOW TABLES")).fetchall()
        actual = [r[0] for r in rows]
    return SQLDatabase(
        engine=_engine,
        sample_rows_in_table_info=3,
        include_tables=actual,
    ), set(actual)

# ═══════════════════════════════════════════════════════════════════════════════
#  LLM + AGENT
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_agent(_db):
    xai_api_key = get_secret("XAI_API_KEY", "")
    if not xai_api_key:
        st.error("❌ XAI_API_KEY not set.")
        st.stop()

    llm = ChatOpenAI(
        model="grok-3-mini",
        api_key=xai_api_key,
        base_url="https://api.x.ai/v1",
        temperature=0,
        max_tokens=2048,
    )

    toolkit = SQLDatabaseToolkit(db=_db, llm=llm)

    system_message = """You are an expert SQL analyst for a Brazilian e-commerce business.
Database: Olist Brazilian E-Commerce (MySQL on TiDB Serverless)

TABLES:
  • customers        — customer_id, customer_unique_id, city, state, zip_code
  • sellers          — seller_id, city, state, zip_code
  • products         — product_id, category, weight_g, length_cm, height_cm, width_cm, photos_qty
  • orders           — order_id, customer_id, order_status, order_purchase_timestamp, delivery dates
  • order_items      — order_id, order_item_id, product_id, seller_id, price, freight_value
  • order_reviews    — review_id, order_id, review_score (1-5), review_comment_message
  • payments         — order_id, payment_type, payment_installments, payment_value

VIEWS (use for aggregations — much faster):
  • vw_sales_by_category      — category, total_orders, total_items_sold, total_revenue,
                                avg_item_price, total_freight, avg_review_score, unique_sellers
  • vw_customer_order_history — customer_unique_id, order_id, order_status,
                                product_category, price, payment_type, review_score

RULES:
1. Write syntactically correct MySQL SQL only.
2. Prefer vw_sales_by_category for category revenue/review questions.
3. Prefer vw_customer_order_history for customer history questions.
4. Always LIMIT 200 unless user requests otherwise.
5. After retrieving data, give a clear concise business interpretation.
6. Never expose credentials or connection strings.
"""

    return create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type="openai-tools",
        verbose=True,
        max_iterations=15,
        max_execution_time=60,
        handle_parsing_errors=True,
        system_message=system_message,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def run_sql_safe(engine, query: str):
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            return pd.DataFrame(result.fetchall(), columns=result.keys())
    except Exception:
        return None


def extract_last_select(steps: list):
    last = None
    for step in steps:
        try:
            action, _ = step
            ti = getattr(action, "tool_input", "")
            if isinstance(ti, dict):
                ti = ti.get("query", "")
            if isinstance(ti, str) and ti.strip().upper().startswith("SELECT"):
                last = ti.strip()
        except Exception:
            continue
    return last


def try_auto_chart(df: pd.DataFrame):
    if df is None or df.empty or len(df) < 2:
        return None
    num  = df.select_dtypes(include="number").columns.tolist()
    cat  = df.select_dtypes(exclude="number").columns.tolist()
    date = [c for c in df.columns
            if any(k in c.lower() for k in ["date", "timestamp", "month", "year"])]

    if date and num:
        try:
            df[date[0]] = pd.to_datetime(df[date[0]])
            return px.line(df.sort_values(date[0]), x=date[0], y=num[0],
                           template="plotly_dark",
                           color_discrete_sequence=["#f59e0b"],
                           title=f"{num[0]} over time")
        except Exception:
            pass

    if cat and num and len(df) <= 30:
        return px.bar(df, x=cat[0], y=num[0],
                      template="plotly_dark",
                      color_discrete_sequence=["#f59e0b"],
                      title=f"{num[0]} by {cat[0]}")

    if len(num) >= 2 and len(df) <= 500:
        return px.scatter(df, x=num[0], y=num[1],
                          template="plotly_dark",
                          color_discrete_sequence=["#f59e0b"],
                          title=f"{num[0]} vs {num[1]}")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  SAMPLE QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLES = [
    "What are the top 5 product categories by total revenue?",
    "Show me monthly order count trends for 2018.",
    "Which state has the most customers?",
    "What is the average review score per product category?",
    "List the top 10 sellers by number of items sold.",
    "What percentage of orders were delivered on time?",
    "What is the most common payment method?",
    "Show average freight cost by seller state.",
    "How many orders were placed each month in 2017?",
    "What is the average order value by payment type?",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  CONNECTION  (do this early so sidebar shows live status)
# ═══════════════════════════════════════════════════════════════════════════════

engine      = None
actual_tbls = set()
db_error    = None
obj_count   = 0

try:
    engine = get_engine()
    with engine.connect() as conn:
        # Count actual objects in the current database
        obj_count = conn.execute(
            text("SELECT COUNT(*) FROM information_schema.TABLES "
                 "WHERE TABLE_SCHEMA = DATABASE()")
        ).scalar()
        actual_rows = conn.execute(text("SHOW TABLES")).fetchall()
        actual_tbls = {r[0] for r in actual_rows}
except Exception as e:
    db_error = str(e)

xai_ok = bool(get_secret("XAI_API_KEY"))


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sb-logo">⚡ SQL Brain</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sb-tagline">Natural Language → SQL → Insight</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="sb-meta">'
        f'<b>Dataset</b> Olist Brazilian E-Commerce<br>'
        f'<b>Database</b> TiDB Serverless · MySQL 8<br>'
        f'<b>LLM</b> Grok-3-mini · xAI<br>'
        f'<b>Objects</b> {obj_count} in DB '
        f'({len(actual_tbls & {"customers","sellers","products","orders","order_items","order_reviews","payments"})} tables '
        f'+ {len(actual_tbls & {"vw_sales_by_category","vw_customer_order_history"})} views)'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)

    st.markdown(
        '<p style="font-size:0.72rem;color:#64748b;font-weight:600;'
        'text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem">'
        '💡 Quick questions</p>',
        unsafe_allow_html=True,
    )
    for q in SAMPLES:
        if st.button(q, key=f"sq_{q[:28]}"):
            st.session_state.pending_question = q

    st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
    show_sql   = st.toggle("Show generated SQL",  value=True)
    show_table = st.toggle("Show result table",    value=True)
    show_chart = st.toggle("Auto-render chart",    value=True)

    st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        '<p style="font-size:0.68rem;color:#374151;margin-top:1.5rem;text-align:center">'
        'LangChain · Streamlit · TiDB · xAI Grok'
        '</p>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="hero-wrap">'
    '<div class="hero-title">SQL<span>/</span>Brain</div>'
    '<div class="hero-sub">Natural Language → SQL → Business Insight · Olist E-Commerce</div>'
    '</div>',
    unsafe_allow_html=True,
)

# Status chips
if db_error:
    db_chip = f'<span class="chip warn">❌ DB Error</span>'
else:
    db_chip = f'<span class="chip ok">✅ Connected · {obj_count} objects</span>'

xai_chip = (
    '<span class="chip ok">✅ XAI key set</span>' if xai_ok
    else '<span class="chip warn">❌ XAI key missing</span>'
)

# Show which tables were found
found_tables = actual_tbls & {
    "customers","sellers","products","orders",
    "order_items","order_reviews","payments",
    "vw_sales_by_category","vw_customer_order_history"
}
tbl_chip = (
    f'<span class="chip info">📋 {len(found_tables)}/9 schema objects found</span>'
    if engine else ""
)

st.markdown(
    f'<div class="chip-row">{db_chip}{xai_chip}{tbl_chip}</div>',
    unsafe_allow_html=True,
)

# ── Diagnostic warning if tables are missing ─────────────────────────────────
if engine and len(found_tables) < 7:
    missing = {
        "customers","sellers","products","orders",
        "order_items","order_reviews","payments"
    } - actual_tbls
    st.warning(
        f"⚠️ **{len(missing)} base tables missing** from the connected database: "
        f"`{'`, `'.join(sorted(missing))}`\n\n"
        f"**Connected DB has these objects:** `{'`, `'.join(sorted(actual_tbls)) or 'none'}`\n\n"
        f"Check that `TIDB_DB = ecommerce` in **Streamlit Cloud → Settings → Secrets** "
        f"and click **Reboot app** after saving."
    )

if db_error:
    st.error(f"Database connection failed: {db_error}")
    st.info("Go to Streamlit Cloud → Settings → Secrets and verify all TIDB_* values.")
    st.stop()

if not xai_ok:
    st.error("XAI_API_KEY missing. Add it in Streamlit Cloud → Settings → Secrets.")
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
            f'<div class="msg-wrap">'
            f'<div class="msg-user">'
            f'<div class="msg-label">You</div>'
            f'{msg["content"]}'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="msg-wrap">'
            f'<div class="msg-bot">'
            f'<div class="msg-label">⚡ SQL Brain</div>'
            f'{msg["content"]}'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        if show_sql and msg.get("sql"):
            st.markdown(
                f'<div class="sql-wrap">'
                f'<div class="sql-header">⚡ Generated SQL</div>'
                f'<pre class="sql-body">{msg["sql"]}</pre>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if show_table and msg.get("df") is not None:
            st.dataframe(msg["df"], use_container_width=True, height=240)
        if show_chart and msg.get("chart") is not None:
            st.plotly_chart(msg["chart"], use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CHAT INPUT
# ═══════════════════════════════════════════════════════════════════════════════

user_question = None
if st.session_state.pending_question:
    user_question = st.session_state.pending_question
    st.session_state.pending_question = None

chat_input = st.chat_input("Ask a business question about the e-commerce data …")
if chat_input:
    user_question = chat_input


# ═══════════════════════════════════════════════════════════════════════════════
#  AGENT INVOCATION
# ═══════════════════════════════════════════════════════════════════════════════

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.spinner("⚡ Thinking …"):
        try:
            lc_db, _ = get_langchain_db(engine)
            agent    = get_agent(lc_db)

            response  = agent.invoke({"input": user_question})
            answer    = response.get("output", str(response))
            sql_query = extract_last_select(response.get("intermediate_steps", []))
            df_result = run_sql_safe(engine, sql_query) if sql_query else None
            chart     = try_auto_chart(df_result) if df_result is not None else None

            st.session_state.messages.append({
                "role": "assistant", "content": answer,
                "sql": sql_query, "df": df_result, "chart": chart,
            })

        except Exception as exc:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ Agent error: `{type(exc).__name__}: {exc}`",
                "sql": None, "df": None, "chart": None,
            })

    st.rerun()