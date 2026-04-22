# 🧠 AI-Agent-for-Relational-Database-System

> Natural language → SQL → Business insight  
> Powered by **Grok (xAI)** · **LangChain** · **TiDB Serverless** · **Streamlit**

---

## Project Structure

```
.
├── app.py                    # Streamlit chat application
├── setup_database.py         # Notebook cells (also save as .ipynb)
├── requirements.txt
├── TECHNICAL_REPORT.md
├── .env.example              # Copy → .env for local development
└── .streamlit/
    └── secrets.toml          # Streamlit secrets (never commit!)
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/your-username/text-to-sql-agent.git
cd text-to-sql-agent
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env with your TiDB and xAI credentials
```

### 3. Run the database setup notebook

Open `setup_database.py` as a Jupyter notebook (rename to `.ipynb`) and run all cells:

```bash
jupyter notebook setup_database.ipynb
```

This will:
- Download the Olist dataset via `kagglehub`
- Clean the data with pandas
- Create the schema on TiDB (7 tables, PKs, FKs)
- Insert all data (~100k rows per main table)
- Create indexes, a VIEW, and a STORED PROCEDURE

### 4. Run the Streamlit app locally

```bash
# Create .streamlit/secrets.toml first (copy from .env)
streamlit run app.py
```

---

## Deployment on Streamlit Community Cloud

1. Push this repo to GitHub (make sure `.env` and `secrets.toml` are in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select your repo and `app.py`
4. Under **Settings → Secrets**, paste the contents of `.streamlit/secrets.toml`
5. Deploy!

---

## Database Schema (ER Overview)

```
customers ──< orders ──< order_items >── products
                 │              └──────────── sellers
                 ├──< order_reviews
                 └──< payments
```

**Advanced objects:**
- `vw_sales_by_category` — VIEW for revenue/review aggregation by category
- `sp_customer_order_history(id)` — PROCEDURE returning full order history

---

## Sample Questions

- *"What are the top 5 product categories by total revenue?"*
- *"Show me monthly order trends for 2018."*
- *"Which state has the highest average review score?"*
- *"What percentage of orders were delivered late?"*
- *"Who are the top 10 sellers by items sold?"*

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Grok-3-mini (xAI API) |
| Agent framework | LangChain SQL Agent |
| Database | TiDB Serverless (MySQL 8 compatible) |
| DB driver | PyMySQL + SQLAlchemy |
| UI | Streamlit |
| Data prep | pandas, numpy |
| Charts | Plotly |
| Deployment | Streamlit Community Cloud |
