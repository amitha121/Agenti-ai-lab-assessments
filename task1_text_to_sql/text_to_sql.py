"""
Task 1: Text-to-SQL Workflow
=============================
End-to-end pipeline: natural-language question -> SQL -> executed result.

Pipeline stages (this is the "workflow" the assignment asks for):
  1. SCHEMA STORE   : Each table is described as a short "schema card".
  2. RETRIEVAL       : Embed the question + all schema cards, use cosine
                        similarity to pull only the RELEVANT tables.
                        (In a real enterprise DB with 200+ tables, you
                        can't stuff the whole schema into every prompt —
                        this retrieval step is what keeps prompts small
                        and accurate.)
  3. QUERY GENERATION: The LLM (Groq/Llama-3.3) writes SQL using ONLY the
                        retrieved schema.
  4. SAFETY CHECK     : Only SELECT statements are allowed to execute.
  5. EXECUTION        : Query runs against a local SQLite sample database
                        and results are printed as a table.

Run:
    python text_to_sql.py
"""

import os
import re
import sqlite3
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
from llm_client import chat  # noqa: E402

DB_PATH = os.path.join(os.path.dirname(__file__), "sample.db")

# ---------------------------------------------------------------------------
# 1. Build a small sample enterprise database (runs once)
# ---------------------------------------------------------------------------
def build_sample_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executescript(
        """
        CREATE TABLE departments (
            dept_id INTEGER PRIMARY KEY,
            dept_name TEXT,
            location TEXT
        );

        CREATE TABLE employees (
            emp_id INTEGER PRIMARY KEY,
            name TEXT,
            dept_id INTEGER,
            role TEXT,
            salary INTEGER,
            hire_date TEXT,
            FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
        );

        CREATE TABLE projects (
            project_id INTEGER PRIMARY KEY,
            project_name TEXT,
            dept_id INTEGER,
            budget INTEGER,
            status TEXT,
            FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
        );

        CREATE TABLE sales (
            sale_id INTEGER PRIMARY KEY,
            emp_id INTEGER,
            amount INTEGER,
            sale_date TEXT,
            FOREIGN KEY (emp_id) REFERENCES employees(emp_id)
        );
        """
    )

    cur.executemany(
        "INSERT INTO departments VALUES (?, ?, ?)",
        [
            (1, "Engineering", "Hyderabad"),
            (2, "Security", "Hyderabad"),
            (3, "Sales", "Bengaluru"),
        ],
    )
    cur.executemany(
        "INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, "Amitha Rao", 2, "SOC Analyst", 550000, "2024-06-01"),
            (2, "Sreekari M", 1, "Backend Engineer", 700000, "2023-11-15"),
            (3, "Vikram Singh", 3, "Sales Executive", 480000, "2022-04-10"),
            (4, "Neha Patel", 1, "ML Engineer", 900000, "2021-09-20"),
            (5, "Arjun Reddy", 2, "SIEM Engineer", 620000, "2023-01-05"),
        ],
    )
    cur.executemany(
        "INSERT INTO projects VALUES (?, ?, ?, ?, ?)",
        [
            (1, "Zero Trust Gateway", 2, 2000000, "active"),
            (2, "RAG Pipeline Hardening", 1, 1500000, "active"),
            (3, "SIEM Migration", 2, 900000, "completed"),
        ],
    )
    cur.executemany(
        "INSERT INTO sales VALUES (?, ?, ?, ?)",
        [
            (1, 3, 250000, "2025-01-10"),
            (2, 3, 180000, "2025-02-14"),
            (3, 3, 300000, "2025-03-02"),
        ],
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# 2. Schema cards (the retrieval corpus)
# ---------------------------------------------------------------------------
SCHEMA_CARDS = {
    "departments": "Table departments: dept_id, dept_name, location. "
                   "Stores each company department and its city.",
    "employees": "Table employees: emp_id, name, dept_id, role, salary, "
                 "hire_date. Stores staff records including which "
                 "department they belong to, their job role and pay.",
    "projects": "Table projects: project_id, project_name, dept_id, "
                "budget, status. Stores ongoing/completed department "
                "projects and budgets.",
    "sales": "Table sales: sale_id, emp_id, amount, sale_date. Stores "
             "individual sales transactions made by employees.",
}


def embed_texts(model, texts):
    return model.encode(texts, normalize_embeddings=True)


def retrieve_relevant_tables(question, model, top_k=2):
    """Cosine-similarity retrieval: pick the most relevant schema cards."""
    table_names = list(SCHEMA_CARDS.keys())
    card_texts = list(SCHEMA_CARDS.values())

    card_vecs = embed_texts(model, card_texts)
    q_vec = embed_texts(model, [question])[0]

    scores = card_vecs @ q_vec  # cosine similarity (vectors are normalized)
    ranked = sorted(zip(table_names, scores), key=lambda x: x[1], reverse=True)

    top_tables = [name for name, _ in ranked[:top_k]]
    print("\n[Retrieval] Table relevance scores:")
    for name, score in ranked:
        marker = " <-- retrieved" if name in top_tables else ""
        print(f"    {name:12s} {score:.3f}{marker}")

    return top_tables


# ---------------------------------------------------------------------------
# 3. Query generation
# ---------------------------------------------------------------------------
def generate_sql(question, retrieved_tables):
    schema_block = "\n".join(SCHEMA_CARDS[t] for t in retrieved_tables)

    system = (
        "You are a SQL generation engine for a SQLite database. "
        "Use ONLY the tables/columns given in the schema below. "
        "Return ONLY the SQL query, no explanation, no markdown fences, "
        "no trailing semicolon commentary. The query MUST be a SELECT."
    )
    prompt = f"""Schema:
{schema_block}

Question: {question}

SQL query:"""

    sql = chat(prompt, system=system, temperature=0)
    # Strip accidental markdown fences if the model adds them anyway
    sql = re.sub(r"^```sql|```$|^```", "", sql.strip(), flags=re.MULTILINE).strip()
    return sql


# ---------------------------------------------------------------------------
# 4. Safety check + 5. Execution
# ---------------------------------------------------------------------------
def run_query_safely(sql):
    if not re.match(r"^\s*SELECT", sql, re.IGNORECASE):
        raise ValueError(f"Refusing to run non-SELECT query:\n{sql}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    conn.close()
    return cols, rows


def print_table(cols, rows):
    print("    " + " | ".join(cols))
    print("    " + "-" * (10 * len(cols)))
    for row in rows:
        print("    " + " | ".join(str(v) for v in row))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def answer_question(question, model):
    print(f"\n=== Question: {question} ===")
    retrieved = retrieve_relevant_tables(question, model)
    sql = generate_sql(question, retrieved)
    print(f"\n[Generation] SQL:\n    {sql}")
    try:
        cols, rows = run_query_safely(sql)
        print("\n[Execution] Result:")
        print_table(cols, rows)
    except Exception as e:
        print(f"\n[Execution] Failed: {e}")


if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        build_sample_db()
        print(f"Sample database created at {DB_PATH}")

    print("Loading embedding model (first run downloads ~90MB, then cached)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    sample_questions = [
        "List all employees in the Security department with their roles",
        "What is the total sales amount made by Vikram Singh?",
        "Show all active projects and their budgets",
    ]

    for q in sample_questions:
        answer_question(q, embed_model)

    print("\nDone. Edit `sample_questions` above, or import `answer_question()` "
          "to ask your own questions interactively.")
