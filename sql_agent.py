"""
Task 4: SQL Agent with Tool Use (ReAct pattern)
==================================================
Unlike Task 1 (which does one-shot: retrieve schema -> generate SQL ->
run it), this is a proper ReAct AGENT: the LLM reasons in a loop,
deciding for itself which tool to call next based on what it learned
from the previous tool's result, until it has enough information to
answer.

ReAct = "Reasoning + Acting". At each step the agent:
    Thought      -> reasons about what it needs to do next
    Action       -> calls one tool (function calling)
    Observation  -> receives the tool's result
  ...repeats until it has enough info, then gives a Final Answer.

This matters for questions a one-shot text-to-SQL system can't answer
in a single query -- e.g. "first check what tables exist, then look at
the right one's columns, then decide the query" or multi-step
questions that need one query's result to build the next query.

Tools available to the agent:
    list_tables()             -> names of all tables in the database
    get_schema(table_name)    -> columns of one table
    run_query(sql)            -> executes a SELECT and returns rows

Run:
    python sql_agent.py
"""

import json
import os
import re
import sqlite3
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
from llm_client import get_client, DEFAULT_MODEL  # noqa: E402

DB_PATH = os.path.join(os.path.dirname(__file__), "sample.db")
MAX_STEPS = 8  # safety cap so the agent can't loop forever


# ---------------------------------------------------------------------------
# Sample database (same schema style as Task 1, built independently here)
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
            hire_date TEXT
        );
        CREATE TABLE projects (
            project_id INTEGER PRIMARY KEY,
            project_name TEXT,
            dept_id INTEGER,
            budget INTEGER,
            status TEXT
        );
        CREATE TABLE sales (
            sale_id INTEGER PRIMARY KEY,
            emp_id INTEGER,
            amount INTEGER,
            sale_date TEXT
        );
        """
    )
    cur.executemany(
        "INSERT INTO departments VALUES (?, ?, ?)",
        [(1, "Engineering", "Hyderabad"), (2, "Security", "Hyderabad"),
         (3, "Sales", "Bengaluru")],
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
        [(1, "Zero Trust Gateway", 2, 2000000, "active"),
         (2, "RAG Pipeline Hardening", 1, 1500000, "active"),
         (3, "SIEM Migration", 2, 900000, "completed")],
    )
    cur.executemany(
        "INSERT INTO sales VALUES (?, ?, ?, ?)",
        [(1, 3, 250000, "2025-01-10"), (2, 3, 180000, "2025-02-14"),
         (3, 3, 300000, "2025-03-02")],
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Tools the agent can call
# ---------------------------------------------------------------------------
def tool_list_tables():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cur.fetchall()]
    conn.close()
    return {"tables": tables}


def tool_get_schema(table_name):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    cols = [{"name": r[1], "type": r[2]} for r in cur.fetchall()]
    conn.close()
    if not cols:
        return {"error": f"Table '{table_name}' not found"}
    return {"table": table_name, "columns": cols}


def tool_run_query(sql):
    if not re.match(r"^\s*SELECT", sql, re.IGNORECASE):
        return {"error": "Only SELECT queries are allowed"}
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        conn.close()
        return {"columns": cols, "rows": rows[:20]}  # cap rows returned
    except Exception as e:
        return {"error": str(e)}


TOOL_IMPLEMENTATIONS = {
    "list_tables": lambda args: tool_list_tables(),
    "get_schema": lambda args: tool_get_schema(args["table_name"]),
    "run_query": lambda args: tool_run_query(args["sql"]),
}

# OpenAI/Groq function-calling schema describing the same tools
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "List all table names in the database.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": "Get the column names and types for one table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "Exact table name"}
                },
                "required": ["table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_query",
            "description": "Execute a read-only SELECT SQL query and return rows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A SELECT SQL statement"}
                },
                "required": ["sql"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a ReAct-style database agent. You answer questions
about a SQLite database by reasoning step by step and calling tools -- you do
NOT know the schema in advance, so use list_tables and get_schema to explore
before writing queries. Only call run_query with SELECT statements.

At each turn, briefly state your reasoning (one short sentence) before
calling a tool, or before giving your final answer. When you have enough
information, respond with a clear final answer in plain English (not a tool
call) summarizing the result for the user."""


# ---------------------------------------------------------------------------
# The ReAct loop
# ---------------------------------------------------------------------------
def run_agent(question, verbose=True):
    client = get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    print(f"\n{'='*70}\nQUESTION: {question}\n{'='*70}")

    for step in range(1, MAX_STEPS + 1):
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            temperature=0,
        )
        msg = response.choices[0].message

        # Print the model's reasoning/thought for this step, if any
        if msg.content:
            print(f"\n[Step {step}] Thought: {msg.content.strip()}")

        if not msg.tool_calls:
            # No more tool calls -> this is the final answer
            print(f"\n[FINAL ANSWER]\n{msg.content}")
            return msg.content

        # Append the assistant's tool-call message to history
        messages.append(msg)

        # Execute every requested tool call, feed observations back
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments or "{}")

            print(f"[Step {step}] Action: {fn_name}({fn_args})")
            result = TOOL_IMPLEMENTATIONS[fn_name](fn_args)
            print(f"[Step {step}] Observation: {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

    print("\n[Agent stopped: reached MAX_STEPS without a final answer]")
    return None


if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        build_sample_db()
        print(f"Sample database created at {DB_PATH}")

    sample_questions = [
        "What tables exist in this database, and what does each one store?",
        "Which employee has the highest salary, and which department are they in?",
        "What is the total budget across all active projects?",
    ]

    for q in sample_questions:
        run_agent(q)

    print("\nDone. Edit `sample_questions` above, or call run_agent('your "
          "question') to ask your own things interactively.")
