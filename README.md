# 🤖 Agentic AI Sales Analytics Assistant (RAG + Ollama + Streamlit)

An interactive, chat-based analytics app that lets business users query a sales/churn dataset in natural language and get **tables, bar charts, and short narratives**—with **retrieval-augmented generation (RAG)** for grounding and **Ollama (Llama 3/3.1)** for planning.

The assistant converts free-form questions (e.g., *“Show customers with revenue > 80k and churn probability > 40%”*) into a structured query plan, executes it on your dataset, and renders results in a **ChatGPT-style UI**. It also supports **explicit chart commands** like *“revenue against segments”*.

---

## 🚀 Features

- 💬 **Chat UI** (ChatGPT-like): your message on the right, assistant response on the left.
- 🧠 **Agentic Planner** (Ollama): plans a sequence of steps (query → chart → narration).
- 📦 **RAG-ready**: plug in context passages if/when you add document retrieval.
- 📊 **One-step Charts**: type *“X against Y”* (e.g., *“revenue against churn”*) and get a **bar chart**.
- 🧮 **Robust Querying**: filters, grouping, aggregations, order/limit; schema/alias handling.
- 🧯 **Fail-safes**: automatic threshold relaxation when results are empty; schema sanity utilities.
- ♻️ **Replayable Charts**: chart spec is persisted so charts render again when you scroll up.
- 🧱 **Modular Code**: planner, tools, charts, and UI helpers are cleanly separated for iteration.

---

## 🛠 Tech Stack

| Layer              | Tech / Libs                                   |
|--------------------|-----------------------------------------------|
| Frontend / UI      | Streamlit                                     |
| LLM / Planning     | **Ollama** (Llama 3 / 3.1), JSON planning     |
| Retrieval (RAG)    | Placeholder hooks (ready for your retriever)  |
| Data Wrangling     | Pandas                                        |
| Visualization      | Streamlit `st.bar_chart` (Matplotlib optional)|
| Packaging          | `requirements.txt`                            |

---

## 🧩 Architecture (High-Level)

1. **User Query** → Streamlit chat input  
2. **Axes Shortcut (if matched)** → *“X against Y”* parser → build frame → **bar chart**  
3. **Otherwise** → **Ollama planner** produces a JSON plan with steps:  
   - `query` → run with `utils.tools.tool_query` (filters, group_by, aggregations)  
   - `chart` → render with `utils.charts.render_bar_only`  
   - `narrate` → optional text via `utils.ui_tools.tool_narrate_streamlit`  
4. **Replay** → chart spec saved; rerender charts when scrolling history.

---

## 📁 Project Structure


---

## ⚡️ Quick Start

### 1) Install dependencies
```bash
pip install -r requirements.txt


# install Ollama (see https://ollama.com)
# then pull at least one model:
ollama pull llama3:latest
# or
ollama pull llama3.1:8b

streamlit run app.py

