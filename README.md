# The Research Desk

Live demo

🔗 [https://research-desky.streamlit.app/].

A multi-agent research paper pipeline: give it a topic, and it searches ArXiv, reads
the papers, and writes up what's changed — end to end, orchestrated with LangChain
and served through a Streamlit UI.

```
Streamlit User Input → LangChain Pipeline → Streamlit Dashboard
   (topic, paper count,     1. Search & Fetch     (interactive results,
    output preferences)     2. Reader & Extractor   downloadable brief)
                             3. Synthesis Agent
                             4. Output & Automation
                                (Sheets, Markdown/PDF, Email)
```

## What it does

1. **Search & Fetch** — queries the ArXiv API for the topic, filters by publication
   date, then uses an LLM to pick the papers most relevant to what you asked for.
2. **Reader & Extractor** — downloads each paper's PDF, parses the text, and asks an
   LLM to pull out methodology, datasets, key results, and limitations.
3. **Synthesis Agent** — compares the extracted papers and summarizes trends and gaps
   across all of them.
4. **Output & Automation** — optionally mirrors the run into a Google Sheet, generates
   a downloadable Markdown/PDF brief, and can email the result.

All three LLM steps run on [Groq](https://groq.com/), with a separate model
configurable per step (a fast/cheap model for filtering, a stronger one for synthesis).

## Tech stack

- **UI:** [Streamlit](https://streamlit.io/)
- **Orchestration:** [LangChain](https://python.langchain.com/) (LCEL — plain
  `RunnableLambda`/`RunnableSequence`, no LangGraph)
- **LLMs:** Groq, via `langchain-groq`
- **Papers:** [ArXiv API](https://pypi.org/project/arxiv/) + [`pypdf`](https://pypi.org/project/pypdf/) for parsing
- **Output:** Google Sheets API (optional), Markdown/PDF brief generation, email (optional)

## Project structure

```
.
├── app.py          # Streamlit UI — input form + results dashboard
├── pipeline.py      # LangChain orchestration: the three agent nodes + run_pipeline()
├── outputs.py        # PDF brief generation + Google Sheets integration
└── requirements.txt
```

## Setup

```bash
git clone <your-repo-url>
cd <your-repo>
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### `requirements.txt`

```
streamlit
langchain-core
langchain-groq
arxiv
pypdf
requests
```

Add any packages `outputs.py` needs for PDF generation and the Google Sheets API
(e.g. `reportlab` or `fpdf2`, `gspread`, `google-auth`).

## Running it

```bash
streamlit run app.py
```

Then, in the app:

1. Paste a **Groq API key** into the sidebar (get one at [console.groq.com](https://console.groq.com/)).
2. *(Optional)* Configure the model used per pipeline step under **Model selection**.
3. *(Optional)* Fill in email or Google Sheets settings if you want those outputs.
4. On the **Run Pipeline** tab, enter a topic, pick how many papers to analyze and a
   publication window, and hit **Send the agents to work**.
5. Open the **Dashboard** tab to read the results and download the brief.

## Configuration

Model defaults live at the top of `pipeline.py` and can be overridden per run from the
sidebar's "Model selection (advanced)" panel:

| Step | Default model | Purpose |
|---|---|---|
| Relevance filter | `llama-3.1-8b-instant` | Cheap/fast pass to narrow candidates down to the top matches |
| Extraction | `llama-3.3-70b-versatile` | Reads each paper and pulls structured fields |
| Synthesis | `llama-3.3-70b-versatile` | Compares papers, finds trends and gaps |

Other tunables in `pipeline.py`:

- `FETCH_MULTIPLIER` — how many extra candidates to pull from ArXiv before the
  relevance filter narrows them down (default: 4x the requested paper count).
- `PDF_TEXT_CHAR_LIMIT` — how much parsed PDF text gets sent to the extraction LLM per
  paper (default: 14,000 characters).
- `DATE_RANGE_DAYS` — the publication-window options shown in the UI.

## How the pipeline is wired

`pipeline.py` has no framework-level graph — it's plain LangChain LCEL. Each stage is a
`RunnableLambda` over a shared state dict, piped together:

```python
search_and_fetch | read_and_extract | synthesize
```

`build_pipeline()` returns that composed `RunnableSequence` for anyone who wants to run
the whole thing in one call. `run_pipeline()` — what `app.py` actually calls — invokes
each step individually instead, so the UI's progress bar can update between stages.

## Notes

- The relevance filter and synthesis steps expect the LLM to return raw JSON; a small
  recovery helper (`_safe_json_parse`) strips markdown code fences and falls back
  gracefully if parsing fails.
- If a PDF fails to download or parse, that paper's abstract is used as a fallback
  input to the extraction step rather than failing the whole run.
- Warnings and partial failures (a paper that couldn't be parsed, a Sheets update that
  failed, etc.) are collected into `errors` and surfaced in the dashboard rather than
  stopping the pipeline.
