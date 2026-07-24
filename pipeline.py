
from __future__ import annotations

import io
import json
import re
import datetime as dt
from typing import TypedDict, List, Optional, Callable

import requests
import arxiv
from pypdf import PdfReader
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


GROQ_MODEL_FILTER = "llama-3.1-8b-instant"          
GROQ_MODEL_EXTRACT = "llama-3.3-70b-versatile"      
GROQ_MODEL_SYNTHESIS = "llama-3.3-70b-versatile"    
GROQ_MODEL_QA = "llama-3.3-70b-versatile"            

FETCH_MULTIPLIER = 8          
PDF_TEXT_CHAR_LIMIT = 14000   

DATE_RANGE_DAYS = {
    "Last 7 days": 7,
    "Last 30 days": 30,
    "Last 6 months": 182,
    "Last year": 365,
    "Any time": None,
}


class PipelineState(TypedDict, total=False):
    topic: str
    max_papers: int
    date_range: str
    groq_api_key: str
    model_filter: str
    model_extract: str
    model_synthesis: str

    candidates: List[dict]
    selected_papers: List[dict]
    extracted: List[dict]
    trends: str
    gaps: str
    errors: List[str]


def _progress(state: PipelineState, cb: Optional[Callable], pct: float, msg: str):
    if cb:
        cb(pct, msg)


def _safe_json_parse(text: str, fallback: dict) -> dict:
    """
    LLMs sometimes wrap JSON in prose, markdown code fences, or leave literal
    newlines inside string values (which breaks strict JSON parsing with an
    "invalid control character" error). Try several increasingly forgiving
    strategies before giving up and returning the fallback.
    """
    raw = text.strip()
    candidates = [raw]

    if "```" in raw:
        fenced = raw.split("```")[1]
        if fenced.lower().startswith("json"):
            fenced = fenced[4:].strip()
        candidates.append(fenced.strip())

    for c in list(candidates):
        start, end = c.find("{"), c.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(c[start:end + 1])

    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            pass
        try:
            # Common failure mode: the model put a literal line break inside
            # a string value. Collapsing all raw newlines to spaces fixes
            # this without needing to know which key it happened in.
            return json.loads(c.replace("\n", " "))
        except Exception:
            pass

    # Last resort: regex out the individual keys instead of parsing as JSON.
    recovered = {}
    for key in fallback.keys():
        m = re.search(rf'"{key}"\s*:\s*"(.*?)"\s*[,}}]', raw, re.DOTALL)
        if m:
            recovered[key] = m.group(1).replace("\n", " ").strip()
    if recovered:
        return {**fallback, **recovered}

    return fallback


# --------------------------------------------------------------------------
# Node 1 — Search & Fetch
# --------------------------------------------------------------------------
def search_and_fetch_node(state: PipelineState) -> PipelineState:
    topic = state["topic"]
    max_papers = state["max_papers"]
    date_range = state.get("date_range", "Any time")
    errors = list(state.get("errors", []))

    days = DATE_RANGE_DAYS.get(date_range)
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days) if days else None

    fetch_count = max_papers * FETCH_MULTIPLIER
    candidates = []
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=topic,
            max_results=fetch_count,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        for result in client.results(search):
            if cutoff and result.published < cutoff:
                continue
            candidates.append({
                "title": result.title.strip(),
                "authors": ", ".join(a.name for a in result.authors),
                "published": result.published.strftime("%Y-%m-%d"),
                "summary": result.summary.strip().replace("\n", " "),
                "url": result.entry_id,
                "pdf_url": result.pdf_url,
            })
    except Exception as e:
        errors.append(f"ArXiv search failed: {e}")

    if not candidates:
        return {**state, "candidates": [], "selected_papers": [], "errors": errors}

    # LLM relevance filter: pick the most relevant subset (<= max_papers)
    llm = ChatGroq(
        model=state.get("model_filter", GROQ_MODEL_FILTER),
        api_key=state["groq_api_key"],
        temperature=0,
    )
    listing = "\n".join(
        f"{i}. {c['title']} — {c['summary'][:220]}" for i, c in enumerate(candidates)
    )
    prompt = (
        f"Research topic: \"{topic}\"\n\n"
        f"Candidate papers:\n{listing}\n\n"
        f"Select the {max_papers} papers MOST relevant to the topic above. "
        f"Respond ONLY with JSON: {{\"selected_indices\": [list of integers]}}. "
        f"No prose, no markdown fences."
    )
    try:
        resp = llm.invoke(prompt)
        parsed = _safe_json_parse(resp.content, {"selected_indices": list(range(min(max_papers, len(candidates))))})
        indices = [i for i in parsed.get("selected_indices", []) if 0 <= i < len(candidates)][:max_papers]
        if not indices:
            indices = list(range(min(max_papers, len(candidates))))
    except Exception as e:
        errors.append(f"Relevance filtering failed, falling back to most recent: {e}")
        indices = list(range(min(max_papers, len(candidates))))

    selected = [candidates[i] for i in indices]
    return {**state, "candidates": candidates, "selected_papers": selected, "errors": errors}


# --------------------------------------------------------------------------
# Node 2 — Reader & Extractor
# --------------------------------------------------------------------------
def _download_and_parse_pdf(pdf_url: str) -> str:
    resp = requests.get(pdf_url, timeout=30)
    resp.raise_for_status()
    reader = PdfReader(io.BytesIO(resp.content))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
        if len(text) >= PDF_TEXT_CHAR_LIMIT:
            break
    return text[:PDF_TEXT_CHAR_LIMIT]


def read_and_extract_node(state: PipelineState) -> PipelineState:
    errors = list(state.get("errors", []))
    selected = state.get("selected_papers", [])
    extracted = []

    llm = ChatGroq(
        model=state.get("model_extract", GROQ_MODEL_EXTRACT),
        api_key=state["groq_api_key"],
        temperature=0,
    )

    fallback_fields = {
        "methodology": "Not extracted",
        "datasets": "Not extracted",
        "key_results": "Not extracted",
        "limitations": "Not extracted",
    }

    for paper in selected:
        entry = {**paper}
        try:
            pdf_text = _download_and_parse_pdf(paper["pdf_url"])
        except Exception as e:
            errors.append(f"Could not download/parse PDF for '{paper['title']}': {e}")
            pdf_text = paper.get("summary", "")

        prompt = (
            f"Paper title: {paper['title']}\n\n"
            f"Text (may be truncated):\n{pdf_text}\n\n"
            "Extract the following as JSON with exactly these keys: "
            '"methodology" (1-2 sentences), "datasets" (datasets/benchmarks used), '
            '"key_results" (main quantitative or qualitative findings), '
            '"limitations" (stated or apparent limitations). '
            "Respond ONLY with JSON, no prose, no markdown fences."
        )
        try:
            resp = llm.invoke(prompt)
            fields = _safe_json_parse(resp.content, fallback_fields)
        except Exception as e:
            errors.append(f"Metric extraction failed for '{paper['title']}': {e}")
            fields = fallback_fields

        entry.update({
            "methodology": fields.get("methodology", fallback_fields["methodology"]),
            "datasets": fields.get("datasets", fallback_fields["datasets"]),
            "key_results": fields.get("key_results", fallback_fields["key_results"]),
            "limitations": fields.get("limitations", fallback_fields["limitations"]),
        })
        extracted.append(entry)

    return {**state, "extracted": extracted, "errors": errors}


# --------------------------------------------------------------------------
# Node 3 — Synthesis Agent
# --------------------------------------------------------------------------
def synthesize_node(state: PipelineState) -> PipelineState:
    errors = list(state.get("errors", []))
    extracted = state.get("extracted", [])

    if not extracted:
        return {**state, "trends": "No papers available to synthesize.", "gaps": "N/A", "errors": errors}

    llm = ChatGroq(
        model=state.get("model_synthesis", GROQ_MODEL_SYNTHESIS),
        api_key=state["groq_api_key"],
        temperature=0.2,
    )

    listing = "\n\n".join(
        f"Paper: {p['title']}\n"
        f"Methodology: {p['methodology']}\n"
        f"Datasets: {p['datasets']}\n"
        f"Key results: {p['key_results']}\n"
        f"Limitations: {p['limitations']}"
        for p in extracted
    )
    prompt = (
        f"Topic: \"{state['topic']}\"\n\n"
        f"Here are structured extractions from {len(extracted)} papers:\n\n{listing}\n\n"
        "Compare these papers. Respond ONLY with a single valid JSON object with exactly these keys: "
        '"trends" (2-4 sentences on common approaches/directions across papers), '
        '"gaps" (2-4 sentences on what is missing or under-explored). '
        "Each value must be a single line with no literal line breaks inside the string "
        "(use spaces between sentences, not newlines). No prose outside the JSON, no markdown fences."
    )
    try:
        resp = llm.invoke(prompt)
        parsed = _safe_json_parse(resp.content, {
            "trends": "Could not parse synthesis output.",
            "gaps": "Could not parse synthesis output.",
        })
    except Exception as e:
        errors.append(f"Synthesis failed: {e}")
        parsed = {"trends": "Synthesis step failed.", "gaps": "Synthesis step failed."}

    return {**state, "trends": parsed.get("trends", ""), "gaps": parsed.get("gaps", ""), "errors": errors}



search_and_fetch = RunnableLambda(search_and_fetch_node)
read_and_extract = RunnableLambda(read_and_extract_node)
synthesize = RunnableLambda(synthesize_node)


def build_pipeline() -> RunnableSequence:
    """Composes the three agent nodes into a single LCEL RunnableSequence."""
    return search_and_fetch | read_and_extract | synthesize


# --------------------------------------------------------------------------
# Public entry point used by app.py
# --------------------------------------------------------------------------
def run_pipeline(
    topic: str,
    max_papers: int,
    date_range: str,
    groq_api_key: str,
    model_filter: str = GROQ_MODEL_FILTER,
    model_extract: str = GROQ_MODEL_EXTRACT,
    model_synthesis: str = GROQ_MODEL_SYNTHESIS,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """
    Runs the full LangChain pipeline and returns a result dict shaped for
    the Streamlit dashboard:
        { topic, max_papers, timestamp, papers, trends, gaps, errors }

    Steps are invoked one Runnable at a time (rather than a single
    `build_pipeline().invoke(...)` call) so the UI's progress bar can get an
    update between each stage. `build_pipeline()` is still there for callers
    that just want to run the whole thing in one shot.
    """
    initial_state: PipelineState = {
        "topic": topic,
        "max_papers": max_papers,
        "date_range": date_range,
        "groq_api_key": groq_api_key,
        "model_filter": model_filter,
        "model_extract": model_extract,
        "model_synthesis": model_synthesis,
        "errors": [],
    }

    _progress(initial_state, progress_callback, 0.05, "Searching ArXiv & filtering by relevance...")
    state = search_and_fetch.invoke(initial_state)

    _progress(state, progress_callback, 0.4, "Downloading & parsing PDFs, extracting metrics...")
    state = read_and_extract.invoke(state)

    _progress(state, progress_callback, 0.8, "Synthesizing findings across papers...")
    state = synthesize.invoke(state)

    _progress(state, progress_callback, 1.0, "Done.")

    return {
        "topic": topic,
        "max_papers": max_papers,
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "papers": state.get("extracted", []),
        "trends": state.get("trends", ""),
        "gaps": state.get("gaps", ""),
        "errors": state.get("errors", []),
        # placeholders for step 4 (output & automation), filled in by app.py
        "sheet_updated": False,
        "report_generated": False,
        "email_sent": False,
        "email": "",
    }


# --------------------------------------------------------------------------
# Q&A chat over a finished run
# --------------------------------------------------------------------------
def answer_question(
    result: dict,
    question: str,
    chat_history: list[dict],
    groq_api_key: str,
    model: str = GROQ_MODEL_QA,
) -> str:
    """
    Answers a follow-up question grounded in a finished pipeline result
    (papers + trends + gaps). `chat_history` is a list of
    {"role": "user"|"assistant", "content": str} dicts, oldest first,
    NOT including the current `question`.
    """
    papers = result.get("papers", [])
    if not papers:
        return "No papers have been analyzed yet — run the pipeline first."

    context = "\n\n".join(
        f"Paper: {p.get('title', '')}\n"
        f"Authors: {p.get('authors', '')}\n"
        f"Published: {p.get('published', '')}\n"
        f"Abstract: {p.get('summary', '')}\n"
        f"Methodology: {p.get('methodology', '')}\n"
        f"Datasets: {p.get('datasets', '')}\n"
        f"Key results: {p.get('key_results', '')}\n"
        f"Limitations: {p.get('limitations', '')}\n"
        f"URL: {p.get('url', '')}"
        for p in papers
    )
    system_prompt = (
        f"You are a research assistant answering questions about papers on "
        f"\"{result.get('topic', '')}\". Only use the information below. If the "
        f"answer isn't in it, say you don't have enough information from these papers.\n\n"
        f"Trends across papers: {result.get('trends', '')}\n"
        f"Gaps across papers: {result.get('gaps', '')}\n\n"
        f"Papers:\n{context}"
    )

    llm = ChatGroq(model=model, api_key=groq_api_key, temperature=0.2)

    messages = [SystemMessage(content=system_prompt)]
    for turn in chat_history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=question))

    try:
        resp = llm.invoke(messages)
        return resp.content
    except Exception as e:
        return f"Sorry, that question couldn't be answered: {e}"