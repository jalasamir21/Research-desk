
import datetime as dt
import html

import streamlit as st

from pipeline import (
    run_pipeline,
    answer_question,
    GROQ_MODEL_FILTER,
    GROQ_MODEL_EXTRACT,
    GROQ_MODEL_SYNTHESIS,
)
from outputs import generate_pdf_brief, send_email_brief

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="The Research Desk",
    page_icon="📚",
    layout="wide",
)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
defaults = {
    "run_history": [],       # list of past run summaries
    "pipeline_result": None, # result of the most recent run
    "is_running": False,
    "chat_messages": [],     # Q&A chat history for the current run
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value



CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

:root {
    --ink: #1B2740;
    --ink-soft: #3E4A63;
    --ink-muted: #6B7288;
    --paper: #FBF7EC;
    --paper-dim: #F1E9D3;
    --card: #FFFDF7;
    --rule: #D9C89C;
    --accent: #A8531F;
    --accent-soft: #F1DFC9;
    --gold: #D9A652;
    --gold-soft: #F3DFB4;
    --sage: #43614F;
    --sage-soft: #DCE7DE;
}

.stApp {
    background:
        radial-gradient(circle at 1px 1px, rgba(27,39,64,0.06) 1px, transparent 0) 0 0/16px 16px,
        var(--paper);
}
#MainMenu, footer { visibility: hidden; }

h1, h2, h3 { font-family: 'Fraunces', serif; color: var(--ink); }

/* ---- Masthead hero: a letterpress ticket, not a dark banner ---- */
.masthead {
    background: var(--card);
    border: 1.5px solid var(--ink);
    border-radius: 2px;
    padding: 2.2rem 2.6rem 1.9rem 2.6rem;
    margin-bottom: 1.8rem;
    position: relative;
    box-shadow: 6px 6px 0 var(--gold-soft), 6px 6px 0 1.5px var(--ink);
}
.masthead::before, .masthead::after {
    content: "✦";
    position: absolute;
    color: var(--accent);
    font-size: 0.85rem;
    top: 0.9rem;
}
.masthead::before { left: 0.95rem; }
.masthead::after { right: 0.95rem; }
.masthead .eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.6rem;
    text-align: center;
}
.masthead h1 {
    color: var(--ink);
    font-size: 2.4rem;
    font-weight: 600;
    font-style: italic;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.01em;
    text-align: center;
}
.masthead p {
    color: var(--ink-soft);
    font-size: 0.95rem;
    margin: 0 auto 1.3rem auto;
    max-width: 560px;
    text-align: center;
    font-family: 'IBM Plex Sans', sans-serif;
}
.masthead-rule {
    border: none;
    border-top: 1px dashed var(--rule);
    margin: 0 0 1.1rem 0;
}
.contents-line {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    align-items: center;
    gap: 0.5rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem;
}
.contents-item {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--paper-dim);
    border: 1px solid var(--rule);
    border-radius: 999px;
    color: var(--ink-soft);
    padding: 0.25rem 0.75rem 0.25rem 0.55rem;
}
.contents-item .num {
    background: var(--gold);
    color: var(--ink);
    width: 16px;
    height: 16px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.62rem;
    font-weight: 600;
}
.contents-sep { display: none; }

/* ---- Top Streamlit toolbar: blend into the page instead of a dark bar ---- */
header[data-testid="stHeader"] {
    background: var(--paper) !important;
    box-shadow: none !important;
}
header[data-testid="stHeader"] * {
    color: var(--ink) !important;
    fill: var(--ink) !important;
}
div[data-testid="stToolbar"], div[data-testid="stDecoration"] { background: var(--paper) !important; }
div[data-testid="stToolbarActions"] button {
    background: var(--paper-dim) !important;
    color: var(--ink) !important;
    border: 1px solid var(--rule) !important;
}


/* ---- Step headers (echo the masthead numbering) ---- */
.step-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0.3rem 0 1rem 0;
}
.step-badge {
    background: var(--gold-soft);
    border: 1.5px solid var(--ink);
    color: var(--ink);
    width: 30px;
    height: 30px;
    min-width: 30px;
    border-radius: 3px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 0.95rem;
}
.step-title { font-family: 'Fraunces', serif; font-size: 1.25rem; font-weight: 600; color: var(--ink); }
.step-subtitle { font-size: 0.82rem; color: var(--ink-muted); margin: -0.4rem 0 0.9rem 2.75rem; }

/* ---- Section card wrapper ---- */
.section-card {
    background: var(--card);
    border: 1px solid var(--rule);
    border-radius: 6px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}

/* ---- Metric cards ---- */
div[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--rule);
    border-top: 3px solid var(--accent);
    border-radius: 4px;
    padding: 0.85rem 1rem 0.7rem 1rem;
}
div[data-testid="stMetricLabel"] {
    color: var(--ink-muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
div[data-testid="stMetricValue"] { font-family: 'Fraunces', serif; color: var(--ink); }

/* ---- Buttons ---- */
.stButton > button, [data-testid="stFormSubmitButton"] button, .stDownloadButton > button {
    background: var(--gold);
    color: var(--ink);
    border: 1.5px solid var(--ink);
    border-radius: 4px;
    font-weight: 600;
    font-family: 'IBM Plex Sans', sans-serif;
    padding: 0.55rem 1.2rem;
    box-shadow: 3px 3px 0 var(--ink);
    transition: transform 0.08s ease, box-shadow 0.08s ease, background 0.15s ease;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover, .stDownloadButton > button:hover {
    background: var(--accent-soft);
    color: var(--ink);
    transform: translate(-1px, -1px);
    box-shadow: 4px 4px 0 var(--ink);
}
.stButton > button:active, [data-testid="stFormSubmitButton"] button:active, .stDownloadButton > button:active {
    transform: translate(1px, 1px);
    box-shadow: 1px 1px 0 var(--ink);
}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
    outline: 2px solid var(--ink);
    outline-offset: 2px;
}

/* ---- Expanders as catalog cards ---- */
div[data-testid="stExpander"] {
    background: var(--card);
    border: 1px solid var(--rule);
    border-left: 3px solid var(--accent);
    border-radius: 3px;
    margin-bottom: 0.6rem;
}
div[data-testid="stExpander"] summary {
    font-family: 'Fraunces', serif;
    font-weight: 500;
}

/* ---- Tabs ---- */
button[data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
    color: var(--ink-muted);
}
div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--accent); }
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: var(--paper-dim);
    border-right: 1px solid var(--rule);
}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    color: var(--ink);
}
.sidebar-kicker {
    font-family: 'Fraunces', serif;
    font-style: italic;
    font-size: 1.35rem;
    color: var(--ink);
    margin-bottom: 0.1rem;
}
.sidebar-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--ink-muted);
    margin-bottom: 1.1rem;
}
.run-log-entry {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
    color: var(--ink-soft);
    border-bottom: 1px dotted var(--rule);
    padding: 0.3rem 0;
}

/* ---- Field labels for paper detail rows ---- */
.field-label {
    display: inline-block;
    background: var(--accent-soft);
    color: var(--accent);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border-radius: 3px;
    padding: 0.15rem 0.5rem;
    margin-bottom: 0.3rem;
}

/* ---- Force light widget chrome + ink text everywhere ----
   Streamlit's dark theme defaults (white text, black widget fills) leak
   into custom layouts like this one. Override broadly so every label,
   caption, and control reads in ink-on-paper, never white-on-white. */
.stApp, .stApp p, .stApp span, .stApp label, .stApp li,
div[data-testid="stWidgetLabel"] p, div[data-testid="stWidgetLabel"] label,
div[data-testid="stCaptionContainer"], div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li {
    color: var(--ink);
}
div[data-testid="stCaptionContainer"] { color: var(--ink-muted) !important; }

.stTextInput input, .stTextArea textarea, .stNumberInput input,
div[data-baseweb="select"] > div, div[data-baseweb="input"] {
    background: var(--card) !important;
    color: var(--ink) !important;
    border: 1px solid var(--rule) !important;
    border-radius: 4px !important;
}

/* Chat input (Ask the Desk tab) — st.chat_input lives in a fixed bottom
   container that keeps Streamlit's native dark chrome even under a custom
   light theme, so we target that container explicitly, not just the
   widget's own wrapper. */
div[data-testid="stBottomBlockContainer"],
div[data-testid="stBottom"],
.stChatFloatingInputContainer,
div[data-testid="stChatInput"] {
    background: var(--paper-dim) !important;
}
div[data-testid="stChatInput"] textarea,
div[data-testid="stChatInputTextArea"] {
    background: #2A2A2A !important;
    color: #D8D8D8 !important;
    caret-color: #D8D8D8 !important;
}
div[data-testid="stChatInput"] textarea::placeholder {
    color: #9A9A9A !important;
    opacity: 1 !important;
}
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] span,
div[data-testid="stChatMessage"] div {
    color: var(--ink) !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {
    color: var(--ink-muted) !important;
    opacity: 1 !important;
}
div[data-baseweb="select"] span, div[data-baseweb="select"] div { color: var(--ink) !important; }
div[data-baseweb="popover"] { background: var(--card) !important; }
div[data-baseweb="popover"] li, div[data-baseweb="popover"] li span {
    color: var(--ink) !important;
    background: var(--card) !important;
}
div[data-baseweb="popover"] li:hover { background: var(--paper-dim) !important; }

/* Sliders */
div[data-testid="stSlider"] div[role="slider"] {
    background-color: var(--accent) !important;
    border-color: var(--accent) !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
    background: var(--accent) !important;
}
div[data-testid="stTickBarMin"], div[data-testid="stTickBarMax"] { color: var(--ink-muted) !important; }
div[data-testid="stThumbValue"] { color: var(--ink) !important; }

/* Checkboxes */
div[data-testid="stCheckbox"] label span[data-baseweb="checkbox"] > div {
    background: var(--card) !important;
    border-color: var(--ink-muted) !important;
}
div[data-testid="stCheckbox"] label span[aria-checked="true"] > div {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}

/* File uploader */
section[data-testid="stFileUploaderDropzone"] {
    background: var(--card) !important;
    border: 1px dashed var(--rule) !important;
}
section[data-testid="stFileUploaderDropzone"] span, section[data-testid="stFileUploaderDropzone"] small {
    color: var(--ink-soft) !important;
}

/* Password-visibility / help icon buttons */
button[title="Show password text"], button[title="Hide password text"] { color: var(--ink-muted) !important; }
hr { border-color: var(--rule); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def step_header(number: str, title: str, subtitle: str = ""):
    """Renders a numbered step badge + title, echoing the masthead's contents line."""
    st.markdown(
        f"""
        <div class="step-header">
            <div class="step-badge">{number}</div>
            <div class="step-title">{title}</div>
        </div>
        {f'<div class="step-subtitle">{subtitle}</div>' if subtitle else ''}
        """,
        unsafe_allow_html=True,
    )


def field_row(label: str, value: str):
    """Renders a labeled field with a small mono-tag label above the value."""
    st.markdown(f'<span class="field-label">{label}</span>', unsafe_allow_html=True)
    st.write(value)


# --------------------------------------------------------------------------
# Sidebar — configuration / API keys
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="sidebar-kicker">The Desk</div>'
        '<div class="sidebar-sub">Configuration &amp; Integrations</div>',
        unsafe_allow_html=True,
    )

    groq_api_key = st.text_input(
        "Groq API Key", type="password",
        help="Used by all three LLM agents (filtering, extraction, synthesis).",
    )

    with st.expander("Model selection (advanced)"):
        groq_model_options = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ]
        model_filter = st.selectbox(
            "Relevance filter model — step 1", groq_model_options,
            index=groq_model_options.index(GROQ_MODEL_FILTER) if GROQ_MODEL_FILTER in groq_model_options else 0,
        )
        model_extract = st.selectbox(
            "Extraction model — step 2", groq_model_options,
            index=groq_model_options.index(GROQ_MODEL_EXTRACT) if GROQ_MODEL_EXTRACT in groq_model_options else 0,
        )
        model_synthesis = st.selectbox(
            "Synthesis model — step 3", groq_model_options,
            index=groq_model_options.index(GROQ_MODEL_SYNTHESIS) if GROQ_MODEL_SYNTHESIS in groq_model_options else 0,
        )

    st.divider()
    st.subheader("Email")
    st.caption("For auto-sending the finished brief.")
    smtp_sender = st.text_input("Sender email", placeholder="you@example.com")
    smtp_password = st.text_input("Sender app password", type="password")

    st.divider()
    if st.session_state.run_history:
        st.subheader("Run log")
        for run in reversed(st.session_state.run_history):
            st.markdown(
                f'<div class="run-log-entry">{run["timestamp"]} &nbsp;·&nbsp; '
                f'"{run["topic"]}" &nbsp;·&nbsp; {run["max_papers"]} papers</div>',
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------
# Main — masthead header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="masthead">
        <div class="eyebrow">Agent Pipeline · Vol. I</div>
        <h1>The Research Desk</h1>
        <p>A standing desk of Groq-powered agents that search ArXiv, read the
        papers, and write up what's changed.</p>
        <hr class="masthead-rule">
        <div class="contents-line">
            <span class="contents-item"><span class="num">1</span> User Input</span>
            <span class="contents-item"><span class="num">2</span> Search &amp; Fetch</span>
            <span class="contents-item"><span class="num">3</span> Read &amp; Extract</span>
            <span class="contents-item"><span class="num">4</span> Synthesize</span>
            <span class="contents-item"><span class="num">5</span> PDF / Email</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_run, tab_dashboard, tab_qa = st.tabs(["Run Pipeline", "Dashboard", "Ask the Desk"])

# --------------------------------------------------------------------------
# Tab 1 — Run Pipeline (the input form from the diagram)
# --------------------------------------------------------------------------
with tab_run:
    step_header("1", "Assign the desk a topic", "Topic, paper count, and output preferences")

    with st.form("pipeline_form"):
        col1, col2 = st.columns(2)

        with col1:
            topic = st.text_input(
                "Research topic",
                placeholder="e.g. Retrieval-Augmented Generation for code synthesis",
            )
            max_papers = st.slider("Max papers to analyze", min_value=1, max_value=20, value=5)

        with col2:
            email = st.text_input("Notification email", placeholder="you@example.com")
            date_range = st.selectbox(
                "Publication window",
                ["Last 7 days", "Last 30 days", "Last 6 months", "Last year", "Any time"],
                index=2,
            )

        st.markdown(
            '<div style="font-family:\'Fraunces\',serif; font-weight:600; font-size:1.02rem; '
            'margin-top:0.7rem; margin-bottom:0.4rem; color:var(--ink);">Output &amp; automation</div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            generate_report = st.checkbox("Generate Markdown/PDF brief", value=True)
        with c2:
            send_email = st.checkbox("Send email brief", value=False)

        submitted = st.form_submit_button("Send the agents to work", use_container_width=True)

    if submitted:
        # ---- basic validation ----
        errors = []
        if not topic.strip():
            errors.append("Enter a research topic before running the pipeline.")
        if not groq_api_key:
            errors.append("Add your Groq API key in the sidebar.")
        if send_email and not email.strip():
            errors.append("Add a notification email, or uncheck 'Send email brief'.")
        if send_email and not smtp_sender.strip():
            errors.append("Add a sender email in the sidebar, or uncheck 'Send email brief'.")
        if send_email and not smtp_password:
            errors.append("Add a sender app password in the sidebar, or uncheck 'Send email brief'.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            st.session_state.is_running = True
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def _cb(pct, msg):
                progress_bar.progress(pct)
                status_text.info(msg)

            result = run_pipeline(
                topic=topic,
                max_papers=max_papers,
                date_range=date_range,
                groq_api_key=groq_api_key,
                model_filter=model_filter,
                model_extract=model_extract,
                model_synthesis=model_synthesis,
                progress_callback=_cb,
            )
            # Step 4 (output & automation)
            result["email"] = email
            result["report_generated"] = generate_report
            result["email_sent"] = False  # overwritten below if send_email is checked

            if send_email:
                status_text.info("Emailing the brief...")
                email_pdf_bytes = generate_pdf_brief(result)
                email_success, email_msg = send_email_brief(
                    result,
                    email_pdf_bytes,
                    smtp_sender.strip(),
                    smtp_password,
                    email.strip(),
                )
                result["email_sent"] = email_success
                result["email_message"] = email_msg
                if email_success:
                    st.success(email_msg)
                else:
                    st.error(email_msg)

            st.session_state.pipeline_result = result
            st.session_state.run_history.append(result)
            st.session_state.is_running = False
            st.session_state.chat_messages = []

            status_text.success("Run complete — open the Dashboard tab to read it.")

# --------------------------------------------------------------------------
# Tab 2 — Dashboard (interactive results & metrics)
# --------------------------------------------------------------------------
with tab_dashboard:
    step_header("5", "Read the brief", "Interactive results & metrics")

    result = st.session_state.pipeline_result
    if not result:
        st.info("Nothing on the desk yet. Head to **Run Pipeline** and give it a topic.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Topic", result["topic"])
        m2.metric("Papers analyzed", len(result["papers"]))
        m3.metric("Email sent", "✓" if result["email_sent"] else "—")

        if result.get("errors"):
            with st.expander(f"{len(result['errors'])} warning(s) during this run"):
                for e in result["errors"]:
                    st.warning(e)

        st.markdown(
            '<div style="font-family:\'Fraunces\',serif; font-size:1.15rem; font-weight:600; '
            'margin:1.5rem 0 0.7rem 0; color:var(--ink);">Papers</div>',
            unsafe_allow_html=True,
        )
        if not result["papers"]:
            st.info("No papers matched this topic and date range — try widening the window.")
        for i, p in enumerate(result["papers"], start=1):
            with st.expander(f"{i:02d}  ·  {p['title']}"):
                st.write(f"**Authors:** {p['authors']}")
                st.write(f"**Published:** {p['published']}")
                st.write(f"**Abstract:** {p['summary']}")
                st.markdown(f"[View on ArXiv ↗]({p['url']})")
                st.markdown("<hr style='margin:0.8rem 0;'>", unsafe_allow_html=True)
                fc1, fc2 = st.columns(2)
                with fc1:
                    field_row("Methodology", p.get("methodology", "N/A"))
                    field_row("Key Results", p.get("key_results", "N/A"))
                with fc2:
                    field_row("Datasets", p.get("datasets", "N/A"))
                    field_row("Limitations", p.get("limitations", "N/A"))

        st.markdown(
            '<div style="font-family:\'Fraunces\',serif; font-size:1.15rem; font-weight:600; '
            'margin:1.5rem 0 0.7rem 0; color:var(--ink);">Synthesis</div>',
            unsafe_allow_html=True,
        )
        trends_html = html.escape(result["trends"] or "N/A").replace("\n", "<br>")
        gaps_html = html.escape(result["gaps"] or "N/A").replace("\n", "<br>")
        st.markdown(
            '<div class="section-card">'
            '<span class="field-label">Trends</span>'
            f'<div style="margin-bottom:1rem;">{trends_html}</div>'
            '<span class="field-label">Gaps</span>'
            f'<div>{gaps_html}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        if result["report_generated"]:
            papers_md = "\n\n".join(
                f"### {p['title']}\n"
                f"*{p['authors']} — {p['published']}*\n\n"
                f"- **Methodology:** {p.get('methodology', 'N/A')}\n"
                f"- **Datasets:** {p.get('datasets', 'N/A')}\n"
                f"- **Key results:** {p.get('key_results', 'N/A')}\n"
                f"- **Limitations:** {p.get('limitations', 'N/A')}\n"
                f"- [ArXiv link]({p['url']})"
                for p in result["papers"]
            )
            brief_md = (
                f"# Research Brief: {result['topic']}\n\n"
                f"Generated: {result['timestamp']}\n\n"
                f"## Trends\n{result['trends']}\n\n"
                f"## Gaps\n{result['gaps']}\n\n"
                f"## Papers\n\n{papers_md}\n"
            )

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "Download Markdown brief",
                    data=brief_md,
                    file_name=f"brief_{result['topic'].replace(' ', '_')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with dl2:
                pdf_bytes = generate_pdf_brief(result)
                st.download_button(
                    "Download PDF brief",
                    data=pdf_bytes,
                    file_name=f"brief_{result['topic'].replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

# --------------------------------------------------------------------------
# Tab 3 — Ask the Desk (Q&A chat over the last finished run)
# --------------------------------------------------------------------------
with tab_qa:
    step_header("6", "Ask the desk", "Chat about the papers, trends, and gaps from your last run")

    result = st.session_state.pipeline_result
    if not result or not result.get("papers"):
        st.info("Nothing to ask about yet — run the pipeline first.")
    elif not groq_api_key:
        st.warning("Add your Groq API key in the sidebar to chat.")
    else:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        question = st.chat_input("Ask about these papers, the trends, or the gaps...")
        if question:
            st.session_state.chat_messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = answer_question(
                        result,
                        question,
                        st.session_state.chat_messages[:-1],
                        groq_api_key,
                    )
                st.write(answer)
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})

        if st.session_state.chat_messages:
            if st.button("Clear chat"):
                st.session_state.chat_messages = []
                st.rerun()