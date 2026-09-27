import html
import os
import random
import re
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from ingestion.pinecone import upload_to_pinecone
from retrieval.rag import (
    build_retriever, search_documents, validate_context,
    generate_answer, validate_answer,
)

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(page_title="Paper Intelligence", page_icon="◆", layout="wide")

THINKING_LINES = [
    "Reading between the lines…",
    "Cross-referencing the paper…",
    "Tracing the argument…",
    "Weighing the evidence…",
]

FALLBACK_QUESTIONS = [
    "What problem does this paper try to solve?",
    "What methodology did the authors use?",
    "What are the main findings?",
]

# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
    --bg:#0b0d12; --panel:rgba(255,255,255,.045); --panel-solid:#12151d; --border:rgba(255,255,255,.09);
    --text:#edeff3; --soft:#9aa0ae; --faint:#5c6270; --gold:#d7a854; --gold-soft:rgba(215,168,84,.13);
    --ok:#4fbf8b; --ok-bg:rgba(79,191,139,.12); --err:#e07a6f; --err-bg:rgba(224,122,111,.12);
}
html, body, .stApp { background:var(--bg) !important; color:var(--text); font-family:'Space Grotesk',sans-serif; }
#MainMenu, footer, header {visibility:hidden;}
::-webkit-scrollbar { width:9px; } ::-webkit-scrollbar-thumb { background:#242833; border-radius:6px; }
.block-container { max-width:840px; padding-top:26px; padding-bottom:130px; }
section[data-testid="stSidebar"] { background:var(--panel-solid); border-right:1px solid var(--border); }
section[data-testid="stSidebar"] .block-container { padding-top:34px; }

/* ---- hero / aurora ---- */
.hero-wrap { position:relative; text-align:center; padding:38px 10px 30px; overflow:hidden; border-radius:20px; margin-bottom:26px; }
.blob { position:absolute; border-radius:50%; filter:blur(60px); opacity:.35; animation:drift 14s ease-in-out infinite alternate; z-index:0; }
.blob1 { width:260px; height:260px; background:#7a4fbf; top:-90px; left:8%; }
.blob2 { width:220px; height:220px; background:#d7a854; bottom:-90px; right:10%; animation-delay:2s; }
@keyframes drift { from{transform:translate(0,0) scale(1);} to{transform:translate(20px,-18px) scale(1.12);} }
.hero-inner { position:relative; z-index:1; }
.eyebrow { font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:.22em; color:var(--gold); margin-bottom:14px; }
.hero-inner h1 { font-family:'Fraunces',serif; font-weight:600; font-size:46px; margin:0 0 12px; letter-spacing:-.01em; }
.hero-inner p { color:var(--soft); font-size:15px; max-width:460px; margin:0 auto; line-height:1.6; }

/* ---- top strip (chat mode) ---- */
.topstrip { display:flex; align-items:center; gap:10px; font-family:'JetBrains Mono',monospace; font-size:12px; color:var(--soft); margin-bottom:22px; padding-bottom:14px; border-bottom:1px solid var(--border); }
.topstrip .dot { width:7px; height:7px; border-radius:50%; background:var(--ok); box-shadow:0 0 8px var(--ok); }
.topstrip b { color:var(--text); font-family:'Fraunces',serif; font-size:15px; font-weight:600; }

/* ---- upload dropzone ---- */
.dropzone { background:var(--panel); border:1.5px dashed var(--border); border-radius:16px; padding:30px; text-align:center; margin-bottom:14px; transition:.2s; }
.dropzone:hover { border-color:var(--gold); }
.dropzone h3 { font-family:'Fraunces',serif; font-size:19px; margin:8px 0 4px; }
.dropzone p { color:var(--faint); font-size:13px; margin:0; }
[data-testid="stFileUploader"] section {
    background:transparent;
    border:none;
    pointer-events:none;
}

[data-testid="stFileUploader"] button {
    background:var(--gold) !important;
    color:#161208 !important;
    border:none !important;
    border-radius:8px !important;
    font-weight:600 !important;
    pointer-events:auto !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] {
    color:var(--soft) !important;
    pointer-events:none;
}

/* ---- sidebar dashboard ---- */
.stat-card { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:14px 16px; margin-bottom:10px; }
.stat-card .k { font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:.08em; color:var(--faint); margin-bottom:4px; }
.stat-card .v { font-family:'Fraunces',serif; font-size:20px; color:var(--text); }
.stat-card .v.small { font-family:'JetBrains Mono',monospace; font-size:12px; word-break:break-word; }
.side-label { font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:.1em; color:var(--faint); margin:18px 0 8px 2px; }

/* ---- buttons (global pill style) ---- */
.stButton > button { background:var(--panel); border:1px solid var(--border); border-radius:10px; color:var(--text); font-family:'Space Grotesk',sans-serif; font-size:13.5px; transition:.15s; }
.stButton > button:hover { border-color:var(--gold); background:var(--gold-soft); color:var(--gold); }

/* ---- chat rows ---- */
.row { display:flex; gap:10px; margin:16px 0; animation:rise .35s ease; }
.row.user { flex-direction:row-reverse; }
@keyframes rise { from{opacity:0; transform:translateY(6px);} to{opacity:1; transform:translateY(0);} }
.avatar { flex-shrink:0; width:30px; height:30px; border-radius:9px; display:flex; align-items:center; justify-content:center; font-family:'JetBrains Mono',monospace; font-size:10px; font-weight:600; }
.avatar.user { background:linear-gradient(135deg,#4a4f5c,#2c2f38); color:var(--text); }
.avatar.ai { background:linear-gradient(135deg,var(--gold),#8a5a2b); color:#1a1206; }
.bubble { max-width:78%; padding:13px 16px; border-radius:14px; font-size:14.5px; line-height:1.65; }
.bubble.user { background:#1c1f28; border:1px solid var(--border); border-top-right-radius:3px; }
.bubble.ai { background:var(--panel); border:1px solid var(--border); border-top-left-radius:3px; }
.chips { margin-top:10px; }
.chip { display:inline-block; background:var(--gold-soft); color:var(--gold); border-radius:20px; padding:2px 10px; margin:3px 5px 0 0; font-family:'JetBrains Mono',monospace; font-size:10.5px; }

/* ---- empty state ---- */
.empty { text-align:center; padding:26px 0 6px; }
.empty .big { font-family:'Fraunces',serif; font-size:20px; }
.empty .small { color:var(--faint); font-size:13px; margin-top:4px; }

/* ---- misc ---- */
[data-testid="stChatInput"] textarea { border-radius:12px !important; border:1px solid var(--border) !important; background:var(--panel) !important; color:var(--text) !important; }
div[data-testid="stExpander"] { background:var(--panel); border:1px solid var(--border); border-radius:12px; }
[data-testid="stSpinner"] { color:var(--gold); }
.stAlert { border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================

if "processed" not in st.session_state:
    st.session_state.update(
        processed=False, retriever=None, doc_id=None, filename=None,
        history=[], suggestions=[], feedback={},
    )

# ============================================================
# HELPERS
# ============================================================


def format_text(t):
    t = html.escape(t or "")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return t.replace("\n", "<br>")


def build_overview_context(retriever, max_chunks=8):
    chunks = retriever.get("chunks", [])
    if not chunks:
        return ""
    if len(chunks) <= max_chunks:
        sample = chunks
    else:
        step = len(chunks) / max_chunks
        sample = [chunks[int(i * step)] for i in range(max_chunks)]
    return "\n\n".join(f'Page {c.get("page_number","?")}:\n{c.get("text","")}' for c in sample)


def generate_suggested_questions(retriever):
    try:
        context = build_overview_context(retriever)
        if not context:
            return FALLBACK_QUESTIONS
        prompt = (
            "Based only on the excerpts below, write exactly 3 specific, concrete "
            "questions a reader could ask about this paper (about its problem, "
            "method, or results). Return only the 3 questions, one per line, "
            "numbered 1-3. No other text."
        )
        raw = generate_answer(prompt, context, [])
        qs = []
        for line in raw.splitlines():
            cleaned = re.sub(r"^[\-\*\d\.\)]+\s*", "", line.strip()).strip()
            if cleaned.endswith("?") and 8 <= len(cleaned) <= 160:
                qs.append(cleaned)
        return qs[:3] if len(qs) >= 2 else FALLBACK_QUESTIONS
    except Exception:
        return FALLBACK_QUESTIONS


def doc_stats(retriever):
    chunks = retriever.get("chunks", [])
    pages = {c.get("page_number") for c in chunks if c.get("page_number") is not None}
    return len(chunks), len(pages)


def reset_document():
    st.session_state.update(
        processed=False, retriever=None, doc_id=None, filename=None,
        history=[], suggestions=[], feedback={},
    )


# ============================================================
# UPLOAD SCREEN
# ============================================================

if not st.session_state.processed:

    st.markdown("""
    <div class="hero-wrap">
        <div class="blob blob1"></div><div class="blob blob2"></div>
        <div class="hero-inner">
            <div class="eyebrow">PAPER INTELLIGENCE</div>
            <h1>Read faster. Ask anything.</h1>
            <p>Drop in a research paper and interrogate it — methodology,
            findings, contributions — every answer traced back to the page it came from.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="dropzone"><h3>◆ Upload a paper</h3><p>PDF only · ready when you are</p></div>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded_file:
        pdf_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(uploaded_file.getbuffer())
                pdf_path = f.name

            with st.status("Processing your paper", expanded=True) as status:
                st.write("Extracting text & building embeddings…")
                doc_id, records = upload_to_pinecone(pdf_path)

                st.write("Assembling the retriever index…")
                retriever = build_retriever(pdf_path, doc_id, records)

                st.write("Curating a few starter questions…")
                suggestions = generate_suggested_questions(retriever)

                status.update(label="Ready to explore", state="complete")

            st.session_state.update(
                processed=True, retriever=retriever, doc_id=doc_id,
                filename=uploaded_file.name, history=[],
                suggestions=suggestions, feedback={},
            )
            st.rerun()

        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Something went wrong: {e}")
        finally:
            if pdf_path and os.path.exists(pdf_path):
                os.remove(pdf_path)

# ============================================================
# CHAT SCREEN
# ============================================================

else:

    retriever = st.session_state.retriever
    n_chunks, n_pages = doc_stats(retriever)

    # ---- sidebar dashboard ----
    with st.sidebar:
        st.markdown('<div class="eyebrow" style="text-align:left">◆ PAPER INTELLIGENCE</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-card"><div class="k">DOCUMENT</div><div class="v small">{html.escape(st.session_state.filename or "")}</div></div>
        <div class="stat-card"><div class="k">PAGES</div><div class="v">{n_pages}</div></div>
        <div class="stat-card"><div class="k">INDEXED PASSAGES</div><div class="v">{n_chunks}</div></div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="side-label">QUICK PROMPTS</div>', unsafe_allow_html=True)
        for i, q in enumerate(st.session_state.suggestions):
            if st.button(q, key=f"side_sugg_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

        st.markdown('<div class="side-label">SESSION</div>', unsafe_allow_html=True)
        if st.button("🗑️ Clear chat", use_container_width=True, disabled=not st.session_state.history):
            st.session_state.history, st.session_state.feedback = [], {}
            st.rerun()
        if st.button("↺ New paper", use_container_width=True):
            reset_document()
            st.rerun()

    # ---- top strip ----
    st.markdown(f"""
    <div class="topstrip"><span class="dot"></span><b>Paper Intelligence</b>
    &nbsp;·&nbsp; {html.escape(st.session_state.filename or "")}</div>
    """, unsafe_allow_html=True)

    # ---- conversation ----
    for i, msg in enumerate(st.session_state.history):

        st.markdown(f'<div class="row user"><div class="avatar user">YOU</div>'
                     f'<div class="bubble user">{format_text(msg["user"])}</div></div>', unsafe_allow_html=True)

        chips = ""
        if msg.get("sources"):
            chips = '<div class="chips">' + "".join(f'<span class="chip">p.{p}</span>' for p in msg["sources"]) + "</div>"

        st.markdown(f'<div class="row"><div class="avatar ai">AI</div>'
                     f'<div class="bubble ai">{format_text(msg["assistant"])}{chips}</div></div>', unsafe_allow_html=True)

        fb1, fb2, _ = st.columns([1, 1, 10])
        current = st.session_state.feedback.get(i)
        with fb1:
            if st.button("👍" if current != "up" else "✅", key=f"up_{i}"):
                st.session_state.feedback[i] = "up"
                st.rerun()
        with fb2:
            if st.button("👎" if current != "down" else "✅", key=f"down_{i}"):
                st.session_state.feedback[i] = "down"
                st.rerun()

    if not st.session_state.history:
        st.markdown("""
        <div class="empty"><div class="big">Ask anything about this paper</div>
        <div class="small">Try a quick prompt from the sidebar, or type your own below</div></div>
        """, unsafe_allow_html=True)

    # ---- input ----
    question = st.chat_input("Ask a question about this paper…")
    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")

    if question:
        history = st.session_state.history

        with st.spinner(random.choice(THINKING_LINES)):
            top_chunks, reranked = search_documents(question, retriever, final_k=5)

        valid, message = validate_context(top_chunks, reranked, retriever["chunks"])

        if not valid:
            st.error(message)
        else:
            context, pages = "", []
            for rank, (index, score) in enumerate(top_chunks):
                chunk = retriever["chunks"][index]
                pages.append(chunk["page_number"])
                context += (f'\n[Chunk {rank+1}]\nDocument: {chunk["doc_id"]}\n'
                            f'Page: {chunk["page_number"]}\nText:\n{chunk["text"]}\n')

            with st.spinner("Drafting a grounded answer…"):
                answer = generate_answer(question, context, history)

            valid, message = validate_answer(answer, context)
            if not valid:
                st.error(message)
            else:
                history.append({
                    "user": question, "assistant": answer,
                    "sources": sorted(set(pages), key=lambda p: (p is None, p)),
                })
                st.session_state.history = history
                st.rerun()