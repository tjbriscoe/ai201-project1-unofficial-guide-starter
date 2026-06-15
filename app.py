
import gradio as gr
from pipeline import retrieve
from generation import ask
import chromadb
import pipeline

import os
import chromadb
from pipeline import (
    SOURCES, load_source, chunk_text,
    embed_and_store, retrieve,
    chroma_client, collection, _bm25_chunks
)

# run ingestion on startup if store is empty
def initialize_pipeline():
    existing = collection.count()
    if existing == 0:
        print("[startup] Vector store empty — running ingestion...")
        all_chunks = []
        for entry in SOURCES:
            try:
                raw = load_source(entry["url"], entry["source_type"])
                if raw == "MANUAL_LOAD_REQUIRED":
                    continue
                chunks = chunk_text(raw, entry["meta"])
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"[startup] Error loading {entry['url']}: {e}")
        if all_chunks:
            embed_and_store(all_chunks)
            print(f"[startup] Stored {len(all_chunks)} chunks.")
    else:
        print(f"[startup] Found {existing} chunks in store — skipping ingestion.")

initialize_pipeline()

def handle_query(question: str, k: int) -> tuple[str, str]:
    """py
    End-to-end handler: retrieve chunks then generate grounded answer.
    Returns (answer, sources) for Gradio outputs.
    """
    if not question.strip():
        return "Enter a question above.", ""

    chunks = retrieve(question.strip(), k=int(k))

    if not chunks:
        return (
            "No sources retrieved. Run pipeline.py first to populate the vector store.",
            "",
        )

    result = ask(question.strip(), chunks)

    # format sources as bullet list
    sources_text = "\n".join(f"• {s}" for s in result["sources"])

    # format retrieved chunks for inspection
    chunks_text = ""
    for i, chunk in enumerate(chunks):
        chunks_text += f"[{i+1}] {chunk.get('source', '?')} — score: {chunk.get('fused_score', 0):.4f}\n"
        chunks_text += f"    {chunk['text'][:200]}...\n\n"

    return result["answer"], sources_text, chunks_text


# ── eval queries for quick testing ────────────────────────────────────────────

EVAL_QUERIES = [
    "What do students say about job outcomes after graduating from Informatics?",
    "How do students rate professor availability and helpfulness in CINF?",
    "Is the online Informatics BS considered equivalent to in-person?",
    "What professors are mentioned on RateMyProfessor and what do students say?",
    "What reasons do students give for switching out of the Informatics major?",
]

# ── gradio interface ───────────────────────────────────────────────────────────

with gr.Blocks(title="UAlbany Informatics Program Q&A") as demo:

    gr.Markdown("# 🎓 UAlbany Informatics Program Q&A")
    gr.Markdown(
        "Ask questions about the program based on real student reviews, "
        "Reddit threads, and professor ratings. Answers are grounded in "
        "retrieved sources only."
    )

    with gr.Row():
        with gr.Column(scale=4):
            question_input = gr.Textbox(
                label="Your question",
                placeholder="e.g. What do students say about job outcomes after graduating?",
                lines=2,
            )
        with gr.Column(scale=1):
            k_slider = gr.Slider(
                minimum=3,
                maximum=10,
                value=6,
                step=1,
                label="Sources (k)",
            )

    ask_btn = gr.Button("Ask", variant="primary")

    answer_output = gr.Textbox(
        label="Answer",
        lines=8,
        interactive=False,
    )

    sources_output = gr.Textbox(
        label="Sources cited",
        lines=4,
        interactive=False,
    )

    with gr.Accordion("Retrieved chunks — inspect retrieval", open=False):
        chunks_output = gr.Textbox(
            label="",
            lines=12,
            interactive=False,
        )

    gr.Markdown("### Evaluation queries")
    gr.Markdown("Click any query to run it directly.")

    with gr.Row():
        for q in EVAL_QUERIES[:3]:
            gr.Button(q, size="sm").click(
                fn=lambda x=q: handle_query(x, 6),
                outputs=[answer_output, sources_output, chunks_output],
            )

    with gr.Row():
        for q in EVAL_QUERIES[3:]:
            gr.Button(q, size="sm").click(
                fn=lambda x=q: handle_query(x, 6),
                outputs=[answer_output, sources_output, chunks_output],
            )

    # wire up main button and enter key
    ask_btn.click(
        fn=handle_query,
        inputs=[question_input, k_slider],
        outputs=[answer_output, sources_output, chunks_output],
    )
    question_input.submit(
        fn=handle_query,
        inputs=[question_input, k_slider],
        outputs=[answer_output, sources_output, chunks_output],
    )

demo.launch()