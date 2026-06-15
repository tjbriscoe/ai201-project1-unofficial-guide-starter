
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a research assistant helping students evaluate UAlbany's 
Informatics program. You answer questions based only on the provided student reviews, 
Reddit comments, and professor ratings. Be direct and specific. When evidence is thin 
or contradictory, say so clearly. Never invent information not present in the sources."""


def map_chunk(query: str, chunk: dict) -> dict:
    """
    Extract the single most relevant claim from a chunk for the given query.
    Returns dict with 'claim' (str) and 'source' (str).
    """
    prompt = f"""Query: {query}

Source text:
{chunk['text']}

Extract the single most relevant claim from this text that helps answer the query.
If this text is not relevant to the query, reply with exactly: NOT_RELEVANT

Reply with only the claim or NOT_RELEVANT. No preamble."""

    response  = client.chat.completions.create(
    model="llama3-8b-8192",
    max_tokens=200,
    messages=[{"role": "user", "content": prompt}],
)
    

    claim = response.choices[0].message.content.strip()
    
    return {
        "claim": claim,
        "source": chunk.get("source", "unknown"),
        "url": chunk.get("url", ""),
        "topic_tag": chunk.get("topic_tag", ""),
        "relevant": claim != "NOT_RELEVANT",
    }


def reduce_claims(query: str, mapped: list[dict]) -> dict:
    """
    Synthesize extracted claims into a final answer with citations.
    Returns dict with 'answer' (str) and 'sources' (list of URLs).
    """
    relevant = [m for m in mapped if m["relevant"]]

    if len(relevant) < 2:
        return {
            "answer": (
                "There is not enough evidence in the available sources to "
                "confidently answer this question. Try rephrasing your query "
                "or check that your source files have been loaded correctly."
            ),
            "sources": [],
        }

    claims_text = ""
    for i, m in enumerate(relevant):
        claims_text += f"[{i+1}] Source: {m['source']} ({m['url']})\n"
        claims_text += f"    Claim: {m['claim']}\n\n"

    prompt = f"""Query: {query}

Extracted claims from student sources:
{claims_text}

Write a clear, direct answer to the query using these claims. Follow these rules:
- Cite each claim inline using its number like [1] or [2]
- If claims contradict each other, note the disagreement explicitly
- If only one or two sources address the query, flag that evidence is limited
- Do not invent any information not present in the claims above
- Write in plain prose, no bullet points, 3-5 sentences maximum"""

    response = client.chat.completions.create(
    model="llama3-8b-8192",
    max_tokens=500,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ],
)

    answer = response.choices[0].message.content.strip()
    sources = list({m["url"] for m in relevant if m["url"]})

    return {"answer": answer, "sources": sources}


def ask(query: str, chunks: list[dict]) -> dict:
    """
    Full map-reduce generation pipeline.
    
    Args:
        query:  The user's question
        chunks: List of chunk dicts from retrieve()
    
    Returns:
        {
            "answer":  str,         — synthesized answer with inline citations
            "sources": list[str],   — list of source URLs cited
            "mapped":  list[dict],  — raw map step output for debugging
        }
    """
    if not chunks:
        return {
            "answer": "No chunks were retrieved. Check that your vector store is populated.",
            "sources": [],
            "mapped": [],
        }

    # map step — extract one claim per chunk
    mapped = [map_chunk(query, chunk) for chunk in chunks]

    # reduce step — synthesize into final answer
    result = reduce_claims(query, mapped)
    result["mapped"] = mapped

    return result


# ── main test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pipeline import retrieve, embed_and_store, chunk_text, load_source
    import time

    # quick smoke test with one query
    # assumes pipeline has already been run and ChromaDB is populated
    test_query = "What do students say about job outcomes after graduating from Informatics?"

    print(f"Query: {test_query}\n")
    chunks = retrieve(test_query, k=6)

    if not chunks:
        print("No chunks retrieved — run pipeline.py first to populate the vector store.")
    else:
        result = generate_answer(test_query, chunks)

        print("ANSWER:")
        print(result["answer"])
        print("\nSOURCES:")
        for url in result["sources"]:
            print(f"  {url}")

        print("\nMAP STEP DEBUG:")
        for i, m in enumerate(result["mapped"]):
            status = "✓" if m["relevant"] else "✗"
            print(f"  [{status}] {m['source']}: {m['claim'][:100]}")